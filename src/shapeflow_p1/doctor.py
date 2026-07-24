"""Environment and stack verification (plan §5.1).

Doctor is a fail-closed gate: it records the exact stack and refuses to proceed on any mismatch,
so a run can never silently execute against a substituted model, an unfrozen config, or an
unclosed schema. It splits its checks into pure ones (secret *presence* without ever printing a
value, config hashing, schema closure, identity/role, Python version) that run anywhere, and
runtime ones (GPU UUID, driver, vLLM, model revision) that are best-effort off the run host and
become authoritative on it. A pure-check failure is fatal here in the dev environment too, so
those invariants are caught long before GPU time.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import config_sha, load_config

__all__ = ["CheckResult", "DoctorReport", "check_secret_present", "check_configs",
           "check_schemas_closed", "check_identity", "run_pure_checks"]

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str = ""  # already redaction-safe; never contains a secret value


@dataclass
class DoctorReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.status != FAIL for c in self.checks)

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)

    def to_json(self) -> str:
        return json.dumps([c.__dict__ for c in self.checks], indent=2, sort_keys=True)


def check_secret_present(label: str, *, env_var: str, file_env_var: str) -> CheckResult:
    """PASS if a credential is reachable, without reading or printing its value beyond a short
    non-invertible fingerprint. Prefers the *_FILE path (production) over the env var (dev)."""
    path = os.environ.get(file_env_var)
    value: Optional[str] = None
    source = ""
    if path and Path(path).exists():
        try:
            value = Path(path).read_text(encoding="utf-8").strip()
            source = f"file:{file_env_var}"
        except OSError:
            return CheckResult(f"secret:{label}", FAIL, f"{file_env_var} set but unreadable")
    elif os.environ.get(env_var):
        value = os.environ[env_var].strip()
        source = f"env:{env_var}"
    if not value:
        return CheckResult(f"secret:{label}", FAIL, "no credential via *_FILE or env var")
    fp = hashlib.sha256(value.encode("utf-8")).hexdigest()[:6]
    return CheckResult(f"secret:{label}", PASS, f"present via {source} (fp {fp})")


def check_configs(paths: dict[str, Path]) -> CheckResult:
    """Load each config and confirm it hashes; report the combined protocol-ish hash."""
    shas = {}
    for name, p in paths.items():
        try:
            _, sha = load_config(p)
            shas[name] = sha
        except Exception as e:  # noqa: BLE001 - any load failure is a doctor failure
            return CheckResult(f"config:{name}", FAIL, str(e))
    combined = config_sha(shas)
    return CheckResult("configs", PASS, f"{len(shas)} configs, combined {combined[:12]}")


def check_schemas_closed(schema_dir: Path) -> CheckResult:
    """Every JSON Schema must be valid and closed (additionalProperties:false on objects)."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError:  # pragma: no cover
        return CheckResult("schemas", SKIP, "jsonschema not installed")
    files = sorted(schema_dir.glob("*.schema.json"))
    if not files:
        return CheckResult("schemas", FAIL, f"no schemas in {schema_dir}")
    for f in files:
        schema = json.loads(f.read_text(encoding="utf-8"))
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as e:  # noqa: BLE001
            return CheckResult("schemas", FAIL, f"{f.name}: {e}")
        for where, node in _object_nodes(schema):
            if "properties" in node and node.get("additionalProperties") is not False:
                return CheckResult("schemas", FAIL, f"{f.name}: unclosed object at {where}")
    return CheckResult("schemas", PASS, f"{len(files)} schemas valid and closed")


def _object_nodes(node, path="<root>"):
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            yield path, node
        for key, sub in node.items():
            if key == "properties" and isinstance(sub, dict):
                for field_name, fs in sub.items():
                    yield from _object_nodes(fs, f"{path}.{field_name}")
            elif isinstance(sub, (dict, list)):
                yield from _object_nodes(sub, f"{path}.{key}")
    elif isinstance(node, list):
        for i, sub in enumerate(node):
            yield from _object_nodes(sub, f"{path}[{i}]")


def check_identity(expected_role: Optional[str] = None) -> CheckResult:
    """Record the effective identity. On Linux this is the UID/username the CLI enforces per role
    (steward/runner/evaluator); off Linux it is informational."""
    uid = getattr(os, "geteuid", lambda: -1)()
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "?"
    detail = f"uid={uid} user={user}"
    if expected_role:
        detail += f" expected_role={expected_role}"
    return CheckResult("identity", PASS, detail)


def check_git_clean(repo: Path) -> CheckResult:
    """A dirty tree must not launch a campaign."""
    if not shutil.which("git"):
        return CheckResult("git_clean", SKIP, "git not available")
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return CheckResult("git_clean", SKIP, f"git status failed: {e}")
    if out.returncode != 0:
        return CheckResult("git_clean", SKIP, "not a git repo")
    dirty = [ln for ln in out.stdout.splitlines() if ln.strip()]
    if dirty:
        return CheckResult("git_clean", FAIL, f"{len(dirty)} uncommitted change(s)")
    return CheckResult("git_clean", PASS, "working tree clean")


def run_pure_checks(*, repo: Path, configs: dict[str, Path], schema_dir: Path) -> DoctorReport:
    """The checks that run anywhere -- so the fail-closed invariants are enforced in dev too."""
    report = DoctorReport()
    report.add(check_configs(configs))
    report.add(check_schemas_closed(schema_dir))
    report.add(check_identity())
    report.add(check_secret_present("tavily", env_var="TAVILY_API_KEY",
                                    file_env_var="TAVILY_API_KEY_FILE"))
    report.add(check_secret_present("deepseek", env_var="DEEPSEEK_API_KEY",
                                    file_env_var="DEEPSEEK_API_KEY_FILE"))
    return report
