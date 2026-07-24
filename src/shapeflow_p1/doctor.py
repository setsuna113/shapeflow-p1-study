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
        """Green means every check PASSed. A SKIP is not a pass.

        Counting SKIP as success is how ``doctor: ok`` was obtained on a machine with no GPU,
        no vLLM and no model -- every runtime check skipped, and the absence of evidence read
        as evidence of correctness. A check that could not run has not been satisfied.
        """
        return bool(self.checks) and all(c.status == PASS for c in self.checks)

    @property
    def skipped(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == SKIP]

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


# Role -> the account that role must run as. The campaign never runs as root: the plan allows
# root only for one-time unit/ACL installation (section 5.2).
ROLE_USERS = {
    "provider": "sfprovider",
    "runner": "sfrunner",
    "infer": "sfinfer",
    "steward": "sfsteward",
    "evaluator": "sfevaluator",
}


def check_identity(expected_role: Optional[str] = None) -> CheckResult:
    """Verify the effective identity against the role this process claims.

    This used to return PASS unconditionally -- it recorded a uid and asserted nothing, so the
    separation between provider (holds credentials), runner (never does) and evaluator (holds
    the answer key) was documented but not enforced. With no expected role it now SKIPs, which
    no longer counts as green.
    """
    uid = getattr(os, "geteuid", lambda: -1)()
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "?"
    if expected_role is None:
        return CheckResult("identity", SKIP, f"uid={uid} user={user}; no role asserted")
    want = ROLE_USERS.get(expected_role)
    if want is None:
        return CheckResult("identity", FAIL, f"unknown role {expected_role!r}")
    if uid == 0:
        return CheckResult(
            "identity", FAIL,
            f"running as root; role {expected_role!r} must run as {want} (root is permitted "
            "only for one-time unit/ACL installation)",
        )
    if user != want:
        return CheckResult(
            "identity", FAIL, f"role {expected_role!r} must run as {want}, but user={user}"
        )
    return CheckResult("identity", PASS, f"uid={uid} user={user} role={expected_role}")


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


def check_stack_manifest(repo: Path, stack_config: Path) -> CheckResult:
    """Compare the live stack against a manifest a steward froze BEFORE approval.

    Read-only, always. A doctor that wrote the manifest itself at launch would take whatever it
    happened to observe -- including an already-drifted engine, driver or model -- and launder
    it into a legitimate baseline, which inverts the purpose of the check.

    **Unconditionally FAIL until the live verifier exists.** Passing on "a JSON file is present
    with the right keys" is a gate in name only: a hand-written manifest with three arbitrary
    strings satisfied it. Until this compares the manifest's digest, its schema, and each live
    value (GPU UUID, driver, model Merkle root, vLLM package tree, attention backend, unit
    flags) against the running host, the honest status is that the check has not been built --
    not that the stack is fine. A gate that cannot fail is not a gate.
    """
    manifest_path = repo / "protocol" / "stack_manifest.json"
    declared, _ = load_config(stack_config)
    unresolved = [
        f"{section}.{key}"
        for section, block in declared.items()
        if isinstance(block, dict)
        for key, value in block.items()
        if isinstance(value, str) and value.startswith("@")
    ]
    if not manifest_path.exists():
        return CheckResult(
            "stack_manifest", FAIL,
            f"protocol/stack_manifest.json missing; {len(unresolved)} field(s) unresolved. "
            "Run `shapeflow-p1 freeze-stack` as the steward before approval -- doctor will "
            "not freeze it at launch.",
        )

    from .ops.live_stack import attention_backend_from_log, compare, observe

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return CheckResult("stack_manifest", FAIL, f"manifest unreadable: {e}")

    engine_pid = _engine_pid()
    observation = observe(
        gpu_uuid=str(declared.get("host", {}).get("gpu_uuid", "")),
        model_dir=Path(str(declared.get("model", {}).get("path", ""))),
        vllm_python=Path(str(declared.get("engine", {}).get("vllm_venv", ""))) / "bin" / "python",
        engine_pid=engine_pid,
    )
    # The attention backend is only knowable from a served engine: vLLM picks it at startup and
    # records it in its own log. Read the same way the freeze read it, so doctor verifies the
    # engine running *now* chose the backend the manifest froze -- an engine restarted onto a
    # different backend is caught here.
    engine_log = _engine_log(repo)
    if engine_log:
        backend = attention_backend_from_log(engine_log)
        if backend:
            observation.values["attention_backend"] = backend
    layer = declared.get("isolation", {}).get("causal", {})
    expected_flags = [
        f"--max-num-seqs", str(layer.get("max_num_seqs", 1)),
    ] if engine_pid else []
    if engine_pid and layer.get("enable_prefix_caching") is False:
        expected_flags.append("--no-enable-prefix-caching")

    problems = compare(manifest, declared, observation,
                       expected_engine_flags=expected_flags)
    if problems:
        return CheckResult("stack_manifest", FAIL,
                           f"{len(problems)} mismatch(es): " + "; ".join(problems[:4]))
    return CheckResult(
        "stack_manifest", PASS,
        f"manifest {manifest.get('manifest_sha256', '')[:12]} matches the live stack"
        + (f" (engine pid {engine_pid})" if engine_pid else " (engine not running)"),
    )


def _engine_pid() -> Optional[int]:
    """The pid serving the causal engine, if one is running.

    Read from the environment the unit sets rather than guessed from a process name: guessing
    could match a foreign vLLM on a shared host, and this study never touches a process that is
    not its own.
    """
    raw = os.environ.get("SHAPEFLOW_ENGINE_PID", "").strip()
    if not raw.isdigit():
        return None
    pid = int(raw)
    return pid if Path(f"/proc/{pid}").exists() else None


def _engine_log(repo: Path) -> Optional[Path]:
    """The causal engine's own startup log, for the attention backend it chose.

    Unlike the pid, a log *path* under our own repo is safe to default: it names a file this
    study writes, never a foreign process, so there is nothing to accidentally match. The
    supervisor writes logs/<name>.log, and the causal engine's name is vllm-causal;
    SHAPEFLOW_ENGINE_LOG overrides it. Returned only if it exists, so a missing log leaves the
    backend unobservable and the stack check fails closed rather than recording a guess.
    """
    raw = os.environ.get("SHAPEFLOW_ENGINE_LOG", "").strip()
    candidate = Path(raw) if raw else Path(repo) / "logs" / "vllm-causal.log"
    return candidate if candidate.exists() else None


def run_pure_checks(
    *, repo: Path, configs: dict[str, Path], schema_dir: Path, role: Optional[str] = None
) -> DoctorReport:
    """The checks that run anywhere -- so the fail-closed invariants are enforced in dev too."""
    report = DoctorReport()
    report.add(check_configs(configs))
    report.add(check_schemas_closed(schema_dir))
    report.add(check_identity(role))
    report.add(check_secret_present("tavily", env_var="TAVILY_API_KEY",
                                    file_env_var="TAVILY_API_KEY_FILE"))
    report.add(check_secret_present("deepseek", env_var="DEEPSEEK_API_KEY",
                                    file_env_var="DEEPSEEK_API_KEY_FILE"))
    return report
