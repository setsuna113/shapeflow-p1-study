"""Observing the live stack, freezing it once, and comparing against it read-only.

The two halves are deliberately different programs.

:func:`freeze_stack` is run **by the steward, before approval**. It resolves every
``@STEWARD_FREEZES@`` field into ``protocol/stack_manifest.json`` and that file then enters the
approval binding.

:func:`compare` is what doctor runs at launch, and it only ever *reads*. A doctor that wrote the
manifest itself would take whatever it happened to observe -- including an already-drifted
driver, an engine started with the wrong flags, or a different model revision -- and launder it
into a legitimate baseline. That inverts the purpose of the check, so the writer and the checker
are separate commands run by separate identities.

What is observed is chosen so a substitution cannot hide:

- the GPU by **UUID**, never index, because indices renumber;
- the driver and the runtime CUDA separately from ``nvidia-smi``'s "CUDA Version", which is the
  driver's maximum supported capability and legitimately differs;
- vLLM's distribution version *and* its ``__version__`` *and* the hash of its installed tree,
  because a cu129 wheel and the upstream release report different strings and comparing against
  either alone silently accepts the other;
- the model's revision and a Merkle root over the weight, config and tokenizer files actually
  loaded, so a swapped shard is visible even at an unchanged revision;
- the engine's serving flags from the running process's own command line, because the flags that
  matter (``--max-num-seqs 1``, prefix caching off) are what makes the causal layer causal, and a
  config file saying so proves nothing about the process that is actually serving.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from ..canonical import canonical_json
from ..hashing import sha256_hex

__all__ = ["StackObservation", "observe", "freeze_stack", "compare", "StackError",
           "engine_flags_of", "model_merkle_root"]

PLACEHOLDER = "@STEWARD_FREEZES@"


class StackError(RuntimeError):
    pass


@dataclass
class StackObservation:
    values: dict = field(default_factory=dict)
    unavailable: list = field(default_factory=list)

    def get(self, key: str) -> Optional[str]:
        return self.values.get(key)


def _run(cmd: Sequence[str], *, timeout: float = 30.0) -> Optional[str]:
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def model_merkle_root(model_dir: Path) -> str:
    """A digest over the files the engine actually loads, in a fixed order.

    Weights, config and tokenizer only: a README changing must not invalidate a stack, and a
    weight shard changing must not be invisible.
    """
    model_dir = Path(model_dir)
    suffixes = (".safetensors", ".json", ".txt", ".model")
    leaves = []
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if path.name.lower().startswith("readme"):
            continue
        digest = sha256_hex(path.read_bytes()) if path.stat().st_size < 64 * 1024 * 1024 \
            else _stream_sha256(path)
        leaves.append(sha256_hex(canonical_json({
            "name": str(path.relative_to(model_dir)), "sha256": digest,
        })))
    from ..hashing import merkle_root

    return merkle_root(leaves)


def _stream_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def engine_flags_of(pid: int) -> list[str]:
    """The serving flags of a running engine, from its own command line.

    Read from the process rather than from a config file: the flags that decide whether the
    causal layer is causal are the ones the serving process was actually started with.
    """
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError as e:
        raise StackError(f"cannot read the command line of pid {pid}: {e}") from e
    return [part for part in raw.decode("utf-8", "replace").split("\0") if part]


def _gpu_rows() -> list[dict]:
    out = _run(["nvidia-smi",
                "--query-gpu=uuid,name,driver_version,memory.total",
                "--format=csv,noheader"])
    if not out:
        return []
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            rows.append({"uuid": parts[0], "name": parts[1], "driver_version": parts[2],
                         "memory_total": parts[3]})
    return rows


def observe(
    *,
    gpu_uuid: str = "",
    model_dir: Optional[Path] = None,
    vllm_python: Optional[Path] = None,
    engine_pid: Optional[int] = None,
) -> StackObservation:
    """Read the live stack. Missing values are recorded as unavailable, never guessed."""
    observation = StackObservation()

    rows = _gpu_rows()
    if not rows:
        observation.unavailable.append("gpu")
    else:
        chosen = next((r for r in rows if r["uuid"] == gpu_uuid), rows[0])
        observation.values["gpu_uuid"] = chosen["uuid"]
        observation.values["gpu_name"] = chosen["name"]
        observation.values["driver_version"] = chosen["driver_version"]
        observation.values["gpu_count"] = str(len(rows))

    capability = _run(["nvidia-smi", "--query", "--display=COMPUTE"])
    if capability is None:
        observation.unavailable.append("driver_cuda_capability")

    if model_dir and Path(model_dir).exists():
        observation.values["model_artifact_merkle_root"] = model_merkle_root(Path(model_dir))
        revision = _hf_revision(Path(model_dir))
        if revision:
            observation.values["model_revision"] = revision
        else:
            observation.unavailable.append("model_revision")
    else:
        observation.unavailable.append("model")

    if vllm_python and Path(vllm_python).exists():
        probe = _run([str(vllm_python), "-c", _VLLM_PROBE], timeout=180)
        if probe:
            try:
                observation.values.update(json.loads(probe))
            except json.JSONDecodeError:
                observation.unavailable.append("vllm_probe")
        else:
            observation.unavailable.append("vllm_probe")
    else:
        observation.unavailable.append("vllm")

    if engine_pid:
        try:
            observation.values["engine_flags"] = " ".join(engine_flags_of(engine_pid))
        except StackError:
            observation.unavailable.append("engine_flags")

    observation.values["python"] = ".".join(
        str(p) for p in __import__("sys").version_info[:2])
    observation.values["timezone"] = os.environ.get("TZ", "")
    observation.values["pythonhashseed"] = os.environ.get("PYTHONHASHSEED", "")
    return observation


_VLLM_PROBE = """
import json, hashlib, pathlib, sys
out = {}
try:
    import vllm
    out["vllm_dunder_version"] = getattr(vllm, "__version__", "")
    root = pathlib.Path(vllm.__file__).parent
    leaves = []
    for p in sorted(root.rglob("*.py")):
        leaves.append(hashlib.sha256(p.read_bytes()).hexdigest())
    h = hashlib.sha256()
    for leaf in leaves:
        h.update(bytes.fromhex(leaf))
    out["vllm_package_tree_sha256"] = h.hexdigest()
except Exception as e:
    out["vllm_error"] = type(e).__name__
try:
    from importlib.metadata import version
    out["vllm_distribution_version"] = version("vllm")
except Exception:
    pass
try:
    import torch
    out["torch_version"] = torch.__version__
    out["torch_compiled_cuda"] = torch.version.cuda or ""
except Exception as e:
    out["torch_error"] = type(e).__name__
print(json.dumps(out, sort_keys=True))
"""


def attention_backend_from_log(log_path: Path) -> str:
    """Read the attention backend the engine actually chose, from its own startup log.

    Not from an environment variable and not from a default: vLLM selects a backend at startup
    based on the hardware and the build, and which one it chose is part of what makes two runs
    comparable. An unreadable log yields an empty string, and the freeze then refuses rather
    than recording a plausible guess.

    Returns the normalized backend *name* (``FLASH_ATTN``, ``FLASHINFER``, ...), not the raw log
    line: the line carries a pid and a timestamp that change on every restart, and a manifest
    field that moved every restart could never match.

    Reads the **last** matching line, not the first. The supervisor appends to one log across
    restarts, so the file accumulates a "Using X attention backend" line per launch; the backend
    that matters is the one the engine running *now* chose. Returning the first line would pin the
    answer to the oldest launch, which is exactly the drift the doctor check exists to catch.
    """
    import re

    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    named = re.compile(r"Using\s+([A-Z][A-Z0-9_]+)\s+attention backend", re.IGNORECASE)
    found = ""
    for line in text.splitlines():
        match = named.search(line)
        if match:
            found = match.group(1).upper()
    if found:
        return found
    # Fall back to a recognised token if the exact phrasing differs across vLLM versions.
    tokens = ("FLASH_ATTN", "FLASHINFER", "XFORMERS", "FLASHMLA", "TRITON_ATTN", "TORCH_SDPA")
    for line in text.splitlines():
        for token in tokens:
            if token.lower() in line.lower():
                return token
    return ""


def _hf_revision(model_dir: Path) -> str:
    """Recover the commit the local snapshot came from, from HF download metadata.

    Each ``.metadata`` file's *first* line is the repository commit; the second is that file's
    own blob etag. Only the first line is read, and every file must agree -- a snapshot whose
    files came from two different commits is not one revision, and returning either of them
    would pin the stack to something that never existed as a whole.
    """
    meta_dir = model_dir / ".cache" / "huggingface" / "download"
    revisions = set()
    if meta_dir.exists():
        for path in meta_dir.rglob("*.metadata"):
            first = path.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
            token = first[0].strip() if first else ""
            if len(token) == 40 and all(c in "0123456789abcdef" for c in token):
                revisions.add(token)
    if len(revisions) == 1:
        return revisions.pop()
    return ""


def freeze_stack(repo: Path, declared: dict, observation: StackObservation,
                 *, frozen_at_utc: str, extra: Optional[dict] = None) -> dict:
    """Resolve every placeholder into a manifest. Steward-only, before approval, write-once."""
    unresolved = [
        f"{section}.{key}"
        for section, block in declared.items() if isinstance(block, dict)
        for key, value in block.items()
        if isinstance(value, str) and value.startswith("@")
    ]
    resolved: dict = {}
    missing: list[str] = []
    for field_path in unresolved:
        _, _, key = field_path.partition(".")
        observed = observation.get(_OBSERVED_KEY.get(key, key))
        if observed is None:
            missing.append(field_path)
        else:
            resolved[field_path] = observed
    if missing:
        raise StackError(
            f"cannot freeze: {missing} are not observable on this host. A manifest with an "
            "unresolved field is not a baseline, and doctor must not fill it in later."
        )

    body = {
        "frozen_at_utc": frozen_at_utc,
        "resolved": dict(sorted(resolved.items())),
        "observed": dict(sorted(observation.values.items())),
        "unavailable": sorted(observation.unavailable),
        "declared_sha256": sha256_hex(canonical_json(declared)),
    }
    if extra:
        body["extra"] = dict(sorted(extra.items()))
    body["manifest_sha256"] = sha256_hex(canonical_json(body))

    path = Path(repo) / "protocol" / "stack_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("manifest_sha256") != body["manifest_sha256"]:
            raise StackError(
                f"{path} already froze a different stack "
                f"({existing.get('manifest_sha256')!r} vs {body['manifest_sha256']!r}); a "
                "changed stack is a new protocol version, not an overwritten manifest"
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return body


#: Manifest field -> the observation key that answers it.
_OBSERVED_KEY = {
    "artifact_merkle_root": "model_artifact_merkle_root",
    "vllm_package_tree_sha256": "vllm_package_tree_sha256",
    "attention_backend": "attention_backend",
}


def compare(manifest: dict, declared: dict, observation: StackObservation,
            *, expected_engine_flags: Sequence[str] = ()) -> list[str]:
    """Read-only comparison. Returns the mismatches, naming each field.

    Never writes, never fills in a missing value, and treats an unobservable field as a
    mismatch rather than a pass: a check that could not run has not been satisfied.
    """
    problems: list[str] = []

    if sha256_hex(canonical_json(declared)) != manifest.get("declared_sha256"):
        problems.append(
            "configs/stack.yaml changed since the manifest was frozen; the approval no longer "
            "describes this configuration"
        )

    frozen = manifest.get("resolved", {})
    for field_path, expected in sorted(frozen.items()):
        _, _, key = field_path.partition(".")
        actual = observation.get(_OBSERVED_KEY.get(key, key))
        if actual is None:
            problems.append(f"{field_path}: not observable on this host (frozen {expected[:16]})")
        elif actual != expected:
            problems.append(f"{field_path}: live {actual[:24]!r} != frozen {expected[:24]!r}")

    for section, block in sorted(declared.items()):
        if not isinstance(block, dict):
            continue
        for key, value in sorted(block.items()):
            if not isinstance(value, str) or value.startswith("@"):
                continue
            observed = observation.get(_LIVE_FIELDS.get(f"{section}.{key}", ""))
            if observed is None:
                continue
            if observed != value:
                problems.append(f"{section}.{key}: live {observed!r} != declared {value!r}")

    flags = observation.get("engine_flags")
    if expected_engine_flags:
        if flags is None:
            problems.append(
                "engine flags not observable: the flags that make the causal layer causal must "
                "be read from the serving process, not assumed from a config"
            )
        else:
            for flag in expected_engine_flags:
                if flag not in flags:
                    problems.append(f"engine is not serving with {flag!r}: {flags[:200]}")
    return problems


#: declared field -> observation key, for the values that can be checked live.
_LIVE_FIELDS = {
    "host.gpu_uuid": "gpu_uuid",
    "host.gpu_name": "gpu_name",
    "host.driver_version": "driver_version",
    "host.python": "python",
    "model.revision": "model_revision",
    "engine.vllm_dunder_version": "vllm_dunder_version",
    "engine.vllm_distribution_version": "vllm_distribution_version",
    "engine.torch_version": "torch_version",
    "engine.torch_compiled_cuda": "torch_compiled_cuda",
}
