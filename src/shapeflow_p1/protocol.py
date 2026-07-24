"""What the campaign is approved to do, and how that is checked.

The protocol SHA is the digest of the **tracked protocol document**, computed here. It used to
come from an environment variable, which meant the launch gate compared a number the caller
supplied to a number the caller supplied -- a check that cannot fail. It is a fact about a file
in the repository, so it is read from that file.

The approval binds everything a result depends on. Pinning only the protocol, budget and
thresholds left the rest free to move under an approval that still verified: the variant
registry could gain an arm, the stack manifest could be replaced, the patch could change what
the graph does. All of it is in the digest now, so a change to any of them invalidates the
approval rather than silently running under it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .canonical import canonical_json
from .config import load_config
from .hashing import sha256_hex

__all__ = [
    "PROTOCOL_DOCUMENT",
    "ProtocolBinding",
    "ApprovalError",
    "protocol_sha",
    "compute_binding",
    "verify_approval_file",
    "write_approval_file",
]

PROTOCOL_DOCUMENT = "protocol/SHAPEFLOW_P1_WEEK1_CODING_PLAN_v0.1_2026-07-24.md"


class ApprovalError(RuntimeError):
    """The approval does not bind the configuration that is about to run."""


def protocol_sha(repo: Path) -> str:
    """Digest of the tracked protocol document.

    Read from the file, never from the environment. A SHA supplied by the caller and then
    compared against itself is a gate that cannot fail, and this one guards every threshold the
    verdict depends on.
    """
    path = Path(repo) / PROTOCOL_DOCUMENT
    if not path.exists():
        raise ApprovalError(
            f"{PROTOCOL_DOCUMENT} is not in the repository. The protocol of record must be "
            "tracked: an untracked document cannot be hashed, and a campaign whose thresholds "
            "are not pinned to a specific text is not pre-registered."
        )
    return sha256_hex(path.read_bytes())


@dataclass(frozen=True)
class ProtocolBinding:
    """Every input whose change would change what the campaign means."""

    protocol_sha: str
    decision_thresholds_sha: str
    budget_sha: str
    variants_sha: str
    stack_sha: str
    stack_manifest_sha: str
    # The campaign, acquisition, corpus and judge configs decide the arm set, the size and
    # composition of the frozen world, how many tasks exist and which model scores them. A
    # result depends on all four, so an approval that did not pin them would still verify while
    # describing a different experiment.
    week1_sha: str
    acquisition_sha: str
    task_source_sha: str
    judge_sha: str
    vendor_commit: str
    patched_tree_sha: str
    approved_commit: str

    def content(self) -> dict:
        return {
            "protocol_sha": self.protocol_sha,
            "decision_thresholds_sha": self.decision_thresholds_sha,
            "budget_sha": self.budget_sha,
            "variants_sha": self.variants_sha,
            "stack_sha": self.stack_sha,
            "stack_manifest_sha": self.stack_manifest_sha,
            "week1_sha": self.week1_sha,
            "acquisition_sha": self.acquisition_sha,
            "task_source_sha": self.task_source_sha,
            "judge_sha": self.judge_sha,
            "vendor_commit": self.vendor_commit,
            "patched_tree_sha": self.patched_tree_sha,
            "approved_commit": self.approved_commit,
        }

    @property
    def digest(self) -> str:
        return sha256_hex(canonical_json(self.content()))


def _sha_of(path: Path) -> str:
    return sha256_hex(path.read_bytes()) if path.exists() else ""


def compute_binding(repo: Path, *, approved_commit: str = "") -> ProtocolBinding:
    """Read the live configuration and produce the binding it implies."""
    repo = Path(repo)
    configs = repo / "configs"
    _, decision = load_config(configs / "decision.yaml")
    _, budget = load_config(configs / "budget_v1.yaml")
    _, variants = load_config(configs / "variants.yaml")
    _, stack = load_config(configs / "stack.yaml")
    _, week1 = load_config(configs / "week1.yaml")
    _, acquisition = load_config(configs / "acquisition.yaml")
    _, task_source = load_config(configs / "task_source.yaml")
    _, judge = load_config(configs / "judge.yaml")
    manifest = repo / "protocol" / "stack_manifest.json"
    return ProtocolBinding(
        protocol_sha=protocol_sha(repo),
        decision_thresholds_sha=decision,
        budget_sha=budget,
        variants_sha=variants,
        stack_sha=stack,
        stack_manifest_sha=_sha_of(manifest),
        week1_sha=week1,
        acquisition_sha=acquisition,
        task_source_sha=task_source,
        judge_sha=judge,
        vendor_commit=read_vendor_pin(repo),
        patched_tree_sha=(repo / "patches" / "patched_tree.sha256").read_text().strip()
        if (repo / "patches" / "patched_tree.sha256").exists() else "",
        approved_commit=approved_commit,
    )


class VendorCommitUnobservable(ApprovalError):
    """The vendor pin could not be read. Never a value, and never an empty string.

    An unreadable pin used to become ``""``, which was then written into the approval and
    compared against ``""`` -- so the one field that says which upstream this study forked
    verified successfully while saying nothing. The approval on disk carries exactly that.
    """


def _git(repo: Path, *args: str) -> "subprocess.CompletedProcess":
    import subprocess

    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30,
    )


def read_vendor_pin(repo: Path) -> str:
    """The vendor commit this repository pins, read from the parent's gitlink.

    Two reasons it comes from here rather than from ``git -C vendor/... rev-parse HEAD``:

    The gitlink is what the repository *pins*; the submodule's own HEAD is whatever it
    happens to be checked out at. Reading the latter means a submodule someone moved reports
    itself as the pin, and drift becomes undetectable by construction.

    And it is the same value under every identity. The submodule read went through git's
    ownership check on a directory the low-privilege roles do not own, so it returned the
    real commit for the owner and an empty string for everyone else -- which is why the
    approval was minted with an empty vendor pin and still verified.
    """
    import subprocess

    try:
        out = _git(repo, "ls-files", "-s", "--", "vendor/open_deep_research")
    except (OSError, subprocess.SubprocessError) as e:
        raise VendorCommitUnobservable(
            f"cannot run git to read the vendor pin: {type(e).__name__}: {e}"
        ) from e
    if out.returncode != 0:
        raise VendorCommitUnobservable(
            f"git could not read the vendor gitlink: {out.stderr.strip()[:200]}"
        )
    fields = out.stdout.split()
    if len(fields) < 2 or fields[0] != "160000":
        raise VendorCommitUnobservable(
            "vendor/open_deep_research is not a gitlink in this repository "
            f"(git said {out.stdout.strip()[:120]!r})"
        )
    return fields[1]


def read_head_commit(repo: Path) -> str:
    """The commit that is checked out right now."""
    import subprocess

    try:
        out = _git(repo, "rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError) as e:
        raise ApprovalError(
            f"cannot run git to read HEAD: {type(e).__name__}: {e}") from e
    if out.returncode != 0:
        raise ApprovalError(f"git could not read HEAD: {out.stderr.strip()[:200]}")
    return out.stdout.strip()


def approvals_dir(repo: Path) -> Path:
    return Path(repo) / "protocol" / "approvals"


def approval_chain(repo: Path) -> list[dict]:
    """Every approval ever recorded, oldest first.

    A chain rather than a file. One mutable path can only ever show the current answer, so
    the question "what was approved when this ran?" has no artifact behind it -- and the
    refusal-to-overwrite that guarded it is one `rm` away from being no guard at all.
    """
    index = approvals_dir(repo) / "INDEX.json"
    if not index.exists():
        return []
    try:
        body = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ApprovalError(f"the approval index is unreadable: {e}") from e
    return list(body.get("approvals") or [])


def _write_atomic(path: Path, body: dict) -> None:
    """Write through a temp file and rename, so a reader never sees half an approval."""
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_approval_file(
    repo: Path,
    *,
    approved_at_utc: str,
    approved_commit: str = "",
    approval_path: Optional[Path] = None,
) -> ProtocolBinding:
    """Record the approval for the configuration that is live right now.

    This does not *grant* an approval -- protocol v0.1 already fixed the mode, the budgets and
    the thresholds, and this only records their hashes so a later edit is detectable.

    It appends. A new configuration mints a new numbered file in ``protocol/approvals/`` and
    a new entry in its index; nothing is ever rewritten in place, so the approval a past run
    was checked against remains readable after the next one is recorded.
    """
    from .experiment.freeze import APPROVAL_MODE_AUTO

    repo = Path(repo)
    approval_path = approval_path or (repo / "protocol" / "launch_approval.json")
    binding = compute_binding(repo, approved_commit=approved_commit or read_head_commit(repo))
    chain = approval_chain(repo)
    if chain and chain[-1].get("binding_sha256") == binding.digest:
        return binding              # already recorded for exactly this configuration
    if approval_path.exists():
        existing = json.loads(approval_path.read_text(encoding="utf-8"))
        if existing.get("binding_sha256") == binding.digest:
            return binding
    body = {
        "approval_mode": APPROVAL_MODE_AUTO,
        "approval_source_date": "2026-07-24",
        "approved_at_utc": approved_at_utc,
        "approved_commit": binding.approved_commit,
        "binding": binding.content(),
        "binding_sha256": binding.digest,
        "claim_scope": "FORMATIVE_ONLY",
        "note": (
            "Protocol v0.1 auto-launch: the user's 2026-07-24 instruction to build and start "
            "running is the launch authorization. Every hard gate still fails closed. The "
            "corpus is FORMATIVE_MACHINE_AUTHORED and no result under this approval may be "
            "presented as confirmatory."
        ),
    }
    sequence = len(chain) + 1
    versioned = approvals_dir(repo) / f"{sequence:04d}-{binding.digest[:12]}.json"
    body["sequence"] = sequence
    _write_atomic(versioned, body)
    _write_atomic(approvals_dir(repo) / "INDEX.json", {
        "approvals": [
            *chain,
            {
                "sequence": sequence,
                "file": versioned.name,
                "binding_sha256": binding.digest,
                "approved_at_utc": approved_at_utc,
                "approved_commit": body["approved_commit"],
            },
        ],
    })
    # The single path stays as the one the launch gate reads, and is now a copy of the
    # newest link in the chain rather than the only record that it existed.
    _write_atomic(approval_path, body)
    return binding


def verify_approval_file(repo: Path, approval_path: Optional[Path] = None) -> ProtocolBinding:
    """Raise unless the approval on disk binds exactly the live configuration.

    Every mismatch names the field. A campaign that ran under a stale approval would have a
    pre-registration that does not describe it, which is indistinguishable from having none.
    """
    repo = Path(repo)
    approval_path = approval_path or (repo / "protocol" / "launch_approval.json")
    if not approval_path.exists():
        raise ApprovalError(f"{approval_path} missing; nothing authorises this campaign")
    try:
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ApprovalError(f"{approval_path} is not readable JSON: {e}") from e

    from .experiment.freeze import APPROVAL_MODES

    mode = approval.get("approval_mode")
    if mode not in APPROVAL_MODES:
        raise ApprovalError(
            f"approval_mode {mode!r} is not authorized by protocol v0.1 "
            f"(authorized: {sorted(APPROVAL_MODES)})"
        )

    # Read from git, not copied out of the file being checked. Seeding the live binding
    # from the approval and then excluding that field from the comparison made this a
    # self-check: it could not fail, and a campaign could run thirteen commits past the
    # tree its approval describes.
    live = compute_binding(repo, approved_commit=read_head_commit(repo))
    pinned = approval.get("binding") or {}
    mismatches = [
        f"  {key}: approval pins {pinned.get(key)!r}, live is {value!r}"
        for key, value in live.content().items()
        if pinned.get(key) != value
    ]
    if mismatches:
        raise ApprovalError(
            "the approval does not bind the live configuration:\n" + "\n".join(mismatches)
        )
    if approval.get("binding_sha256") != live.digest:
        raise ApprovalError(
            f"binding digest mismatch: approval {approval.get('binding_sha256')!r}, "
            f"live {live.digest!r}"
        )
    return live
