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
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .canonical import canonical_json
from .config import load_config
from .fsmode import chmod_shared
from .hashing import sha256_hex

__all__ = [
    "PROTOCOL_DOCUMENT",
    "ProtocolBinding",
    "ApprovalError",
    "default_approval_path",
    "protocol_sha",
    "compute_binding",
    "verify_approval_file",
    "verified_execution_binding",
    "write_approval_file",
]

#: The protocol of record. Freeze-1 is not deleted and not superseded -- it is a completed study
#: whose artifacts remain verifiable under the bindings they were produced with, reachable via
#: `--ran-under-binding`. Freeze-2 is what is currently being executed, so it is what the live
#: binding must hash: an approval naming the finished study would verify while describing a
#: different experiment.
PROTOCOL_DOCUMENT = "protocol/SHAPEFLOW_FREEZE_2.md"


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
    # Two prereg fields, not one, and deliberately so. Resolving a numeric slot during Phase 0
    # legitimately changes the document, so a single digest would invalidate the approval on
    # every fill. The template locks the *structure* and is stable across Phase 0; the frozen
    # document is empty until Phase 0 exit and immutable after. G0's first criterion is exactly
    # that the second is non-empty with no PENDING slots left.
    prereg_template_sha: str
    prereg_frozen_sha: str
    # The gate criteria and the contract documents decide what "passed" means and what the
    # broker is allowed to do. Both are empty until they exist, and binding them from the start
    # means the approval invalidates the moment they appear -- which is correct, because a
    # campaign with gates is not the campaign without them.
    gates_sha: str
    contracts_sha: str
    # What every arm retrieves, and the benchmark provenance it retrieves from.
    retrieval_freeze_sha: str
    bench_manifest_sha: str
    budget_sha: str
    variants_sha: str
    stack_sha: str
    stack_manifest_sha: str
    # The campaign, retrieval and judge configs decide the arm set, what every arm retrieves
    # and ranks, and which model scores the results. A result depends on all three, so an
    # approval that did not pin them would still verify while describing a different experiment.
    week1_sha: str
    retrieval_sha: str
    judge_sha: str
    vendor_commit: str
    patched_tree_sha: str
    approved_commit: str

    def content(self) -> dict:
        return {
            "protocol_sha": self.protocol_sha,
            "prereg_template_sha": self.prereg_template_sha,
            "prereg_frozen_sha": self.prereg_frozen_sha,
            "gates_sha": self.gates_sha,
            "contracts_sha": self.contracts_sha,
            "retrieval_freeze_sha": self.retrieval_freeze_sha,
            "bench_manifest_sha": self.bench_manifest_sha,
            "budget_sha": self.budget_sha,
            "variants_sha": self.variants_sha,
            "stack_sha": self.stack_sha,
            "stack_manifest_sha": self.stack_manifest_sha,
            "week1_sha": self.week1_sha,
            "retrieval_sha": self.retrieval_sha,
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


def _sha_of_tree(directory: Path, pattern: str) -> str:
    """A digest over every matching file in ``directory``, by relative name and bytes.

    An absent or empty directory hashes to the digest of an empty mapping rather than to the
    empty string, so "there are no gate files" is a *stated* fact that the approval binds. If it
    collapsed to "" the first gate file to appear would be indistinguishable from a file that had
    always been there.
    """
    directory = Path(directory)
    body = {
        str(path.relative_to(directory)): sha256_hex(path.read_bytes())
        for path in sorted(directory.rglob(pattern)) if path.is_file()
    } if directory.exists() else {}
    return sha256_hex(canonical_json(body))


def compute_binding(repo: Path, *, approved_commit: str = "") -> ProtocolBinding:
    """Read the live configuration and produce the binding it implies."""
    repo = Path(repo)
    configs = repo / "configs"
    _, prereg_template = load_config(configs / "prereg.yaml")
    _, budget = load_config(configs / "budget_v1.yaml")
    _, variants = load_config(configs / "variants.yaml")
    _, stack = load_config(configs / "stack.yaml")
    _, week1 = load_config(configs / "week1.yaml")
    _, retrieval = load_config(configs / "retrieval.yaml")
    _, judge = load_config(configs / "judge.yaml")
    manifest = repo / "protocol" / "stack_manifest.json"
    return ProtocolBinding(
        protocol_sha=protocol_sha(repo),
        prereg_template_sha=prereg_template,
        prereg_frozen_sha=_sha_of(repo / "protocol" / "prereg.lock.json"),
        gates_sha=_sha_of_tree(configs / "gates", "*.yaml"),
        contracts_sha=_sha_of_tree(repo / "src" / "shapeflow" / "contracts" / "docs", "*.md"),
        retrieval_freeze_sha=_sha_of(repo / "protocol" / "retrieval_freeze.json"),
        bench_manifest_sha=_sha_of(repo / "protocol" / "bench_manifest.json"),
        budget_sha=budget,
        variants_sha=variants,
        stack_sha=stack,
        stack_manifest_sha=_sha_of(manifest),
        week1_sha=week1,
        retrieval_sha=retrieval,
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


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
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


def default_approval_path(repo: Path) -> Path:
    """Resolve the mutable current-approval pointer outside the approved Git tree.

    An approval cannot be a tracked input to the commit it approves: writing it dirties that
    tree, while committing it changes HEAD and invalidates ``approved_commit``.  The host
    therefore supplies ``SHAPEFLOW_APPROVAL_FILE`` (or ``SHAPEFLOW_DATA_ROOT`` as a base).
    """
    explicit = os.environ.get("SHAPEFLOW_APPROVAL_FILE", "").strip()
    if explicit:
        candidate = Path(explicit).expanduser()
    else:
        data_root = os.environ.get("SHAPEFLOW_DATA_ROOT", "").strip()
        if not data_root:
            raise ApprovalError(
                "no external approval store configured; set SHAPEFLOW_APPROVAL_FILE or "
                "SHAPEFLOW_DATA_ROOT"
            )
        candidate = Path(data_root).expanduser() / "approvals" / "launch_approval.json"
    return _external_approval_path(repo, candidate)


def _external_approval_path(repo: Path, approval_path: Path) -> Path:
    repo_root = Path(repo).resolve()
    candidate = Path(approval_path).expanduser().resolve()
    if candidate == repo_root or repo_root in candidate.parents:
        raise ApprovalError(
            f"approval artifact {candidate} is inside the approved execution tree {repo_root}; "
            "that creates a self-referential approval"
        )
    return candidate


def approvals_dir(repo: Path, approval_path: Optional[Path] = None) -> Path:
    current = (
        _external_approval_path(repo, approval_path)
        if approval_path is not None
        else default_approval_path(repo)
    )
    return current.parent / "history"


def approval_chain(repo: Path, approval_path: Optional[Path] = None) -> list[dict]:
    """Every approval ever recorded, oldest first.

    A chain rather than a file. One mutable path can only ever show the current answer, so
    the question "what was approved when this ran?" has no artifact behind it -- and the
    refusal-to-overwrite that guarded it is one `rm` away from being no guard at all.
    """
    index = approvals_dir(repo, approval_path) / "INDEX.json"
    if not index.exists():
        return []
    try:
        body = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ApprovalError(f"the approval index is unreadable: {e}") from e
    return list(body.get("approvals") or [])


def approval_chain_digests(
    repo: Path, approval_path: Optional[Path] = None
) -> frozenset[str]:
    """Every execution binding this repository has ever approved.

    The evaluator needs it because a work key is namespaced by the binding and the binding moves
    with every commit, so grading a finished run from a tree that has advanced means naming the
    namespace the run actually wrote under. Answering that from the chain rather than from the
    caller is what keeps it an audit question instead of a free parameter.
    """
    return frozenset(
        str(link.get("binding_sha256"))
        for link in approval_chain(repo, approval_path)
        if link.get("binding_sha256")
    )


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
        # Widen from mkstemp's 0600 before the rename: the approval must never be visible at
        # its published path with an ACL mask that denies the roles that have to verify it.
        # ``smoke``, ``run-screen``, ``preflight`` and doctor's phase record all run as
        # sfrunner and call verified_execution_binding. See shapeflow.fsmode.
        chmod_shared(tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _write_new(path: Path, body: dict) -> None:
    """Create one immutable history link, accepting only an identical existing link."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(body, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        chmod_shared(path)
        return
    except FileExistsError:
        pass
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ApprovalError(f"immutable approval link {path} is unreadable: {exc}") from exc
    if existing != body:
        raise ApprovalError(f"immutable approval link {path} already contains different bytes")


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

    It appends. A new configuration mints a new numbered file in the configured external
    approval store's ``history/`` directory and a new entry in its index; nothing is ever
    rewritten in place, so the approval a past run was checked against remains readable after
    the next one is recorded.
    """
    from .experiment.freeze import APPROVAL_MODE_AUTO

    repo = Path(repo)
    approval_path = (
        _external_approval_path(repo, approval_path)
        if approval_path is not None
        else default_approval_path(repo)
    )
    _require_clean_execution_tree(repo)
    head = read_head_commit(repo)
    if approved_commit and approved_commit != head:
        raise ApprovalError(
            f"cannot approve commit {approved_commit!r} while clean execution HEAD is {head!r}")
    binding = compute_binding(repo, approved_commit=head)
    # Close the check/read/write race: if any approved byte or HEAD changed while the binding
    # was being computed, do not publish a credential for the mixed observation.
    _require_clean_execution_tree(repo)
    if read_head_commit(repo) != binding.approved_commit:
        raise ApprovalError("execution HEAD changed while the approval binding was computed")
    chain = approval_chain(repo, approval_path)
    if chain and chain[-1].get("binding_sha256") == binding.digest:
        link = approvals_dir(repo, approval_path) / str(chain[-1].get("file") or "")
        try:
            body = json.loads(link.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ApprovalError(
                f"latest immutable approval link {link} is unreadable: {exc}") from exc
        if body.get("binding_sha256") != binding.digest:
            raise ApprovalError(f"latest immutable approval link {link} does not verify")
        _write_atomic(approval_path, body)
        return binding              # already recorded for exactly this configuration
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
    history = approvals_dir(repo, approval_path)
    versioned = history / f"{sequence:04d}-{binding.digest[:12]}.json"
    body["sequence"] = sequence
    _write_new(versioned, body)
    _write_atomic(history / "INDEX.json", {
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
    approval_path = (
        _external_approval_path(repo, approval_path)
        if approval_path is not None
        else default_approval_path(repo)
    )
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


def _require_clean_execution_tree(repo: Path) -> None:
    status = _git(Path(repo), "status", "--porcelain", "--untracked-files=all")
    if status.returncode != 0:
        raise ApprovalError(
            f"cannot establish a clean execution tree: {status.stderr.strip()[:200]}")
    if status.stdout.strip():
        first = status.stdout.splitlines()[0]
        raise ApprovalError(
            "the execution tree has tracked or untracked changes after its approved commit "
            f"(first entry: {first!r})"
        )


def verified_execution_binding(
    repo: Path,
    *,
    expected_digest: Optional[str] = None,
    approval_path: Optional[Path] = None,
) -> ProtocolBinding:
    """Return the live, approved execution identity and optionally bind a caller claim.

    ``protocol_sha`` names only the protocol document.  It deliberately does not change when
    an arm, stack, corpus, judge, patch, or approved commit changes.  Campaign work therefore
    uses :attr:`ProtocolBinding.digest` as its namespace.  The optional expected digest is the
    resume/launcher claim; it is compared with the independently re-derived approval binding,
    never with another caller-supplied value.  A clean tree is part of that claim: otherwise
    ``approved_commit`` would name HEAD while the interpreter executes different source bytes.
    """
    _require_clean_execution_tree(Path(repo))
    binding = verify_approval_file(repo, approval_path)
    if expected_digest is not None and expected_digest != binding.digest:
        raise ApprovalError(
            f"execution binding mismatch: caller supplied {expected_digest!r}, "
            f"approved live binding is {binding.digest!r}"
        )
    return binding
