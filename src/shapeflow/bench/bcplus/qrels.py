"""The BrowseComp-Plus answer key. **EVALUATOR-ONLY -- importing this from the treatment path
is the failure this module exists to prevent.**

Everything here is oracle material: which documents are gold, which are evidence, which are hard
negatives, and what the short answer is. AGENTS.md §2 states the rule this module is the concrete
instance of -- a selector, aggregator, preflight, broker feature or predictor feature that could
read any of it would be scored on its ability to read the answer key, and the resulting number
would look excellent and mean nothing. There is no legitimate reason for a module under
``shapeflow.p1``, ``shapeflow.strategies``, ``shapeflow.broker`` or ``shapeflow.odr`` to import
this file, and the firewall check keys on :data:`EVALUATOR_ONLY` to say so statically.

The static check is the primary enforcement. :func:`assert_evaluator_process` is the second
layer, for the one case a static import graph cannot see: a treatment process reaching this
module through an indirection. It runs at import.

Loading is fail-closed on every malformed row rather than skipping it. That is the single most
important property of this file: a qrels file that silently lost half its rows leaves fewer
documents marked relevant, so every recall number computed against it comes out *higher*, and
nothing downstream can tell the difference between "the run retrieved the right documents" and
"the answer key was truncated". Row counts are checked against the frozen vintage for the same
reason.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

from ...canonical import canonical_json
from ...hashing import sha256_hex

__all__ = [
    "EVALUATOR_ONLY",
    "LeakageFirewallError",
    "TREATMENT_IDENTITIES",
    "assert_evaluator_process",
    "QrelsError",
    "UnjudgedQuery",
    "UnknownQuery",
    "QrelKind",
    "Qrels",
    "EvaluatorQueryView",
    "EvaluatorQuerySet",
    "BCPLUS_QREL_FILES",
    "BCPLUS_QUERY_COUNT",
    "BCPLUS_EVALUATOR_QUERY_FILE",
    "load_qrels",
    "load_bcplus_qrels",
    "load_evaluator_queries",
    "load_bcplus_evaluator_queries",
    "assert_docsets_agree",
]

#: Firewall marker. The static leakage check keys on this module-level name: any module on the
#: treatment path that (transitively) imports a module declaring it is a firewall breach. It is a
#: constant rather than a comment so the check is a fact about the code and not about who read it.
EVALUATOR_ONLY = True

#: The service accounts that run treatment work, and the short role names the launcher and
#: :mod:`shapeflow.doctor` use for the same accounts. Separation is by uid on the run host (see
#: ``tests/integration/test_uid_isolation.py``); this is the same boundary expressed in-process.
#:
#: All three of these are treatment, not just the researcher process. ``sfinfer`` hosts vLLM and
#: therefore the broker extension, which is the component AGENTS.md §2 names first; ``sfprovider``
#: runs the search backend. Listing only ``sfrunner`` would leave the two processes that actually
#: make the admission and retrieval decisions free to load the answer key. Both spellings are
#: listed because ``sfsupervise.sh <role>`` and ``doctor.ROLE_USERS`` disagree about which one a
#: role is called, and a guard that only recognises one of the two is off by a string.
TREATMENT_IDENTITIES = frozenset({
    "sfrunner", "runner",
    "sfinfer", "infer",
    "sfprovider", "provider",
})


class LeakageFirewallError(RuntimeError):
    """Evaluator-only material was reached from a process that must not see it."""


class QrelsError(RuntimeError):
    """The answer key cannot be read as the benchmark specifies it."""


class UnjudgedQuery(KeyError):
    """A query has no judgements at all.

    Distinct from "judged, nothing relevant" on purpose. Collapsing the two is how an unjudged
    query silently becomes a recall of 0.0 (or of 1.0, depending on which way the arithmetic
    falls), and either way it enters the mean as if it were evidence.
    """


class UnknownQuery(KeyError):
    """A query id that is not in the evaluator record set. Never a default record."""


class QrelKind(str, Enum):
    """Which of the benchmark's two qrels files a :class:`Qrels` came from.

    Carried on the object because gold recall and evidence recall are different endpoints with
    different denominators, and a mislabelled set would be compared against the wrong one.
    """

    GOLD = "gold"
    EVIDENCE = "evidence"


#: The frozen vintage: filename -> (kind, exact row count). The counts are part of the data
#: freeze, so a file that no longer has them is a different artifact and stops the run rather
#: than quietly rescaling every recall denominator.
BCPLUS_QREL_FILES: Mapping[str, tuple[QrelKind, int]] = {
    "qrel_golds.txt": (QrelKind.GOLD, 2407),
    "qrel_evidence.txt": (QrelKind.EVIDENCE, 5064),
}

#: BrowseComp-Plus ships 830 queries. Same reasoning as the row counts above.
BCPLUS_QUERY_COUNT = 830

#: The evaluator record file inside the benchmark's data directory. Named here so
#: :func:`load_bcplus_evaluator_queries` is the entry point that pins both the filename and the
#: frozen query count, exactly as :func:`load_bcplus_qrels` pins the two qrels files.
BCPLUS_EVALUATOR_QUERY_FILE = "browsecomp_plus_decrypted.jsonl"

#: TREC's second column is the iteration marker and carries no information, but requiring it to
#: be one of these catches the failure it is worth catching: a file whose columns are in some
#: other order still parses as four whitespace-separated fields, and would then be read with
#: docids in the qid position.
_ITERATION_MARKERS = frozenset({"q0", "0"})


def _declared_role() -> str:
    """The role this process *claims*, from the environment. A claim, not a fact."""
    return (os.environ.get("SHAPEFLOW_ROLE") or "").strip()


def _account_name() -> str:
    """The OS account this process actually runs as, or "" if it cannot be resolved."""
    try:
        import pwd

        return pwd.getpwuid(os.geteuid()).pw_name
    except Exception:  # noqa: BLE001 -- an unresolvable uid is handled by the caller's policy
        return (os.environ.get("USER") or os.environ.get("LOGNAME") or "").strip()


def _process_identity() -> str:
    """Best available name for the identity this process runs as."""
    return _declared_role() or _account_name()


def assert_evaluator_process(identity: Optional[str] = None) -> str:
    """Refuse to be imported by a treatment identity.

    The declared role and the OS account are *both* checked, and either one naming a treatment
    identity raises. An earlier version let ``SHAPEFLOW_ROLE`` shadow the account: exporting any
    unrecognised string in the runner's environment then made this guard pass while the process
    was still uid ``sfrunner``. A guard whose only input is a variable the guarded process sets
    for itself is not a guard, and the failure is silent -- the answer key loads and nothing says
    so.

    Deliberately *not* fail-closed on an unresolvable identity, which is the one place in this
    file that rule is not applied. The authoritative firewall is the static import check; this
    guard exists only to catch a treatment process that reached the module dynamically, where the
    static check cannot see it. Refusing to import on a host whose uid is missing from
    ``/etc/passwd`` would take the evaluator offline without catching anything -- it would trade a
    real outage for no additional protection. An identity that positively *is* a treatment
    account is unambiguous, and that is what raises.
    """
    if identity is None:
        candidates = (("declared role", _declared_role()), ("account", _account_name()))
    else:
        candidates = (("identity", identity),)
    for source, who in candidates:
        if who in TREATMENT_IDENTITIES:
            raise LeakageFirewallError(
                f"{source} {who!r} is a treatment identity and must never load the "
                "BrowseComp-Plus answer key: qrels, gold answers and the negative sets are "
                "evaluator-only (AGENTS.md §2). If a treatment component needs this, the "
                "component is wrong, not the firewall."
            )
    return _process_identity() if identity is None else identity


assert_evaluator_process()


@dataclass(frozen=True)
class Qrels:
    """qid -> the docids judged relevant, plus everything needed to attribute the number.

    ``judged`` holds every docid that appears with *any* relevance grade, ``relevant_by_query``
    only those with a positive grade. Both are needed: a query judged with zero relevant
    documents is a real state in TREC data and is not the same state as an unjudged query.
    """

    kind: QrelKind
    source_name: str
    source_sha256: str
    n_rows: int
    relevant_by_query: Mapping[str, frozenset[str]]
    judged_by_query: Mapping[str, frozenset[str]]

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[tuple[str, str, int]],
        *,
        kind: QrelKind,
        source_name: str,
        source_sha256: str = "",
    ) -> Qrels:
        """Build from ``(qid, docid, relevance)`` triples, rejecting duplicate judgements.

        Used by :func:`load_qrels` and directly by tests, so the type under test is the type the
        run uses rather than a stand-in that could disagree with it.
        """
        relevant: dict[str, set[str]] = {}
        judged: dict[str, set[str]] = {}
        seen: dict[tuple[str, str], int] = {}
        n_rows = 0
        for qid, docid, rel in rows:
            n_rows += 1
            key = (qid, docid)
            if key in seen:
                raise QrelsError(
                    f"{source_name}: ({qid}, {docid}) is judged twice "
                    f"(grades {seen[key]} and {rel}). A repeated judgement means the file was "
                    "concatenated or edited; its row count no longer identifies its vintage and "
                    "one of the two grades is being silently discarded."
                )
            seen[key] = rel
            judged.setdefault(qid, set()).add(docid)
            if rel > 0:
                relevant.setdefault(qid, set()).add(docid)
        if not n_rows:
            raise QrelsError(
                f"{source_name}: no judgements at all. An empty answer key is never a valid "
                "answer key -- every query would read as unjudged and every recall denominator "
                "would vanish."
            )
        # A query with rows but no positive grade must still be present, so recall can report
        # NO_RELEVANT rather than falling through to "this query was never judged".
        for qid in judged:
            relevant.setdefault(qid, set())
        return cls(
            kind=kind,
            source_name=source_name,
            source_sha256=source_sha256,
            n_rows=n_rows,
            relevant_by_query={q: frozenset(d) for q, d in sorted(relevant.items())},
            judged_by_query={q: frozenset(d) for q, d in sorted(judged.items())},
        )

    def is_judged(self, query_id: str) -> bool:
        return query_id in self.judged_by_query

    def relevant(self, query_id: str) -> frozenset[str]:
        """The relevant docids, or raise. Never an empty set standing in for "no judgements"."""
        try:
            return self.relevant_by_query[query_id]
        except KeyError:
            raise UnjudgedQuery(
                f"{query_id!r} has no rows in {self.source_name}; it is unjudged, which is not "
                "the same as having no relevant documents"
            ) from None

    def query_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.judged_by_query))

    def counts(self) -> dict:
        """The shape of the answer key, for the manifest and for a human to sanity-check."""
        per_query = [len(self.relevant_by_query[q]) for q in sorted(self.relevant_by_query)]
        n_relevant = sum(per_query)
        return {
            "kind": self.kind.value,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "n_rows": self.n_rows,
            "n_queries": len(self.judged_by_query),
            "n_relevant": n_relevant,
            "n_nonrelevant": self.n_rows - n_relevant,
            "min_relevant_per_query": min(per_query) if per_query else 0,
            "max_relevant_per_query": max(per_query) if per_query else 0,
            "n_queries_without_relevant": sum(1 for n in per_query if n == 0),
        }

    def content(self) -> dict:
        """The judgements themselves, and nothing about the file they arrived in.

        Deliberately excludes the filename, the byte digest and the row order: those identify the
        *artifact*, and ``source_sha256`` already carries them. What :attr:`digest` has to answer
        is "were these two numbers scored against the same judgements", and a copy of the key
        under another name, or with its rows in another order, is the same judgements.
        """
        return {
            "kind": self.kind.value,
            "relevant": {q: sorted(d) for q, d in sorted(self.relevant_by_query.items())},
            "judged": {q: sorted(d) for q, d in sorted(self.judged_by_query.items())},
        }

    @property
    def digest(self) -> str:
        """Identity of this answer key, so a recall number records what it was scored against."""
        return sha256_hex(canonical_json(self.content()))


def _parse_qrels_text(text: str, *, source_name: str) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            # A blank line is whitespace in a text file, not a lost judgement. Anything with
            # content on it must parse.
            continue
        fields = line.split()
        if len(fields) != 4:
            raise QrelsError(
                f"{source_name}:{lineno}: expected 4 whitespace-separated TREC fields "
                f"(qid Q0 docid rel), found {len(fields)}: {line!r}. Skipping the line would "
                "drop a judgement and make recall read higher than it is."
            )
        qid, marker, docid, rel_text = fields
        if marker.lower() not in _ITERATION_MARKERS:
            raise QrelsError(
                f"{source_name}:{lineno}: second field is {marker!r}, not the TREC iteration "
                "marker. The columns cannot be assumed to be in the documented order, so the "
                "qid and docid positions are not trustworthy."
            )
        try:
            rel = int(rel_text)
        except ValueError:
            raise QrelsError(
                f"{source_name}:{lineno}: relevance {rel_text!r} is not an integer"
            ) from None
        if not qid or not docid:
            raise QrelsError(f"{source_name}:{lineno}: empty qid or docid in {line!r}")
        for name, token in (("qid", qid), ("docid", docid)):
            # A byte-order mark or any other zero-width format character is not whitespace, so it
            # survives strip() and split() and becomes part of the identifier. Two files
            # concatenated with `cat` put one in the middle of the stream, and the qid on that row
            # then silently forks into a second, near-identical id: its judgements leave the real
            # query's relevant set, the row count still matches the frozen vintage, and recall for
            # that query comes out *higher* against the smaller denominator. That is precisely the
            # invisible truncation this loader exists to refuse, arriving through the one door the
            # row-count check cannot see.
            if not token.isprintable():
                raise QrelsError(
                    f"{source_name}:{lineno}: {name} {token!r} contains a non-printable or "
                    "zero-width character (a byte-order mark from a concatenated file is the "
                    "usual cause); it would read as a different identifier and quietly shrink a "
                    "recall denominator"
                )
        rows.append((qid, docid, rel))
    return rows


def load_qrels(
    path: Path, *, kind: QrelKind, expect_rows: Optional[int] = None
) -> Qrels:
    """Read one TREC qrels file. Missing file, malformed row, or wrong vintage all raise."""
    path = Path(path)
    if not isinstance(kind, QrelKind):
        raise QrelsError(f"kind must be a QrelKind, got {kind!r}")
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        # Explicitly not an empty Qrels: "the file is not there" and "nothing is relevant" are
        # different facts and only one of them is a legitimate score.
        raise QrelsError(
            f"qrels file {path} does not exist; refusing to score against an absent answer key"
        ) from None
    # utf-8-sig strips a single leading byte-order mark, which is a transport artefact of the
    # editor that last touched the file and not a judgement. It strips only the leading one, so a
    # mark in the middle of the stream still reaches the per-row check above and stops the load.
    rows = _parse_qrels_text(data.decode("utf-8-sig"), source_name=path.name)
    if expect_rows is not None and len(rows) != expect_rows:
        raise QrelsError(
            f"{path.name}: {len(rows)} judgement rows, the frozen vintage has {expect_rows}. "
            "This is a different answer key; recall computed against it is not comparable with "
            "anything already measured."
        )
    return Qrels.from_rows(
        rows, kind=kind, source_name=path.name, source_sha256=sha256_hex(data)
    )


def load_bcplus_qrels(
    directory: Path, *, expect_frozen_rows: bool = True
) -> dict[QrelKind, Qrels]:
    """Load both BrowseComp-Plus qrels files from ``directory``, checked against the freeze."""
    directory = Path(directory)
    out: dict[QrelKind, Qrels] = {}
    for name in sorted(BCPLUS_QREL_FILES):
        kind, rows = BCPLUS_QREL_FILES[name]
        out[kind] = load_qrels(
            directory / name, kind=kind, expect_rows=rows if expect_frozen_rows else None
        )
    return out


@dataclass(frozen=True)
class EvaluatorQueryView:
    """Everything the evaluator knows about one query, and nothing anyone else may know.

    The benchmark's decrypted record also carries the full text of every gold, evidence and
    negative document. Those bodies are dropped at load: the evaluator scores by docid, so the
    text buys nothing, and keeping it means any accidental serialization of this object writes
    gold document bodies into an artifact that something on the treatment path might later read.
    """

    query_id: str
    query: str
    answer: str
    gold_docids: frozenset[str]
    evidence_docids: frozenset[str]
    negative_docids: frozenset[str]

    def content(self) -> dict:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "answer": self.answer,
            "gold_docids": sorted(self.gold_docids),
            "evidence_docids": sorted(self.evidence_docids),
            "negative_docids": sorted(self.negative_docids),
        }


class EvaluatorQuerySet:
    """query_id -> :class:`EvaluatorQueryView`, with the source digest that identifies it."""

    def __init__(
        self,
        views: Mapping[str, EvaluatorQueryView],
        *,
        source_name: str = "",
        source_sha256: str = "",
    ) -> None:
        self._views = dict(views)
        self.source_name = source_name
        self.source_sha256 = source_sha256

    def __len__(self) -> int:
        return len(self._views)

    def __contains__(self, query_id: object) -> bool:
        return query_id in self._views

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._views))

    def get(self, query_id: str) -> EvaluatorQueryView:
        try:
            return self._views[query_id]
        except KeyError:
            raise UnknownQuery(
                f"{query_id!r} has no evaluator record in {self.source_name or '<memory>'}; a "
                "task with no ground truth cannot be graded, and must not be scored as if it had "
                "been"
            ) from None

    def query_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._views))

    def counts(self) -> dict:
        return {
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "n_queries": len(self._views),
            "n_gold_docs": sum(len(v.gold_docids) for v in self._views.values()),
            "n_evidence_docs": sum(len(v.evidence_docids) for v in self._views.values()),
            "n_negative_docs": sum(len(v.negative_docids) for v in self._views.values()),
        }

    @property
    def digest(self) -> str:
        return sha256_hex(
            canonical_json(
                {
                    "source_name": self.source_name,
                    "source_sha256": self.source_sha256,
                    "queries": [self._views[q].content() for q in sorted(self._views)],
                }
            )
        )


def _require_text(value: Any, *, name: str, where: str) -> str:
    """A non-empty string, or stop.

    Applied to the gold answer as well as to the ids: grading against an empty key does not make
    a task hard, it makes every response the judge feels generous about correct.
    """
    if not isinstance(value, str) or not value.strip():
        raise QrelsError(f"{where}: {name} is missing or empty ({value!r})")
    return value


def _docids(entries: Any, *, where: str, field: str) -> frozenset[str]:
    if entries is None or not isinstance(entries, list):
        raise QrelsError(
            f"{where}: {field} is {type(entries).__name__}, expected a list of "
            "{docid, text, url} objects"
        )
    out: set[str] = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise QrelsError(f"{where}: {field}[{i}] is not an object")
        docid = entry.get("docid")
        if not isinstance(docid, str) or not docid:
            raise QrelsError(f"{where}: {field}[{i}] has no usable docid ({docid!r})")
        out.add(docid)
    return frozenset(out)


def load_evaluator_queries(
    path: Path, *, expect_queries: Optional[int] = None
) -> EvaluatorQuerySet:
    """Read ``browsecomp_plus_decrypted.jsonl``. EVALUATOR-ONLY, and fail-closed per line.

    A record that does not parse stops the load. The alternative -- skipping it -- silently
    shrinks the denominator of the primary quality endpoint, and an accuracy computed over the
    subset of tasks whose ground truth happened to parse is not the accuracy that was
    pre-registered.
    """
    path = Path(path)
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise QrelsError(
            f"evaluator query file {path} does not exist; there is no ground truth to grade "
            "against and no default that could stand in for it"
        ) from None

    views: dict[str, EvaluatorQueryView] = {}
    for lineno, raw in enumerate(data.decode("utf-8-sig").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        where = f"{path.name}:{lineno}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise QrelsError(f"{where}: not valid JSON ({exc.msg})") from None
        if not isinstance(record, dict):
            raise QrelsError(f"{where}: top-level value is not an object")

        query_id = _require_text(record.get("query_id"), name="query_id", where=where)
        query = _require_text(record.get("query"), name="query", where=where)
        answer = _require_text(record.get("answer"), name="answer", where=where)

        gold = _docids(record.get("gold_docs"), where=where, field="gold_docs")
        evidence = _docids(record.get("evidence_docs"), where=where, field="evidence_docs")
        negative = _docids(record.get("negative_docs"), where=where, field="negative_docs")
        contradiction = (gold | evidence) & negative
        if contradiction:
            # A document cannot be both the evidence for the answer and a hard negative. If the
            # record says it is, the interference-precision label and the recall numerator would
            # disagree about the same document and neither could be believed.
            raise QrelsError(
                f"{where}: {len(contradiction)} docid(s) are both relevant and hard negatives, "
                f"e.g. {sorted(contradiction)[:3]}"
            )
        if query_id in views:
            raise QrelsError(
                f"{where}: query_id {query_id!r} appears twice; one of the two ground truths "
                "would silently win"
            )
        views[query_id] = EvaluatorQueryView(
            query_id=query_id,
            query=query,
            answer=answer,
            gold_docids=gold,
            evidence_docids=evidence,
            negative_docids=negative,
        )

    if not views:
        raise QrelsError(f"{path}: no evaluator records; refusing to grade against nothing")
    if expect_queries is not None and len(views) != expect_queries:
        raise QrelsError(
            f"{path.name}: {len(views)} queries, the frozen vintage has {expect_queries}. "
            "A different query set is a different benchmark."
        )
    return EvaluatorQuerySet(
        views, source_name=path.name, source_sha256=sha256_hex(data)
    )


def load_bcplus_evaluator_queries(
    directory: Path, *, expect_frozen_queries: bool = True
) -> EvaluatorQuerySet:
    """Load the evaluator records from ``directory``, checked against the freeze by default.

    The counterpart of :func:`load_bcplus_qrels`, and it exists because the frozen check has to be
    the default rather than an argument somebody remembers. :func:`load_evaluator_queries` takes
    ``expect_queries=None``, so a caller that simply loads the file accepts a truncated record set
    without a word -- and a query set that lost records shrinks the denominator of the *primary*
    quality endpoint, which is the one number the study is about.
    """
    return load_evaluator_queries(
        Path(directory) / BCPLUS_EVALUATOR_QUERY_FILE,
        expect_queries=BCPLUS_QUERY_COUNT if expect_frozen_queries else None,
    )


def assert_docsets_agree(
    queries: EvaluatorQuerySet, qrels: Qrels, *, query_ids: Optional[Sequence[str]] = None
) -> None:
    """Check the two ground-truth artifacts describe the same world.

    The qrels files and the decrypted jsonl are two encodings of one annotation. If they
    disagree, they are from different vintages, and every recall number is computed against one
    of them while every accuracy number is computed against the other -- an inconsistency that no
    downstream check can see, because each artifact is internally consistent.
    """
    field = "gold_docids" if qrels.kind is QrelKind.GOLD else "evidence_docids"
    scoped = query_ids is not None
    ids = tuple(sorted(query_ids)) if scoped else queries.query_ids()
    problems: list[str] = []
    if not scoped:
        # Both directions, or the check only proves containment. A qrels file carrying queries the
        # evaluator records have never heard of is the same vintage mismatch as the reverse, and
        # only this direction catches an answer key from an older release that still covers every
        # query in the current one.
        # Every orphan is recorded, not just the first few: the raise below reports len(problems)
        # as the number of disagreeing queries, so a capped list here would understate the size of
        # the mismatch. Only the rendered message is truncated.
        problems.extend(
            f"{query_id}: judged in {qrels.source_name}, absent from the evaluator records"
            for query_id in sorted(qrels.query_ids())
            if query_id not in queries
        )
    for query_id in ids:
        view = queries.get(query_id)
        if not qrels.is_judged(query_id):
            problems.append(f"{query_id}: judged in the jsonl, absent from {qrels.source_name}")
            continue
        from_jsonl = getattr(view, field)
        from_qrels = qrels.relevant(query_id)
        if from_jsonl != from_qrels:
            only_json = sorted(from_jsonl - from_qrels)[:3]
            only_qrels = sorted(from_qrels - from_jsonl)[:3]
            problems.append(
                f"{query_id}: {len(from_jsonl)} in jsonl vs {len(from_qrels)} in "
                f"{qrels.source_name} (jsonl-only e.g. {only_json}, qrels-only e.g. {only_qrels})"
            )
    if problems:
        raise QrelsError(
            f"{len(problems)} quer(y|ies) disagree between the evaluator records and "
            f"{qrels.source_name}: " + "; ".join(problems[:5])
        )
