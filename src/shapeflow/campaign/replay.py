"""Replay stored boundary checkpoints through the production selection path.

A completed campaign leaves behind every gather batch it ever formed: `runner._store_checkpoint`
persists an `HCheckpoint` for *every* arm, captured inside `researcher_tools` after
`asyncio.gather` and before any strategy runs. Those checkpoints are therefore
treatment-independent -- the same frozen batch a P0 cell saw is the one a P1 cell saw -- which
makes them a corpus of real decision points that can be re-offered to a new selector without
running the agent again.

That matters because the alternative is re-running the graph, and the graph is where all the GPU
time goes. A selector shootout over stored checkpoints costs one request per batch for an LLM
selector and nothing at all for a CPU one.

**The one thing a checkpoint does not carry is the page bytes.** It names a page by
`raw_content_id`, which is `sha256(vendor's truncated bytes)` and nothing else; the bytes lived in
an in-memory `PageRegistry` that the run threw away. So they are rebuilt here, along the same
three hops the search seam used to produce them:

    occurrence_id --(the run's own retrieval trace)--> docid
    docid --(CorpusStore)--> the whole document
    document --(apply_shared_budget)--> exactly what vendor was handed

and then **the result must hash to the recorded `raw_content_id`**. That check is the whole point.
Without it this module would be substituting plausible bytes for frozen ones, and a selector
scored on a different document than the one it was compared against still publishes, still looks
healthy, and is measuring nothing. A page whose hash disagrees is reported unrecoverable; it is
never guessed at, and never silently replaced by its snippet.

Treatment-side by construction. It joins on `occurrence_id` and `docid`, both of which the
retrieval path already hands the agent. The benchmark's relevance labels are not here and must
not be: scoring lives on the evaluator side of the leakage firewall, keyed by the same ids.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence

from ..hashing import sha256_hex

__all__ = [
    "BatchReconstruction",
    "OccurrenceIndex",
    "PageReconstructor",
    "PageResolution",
    "ViewMeasurement",
    "ViewProbeDeclined",
    "iter_checkpoints",
    "load_occurrence_index",
    "measure_whole_batch_view",
    "outcome_record",
    "quantiles",
    "run_trial",
    "trial_key",
]

#: A page that arrived with no raw content at all. Vendor falls back to the short snippet in that
#: case and so does every arm, so there is nothing to rebuild and nothing to verify -- but it is
#: still counted, because a batch that is *all* snippet offers a selector no spans.
SNIPPET_ONLY = "SNIPPET_ONLY"
#: The bytes came back and hashed to the id the checkpoint recorded.
VERIFIED = "VERIFIED"
#: The run's retrieval trace never mentioned this occurrence, so there is no docid to look up.
NO_DOCID = "NO_DOCID"
#: The trace named a docid the corpus does not hold -- index and corpus of different vintages.
NO_DOCUMENT = "NO_DOCUMENT"
#: The bytes came back and hashed to something else. The loudest possible failure: it means the
#: shared budget, the corpus or the truncation rule is not the one the run used.
HASH_MISMATCH = "HASH_MISMATCH"


@dataclass(frozen=True)
class PageResolution:
    """One page of one gather batch, and whether its frozen bytes could be recovered."""

    raw_content_id: str
    occurrence_id: str
    status: str
    docid: str = ""
    text: str = ""

    @property
    def usable(self) -> bool:
        """Whether this page can be offered to a selector as the batch's own bytes."""
        return self.status in {VERIFIED, SNIPPET_ONLY}


@dataclass(frozen=True)
class BatchReconstruction:
    """Every page of one H checkpoint, resolved or explained."""

    checkpoint_digest: str
    task_id: str
    siblings: int
    pages: tuple[PageResolution, ...]

    @property
    def complete(self) -> bool:
        """True only when every page is usable.

        All-or-nothing on purpose. A batch missing one page is a *different* batch: the selector
        would be choosing from a smaller candidate set than the arm it is compared against, and
        the difference would show up as a quality result rather than as a missing page.
        """
        return all(page.usable for page in self.pages)

    @property
    def content_pages(self) -> tuple[PageResolution, ...]:
        return tuple(p for p in self.pages if p.status != SNIPPET_ONLY)

    def registry_prefill(self) -> tuple[dict[str, str], dict[str, str]]:
        """The two maps `PageRegistry.prefill` wants, for the verified pages."""
        text_by_id = {p.raw_content_id: p.text for p in self.pages if p.status == VERIFIED}
        occurrence_by_id = {
            p.raw_content_id: p.occurrence_id for p in self.pages if p.status == VERIFIED
        }
        return text_by_id, occurrence_by_id

    def status_counts(self) -> dict[str, int]:
        return dict(Counter(page.status for page in self.pages))


class OccurrenceIndex:
    """`occurrence_id -> docid`, recovered from the retrieval traces the campaign recorded.

    `SearchRecord` carries no docid -- deliberately, because a docid is what joins a retrieved
    document to the benchmark's relevance labels and the treatment path must not hold that join.
    What it does carry is `occurrence_id`, and `RetrievalClient.search_rows` appends the two as
    parallel arrays to a per-cell trace which `runner` persists inside the cell output. So the
    mapping exists exactly once per run, in the artifact, and this reads it back.

    First writer wins, and a later disagreement is fatal rather than overwritten: one occurrence
    resolving to two documents would mean the id is not identifying what it claims to.
    """

    def __init__(self) -> None:
        self._docid: dict[str, str] = {}
        #: Traces seen, for the census to report coverage rather than assert it.
        self.rows = 0

    def __len__(self) -> int:
        return len(self._docid)

    def __contains__(self, occurrence_id: object) -> bool:
        return occurrence_id in self._docid

    def add_trace(self, trace: Sequence[Mapping]) -> None:
        for row in trace:
            docids = row.get("docids") or ()
            occurrence_ids = row.get("occurrence_ids") or ()
            if len(docids) != len(occurrence_ids):
                # The two are written from one row list in one pass, so a length disagreement
                # means the artifact was assembled by something other than the code that wrote
                # it, and pairing them by position would invent a mapping.
                raise ValueError(
                    f"retrieval trace row for query {row.get('query')!r} has {len(docids)} "
                    f"docids and {len(occurrence_ids)} occurrence ids; they are recorded as "
                    "parallel arrays and cannot be paired")
            for occurrence_id, docid in zip(occurrence_ids, docids):
                self.rows += 1
                existing = self._docid.get(str(occurrence_id))
                if existing is None:
                    self._docid[str(occurrence_id)] = str(docid)
                elif existing != str(docid):
                    raise ValueError(
                        f"occurrence {occurrence_id} resolves to both {existing} and {docid}; "
                        "an occurrence id identifies one retrieved document")

    def docid_for(self, occurrence_id: str) -> Optional[str]:
        return self._docid.get(str(occurrence_id))


def _object_keys(object_store_root: Path) -> Iterator[str]:
    """Every key in an `ObjectStore`, from the shard layout `root/ab/cd/<key>.zst`."""
    for path in sorted(object_store_root.glob("*/*/*.zst")):
        yield path.stem


def load_occurrence_index(object_store_roots: Sequence[Path]) -> OccurrenceIndex:
    """Build the occurrence to docid map from every cell output in the given stores.

    Enumerated from the store's own shard layout rather than from a ledger query. The ledger is
    per lane and its `artifacts` rows would have to agree with the blobs on disk; the blobs are
    the thing being read, so reading them directly removes a way for the two to disagree.
    """
    from ..object_store import CorruptObject, ObjectStore

    index = OccurrenceIndex()
    for root in object_store_roots:
        root = Path(root)
        if not root.exists():
            continue
        store = ObjectStore(root)
        for key in _object_keys(root):
            try:
                raw = store.get_bytes(key)
            except (KeyError, CorruptObject):
                # A corrupt blob is loud where it is *used*. Here it only means one cell's trace
                # is unavailable, which shows up as NO_DOCID on its pages and is counted.
                continue
            try:
                body = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                continue
            trace = body.get("retrieval_trace") if isinstance(body, dict) else None
            if isinstance(trace, list) and trace:
                index.add_trace(trace)
    return index


def iter_checkpoints(checkpoint_roots: Sequence[Path], *, kind: str = "H") -> Iterator[dict]:
    """Every distinct stored checkpoint of one kind, in digest order.

    Yields the raw document rather than an `HCheckpoint`, because the census wants to count what
    is there before deciding anything can be built from it.

    Deduplicated by digest across roots. Lanes partition tasks, so today no checkpoint appears
    in two of them -- 2,523 files, 2,523 distinct digests -- but that is an observation about
    one campaign, not a property of the layout, and a resumed or migrated task could break it.
    Counting one batch twice would inflate every rate this walk feeds, including the mechanical
    routeability ceiling, and it would do so invisibly because both copies are valid.
    """
    seen: set[str] = set()
    for root in checkpoint_roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in sorted(root.glob("*/*.json")):
            if path.stem in seen:
                continue
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeDecodeError, OSError):
                continue
            seen.add(path.stem)
            if document.get("kind") == kind:
                yield document


@dataclass
class PageReconstructor:
    """Rebuild the exact bytes a gather batch's pages were frozen as, or say why not.

    `budget` and `tokenizer` must be the ones the run used: the recorded `raw_content_id` is the
    hash of the text *after* the shared budget, so a different character cap or a different
    tokenizer produces different bytes and every page fails verification. That is the intended
    behaviour -- it is a mismatch detector, not an inconvenience.
    """

    corpus: object                       # CorpusStore
    budget: object                       # SharedContentBudget
    tokenizer: object
    occurrences: OccurrenceIndex
    #: `docid -> truncated text`. Keyed by document, not by content id, and deliberately: the
    #: same bytes can belong to two docids (a corpus may hold duplicates), and the docid is the
    #: key the evaluator later joins relevance labels on. Caching a *resolution* by content id
    #: would let the first occurrence's docid be reported for a later occurrence of identical
    #: bytes, which is a misattribution no downstream check could see. What is cached here is
    #: only the expensive part -- re-truncating a document that recurs across many queries.
    _text_by_docid: dict = field(default_factory=dict, repr=False)

    def resolve(self, *, raw_content_id: Optional[str], occurrence_id: str) -> PageResolution:
        from ..evidence.shared_view import apply_shared_budget

        if not raw_content_id:
            return PageResolution(raw_content_id="", occurrence_id=str(occurrence_id),
                                  status=SNIPPET_ONLY)

        docid = self.occurrences.docid_for(occurrence_id)
        if docid is None:
            return PageResolution(raw_content_id=raw_content_id,
                                  occurrence_id=str(occurrence_id), status=NO_DOCID)

        truncated = self._text_by_docid.get(docid)
        if truncated is None:
            try:
                document = self.corpus.get(docid)
            except KeyError:
                return PageResolution(raw_content_id=raw_content_id,
                                      occurrence_id=str(occurrence_id), status=NO_DOCUMENT,
                                      docid=docid)
            truncated = apply_shared_budget(document.text, self.budget, self.tokenizer).text
            self._text_by_docid[docid] = truncated

        # Verified on every page, never once per document: the check is what makes this a
        # reconstruction, and it is a hash of at most `max_chars` of text.
        if sha256_hex(truncated.encode("utf-8")) != raw_content_id:
            # Not recoverable by trying harder. The id is the hash of what vendor was handed, so
            # a mismatch means this reconstruction is not that -- and publishing spans of the
            # wrong document is the one failure that would look like a working arm.
            return PageResolution(raw_content_id=raw_content_id,
                                  occurrence_id=str(occurrence_id), status=HASH_MISMATCH,
                                  docid=docid)

        return PageResolution(raw_content_id=raw_content_id, occurrence_id=str(occurrence_id),
                              status=VERIFIED, docid=docid, text=truncated)

    def rebuild(self, document: Mapping) -> BatchReconstruction:
        """Resolve every page of one stored H checkpoint document."""
        pages: list[PageResolution] = []
        result_sets = document.get("search_result_sets") or ()
        for entry in result_sets:
            results = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 else ()
            for result in results or ():
                pages.append(self.resolve(
                    raw_content_id=result.get("raw_content_id"),
                    occurrence_id=result.get("source_occurrence_id") or "",
                ))
        return BatchReconstruction(
            checkpoint_digest=str(document.get("digest", "")),
            task_id=str(document.get("task_id", "")),
            siblings=len(result_sets),
            pages=tuple(pages),
        )


@dataclass(frozen=True)
class ViewMeasurement:
    """What one whole-batch candidate view would cost to *offer*, before anyone selects.

    Three numbers decide whether the contract's shape is buildable at all on this workload:
    `candidates` against the selector schema's 64-item publication cap, `prompt_tokens` against
    the engine's context window, and `pages` to stratify both by batch size.
    """

    checkpoint_digest: str
    task_id: str
    pages: int
    candidates: int
    prompt_tokens: int
    view_sha256: str


class ViewProbeDeclined(Exception):
    """Raised by the census probe once it has measured. Never a real selector failure."""


class _ViewProbe:
    """A selector that measures the view it is offered and then declines to select.

    Declining rather than selecting keeps the census a measurement: no engine is called, no
    aggregation runs, and the batch ends in the ordinary P1-failure path. What is measured is
    the *offered* view, built by `run_selection` exactly as production builds it -- which is the
    only reason these numbers describe the real mechanism rather than a reimplementation of it.
    """

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        self.measured: list[tuple[int, int, str]] = []

    async def select(self, *, task_ctx, view):
        self.measured.append((
            len(view.candidates),
            self._tokenizer.count(view.prompt_bytes.decode("utf-8")),
            view.view_sha256,
        ))
        raise ViewProbeDeclined()


async def measure_whole_batch_view(
    *, document: Mapping, reconstruction: BatchReconstruction, tokenizer,
    chunker: str, token_budget: int, chunk_max_tokens: Optional[int] = None,
    topic: str = "",
) -> Optional[ViewMeasurement]:
    """Build the whole-batch candidate view for one checkpoint and measure it.

    Returns None when the batch offers no spans at all -- every page snippet-only -- because
    there is no view to measure and counting it as a zero-token view would understate the
    distribution the LLM arms actually face.
    """
    from types import SimpleNamespace

    from ..strategies.page_h import PageSelectionError, PageSelectionStrategy, PageStrategyConfig
    from ..odr.checkpoints import from_document

    checkpoint = from_document(dict(document))
    text_by_id, occurrence_by_id = reconstruction.registry_prefill()
    probe = _ViewProbe(tokenizer)
    # `chunk_max_tokens` defaults to whatever `PageStrategyConfig` declares, because that is what
    # `StrategyFactory._page_config` leaves it at: the census must chunk the way the campaign
    # chunked, and naming a number here would be a second place for it to be decided.
    chunking = {} if chunk_max_tokens is None else {"chunk_max_tokens": chunk_max_tokens}
    strategy = PageSelectionStrategy(
        PageStrategyConfig(
            variant_id="CENSUS", chunker=chunker, scope="whole_batch", contract="P1_ID",
            aggregation="stable_union_v1", token_budget=token_budget, **chunking,
        ),
        selector=probe,
        tokenizer=tokenizer,
        raw_text_for=lambda content_id: text_by_id.get(content_id, ""),
        occurrence_for=lambda content_id: occurrence_by_id.get(content_id, content_id),
    )
    try:
        await strategy.transform_tool_batch(
            task_ctx=SimpleNamespace(research_topic=topic, selected_token_budget=token_budget),
            checkpoint=checkpoint,
        )
    except PageSelectionError:
        pass
    if not probe.measured:
        return None
    candidates, prompt_tokens, view_sha = probe.measured[0]
    return ViewMeasurement(
        checkpoint_digest=reconstruction.checkpoint_digest,
        task_id=reconstruction.task_id,
        pages=len(reconstruction.content_pages),
        candidates=candidates,
        prompt_tokens=prompt_tokens,
        view_sha256=view_sha,
    )


@dataclass(frozen=True)
class ReplaySetup:
    """Everything a replay needs, loaded once.

    One definition because the census and the shootout must replay *the same world*: the same
    corpus shards, the same shared-content budget and the same tokenizer. Two call sites each
    assembling this from config would agree until one of them was updated.
    """

    corpus: object
    tokenizer: object
    budget: object
    occurrences: OccurrenceIndex
    reconstructor: PageReconstructor
    checkpoint_roots: tuple[Path, ...]
    object_store_roots: tuple[Path, ...]

    def provenance(self) -> dict:
        from ..evidence.model_tokenizer import tokenizer_sha256

        return {
            "checkpoint_roots": [str(p) for p in self.checkpoint_roots],
            "object_store_roots": [str(p) for p in self.object_store_roots],
            "corpus_documents": len(self.corpus),
            "corpus_shard_sha256": list(self.corpus.shard_sha256),
            "tokenizer_sha256": tokenizer_sha256(self.tokenizer),
            "shared_content_budget": {"max_chars": self.budget.max_chars,
                                      "max_tokens": self.budget.max_tokens},
            "occurrences_indexed": len(self.occurrences),
        }


def build_replay_setup(settings, corpus_dir: Path, *, echo=None) -> ReplaySetup:
    """Load the frozen world a replay reads from, in the form the run recorded it."""
    from ..evidence.model_tokenizer import load_frozen_tokenizer
    from ..evidence.shared_view import budget_from_settings
    from ..retrieval.corpus import load_corpus

    say = echo or (lambda _message: None)
    root = Path(settings.data_root)
    # Lane-scoped by design: four runners share nothing but the provider and the corpus, so one
    # campaign's checkpoints and cell outputs are spread across whichever lanes ran it.
    checkpoint_roots = tuple(sorted(root.glob("runner*/checkpoints")))
    object_store_roots = tuple(sorted(root.glob("runner*/object_store")))
    if not checkpoint_roots:
        raise FileNotFoundError(f"no runner*/checkpoints under {root}")

    say(f"loading corpus from {corpus_dir} ...")
    corpus = load_corpus(corpus_dir)
    tokenizer = load_frozen_tokenizer(settings)
    budget = budget_from_settings(settings)
    say(f"corpus: {len(corpus)} documents; budget {budget.max_chars} chars / "
        f"{budget.max_tokens} tokens")

    say(f"indexing retrieval traces from {len(object_store_roots)} object store(s) ...")
    occurrences = load_occurrence_index(object_store_roots)
    say(f"occurrence index: {len(occurrences)} occurrences from {occurrences.rows} rows")

    return ReplaySetup(
        corpus=corpus, tokenizer=tokenizer, budget=budget, occurrences=occurrences,
        reconstructor=PageReconstructor(
            corpus=corpus, budget=budget, tokenizer=tokenizer, occurrences=occurrences),
        checkpoint_roots=checkpoint_roots,
        object_store_roots=object_store_roots,
    )


def trial_key(
    *, execution_binding: str, checkpoint_digest: str, variant_id: str, seed: int,
    tokenizer_sha256: str,
) -> str:
    """Content address for one variant's replay of one boundary.

    Reuses `odr.checkpoints.fork_key` rather than inventing a scheme, and folds everything that
    could change the answer into its `prompt_renderer_version` slot: the tokenizer identity, the
    selector prompt bundle and the renderer grouping version. Resume is then "the file exists",
    with no risk of reusing a result produced under a different prompt or a different renderer.
    """
    from ..odr.checkpoints import fork_key
    from ..p1.prompts import PROMPT_BUNDLE_VERSION
    from ..p1.renderer import RENDERER_GROUPING_VERSION

    return fork_key(
        protocol_sha=execution_binding,
        boundary_id=checkpoint_digest,
        variant_id=variant_id,
        seed=seed,
        prompt_renderer_version=(
            f"{PROMPT_BUNDLE_VERSION}|{RENDERER_GROUPING_VERSION}|{tokenizer_sha256}"),
    )


def outcome_record(outcome) -> dict:
    """The measurable half of one `SelectionOutcome`, as JSON.

    Structural only: span ids, occurrence ids, token counts and the work spent. No relevance
    label appears here and none may -- the evaluator joins these records to the benchmark's own
    sets by occurrence id, on the far side of the leakage firewall.
    """
    failure = outcome.failure
    return {
        "ok": bool(outcome.ok),
        "failure_reason": failure.reason if failure else "",
        "failure_detail": failure.detail if failure else "",
        "selector_attempted": bool(outcome.selector_attempted),
        "view_sha256": outcome.view_sha256,
        "contract": outcome.contract,
        "aggregation": outcome.aggregation,
        "chunker": outcome.chunker,
        "tokenizer_sha256": outcome.tokenizer_sha256,
        "offered": outcome.offered,
        "offered_span_ids": list(outcome.offered_span_ids),
        "offered_source_occurrence_ids": list(outcome.offered_source_occurrence_ids),
        "offered_material_tokens": outcome.offered_material_tokens,
        "selected_span_ids": list(outcome.selected_span_ids),
        "staged_span_ids": list(outcome.staged_span_ids),
        "published_span_ids": list(outcome.published_span_ids),
        "dropped_for_budget": outcome.dropped_for_budget,
        "staged_rendered_tokens": outcome.staged_rendered_tokens,
        "published_rendered_tokens": outcome.published_rendered_tokens,
        "publication_handle_map": [list(pair) for pair in outcome.publication_handle_map],
        "publication_map_sha256": outcome.publication_map_sha256,
        "published_relations": [list(r) for r in outcome.published_relations],
        "work": {
            "selector_calls": outcome.work.selector_calls,
            "prompt_tokens": outcome.work.prompt_tokens,
            "completion_tokens": outcome.work.completion_tokens,
            "cpu_seconds": round(outcome.work.cpu_seconds, 6),
            "retries": outcome.work.retries,
        },
    }


async def run_trial(
    *, document: Mapping, reconstruction: BatchReconstruction, strategy, variant_id: str,
    topic: str, token_budget: int,
) -> dict:
    """Replay one gather batch through one bound strategy and record what it did.

    The strategy is a real `PageSelectionStrategy` built by `StrategyFactory`, so chunking, view
    construction, selection, aggregation and preflight are the production code paths rather than
    a reimplementation of them. A `PageSelectionError` is an *outcome*, not an error: it is what
    a whole-batch P1 failure looks like, and the run recorded it the same way.
    """
    from types import SimpleNamespace

    from ..odr.checkpoints import from_document
    from ..strategies.page_h import PageSelectionError

    checkpoint = from_document(dict(document))
    published_text = None
    batch_failure = ""
    try:
        observations = await strategy.transform_tool_batch(
            task_ctx=SimpleNamespace(research_topic=topic, selected_token_budget=token_budget),
            checkpoint=checkpoint,
        )
        published_text = "".join(str(o.content) for o in observations)
    except PageSelectionError as failure:
        batch_failure = failure.failure.reason

    return {
        "checkpoint_digest": reconstruction.checkpoint_digest,
        "task_id": reconstruction.task_id,
        "variant_id": variant_id,
        "pages": len(reconstruction.content_pages),
        # The join key the evaluator needs, carried so it never has to re-derive the retrieval
        # trace. Treatment-visible on both sides: the agent retrieved these documents.
        "page_docids": {p.occurrence_id: p.docid
                        for p in reconstruction.content_pages if p.docid},
        "batch_failure": batch_failure,
        "published_text": published_text,
        "outcomes": [outcome_record(o) for o in strategy.last_outcomes],
    }


def quantiles(values: Sequence[int], points: Sequence[float] = (0.5, 0.9, 0.95, 0.99)) -> dict:
    """Order statistics with the count and extremes, or an explicitly empty summary."""
    ordered = sorted(values)
    if not ordered:
        return {"n": 0}
    summary = {"n": len(ordered), "min": ordered[0], "max": ordered[-1],
               "mean": round(sum(ordered) / len(ordered), 2)}
    for point in points:
        summary[f"p{int(point * 100)}"] = ordered[int(point * (len(ordered) - 1))]
    return summary
