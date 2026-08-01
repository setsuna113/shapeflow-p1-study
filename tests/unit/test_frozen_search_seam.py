"""The seam that decides which world a cell searches.

``install_frozen_search`` rebinds ``open_deep_research.utils.tavily_search_async`` for the
duration of one cell. Everything here exists because the ways this can go wrong are quiet ones:
a binding that is not put back sends the *next* cell to the previous task's corpus and returns
entirely plausible results; a backend installed without the shared budget hands one arm a
FULL_PAGE view of a document the other arm only saw a prefix of, and nothing downstream
complains because the extra bytes look exactly like ordinary page content.

The BrowseComp-Plus dense backend needs torch, a 1.7 GB corpus and a 100k-vector index, so it is
represented here by the smallest object that satisfies the same protocol: something with
``search(query, *, max_results) -> list[SearchRecord]``. That is precisely the contract the seam
is being generalised to accept, so a stand-in that meets it tests the seam and not the corpus.
"""

from __future__ import annotations

import pytest

from shapeflow.campaign.graph_driver import install_frozen_search
from shapeflow.evidence.chunkers import WhitespaceTokenizer
from shapeflow.evidence.shared_view import OVERFLOW_REASON, SharedContentBudget
from shapeflow.object_store import ObjectStore
from shapeflow.world.search_backend import SearchRecord
from shapeflow.world.snapshot_store import SnapshotStore
from shapeflow.world.source_pool import QueryResponse, RawResult, build_source_pool


# --- stand-ins ------------------------------------------------------------------------


class ListBackend:
    """A minimal ``SearchBackend`` over canned records, bounding nothing on its own.

    Bounding nothing is the point: it is what the dense BC+ backend does, since it returns whole
    documents by design. If the seam does not apply the budget, this backend proves it.
    """

    def __init__(self, records: dict[str, list[SearchRecord]]) -> None:
        self._records = records
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, max_results: int) -> list[SearchRecord]:
        self.calls.append((query, max_results))
        return list(self._records.get(query, []))[:max_results]


def record(url: str, raw: str | None, *, occurrence_id: str = "occ-1") -> SearchRecord:
    return SearchRecord(url=url, title="T", content="snippet", raw_content=raw, score=1.0,
                        occurrence_id=occurrence_id)


@pytest.fixture
def vendor_utils():
    import open_deep_research.utils as module

    return module


def make_pool(tmp_path, *, raw: str, url: str = "https://doc.invalid/a"):
    """A one-page frozen task pool, the Week-1 world this seam already served."""
    store = SnapshotStore(ObjectStore(tmp_path))
    responses = [QueryResponse("qs1", "q1", (RawResult(url, "T", 1, "snippet", raw),))]
    pool = build_source_pool("t1", responses, store, fetched_at_utc="2026-07-24T00:00:00Z")
    return pool, store


#: A budget small enough that a short paragraph overflows it, so the token bound is exercised
#: without carrying a 50,000-character fixture around.
TIGHT = SharedContentBudget(max_chars=10_000, max_tokens=5)
WORDS = " ".join(f"w{i}" for i in range(1, 21))


# --- the external backend is installed at all -----------------------------------------


async def test_a_prebuilt_backend_is_installed_and_returns_vendors_shape(vendor_utils):
    """The BC+ backend reaches the graph through the same seam, with vendor's field names."""
    backend = ListBackend({"q": [record("https://a.invalid", "page body")]})
    with install_frozen_search(max_results=3, backend=backend):
        payload = await vendor_utils.tavily_search_async(["q"])

    assert [entry["query"] for entry in payload] == ["q"]
    result = payload[0]["results"][0]
    assert result["url"] == "https://a.invalid"
    assert result["title"] == "T"
    assert result["content"] == "snippet"
    assert result["raw_content"] == "page body"
    assert result["_shapeflow_occurrence_id"] == "occ-1"
    # The seam, not the caller, supplies vendor's default top-k.
    assert backend.calls == [("q", 3)]


async def test_an_empty_result_set_from_a_prebuilt_backend_is_a_real_state(vendor_utils):
    """A miss is a fact about the frozen world; inventing a result would fabricate evidence."""
    backend = ListBackend({})
    with install_frozen_search(max_results=5, backend=backend):
        payload = await vendor_utils.tavily_search_async(["nothing matches this"])
    assert payload[0]["results"] == []


async def test_on_query_still_fires_with_a_prebuilt_backend(vendor_utils):
    """Citation scoring is restricted to what this arm retrieved, so the ids must survive."""
    seen: list[dict] = []
    backend = ListBackend({"q": [
        record("https://a.invalid", "a", occurrence_id="occ-a"),
        record("https://b.invalid", "b", occurrence_id="occ-b"),
    ]})
    with install_frozen_search(max_results=5, backend=backend,
                               on_query=seen.append):
        await vendor_utils.tavily_search_async(["q"])

    assert seen == [{
        "query": "q",
        "result_count": 2,
        "max_results": 5,
        "topic": "general",
        "source_occurrence_ids": ["occ-a", "occ-b"],
    }]


async def test_the_same_query_twice_returns_identical_bytes(vendor_utils):
    """Determinism: the wrapper must not reorder, re-truncate differently, or cache stale text."""
    backend = ListBackend({"q": [record("https://a.invalid", WORDS)]})
    with install_frozen_search(max_results=5, backend=backend,
                               content_budget=TIGHT, tokenizer=WhitespaceTokenizer()):
        first = await vendor_utils.tavily_search_async(["q"])
        second = await vendor_utils.tavily_search_async(["q"])
    assert first == second


# --- the shared budget, whatever the backend ------------------------------------------


async def test_a_backend_that_bounds_nothing_is_bounded_by_the_seam(vendor_utils):
    """A backend returning whole documents must not become a FULL_PAGE arm by omission."""
    backend = ListBackend({"q": [record("https://a.invalid", WORDS)]})
    with install_frozen_search(max_results=5, backend=backend,
                               content_budget=TIGHT,
                               tokenizer=WhitespaceTokenizer()) as handle:
        payload = await vendor_utils.tavily_search_async(["q"])

    assert payload[0]["results"][0]["raw_content"] == "w1 w2 w3 w4 w5"
    # Truncation is recorded, never silent: a page that lost its tail belongs in the write-up.
    assert handle.content_truncations == {
        # char_truncated says whether the character bound ALSO bit. It did not here, so
        # original_tokens is the true page size rather than the size of what survived a clip.
        "occ-1": {"reason": OVERFLOW_REASON, "original_tokens": 20, "kept_tokens": 5,
                  "char_truncated": False},
    }


async def test_both_worlds_are_bounded_to_the_same_bytes(tmp_path, vendor_utils):
    """P0 and P1 start from the same bytes only if the budget does not depend on the backend."""
    pool, snapshots = make_pool(tmp_path, raw=WORDS)
    tokenizer = WhitespaceTokenizer()

    with install_frozen_search(pool, snapshots, max_results=5,
                               content_budget=TIGHT, tokenizer=tokenizer):
        from_pool = await vendor_utils.tavily_search_async(["w1"])

    backend = ListBackend({"w1": [record("https://doc.invalid/a", WORDS)]})
    with install_frozen_search(max_results=5, backend=backend,
                               content_budget=TIGHT, tokenizer=tokenizer):
        from_prebuilt = await vendor_utils.tavily_search_async(["w1"])

    assert from_pool[0]["results"][0]["raw_content"] \
        == from_prebuilt[0]["results"][0]["raw_content"] == "w1 w2 w3 w4 w5"


async def test_the_pool_path_is_unchanged_without_a_budget(tmp_path, vendor_utils):
    """No budget means no engine is involved -- the offline and canary tests rely on this."""
    pool, snapshots = make_pool(tmp_path, raw=WORDS)
    with install_frozen_search(pool, snapshots, max_results=5) as handle:
        payload = await vendor_utils.tavily_search_async(["w1"])

    assert payload[0]["results"][0]["raw_content"] == WORDS
    assert handle.content_truncations == {}


async def test_the_handle_still_answers_for_the_backend_it_wraps(vendor_utils):
    """The BC+ backend's own ``stats()``/``queries_seen`` are read off the yielded handle."""
    backend = ListBackend({"q": [record("https://a.invalid", "body")]})
    with install_frozen_search(max_results=5, backend=backend) as handle:
        await vendor_utils.tavily_search_async(["q"])
        assert handle.calls == [("q", 5)]
        # Private names are not delegated: reaching through the wrapper for a backend's
        # internals would hide a missing ``_backend`` behind unbounded recursion instead of an
        # AttributeError.
        with pytest.raises(AttributeError):
            handle._records  # noqa: B018 - the lookup itself is the assertion


async def test_the_overflow_ledger_is_shared_not_duplicated(tmp_path, vendor_utils):
    """One ledger, or the count of pages that lost their tail is split across two objects."""
    pool, snapshots = make_pool(tmp_path, raw=WORDS)
    with install_frozen_search(pool, snapshots, max_results=5, content_budget=TIGHT,
                               tokenizer=WhitespaceTokenizer()) as handle:
        payload = await vendor_utils.tavily_search_async(["w1"])

    # The frozen pool bounds its pages at construction; the wrapper must not have started a
    # second ledger that reports zero truncations while the real one reports the page.
    assert payload[0]["results"][0]["raw_content"] == "w1 w2 w3 w4 w5"
    assert len(handle.content_truncations) == 1
    (entry,) = handle.content_truncations.values()
    assert entry["reason"] == OVERFLOW_REASON


async def test_a_page_with_no_raw_content_stays_absent(vendor_utils):
    """A page vendor never fetched has no body; the budget must not invent an empty one."""
    backend = ListBackend({"q": [record("https://a.invalid", None)]})
    with install_frozen_search(max_results=5, backend=backend,
                               content_budget=TIGHT, tokenizer=WhitespaceTokenizer()):
        payload = await vendor_utils.tavily_search_async(["q"])
    assert payload[0]["results"][0]["raw_content"] is None


# --- restoration, including the paths that are easy to get wrong ----------------------


async def test_the_original_is_restored_when_the_body_raises(vendor_utils):
    """A cell that raised must not leave the next one searching the previous task's corpus."""
    original = vendor_utils.tavily_search_async
    backend = ListBackend({})
    with pytest.raises(RuntimeError, match="cell exploded"):
        with install_frozen_search(max_results=5, backend=backend):
            assert vendor_utils.tavily_search_async is not original
            raise RuntimeError("cell exploded")
    assert vendor_utils.tavily_search_async is original


async def test_the_original_is_restored_when_the_body_is_cancelled(vendor_utils):
    """Cancellation is a BaseException, not an Exception; a bare ``except Exception`` misses it."""
    import asyncio

    original = vendor_utils.tavily_search_async
    with pytest.raises(asyncio.CancelledError):
        with install_frozen_search(max_results=5, backend=ListBackend({})):
            raise asyncio.CancelledError
    assert vendor_utils.tavily_search_async is original


async def test_nesting_restores_the_true_original(vendor_utils):
    """Installing while installed must hand back the outer world, then the real one."""
    original = vendor_utils.tavily_search_async
    with install_frozen_search(max_results=5, backend=ListBackend({})):
        outer = vendor_utils.tavily_search_async
        with install_frozen_search(max_results=5, backend=ListBackend({})):
            assert vendor_utils.tavily_search_async is not outer
        assert vendor_utils.tavily_search_async is outer
    assert vendor_utils.tavily_search_async is original


async def test_nesting_restores_the_true_original_when_the_inner_cell_raises(vendor_utils):
    """The failure mode is the inner frame leaking its world into the outer cell's searches."""
    original = vendor_utils.tavily_search_async
    with install_frozen_search(max_results=5, backend=ListBackend({})):
        outer = vendor_utils.tavily_search_async
        with pytest.raises(RuntimeError, match="inner exploded"):
            with install_frozen_search(max_results=5, backend=ListBackend({})):
                raise RuntimeError("inner exploded")
        assert vendor_utils.tavily_search_async is outer
    assert vendor_utils.tavily_search_async is original


async def test_a_world_left_installed_by_the_body_is_refused(vendor_utils):
    """An unrecorded world was searched; restoring quietly would make that undetectable."""
    original = vendor_utils.tavily_search_async

    async def foreign(queries, **kwargs):
        return []

    with pytest.raises(RuntimeError, match="searched an unrecorded world"):
        with install_frozen_search(max_results=5, backend=ListBackend({})):
            vendor_utils.tavily_search_async = foreign
    # Refusing is not a reason to leave the process pointing at the foreign world.
    assert vendor_utils.tavily_search_async is original


# --- fail closed on an ambiguous or impossible world ----------------------------------


def test_installing_no_world_at_all_is_refused(vendor_utils):
    original = vendor_utils.tavily_search_async
    with pytest.raises(ValueError, match="installing nothing"):
        with install_frozen_search(max_results=5):
            pass
    assert vendor_utils.tavily_search_async is original


def test_installing_two_worlds_is_refused(tmp_path):
    pool, snapshots = make_pool(tmp_path, raw="body")
    with pytest.raises(ValueError, match="never both"):
        with install_frozen_search(pool, snapshots, max_results=5, backend=ListBackend({})):
            pass


def test_half_a_frozen_pool_is_refused(tmp_path):
    """Snapshots without a pool is a missing argument, not an empty corpus."""
    pool, _ = make_pool(tmp_path, raw="body")
    with pytest.raises(ValueError, match="needs both a pool and a snapshot store"):
        with install_frozen_search(pool, None, max_results=5):
            pass


def test_a_backend_without_a_search_method_is_refused():
    """Fail at the seam, where the message names the object -- not deep inside a running cell."""
    with pytest.raises(TypeError, match="does not implement SearchBackend.search"):
        with install_frozen_search(max_results=5, backend=object()):
            pass


@pytest.mark.parametrize("bad", [0, -1, "5", 5.0, True])
def test_a_non_positive_top_k_is_refused(bad):
    """A zero top-k makes an empty world and an empty query indistinguishable in the trace."""
    with pytest.raises(ValueError, match="max_results must be a positive int"):
        with install_frozen_search(max_results=bad, backend=ListBackend({})):
            pass


def test_a_backend_from_an_evaluator_only_module_is_refused(monkeypatch):
    """A world handed in at runtime leaves no import edge for the static firewall to walk.

    ``tools/ci/check_leakage_firewall.py`` proves that no treatment module *imports* evaluator
    material. A backend constructed elsewhere and passed to this seam bypasses that proof
    entirely, so the seam re-checks the same ``EVALUATOR_ONLY`` marker on the class it is handed.
    """
    import sys
    import types

    module = types.ModuleType("shapeflow_test_answer_key_world")
    module.EVALUATOR_ONLY = True
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class GoldBackend:
        def search(self, query, *, max_results):
            return []

    GoldBackend.__module__ = module.__name__

    with pytest.raises(ValueError, match="marked EVALUATOR_ONLY"):
        with install_frozen_search(max_results=5, backend=GoldBackend()):
            pass


def test_a_backend_whose_defining_module_cannot_be_found_is_refused():
    """Unresolvable is not innocent: this is the last point that asks what world will be served."""

    class Nowhere:
        def search(self, query, *, max_results):
            return []

    Nowhere.__module__ = "a.module.that.was.never.imported"
    with pytest.raises(ValueError, match="cannot tell which module defines"):
        with install_frozen_search(max_results=5, backend=Nowhere()):
            pass


@pytest.mark.parametrize("bad", [0, -1, "5", 5.0, True])
async def test_a_non_positive_top_k_from_the_graph_is_refused(bad, vendor_utils):
    """The install-time check guards a number the graph never uses.

    Vendor's ``tavily_search`` passes its own ``max_results`` on every call to
    ``tavily_search_async`` (``open_deep_research/utils.py``, step 1), so the default validated at
    install time is not what any query is answered with. Before this was checked per call,
    ``tavily_search_async(["q"], max_results=0)`` returned ``[{"query": "q", "results": []}]`` --
    a frozen world that holds nothing, and an empty result set, written identically in the trace.
    """
    backend = ListBackend({"q": [record("https://a.invalid", "body")]})
    with install_frozen_search(max_results=5, backend=backend):
        with pytest.raises(ValueError, match="top-k must be a positive int"):
            await vendor_utils.tavily_search_async(["q"], max_results=bad)
    # The world was never asked: refusing after retrieving would already have served the results.
    assert backend.calls == []


async def test_the_graph_may_not_ask_for_a_top_k_other_than_the_installed_one(vendor_utils):
    """The recorded top-k and the served top-k must be the same number, not two that agree."""
    backend = ListBackend({"q": [record("https://a.invalid", "body")]})
    # Installed with the frozen top-k; asked with vendor's hard-coded 5.
    with install_frozen_search(max_results=3, backend=backend):
        with pytest.raises(ValueError, match="asked the frozen world for top-5"):
            await vendor_utils.tavily_search_async(["q"], max_results=5)
    assert backend.calls == []


async def test_the_graphs_own_tool_cannot_widen_the_frozen_top_k(vendor_utils):
    """End to end through vendor's tool, which is where the divergence actually came from."""
    backend = ListBackend({"q": [record("https://a.invalid", "body")]})
    with install_frozen_search(max_results=3, backend=backend):
        with pytest.raises(ValueError, match="asked the frozen world for top-5"):
            await vendor_utils.tavily_search.ainvoke({"queries": ["q"]})
    assert backend.calls == []


async def test_a_backend_that_returns_more_than_the_top_k_is_refused(vendor_utils):
    """Extra pages look like ordinary results, so only the seam can see the world widened."""

    class OverServingBackend:
        def search(self, query, *, max_results):
            return [record(f"https://a{i}.invalid", "body", occurrence_id=f"occ-{i}")
                    for i in range(max_results + 1)]

    with install_frozen_search(max_results=5, backend=OverServingBackend()):
        with pytest.raises(ValueError, match="returned 6 results for top-5"):
            await vendor_utils.tavily_search_async(["q"], max_results=5)


async def test_a_backend_that_returns_the_wrong_record_type_is_refused(vendor_utils):
    """Dicts cannot be budgeted, so an unbudgeted arm would reach the graph looking normal."""

    class DictBackend:
        def search(self, query, *, max_results):
            return [{"url": "https://a.invalid", "raw_content": WORDS}]

    with install_frozen_search(max_results=5, backend=DictBackend(),
                               content_budget=TIGHT, tokenizer=WhitespaceTokenizer()):
        with pytest.raises(TypeError, match="returned a dict"):
            await vendor_utils.tavily_search_async(["q"])


async def test_a_backend_that_returns_a_generator_is_refused(vendor_utils):
    """A generator is consumable once: the trace's count would be right and the payload empty."""

    class LazyBackend:
        def search(self, query, *, max_results):
            return (r for r in [record("https://a.invalid", "body")])

    with install_frozen_search(max_results=5, backend=LazyBackend()):
        with pytest.raises(TypeError, match="not a list of SearchRecord"):
            await vendor_utils.tavily_search_async(["q"])
