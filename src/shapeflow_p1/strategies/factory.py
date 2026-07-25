"""Building every arm in the registry, and refusing to build one that is not in it.

The factory is the single place a variant id becomes running code. That matters for two reasons.

**Nothing runs that was not pre-registered.** An arm assembled ad hoc somewhere else would be an
unregistered treatment appearing in the results, and the design's balance would silently no
longer describe what ran.

**Every registered arm must be constructible before anything runs.** `configs/variants.yaml`
listed sixteen variants for a long time while production code implemented none of them; a
campaign would have discovered that one arm at a time, mid-run, with budget already spent.
:func:`build_all` instantiates the whole registry up front so that failure happens at startup.

The controls are not optional and not decoration. Without CPU_LEXICAL, "the LLM selector helped"
cannot be separated from "a lexical ranker would have done the same". Without SHORT_PROSE, a
token saving cannot be separated from "we asked for less text". Their absence would leave every
observed effect with at least three explanations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from ..config import load_config
from ..evidence.chunkers import Tokenizer, WhitespaceTokenizer
from ..odr.hooks import StrategyBundle
from .close_visible import CloseSelectionStrategy, CloseStrategyConfig
from .fused import FusedCloseStrategy
from .p0 import VendorCloseStrategy, VendorPageStrategy
from .page_h import PageSelectionStrategy, PageStrategyConfig
from .prose import ProseCloseStrategy, ProsePageStrategy
from .selectors_async import CpuLexicalAsyncSelector, LlmAsyncSelector, ShortProseSelector

__all__ = ["VariantSpec", "StrategyFactory", "UnknownVariant", "load_registry"]


class UnknownVariant(KeyError):
    """A variant id that is not in the frozen registry. Never built."""


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    node: str
    chunker: str
    scope: str
    contract: str
    aggregation: str
    close_mode: str
    is_control: bool = False


def load_registry(configs: Path) -> dict[str, VariantSpec]:
    data, _ = load_config(Path(configs) / "variants.yaml")
    return {
        v["variant_id"]: VariantSpec(
            variant_id=v["variant_id"], node=v["node"], chunker=v["chunker"],
            scope=v["scope"], contract=v["contract"], aggregation=v["aggregation"],
            close_mode=v["close_mode"], is_control=bool(v.get("is_control", False)),
        )
        for v in data["variants"]
    }


# Which node each variant's page/close half belongs to. H+C combinations are composed from the
# two halves rather than declared as a third thing, so a joint arm cannot drift from the single
# arms it is supposed to be the combination of.
_PAGE_NODES = {"WEBPAGE_P1"}
_CLOSE_NODES = {"C_VISIBLE", "C_REGISTRY", "C_FUSED_EXT"}


class StrategyFactory:
    """Turns a frozen VariantSpec into a bound StrategyBundle."""

    def __init__(
        self,
        *,
        registry: dict[str, VariantSpec],
        model_call: Callable,
        tokenizer: Optional[Tokenizer] = None,
        token_budget: int = 512,
        raw_text_for: Optional[Callable[[str], str]] = None,
        occurrence_for: Optional[Callable[[str], str]] = None,
        raw_spans_for: Optional[Callable[[Any], list]] = None,
        snapshot_texts_for: Optional[Callable[[Any], dict]] = None,
        work_sink: Optional[Callable] = None,
    ) -> None:
        self.registry = registry
        self._model_call = model_call
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_budget = token_budget
        self._raw_text_for = raw_text_for or (lambda cid: "")
        self._occurrence_for = occurrence_for or (lambda cid: cid)
        self._raw_spans_for = raw_spans_for
        self._snapshot_texts_for = snapshot_texts_for
        self._work_sink = work_sink

    # --- public ------------------------------------------------------------------------

    def build(self, variant_id: str) -> StrategyBundle:
        spec = self.registry.get(variant_id)
        if spec is None:
            raise UnknownVariant(
                f"{variant_id!r} is not in the frozen registry. An arm that was not "
                "pre-registered must not run: the design's balance would no longer describe "
                "what executed."
            )
        return StrategyBundle(
            variant_id=variant_id,
            page=self._page_half(spec),
            close=self._close_half(spec),
        )

    def build_all(self) -> dict[str, StrategyBundle]:
        """Instantiate the entire registry, so a missing arm fails at startup not mid-run."""
        return {vid: self.build(vid) for vid in sorted(self.registry)}

    def build_joint(self, page_variant: str, close_variant: str) -> StrategyBundle:
        """An H+C arm: the two single arms, composed.

        Built from the same halves the single arms use, so the interaction cell cannot drift
        from the main-effect cells it is meant to combine with.
        """
        page_spec = self.registry[page_variant]
        close_spec = self.registry[close_variant]
        return StrategyBundle(
            variant_id=f"{page_variant}+{close_variant}",
            page=self._page_half(page_spec),
            close=self._close_half(close_spec),
        )

    # --- halves ------------------------------------------------------------------------

    def _page_half(self, spec: VariantSpec):
        if spec.node not in _PAGE_NODES:
            # A close-only arm still runs vendor's page path: the two nodes are measured
            # separately, so a C arm must leave H at P0 or the contrast is confounded.
            return VendorPageStrategy({})
        if spec.contract == "CPU_LEXICAL":
            return self._page_strategy(spec, CpuLexicalAsyncSelector())
        if spec.contract == "SHORT_PROSE":
            return ProsePageStrategy(
                config=self._page_config(spec),
                selector=ShortProseSelector(self._model_call, op_class="PAGE_P1_SHORT_PROSE"),
                tokenizer=self._tokenizer, raw_text_for=self._raw_text_for,
                occurrence_for=self._occurrence_for, work_sink=self._work_sink,
            )
        op = "PAGE_P1_SELECTOR_GLOBAL" if spec.scope == "hierarchical" \
            else "PAGE_P1_SELECTOR_LOCAL"
        return self._page_strategy(spec, LlmAsyncSelector(self._model_call, op_class=op))

    def _page_strategy(self, spec: VariantSpec, selector):
        return PageSelectionStrategy(
            self._page_config(spec), selector=selector, tokenizer=self._tokenizer,
            raw_text_for=self._raw_text_for, occurrence_for=self._occurrence_for,
            work_sink=self._work_sink,
        )

    def _page_config(self, spec: VariantSpec) -> PageStrategyConfig:
        return PageStrategyConfig(
            variant_id=spec.variant_id, chunker=spec.chunker, scope=spec.scope,
            contract=spec.contract if spec.contract.startswith("P1_") else "P1_ID",
            aggregation=spec.aggregation, token_budget=self._token_budget,
        )

    def _close_half(self, spec: VariantSpec):
        if spec.node not in _CLOSE_NODES:
            return VendorCloseStrategy()
        if spec.close_mode == "none":
            # A CPU control on the page node leaves the close boundary at P0, so the two nodes
            # stay separable.
            return VendorCloseStrategy()
        config = CloseStrategyConfig(
            variant_id=spec.variant_id, node=spec.node,
            contract=spec.contract if spec.contract.startswith("P1_") else "P1_ID",
            aggregation=spec.aggregation, close_mode=spec.close_mode,
            token_budget=self._token_budget, chunker=spec.chunker,
        )
        if spec.contract == "CPU_LEXICAL":
            selector = CpuLexicalAsyncSelector()
        elif spec.contract == "SHORT_PROSE":
            return ProseCloseStrategy(
                config=config,
                selector=ShortProseSelector(self._model_call, op_class="COMPRESSOR_SHORT_PROSE"),
                tokenizer=self._tokenizer, work_sink=self._work_sink,
            )
        else:
            selector = LlmAsyncSelector(self._model_call, op_class="COMPRESSOR_P1_SELECTOR")

        if spec.close_mode == "fused_with_fallback":
            # The fused variant changes the stopping policy, not just the reducer, and
            # ResearchComplete has an empty schema -- so it cannot be a close-after hook. It
            # keeps the dedicated selector as the fallback for the exits that never call it.
            return FusedCloseStrategy(
                config=config, selector=selector, tokenizer=self._tokenizer,
                fallback=CloseSelectionStrategy(
                    config=config, selector=selector, tokenizer=self._tokenizer,
                    work_sink=self._work_sink),
                work_sink=self._work_sink,
            )

        # C_REGISTRY gets the raw-span accessors; C_VISIBLE is constructed without them, so it
        # has no means to read raw page bytes rather than a rule saying it should not.
        registry_access = spec.node == "C_REGISTRY"
        return CloseSelectionStrategy(
            config=config, selector=selector, tokenizer=self._tokenizer,
            raw_spans_for=self._raw_spans_for if registry_access else None,
            snapshot_texts_for=self._snapshot_texts_for if registry_access else None,
            work_sink=self._work_sink,
        )
