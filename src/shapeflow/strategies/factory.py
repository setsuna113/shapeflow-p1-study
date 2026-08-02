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
from .p0 import VendorCloseStrategy, VendorPageStrategy
from .page_h import PAGE_SCOPES, PageSelectionStrategy, PageStrategyConfig
from .prose import ProseCloseStrategy, ProsePageStrategy
from .selectors_async import CpuLexicalAsyncSelector, LlmAsyncSelector, ShortProseSelector

__all__ = [
    "VariantSpec",
    "StrategyFactory",
    "UnknownVariant",
    "VariantUnavailable",
    "load_registry",
]


class UnknownVariant(KeyError):
    """A variant id that is not in the frozen registry. Never built."""


class VariantUnavailable(RuntimeError):
    """A registered proposal arm whose claimed treatment is not implemented faithfully."""


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    node: str
    chunker: str
    scope: str
    contract: str
    aggregation: str
    close_mode: str
    selector_backend: str
    publication_path: str
    output_representation: str
    is_control: bool = False
    bridge_token_cap_each: int | None = None
    bridge_token_cap_total: int | None = None
    #: Input-side budget policy: "none" or "prompt_pack_v1". Distinct from the output-side
    #: rendered-token budget; see Amendment 1.
    prompt_admission: str = "none"
    #: Diagnostic arms are measured but may never be promoted to champion.
    diagnostic_only: bool = False
    runnable: bool = True
    unavailable_reason: str = ""


def load_registry(configs: Path) -> dict[str, VariantSpec]:
    data, _ = load_config(Path(configs) / "variants.yaml")
    registry: dict[str, VariantSpec] = {}
    for v in data["variants"]:
        variant_id = v["variant_id"]
        if variant_id in registry:
            raise ValueError(
                f"duplicate variant_id {variant_id!r}; silently overwriting a treatment changes "
                "what the frozen design means"
            )
        spec = VariantSpec(
            variant_id=v["variant_id"], node=v["node"], chunker=v["chunker"],
            scope=v["scope"], contract=v["contract"], aggregation=v["aggregation"],
            close_mode=v["close_mode"],
            selector_backend=v["selector_backend"],
            publication_path=v["publication_path"],
            output_representation=v["output_representation"],
            is_control=bool(v.get("is_control", False)),
            bridge_token_cap_each=v.get("bridge_token_cap_each"),
            bridge_token_cap_total=v.get("bridge_token_cap_total"),
            prompt_admission=str(v.get("prompt_admission", "none")),
            diagnostic_only=bool(v.get("diagnostic_only", False)),
            runnable=bool(v.get("runnable", True)),
            unavailable_reason=str(v.get("unavailable_reason", "")),
        )
        _validate_spec(spec)
        registry[variant_id] = spec
    return registry


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
        #: Input-side budget: room for candidate material in a selector prompt, after the
        #: instructions, schema, completion ceiling and margin are paid for. Distinct from
        #: `token_budget`, which bounds the *rendered output*. Amendment 1.
        prompt_budget: int = 0,
        prompt_window_ceiling: int = 0,
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
        self._prompt_budget = prompt_budget
        self._prompt_window_ceiling = prompt_window_ceiling
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
        if not spec.runnable:
            raise VariantUnavailable(
                f"{variant_id} is registered as planned but not runnable: "
                f"{spec.unavailable_reason}"
            )
        return StrategyBundle(
            variant_id=variant_id,
            page=self._page_half(spec),
            close=self._close_half(spec),
        )

    def build_all(self) -> dict[str, StrategyBundle]:
        """Instantiate every runnable arm; planned arms remain explicitly unavailable.

        A planned extension is useful protocol metadata, but returning a fallback object for it
        makes a null result look like an implemented treatment.  ``build`` rejects those ids;
        startup validates all specs and exercises the runnable subset here.
        """
        return {
            vid: self.build(vid)
            for vid, spec in sorted(self.registry.items())
            if spec.runnable
        }

    def build_joint(self, page_variant: str, close_variant: str) -> StrategyBundle:
        """An H+C arm: the two single arms, composed.

        Built from the same halves the single arms use, so the interaction cell cannot drift
        from the main-effect cells it is meant to combine with.
        """
        page_spec = self._available_spec(page_variant)
        close_spec = self._available_spec(close_variant)
        return StrategyBundle(
            variant_id=f"{page_variant}+{close_variant}",
            page=self._page_half(page_spec),
            close=self._close_half(close_spec),
        )

    def _available_spec(self, variant_id: str) -> VariantSpec:
        spec = self.registry.get(variant_id)
        if spec is None:
            raise UnknownVariant(variant_id)
        if not spec.runnable:
            raise VariantUnavailable(
                f"{variant_id} is registered as planned but not runnable: "
                f"{spec.unavailable_reason}"
            )
        return spec

    # --- halves ------------------------------------------------------------------------

    def _page_half(self, spec: VariantSpec):
        if spec.node not in _PAGE_NODES:
            # A close-only arm still runs vendor's page path: the two nodes are measured
            # separately, so a C arm must leave H at P0 or the contrast is confounded.
            return VendorPageStrategy({})
        if spec.selector_backend == "CPU_LEXICAL":
            return self._page_strategy(spec, CpuLexicalAsyncSelector())
        if spec.contract == "SHORT_PROSE":
            return ProsePageStrategy(
                config=self._page_config(spec),
                selector=ShortProseSelector(self._model_call, op_class="PAGE_P1_SHORT_PROSE"),
                tokenizer=self._tokenizer, raw_text_for=self._raw_text_for,
                occurrence_for=self._occurrence_for, work_sink=self._work_sink,
            )
        local = LlmAsyncSelector(
            self._model_call,
            # The op class follows the *unit*, not the boundary. A whole-batch arm issues one
            # request per gather batch where a per-page arm issues one per page, so recording
            # both under PAGE_P1_SELECTOR_LOCAL would sum two mechanisms into one line of the
            # work ledger and average away the very difference this rebuild measures.
            op_class=("PAGE_P1_SELECTOR_BATCH" if spec.scope == "whole_batch"
                      else "PAGE_P1_SELECTOR_LOCAL"),
            expected_contract=spec.contract,
        )
        if spec.scope == "hierarchical":
            global_selector = LlmAsyncSelector(
                self._model_call, op_class="PAGE_P1_SELECTOR_GLOBAL",
                expected_contract=spec.contract,
            )
            return self._page_strategy(spec, local, global_selector=global_selector)
        return self._page_strategy(spec, local)

    def _page_strategy(self, spec: VariantSpec, selector, *, global_selector=None):
        return PageSelectionStrategy(
            self._page_config(spec), selector=selector, tokenizer=self._tokenizer,
            raw_text_for=self._raw_text_for, occurrence_for=self._occurrence_for,
            global_selector=global_selector,
            work_sink=self._work_sink,
        )

    def _page_config(self, spec: VariantSpec) -> PageStrategyConfig:
        return PageStrategyConfig(
            variant_id=spec.variant_id, chunker=spec.chunker, scope=spec.scope,
            contract=spec.contract if spec.contract.startswith("P1_") else "P1_ID",
            aggregation=spec.aggregation, token_budget=self._token_budget,
            bridge_token_cap_each=spec.bridge_token_cap_each,
            bridge_token_cap_total=spec.bridge_token_cap_total,
            prompt_admission=spec.prompt_admission,
            prompt_budget=self._prompt_budget,
            # The ceiling belongs only to an LLM arm that runs without admission -- the one
            # shape that can build a prompt the engine will refuse. An arm with admission
            # cannot exceed the window by construction, and a CPU arm has no window at all:
            # giving CPU-FULL a ceiling would mark 61.9% of batches infeasible for the one arm
            # that is feasible on all of them, which is the opposite of what it measures.
            prompt_window_ceiling=(
                self._prompt_window_ceiling
                if spec.prompt_admission == "none" and spec.selector_backend == "LLM"
                else 0),
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
            scope=spec.scope,
            bridge_token_cap_each=spec.bridge_token_cap_each,
            bridge_token_cap_total=spec.bridge_token_cap_total,
        )
        if spec.selector_backend == "CPU_LEXICAL":
            selector = CpuLexicalAsyncSelector()
        elif spec.contract == "SHORT_PROSE":
            return ProseCloseStrategy(
                config=config,
                selector=ShortProseSelector(self._model_call, op_class="COMPRESSOR_SHORT_PROSE"),
                tokenizer=self._tokenizer, work_sink=self._work_sink,
            )
        else:
            selector = LlmAsyncSelector(
                self._model_call, op_class="COMPRESSOR_P1_SELECTOR",
                expected_contract=spec.contract,
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


def _validate_spec(spec: VariantSpec) -> None:
    """Reject registry entries whose names and executable semantics disagree."""
    if not spec.variant_id:
        raise ValueError("variant_id must be non-empty")
    if spec.node not in {"P0", *_PAGE_NODES, *_CLOSE_NODES}:
        raise ValueError(f"{spec.variant_id}: unknown node {spec.node!r}")
    if spec.contract not in {
        "VENDOR_PROSE", "SHORT_PROSE", "P1_ID", "P1_TYPED", "P1_BRIDGE",
    }:
        raise ValueError(f"{spec.variant_id}: unknown contract {spec.contract!r}")
    if spec.selector_backend not in {"VENDOR", "LLM", "CPU_LEXICAL"}:
        raise ValueError(
            f"{spec.variant_id}: unknown selector_backend {spec.selector_backend!r}"
        )
    if spec.publication_path not in {
        "VENDOR_PROSE",
        "STRUCTURED_SELECTION",
        "DIRECT_PROSE",
        "PREFIX_PRESERVING_SELECTION",
        "FUSED_SELECTION",
    }:
        raise ValueError(
            f"{spec.variant_id}: unknown publication_path {spec.publication_path!r}"
        )
    if spec.output_representation not in {
        "PROSE", "SHORT_PROSE", "EVIDENCE_IDS", "TYPED_EVIDENCE", "BRIDGED_EVIDENCE",
    }:
        raise ValueError(
            f"{spec.variant_id}: unknown output_representation "
            f"{spec.output_representation!r}"
        )
    if spec.runnable and spec.unavailable_reason:
        raise ValueError(
            f"{spec.variant_id}: runnable arm must not carry unavailable_reason"
        )
    if not spec.runnable and not spec.unavailable_reason:
        raise ValueError(
            f"{spec.variant_id}: non-runnable arm needs an explicit unavailable_reason"
        )

    caps = (spec.bridge_token_cap_each, spec.bridge_token_cap_total)
    if spec.contract == "P1_BRIDGE":
        if any(v is None for v in caps):
            raise ValueError(
                f"{spec.variant_id}: P1_BRIDGE requires bridge_token_cap_each and "
                "bridge_token_cap_total"
            )
        if not (0 < int(caps[0]) <= int(caps[1])):
            raise ValueError(f"{spec.variant_id}: bridge caps require 0 < each <= total")
    elif any(v is not None for v in caps):
        raise ValueError(f"{spec.variant_id}: bridge caps are valid only for P1_BRIDGE")

    if spec.node == "P0":
        expected = (
            "vendor", "vendor", "VENDOR_PROSE", "vendor", "vendor",
            "VENDOR", "VENDOR_PROSE", "PROSE",
        )
        got = (
            spec.chunker, spec.scope, spec.contract, spec.aggregation, spec.close_mode,
            spec.selector_backend, spec.publication_path, spec.output_representation,
        )
        if got != expected:
            raise ValueError(f"{spec.variant_id}: P0 semantics drifted: {got!r}")
        return

    if spec.node in _PAGE_NODES:
        if spec.scope not in PAGE_SCOPES:
            raise ValueError(f"{spec.variant_id}: invalid page scope {spec.scope!r}")
        if spec.close_mode not in {"none", "separate"}:
            raise ValueError(f"{spec.variant_id}: page arm changed close mode")
    else:
        if spec.scope != "per_researcher":
            raise ValueError(f"{spec.variant_id}: close arm must use per_researcher scope")
        if spec.runnable and spec.close_mode not in {"dedicated_selector", "separate"}:
            raise ValueError(
                f"{spec.variant_id}: runnable close_mode {spec.close_mode!r} is not implemented"
            )

    if spec.selector_backend == "CPU_LEXICAL":
        if (
            not spec.is_control
            or spec.contract != "P1_ID"
            or spec.publication_path != "STRUCTURED_SELECTION"
            or spec.output_representation != "EVIDENCE_IDS"
        ):
            raise ValueError(
                f"{spec.variant_id}: CPU_LEXICAL must be a P1_ID structured-selection control"
            )
    if spec.contract == "SHORT_PROSE":
        if (
            not spec.is_control
            or spec.selector_backend != "LLM"
            or spec.publication_path != "DIRECT_PROSE"
            or spec.output_representation != "SHORT_PROSE"
        ):
            raise ValueError(
                f"{spec.variant_id}: SHORT_PROSE must be an LLM direct-prose control"
            )
    if (
        spec.contract.startswith("P1_")
        and spec.is_control
        and spec.selector_backend != "CPU_LEXICAL"
    ):
        raise ValueError(f"{spec.variant_id}: P1 treatment cannot be marked as a control")
