"""Comparing a patched hooks-off run against vendor, byte for byte.

P0 parity is a hard gate: if the patched graph with hooks off does not reproduce vendor,
then any P1 effect we later measure is confounded with an accidental change in the harness,
and no GPU screening may start. This module holds the *comparison* -- the record types and
the diff logic -- which is langchain-free and unit-testable. The trace *generation* (running
both graphs against a mock model and search) lives in the integration test, where the ODR
runtime is importable.

Two equality levels, because the pinned model may or may not be deterministic:

- ``require_output_equality=True`` (mock deterministic model): everything must match,
  including model outputs and the final report. This is the strong fixture the plan calls
  for and the one the gate runs.
- ``require_output_equality=False`` (real sampling): only the *application* bytes must
  match -- the request envelopes (rendered prompt, tools, op class, sampling), the tool
  message bytes, the sequence, and the close reasons. Model outputs may legitimately differ
  run to run, so they are excluded, exactly as the plan permits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "ModelRequest", "ToolMessageRecord", "PublishBatch", "RunTrace",
    "ParityReport", "compare_traces",
]


@dataclass(frozen=True)
class ModelRequest:
    """One model call as the graph issued it. ``prompt_sha256`` digests the exact rendered
    prompt bytes; ``output_sha256`` digests the response (compared only in strong mode)."""

    op_class: str
    prompt_sha256: str
    tools_signature: str  # canonical signature of the bound tool schemas, in order
    model: str
    sampling_signature: str
    output_sha256: str


@dataclass(frozen=True)
class ToolMessageRecord:
    tool_call_id: str
    name: str
    content_sha256: str


@dataclass(frozen=True)
class PublishBatch:
    """One atomic publish: every sibling ToolMessage of one assistant turn, in pinned order.

    The batch boundary is the fact being compared, so it has to be *in* the trace. A flat list
    of tool messages cannot tell ``[[A, B]]`` -- one atomic publish -- from ``[[A], [B]]``, two
    partial ones, and that distinction is the central claim of the whole H design: P1 must
    never publish half a batch, and a P1 failure must take the whole batch to P0 rather than
    leaving a ``[P1(A), P0(B)]`` hybrid.
    """

    messages: tuple[ToolMessageRecord, ...] = ()
    goto: str = ""                    # the Command's goto target
    state_update_sha256: str = ""     # digest of the state update the Command carried
    checkpoint_digest: str = ""       # the H checkpoint this batch was published from
    fallback: str = ""                # "" | "WHOLE_BATCH_P0" | the failure that forced it


@dataclass(frozen=True)
class RunTrace:
    """An ordered record of one graph execution's observable facts.

    Beyond the request envelopes and the final report, it records what the graph *did with*
    each batch: where it went next, what it wrote to state, which checkpoint it forked from,
    and whether it fell back. A patched graph that produced identical bytes while routing
    differently, or while silently falling back, would otherwise compare equal to vendor.
    """

    model_requests: tuple[ModelRequest, ...] = ()
    publish_batches: tuple[PublishBatch, ...] = ()
    close_reasons: tuple[str, ...] = ()
    exceptions: tuple[str, ...] = ()   # type names, in order; a swallowed error is a difference
    final_report_sha256: str = ""

    @property
    def tool_messages(self) -> tuple[ToolMessageRecord, ...]:
        """Flattened view, for assertions that genuinely do not care about batching."""
        return tuple(m for batch in self.publish_batches for m in batch.messages)


@dataclass
class ParityReport:
    diffs: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.diffs

    def add(self, msg: str) -> None:
        self.diffs.append(msg)


def _request_envelope(r: ModelRequest) -> tuple:
    # Everything about a request except the model's output.
    return (r.op_class, r.prompt_sha256, r.tools_signature, r.model, r.sampling_signature)


def compare_traces(
    vendor: RunTrace, patched: RunTrace, *, require_output_equality: bool
) -> ParityReport:
    """Return a report; ``ok`` iff the patched run matches vendor at the required level."""
    report = ParityReport()

    # --- model requests ---
    if len(vendor.model_requests) != len(patched.model_requests):
        report.add(
            f"model request count: vendor={len(vendor.model_requests)} "
            f"patched={len(patched.model_requests)}"
        )
    else:
        for i, (v, p) in enumerate(zip(vendor.model_requests, patched.model_requests)):
            if _request_envelope(v) != _request_envelope(p):
                report.add(f"model request[{i}] envelope differs: {v} != {p}")
            elif require_output_equality and v.output_sha256 != p.output_sha256:
                report.add(
                    f"model request[{i}] output differs (strong mode): "
                    f"{v.output_sha256[:12]} != {p.output_sha256[:12]}"
                )

    # --- publish batches (application bytes AND the batch boundary) ---
    if len(vendor.publish_batches) != len(patched.publish_batches):
        report.add(
            f"publish batch count: vendor={len(vendor.publish_batches)} "
            f"patched={len(patched.publish_batches)} -- a differing batch count means the "
            "patched graph split or merged an atomic publish"
        )
    else:
        for i, (v, p) in enumerate(zip(vendor.publish_batches, patched.publish_batches)):
            if v.messages != p.messages:
                report.add(f"publish batch[{i}] messages differ: {v.messages} != {p.messages}")
            if v.goto != p.goto:
                report.add(f"publish batch[{i}] goto differs: {v.goto!r} != {p.goto!r}")
            if v.state_update_sha256 != p.state_update_sha256:
                report.add(
                    f"publish batch[{i}] state update differs: "
                    f"{v.state_update_sha256[:12]} != {p.state_update_sha256[:12]}"
                )
            if p.fallback:
                report.add(
                    f"publish batch[{i}] fell back ({p.fallback}); a hooks-off run must take "
                    "the vendor path directly, never reach it via a fallback"
                )

    # --- exceptions (a swallowed error is a behavioural difference, not a detail) ---
    if vendor.exceptions != patched.exceptions:
        report.add(
            f"exceptions differ: vendor={vendor.exceptions} patched={patched.exceptions}"
        )

    # --- close reasons (the three exit paths must be taken identically) ---
    if vendor.close_reasons != patched.close_reasons:
        report.add(
            f"close reasons differ: vendor={vendor.close_reasons} patched={patched.close_reasons}"
        )

    # --- final report (strong mode only) ---
    if require_output_equality and vendor.final_report_sha256 != patched.final_report_sha256:
        report.add(
            f"final report differs (strong mode): "
            f"{vendor.final_report_sha256[:12]} != {patched.final_report_sha256[:12]}"
        )

    return report
