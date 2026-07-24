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

__all__ = ["ModelRequest", "ToolMessageRecord", "RunTrace", "ParityReport", "compare_traces"]


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
class RunTrace:
    """An ordered record of one graph execution's observable facts."""

    model_requests: tuple[ModelRequest, ...] = ()
    tool_messages: tuple[ToolMessageRecord, ...] = ()
    close_reasons: tuple[str, ...] = ()
    final_report_sha256: str = ""


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

    # --- tool messages (application bytes, always compared) ---
    if len(vendor.tool_messages) != len(patched.tool_messages):
        report.add(
            f"tool message count: vendor={len(vendor.tool_messages)} "
            f"patched={len(patched.tool_messages)}"
        )
    else:
        for i, (v, p) in enumerate(zip(vendor.tool_messages, patched.tool_messages)):
            if v != p:
                report.add(f"tool message[{i}] differs: {v} != {p}")

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
