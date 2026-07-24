"""Classifying *why* a researcher closed.

The plan's RESEARCHER_CLOSE experiment insists all three exit paths enter the same
treatment, and that the subset which naturally calls ``ResearchComplete`` must not be
mistaken for the whole. So the close reason is a first-class, recorded fact, derived here
from exactly the signals vendor ``researcher_tools`` uses to decide it
(``deep_researcher.py`` lines 451-509):

- ``NO_TOOL_CALL`` -- the researcher emitted no tool calls and no native search. Vendor
  routes straight to ``compress_research`` with no tool outputs.
- ``RESEARCH_COMPLETE`` -- the researcher explicitly called the ``ResearchComplete`` tool.
- ``MAX_REACT_EXCEEDED`` -- the react-tool-call budget ran out.

``None`` means "not a close": the researcher will loop again. Classification is pure and
mirrors vendor precedence, so the patched code and the vendor code agree on which path a
given turn takes -- a prerequisite for P0 parity.
"""

from __future__ import annotations

import enum

__all__ = ["CloseReason", "classify_close"]


class CloseReason(enum.Enum):
    NO_TOOL_CALL = "NO_TOOL_CALL"
    RESEARCH_COMPLETE = "RESEARCH_COMPLETE"
    MAX_REACT_EXCEEDED = "MAX_REACT_EXCEEDED"


def classify_close(
    *,
    has_tool_calls: bool,
    has_native_search: bool,
    research_complete_called: bool,
    tool_call_iterations: int,
    max_react_tool_calls: int,
) -> CloseReason | None:
    """Return the close reason, or None if the researcher continues.

    The branch order reproduces vendor ``researcher_tools``:

    1. No tool calls and no native search is the *early* exit, taken before any tool runs.
    2. Otherwise tools run, then the late exit is checked: ``ResearchComplete`` called, or
       the iteration budget exceeded. ``ResearchComplete`` takes precedence when both hold,
       because it is the explicit signal and the more informative label.
    """
    # Early exit (vendor: `if not has_tool_calls and not has_native_search`).
    if not has_tool_calls and not has_native_search:
        return CloseReason.NO_TOOL_CALL

    # Late exit (vendor: `exceeded_iterations or research_complete_called`).
    exceeded = tool_call_iterations >= max_react_tool_calls
    if research_complete_called:
        return CloseReason.RESEARCH_COMPLETE
    if exceeded:
        return CloseReason.MAX_REACT_EXCEEDED

    # Tools ran and neither late-exit condition held: the researcher loops again.
    return None
