"""Op classes: what kind of work each model request is.

Every model request is tagged with an op class so the work ledger can attribute cost to the
right part of the pipeline and, crucially, keep P0 and P1 costs separable and keep the judge's
cost out of the treatment total entirely. These are the classes from plan §12.1.
"""

from __future__ import annotations

import enum

__all__ = ["OpClass", "TREATMENT_OPS", "JUDGE_OPS", "REMOTE_ALLOWED_OPS", "is_treatment_work"]


class OpClass(enum.Enum):
    # WEBPAGE boundary
    PAGE_P0_SUMMARY = "PAGE_P0_SUMMARY"
    PAGE_P1_SELECTOR_LOCAL = "PAGE_P1_SELECTOR_LOCAL"
    PAGE_P1_SELECTOR_GLOBAL = "PAGE_P1_SELECTOR_GLOBAL"
    # researcher loop
    RESEARCHER_REACT = "RESEARCHER_REACT"
    # RESEARCHER_CLOSE boundary
    COMPRESSOR_P0 = "COMPRESSOR_P0"
    COMPRESSOR_P1_SELECTOR = "COMPRESSOR_P1_SELECTOR"
    # supervisor / writer
    SUPERVISOR_CONTINUE = "SUPERVISOR_CONTINUE"
    FINAL_WRITER = "FINAL_WRITER"
    # evaluator (DeepSeek) -- never treatment work
    JUDGE_ATOMIZE = "JUDGE_ATOMIZE"
    JUDGE_TRUTH = "JUDGE_TRUTH"
    JUDGE_REPORT = "JUDGE_REPORT"
    JUDGE_EXPLAIN = "JUDGE_EXPLAIN"
    # steward (DeepSeek), before any treatment output exists -- corpus authoring and the
    # arm-independent AcquisitionSpec decomposition the plan permits in §3.4. Listed apart from
    # the judge ops because it is spent before the experiment starts, not while scoring it.
    TASK_AUTHOR = "TASK_AUTHOR"


#: Op classes that count as treatment GPU work.
TREATMENT_OPS = frozenset({
    OpClass.PAGE_P0_SUMMARY,
    OpClass.PAGE_P1_SELECTOR_LOCAL,
    OpClass.PAGE_P1_SELECTOR_GLOBAL,
    OpClass.RESEARCHER_REACT,
    OpClass.COMPRESSOR_P0,
    OpClass.COMPRESSOR_P1_SELECTOR,
    OpClass.SUPERVISOR_CONTINUE,
    OpClass.FINAL_WRITER,
})

#: Judge op classes. Reported separately as API cost; never added to treatment GPU work.
JUDGE_OPS = frozenset({OpClass.JUDGE_ATOMIZE, OpClass.JUDGE_TRUTH, OpClass.JUDGE_REPORT,
                       OpClass.JUDGE_EXPLAIN})

#: Everything the remote (DeepSeek) provider is permitted to serve. The complement of this set
#: within OpClass is the treatment path, and a remote model must never appear there: a second
#: model inside the system under measurement would make every arm's result partly its output.
REMOTE_ALLOWED_OPS = JUDGE_OPS | {OpClass.TASK_AUTHOR}


def is_treatment_work(op: OpClass) -> bool:
    return op in TREATMENT_OPS
