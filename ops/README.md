# ops/ — live-campaign analysis scripts

Twenty-two throwaway-looking but hand-written scripts used to watch and analyse the
Week-1 campaign while it ran on sjtu (2026-08-01 → 08-02): arm accounting (`arms.py`,
`fail_by_arm.py`, `paired.py`), progress and ETA (`status.py`, `eta.py`, `trend.py`,
`peek.py`, `count.py`), failure triage (`fails.py`, `why.py`, `overrun.py`),
quality/recall checks (`recall.py`, `pos.py`, `censor.py`), and the steward/evaluator
drivers (`steward.sh`, `evaluate.sh`, `preflight.py`, `bcplus.sh`, `pub.py`,
`states.py`, `ccpu.py`).

They lived unversioned at `sjtu:/storage/nvme/ops` and were written against the running
campaign, so they encode the operational reality of the run in a way the repo proper
does not. Preserved verbatim; not cleaned up.
