"""Statistical analysis and the machine verdict.

The inference here is deliberately conservative and reproducible. Independence is at the
topic/source-cluster level, not the task, so bootstraps resample clusters; repeated seeds are
within-cluster replicates and never add independent n. Randomness is seeded from the protocol
so a re-run reproduces the same intervals. The verdict logic keeps quality and work as
co-primary guards -- neither compensates the other -- and keeps NOT_ESTABLISHED distinct from a
demonstrated absence of effect.
"""
