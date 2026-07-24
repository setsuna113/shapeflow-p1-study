"""Evaluation: truth packets, blind judges, and the quality metrics.

Two hard separations hold across this package. First, evaluation runs under a different
identity than treatment and reads artifacts treatment never sees (TruthPackets, gold facets).
Second, P0 is not gold: every arm is scored against the same frozen TruthPacket, never against
another arm's output. The quality metrics here are pure functions of a TruthPacket and an
arm-blind ReportClaim ledger, so they can be recomputed for every arm from one truth version.
"""
