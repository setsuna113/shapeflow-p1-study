"""Grading: turning an arm's answer into a score against a benchmark's own ground truth.

A grader is the one place an external model may be consulted about outcomes, so it is deliberately
narrow: blinded to arm and variant, fail-closed when the model is unavailable (never imputing a
score), and version-pinned so a grader change is visible as a configuration change rather than as
drift in the results.
"""
