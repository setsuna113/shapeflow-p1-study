"""Benchmarks: the workloads a campaign is measured on, and how their answers are scored.

Each benchmark adapter owns three things and nothing else: how its tasks are enumerated and split,
how its corpus is retrieved, and how an answer is graded. Nothing here may be imported by a
selector, aggregator or preflight -- gold answers, qrels and hard negatives are evaluator material,
and the leakage firewall that keeps them out of the treatment path is enforced by a static check,
not by convention.
"""
