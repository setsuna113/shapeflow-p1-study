"""The P1 treatment path: selectors, output contracts, aggregators, renderer, preflight.

This is the intervention under study -- replacing vendor prose (summarize_webpage /
compress_research) with evidence-ID selection. Everything here runs the same local target
model P0 uses; the only external evaluator (DeepSeek) never touches this path. Preflight
checks structure only and never reads truth, so the treatment can never be tuned against the
answer key it is later scored on.
"""
