"""BrowseComp-Plus: the workload, its splits, and how an answer is scored.

Named in full as the **BC+-corpus full-document ShapeFlow workload**. Retrieval returns complete
documents (~5k words on average) rather than the snippets the official leaderboard setting uses,
because full documents are what the H boundary exists to compress. That is a deliberate deviation
and is declared as one: results here are never placed beside official leaderboard numbers.

The package is split by *who may read what*, not by topic. :mod:`splits` and :mod:`corpus` are
treatment-visible. :mod:`qrels` and the graded answers are evaluator-only, and the firewall
between them is enforced by a static check rather than by convention -- a selector that could see
which documents are gold would be scored on its ability to read the answer key.
"""
