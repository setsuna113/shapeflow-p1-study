"""The frozen world: the seam between a treatment run and the corpus it retrieves from.

A treatment run must never reach the live network, and every arm of a task must see byte-identical
retrieval results -- otherwise P1, which changes the queries a researcher issues, would also change
which pages exist, and the two arms would be compared across two different webs. Everything here
exists to make that property structural rather than conventional.

The seam is :class:`~shapeflow_p1.world.search_backend.SearchBackend`: given a query and a result
count, return :class:`~shapeflow_p1.world.search_backend.SearchRecord` values. What sits behind it
is a deliberate choice per campaign, and the implementations are kept strictly separate so a
treatment run cannot silently fall back to a different world than the one it froze against.

This package was carved out of the Week-1 ``acquire`` package, which was named for live
acquisition. Live acquisition is gone; the frozen-world seam it produced is what survives.
"""
