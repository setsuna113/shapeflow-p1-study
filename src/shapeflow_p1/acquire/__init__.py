"""Acquisition and frozen retrieval.

Tavily is called only during acquisition, to build a task-local frozen source pool. Every
treatment run then retrieves from that pool deterministically, so no two arms can get a
different world from ranking drift or an intermittent API. The three backends here are kept
strictly separate (capture / exact-replay / frozen-corpus) precisely so a treatment run can
never accidentally reach the live network.
"""
