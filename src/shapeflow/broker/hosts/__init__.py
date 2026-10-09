"""Hosts: the one thing that differs between deployments.

A host answers exactly one question the driver cannot: where the snapshot comes from, and what
"commit this decision" means locally. Everything else is :mod:`shapeflow.broker.driver`.

| Host | Snapshot | Status |
|---|---|---|
| ``SimulatedHost`` | scripted, with a discrete-event engine model | full; the B5 substrate |
| ``ProxyHost`` | proxy-local exact state plus a scrape with a measurable age | conformant against a fake engine; replay-driven against a live vLLM 0.24.0 (``scripts/proxy_live.py``) |
| ``EngineHost`` | exact, from inside the scheduler tick | deferred to S2 |

The engine-resident host is deferred rather than absent: vLLM 0.24 exposes ``--scheduler-cls``
and a documented ``SchedulerInterface``, so it is a sidecar rather than a fork of the engine.
"""
