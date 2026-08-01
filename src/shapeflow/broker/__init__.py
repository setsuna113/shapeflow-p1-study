"""The broker: one tick, one atomic decision over form and admission together.

The contract splits cleanly in two, and that split is the whole design:

- **Where the snapshot comes from** differs per host. Inside the engine it is exact; in a proxy
  it is a scrape with a measurable age; in simulation it is scripted.
- **Everything else** -- submission without starting a form, the atomic choice, materializing only
  what was chosen, dispatching P0's sub-requests independently, retrying whole items on staleness
  and failing closed when retries run out -- is host-independent.

So :func:`shapeflow.broker.driver.run_tick` implements steps 1 and 3-6 exactly once, and every
host reuses it verbatim. A host supplies only :class:`~shapeflow.broker.host.BrokerHost`. Nothing
about the six-step semantics is reimplemented per host, which is what stops the simulation and the
real extension from quietly diverging in the direction that flatters the result.
"""
