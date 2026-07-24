"""The campaign: phases, schedule, and the driver that runs the real patched graph.

Everything here composes the existing library rather than reimplementing it -- the ledger,
budget, external-call FSM, strategy factory, checkpoints, block design and randomization all
already exist and are the only implementations allowed.
"""
