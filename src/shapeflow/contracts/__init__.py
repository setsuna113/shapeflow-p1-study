"""The contracts: what the mechanism is, what the broker may do, and what it may look at.

Three documents under ``docs/``, each with executable predicates beside it. The documents are the
contract; the code is one implementation of it.

That distinction is enforced rather than asserted. :mod:`shapeflow.contracts.versions` digests the
documents, the digest enters the execution binding, and every host module declares the digest it
was written against and asserts it at import. A host that drifts from the text cannot be loaded,
so "the simulation, the conformance tests and the real extension reference the same contract"
is a property of the import system instead of a promise in a paper.
"""
