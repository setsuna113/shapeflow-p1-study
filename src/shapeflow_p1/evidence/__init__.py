"""Evidence intermediate representation: chunking, span identity, lineage.

Everything here is built so that a selected span can be reconstructed *exactly* from the
frozen snapshot -- char offsets are the ground truth, token offsets are secondary. A span
whose stored offsets no longer reproduce its recorded text hash is a fatal integrity error,
never a rounding difference, which is why chunkers here never paraphrase or normalize away
bytes: they only cut.
"""
