"""Week-1 kill/keep study for the ShapeFlow P1 nodes WEBPAGE_P1 and RESEARCHER_CLOSE.

Kept import-light on purpose: the CLI, the provider clients and the analysis stack all
pull heavy dependencies, and importing this package must stay cheap enough that the
launch gate's import-origin assertion and the leak tests can run without them.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
