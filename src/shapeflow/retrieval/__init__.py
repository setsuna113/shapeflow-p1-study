"""Dense retrieval over a frozen benchmark corpus.

Split deliberately across a process boundary. The index and the search are pure numpy and live
here, in the study package. The **query encoder does not**: it needs torch and transformers, and
``pyproject.toml`` constrains the study's dependency closure to the pinned vendor's own lock
precisely so the agent framework under measurement cannot be perturbed by our tooling. Pulling a
deep-learning stack into that closure to embed a search query would be the largest uncontrolled
change in the repository.

So the encoder runs in its own interpreter behind :mod:`shapeflow.retrieval.service`, and this
package talks to it over loopback. The cost is a process hop per search; the benefit is that the
system under measurement keeps the dependency graph it was frozen with.

It is also CPU-only, on every host and in every phase. Each GPU already runs a vLLM engine at
0.90 memory utilisation, and S1's headline number is the sustainable arrival rate on that engine
-- an encoder sharing the device would consume SM time that the measurement attributes to
serving, and no amount of care afterwards can subtract it.
"""
