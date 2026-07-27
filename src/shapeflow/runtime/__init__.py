"""Runtime telemetry: op-class tagging, work accounting, and (on the run host) the model
proxy and NVML sampler.

The work-accounting types and computations here are pure and tested in the dev environment.
The pieces that need a live engine -- the OpenAI-compatible proxy that tags and times model
requests, and the NVML power/energy sampler -- are thin wrappers added on the run host; they
feed the same RequestEvent records this module defines, so the accounting logic is validated
independently of the GPU.
"""
