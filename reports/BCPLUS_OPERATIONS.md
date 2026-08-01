# Freeze-1 / BrowseComp-Plus — how the campaign is actually run

The stack that produces the P0-vs-P1 numbers, on `sjtu`. Written down because none of it is
discoverable from the repository alone: the benchmark, the model and the engine venv live outside
the tree, and three of the settings below were each found the hard way.

## The five processes, per lane

Two lanes run concurrently. GPUs 2 and 3 are held by compute processes outside this container's
PID namespace and cannot be used.

| lane | GPU | engine | provider | retrieval | encoder cpuset |
|---|---|---|---|---|---|
| 0 | `GPU-ef013951` (idx 0) | `127.0.0.1:8000` | `127.0.0.1:8787` | `127.0.0.1:8710` | 48–63 |
| 1 | `GPU-d1f5d8c4` (idx 1) | `127.0.0.1:8001` | `127.0.0.1:8788` | `127.0.0.1:8711` | 32–47 |

Lane 0 is the only lane that may reach a paid upstream (`measurement.shards.paid_upstream_lane`),
so grading runs there. Each lane keeps its own runner ledger and object store under
`runner-lane{n}/`; `grade-bcplus --lanes 0,1` reads both.

**Task-atomic sharding.** A block — one task, all its arms and replicates — belongs to exactly
one lane, assigned by `sha256(binding : layer : task_id) % lanes`. P0 on one card and P1 on
another would fold that card's clocks and thermals into the treatment effect, and nothing
downstream could separate them again.

## Start order

```bash
# 1. engines, one per lane (root; drops to sfinfer itself)
SHAPEFLOW_LANE=0 SHAPEFLOW_GPU_UUID=GPU-ef013951-… bash scripts/start_engine.sh
SHAPEFLOW_LANE=1 SHAPEFLOW_GPU_UUID=GPU-d1f5d8c4-… bash scripts/start_engine.sh

# 2. providers, one per lane
SHAPEFLOW_LANE=0 bash scripts/start_provider.sh
SHAPEFLOW_LANE=1 bash scripts/start_provider.sh

# 3. retrieval services, one per lane, CPU only
bash scripts/start_retrieval.sh 0
bash scripts/start_retrieval.sh 1

# 4. re-mint the approval over the live tree (steward)
shapeflow freeze-approval --approved-commit $(git rev-parse HEAD)

# 5. the campaign (runner), per lane
shapeflow run-bcplus --layer <layer> --arms bcplus_arms --shard <lane> --shards 2 \
  --retrieval-url http://127.0.0.1:871<lane> --resume --protocol-sha <binding>
```

## Things that are load-bearing and non-obvious

**The installed graph is a copy.** `open_deep_research` is installed from
`.build/open_deep_research-patched`, not editable. Re-running `scripts/materialize_vendor.sh`
changes nothing until the package is reinstalled, and the approval cannot see the difference,
because `patched_tree_sha` binds the recorded hash *file* rather than the running bytes. This
cost a full campaign of null results once: after the package rename the host went on running
hooks that imported `shapeflow_p1`, vendor's supervisor swallowed the ImportError and returned an
empty note set, and every cell committed with `search_queries = 0` while reporting `ok: true`.
`doctor` now compares the importable tree against the tracked digest and `run-bcplus` refuses to
claim a cell if they differ.

```bash
bash scripts/materialize_vendor.sh
uv pip install --python .venv/bin/python --reinstall-package open_deep_research \
  .build/open_deep_research-patched --no-deps
```

**The encoder runs in a different interpreter.** The study's venv deliberately has no torch — its
dependency closure is pinned to the vendor's lock for everything that touches the graph. The
retrieval service runs from `$BENCHDATA/.venv` (torch 2.13, transformers 5.14) and reaches
`shapeflow.retrieval` through `PYTHONPATH`. That venv also needs `zstandard`, because the
`SearchRecord` import path reaches the object store.

**The engine flags come from `configs/stack.yaml`.** `isolation.causal_native` is the primary
layer: prefix caching off, chunked prefill off, `max_num_seqs 16`. `VLLM_ATTENTION_BACKEND` must
be `FLASH_ATTN` (the default FlashInfer selection JIT-compiles against system CUDA 11 while torch
is built for 12.9), the venv's `bin` must be on `PATH` (flashinfer's JIT needs `ninja`), and
`--enable-auto-tool-choice --tool-call-parser hermes` are required or every tool-bound ODR
request is answered with a 400.

**Any commit re-mints the approval.** `approved_commit` is inside the binding, so every push to
the host needs `freeze-approval` again before a run, and the new digest goes to `--protocol-sha`.

## Where things land

| artifact | path |
|---|---|
| per-lane ledger | `$DATA_ROOT/runner-lane{n}/runs/ledger.sqlite` |
| per-lane object store | `$DATA_ROOT/runner-lane{n}/object_store/` |
| frozen schedule | `$DATA_ROOT/runner-lane{n}/runs/schedules/<run-id>/<layer>.json` |
| frozen blocks | `$DATA_ROOT/runner-lane{n}/runs/bcplus_blocks/<run-id>/` |
| split carve (write-once) | `$DATA_ROOT/runner-lane{n}/runs/bcplus_split_plan.json` |
| run status | `$DATA_ROOT/runner-lane{n}/runs/STATUS_<run-id>.json` |
| analysis | `reports/BCPLUS_<run-id>.{json,md}` |
| competence gate | `reports/gates/RETRIEVAL_COMPETENCE.json` |
