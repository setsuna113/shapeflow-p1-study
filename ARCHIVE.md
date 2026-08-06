# ARCHIVED — 2026-08-06

**Status:** sealed as part of the *deepresearch* line. No further development is planned.
The repository is intentionally left **writable** — the GitHub "Archived" flag was not set —
so work can resume without ceremony.

**Tag:** `archive/2026-08-06` · **Index:** https://github.com/setsuna113/deepresearch-archive

## What this was

The main line. ShapeFlow is a broker that decides, at the vLLM scheduler tick, which
compression form each job uses **and** which jobs are admitted — jointly rather than in
sequence — for an Open-Deep-Research-style agent. Boundaries H (sibling tool-call batch)
and C (researcher close); forms P0 prose vs P1 span-ID selection.

Freeze-1 and Freeze-2 protocols are in `protocol/` and are frozen: amendments append,
they do not rewrite. The Week-1 campaign ran on a 4×RTX-4090 node through 2026-08-05.

## Where the copies went

| Location | Disposition |
|---|---|
| sjtu (4×RTX-4090 node) | **deleted 2026-08-06** after a per-object/per-file redundancy proof; see `provenance/sjtu-2026-08-06/DELETION-MANIFEST.md` in the index repo |
| WSL `/home/lyc/shapeflow-p1-study` | retained — authoritative working copy |
| GitHub | this repository — authoritative published copy |

## To resume

```sh
git clone https://github.com/setsuna113/shapeflow-p1-study
uv sync            # environments were not archived; re-derive from uv.lock
```

Model weights and benchmark corpora were deleted and are re-downloadable
(Qwen3-14B-AWQ, Qwen3-8B-AWQ, Qwen3-Embedding-{0.6B,4B,8B}, `Tevatron/browsecomp-plus`).
Provider API keys used by this work are being rotated — see `CREDENTIAL-ROTATION.md`
in the index repository. No key value is stored anywhere in these repositories.
