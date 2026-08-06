# ShapeFlow Phase 0 — benchmark data preparation

Scripts, manifests and frozen splits for the three external benchmarks the
ShapeFlow experiments sit on. **Data itself is not in this repo** — only code,
checksums and split files.

Prepared on the `sjtu` run host, 2026-07-27/28 UTC. No LLM evaluation was run
and no GPU was used.

## Layout

```
/storage/sata/shapeflow/benchdata/          <- BASE_DIR   (/data/shapeflow -> /storage/sata/shapeflow)
  browsecomp-plus/
    repo/          BrowseComp-Plus checkout (SHA-pinned tarball)
    corpus/        100,195-doc corpus (HF parquet)
    indexes/       prebuilt bm25 + qwen3-embedding-{0.6b,4b,8b}
    data/          decrypted queries/qrels   *** CONTAINS ANSWERS - never commit, never print ***
    runs/          BM25 TREC run files
  drb/repo/        DeepResearch Bench checkout
  drgym/
    README_api.md  DeepResearchGym API spec as observed from sjtu
    api_probe/     connectivity/latency probe records
    researchy_questions/
  splits/          our frozen dev/test split (in git)
  prep-scripts/    this repo
```

`BASE_DIR` is `/storage/sata/shapeflow/benchdata`, **not** the `/data/shapeflow`
of the original brief: `/data` did not exist on this host. `/storage/sata` is a
dedicated 1 TB disk that was completely empty (956 GB free); `/data/shapeflow`
is symlinked to it so the documented path resolves.

## Environment

- Python 3.11.15 in `$BASE_DIR/.venv` (uv), `requirements.lock` pinned here
- **Java 21.0.7** (`openjdk-21-jdk-headless`, apt) at
  `/usr/lib/jvm/java-21-openjdk-amd64` — required by pyserini/Anserini
- pyserini 1.6.0
- `flash-attn` deliberately **not** installed (only needed for dense retrieval /
  agent runs later)
- `HF_ENDPOINT=https://hf-mirror.com` — **huggingface.co is unreachable from
  this host** (TCP timeout). Every HF fetch used the mirror.

## Scripts

All are idempotent (existing output is verified and skipped).

| Script | Purpose |
|---|---|
| `fetch_repo.sh` | Fetch a GitHub repo as a SHA-pinned tarball (git protocol is blocked here — see Deviations) |
| `download_bcplus.sh` | Download BC+ corpus + all prebuilt indexes via the HF mirror |
| `decrypt_bcplus.py` | Decrypt BC+ queries/qrels, reusing the official `transform_decrypt` verbatim |
| `stats_bcplus.py` | Corpus/query counts and evidence/gold distributions (emits counts only, never text) |
| `smoke_bc_plus.py` | 20-query BM25 smoke test, recall@10/@100 — also proves the Java/pyserini chain |
| `bm25_full_run.py` | BM25 over all 830 queries → TREC run file (for official trec_eval) |
| `make_splits.py` | Frozen dev 530 / test 300 split, seed 20260727, byte-reproducible |
| `probe_drgym.py` | DeepResearchGym FineWeb connectivity/latency probe (never touches ClueWeb22) |
| `stats_researchy.py` | Researchy Questions counts + samples |
| `gen_manifest.py` | Build `MANIFEST.json` (sources, revisions, sha256, counts, licences) |

## Reproducing

```bash
export BASE_DIR=/storage/sata/shapeflow/benchdata
export HF_ENDPOINT=https://hf-mirror.com

$BASE_DIR/prep-scripts/fetch_repo.sh texttron/BrowseComp-Plus main $BASE_DIR/browsecomp-plus/repo
$BASE_DIR/prep-scripts/fetch_repo.sh Ayanami0730/deep_research_bench main $BASE_DIR/drb/repo
$BASE_DIR/prep-scripts/download_bcplus.sh

$BASE_DIR/.venv/bin/hf download Tevatron/browsecomp-plus --repo-type=dataset
$BASE_DIR/.venv/bin/python prep-scripts/decrypt_bcplus.py \
    --repo-dir $BASE_DIR/browsecomp-plus/repo \
    --snapshot-dir /root/.cache/huggingface/hub/datasets--Tevatron--browsecomp-plus/snapshots/<rev> \
    --output $BASE_DIR/browsecomp-plus/data/browsecomp_plus_decrypted.jsonl \
    --generate-tsv $BASE_DIR/browsecomp-plus/repo/topics-qrels/queries.tsv

$BASE_DIR/.venv/bin/python prep-scripts/make_splits.py \
    --decrypted $BASE_DIR/browsecomp-plus/data/browsecomp_plus_decrypted.jsonl \
    --out-dev $BASE_DIR/splits/bcplus_dev.txt \
    --out-test $BASE_DIR/splits/bcplus_test.txt \
    --out-manifest $BASE_DIR/splits/splits_manifest.json
```

`make_splits.py` is byte-reproducible: two consecutive runs produced identical
sha256 for both split files.

## Deviations from the original brief

1. **`BASE_DIR` moved** to `/storage/sata/shapeflow/benchdata` (`/data` did not
   exist). `/data/shapeflow` symlink added.
2. **huggingface.co unreachable** → `HF_ENDPOINT=https://hf-mirror.com`
   throughout, recorded in the manifest.
3. **`git clone` from GitHub is blocked** on this host: the git smart-HTTP
   endpoint (`/info/refs?service=git-upload-pack`) times out, while the REST API
   and `codeload.github.com` work normally. Repos are therefore fetched as
   tarballs pinned to the commit SHA the API reports — same reproducibility
   guarantee, and the SHA is recorded.
4. **`decrypt_dataset.py` could not be run as-is.** It calls
   `load_dataset("Tevatron/browsecomp-plus")`, which hangs indefinitely against
   the mirror, and refuses to read a warm cache under `HF_HUB_OFFLINE=1`.
   `decrypt_bcplus.py` downloads the repo with `hf download` and reads the
   cached parquet, importing the official `transform_decrypt` and canary so the
   decryption itself is unchanged. Output verified: 830 records.
5. **Dense indexes were downloaded.** The brief made them conditional on ≥50 GB
   free; the entire index set is only ~5 GB and the disk has 956 GB free.
6. **`qrel_gold.txt` vs `qrel_golds.txt`** — the repo README references
   `topics-qrels/qrel_gold.txt`; the actual file is `qrel_golds.txt`.
7. **`searcher.batch_search()` is broken on this index** — Anserini aborts with
   `queryCount = 830 is not equal to completedTaskCount = 65`. `bm25_full_run.py`
   searches sequentially instead (~140 ms/query).
8. **pyserini 1.6.0 constructs an OpenAI client at import time**, so
   `OPENAI_API_KEY` must be set to something (a placeholder) even for pure BM25.
9. **DeepResearchGym has no `/fetch` endpoint** and **FineWeb now requires an
   API key** (the site still says it does not). See `drgym/README_api.md`.

## Hygiene

`browsecomp-plus/data/browsecomp_plus_decrypted.jsonl` contains gold answers.
It is outside this repo, `.gitignore`d, and no script prints answer or query
text — the stats scripts emit counts only. Test-split answers are never
displayed.
