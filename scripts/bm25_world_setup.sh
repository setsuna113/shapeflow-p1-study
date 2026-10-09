#!/bin/bash
# Build the BC+ BM25 world for scripts/broker_agents_live.py on a fresh host.
#
#   REPO=/work/sf B=/work/benchdata bash scripts/bm25_world_setup.sh
#
# Every download is pinned to the revision benchdata/prep-scripts/MANIFEST.json records. The
# decrypted query file carries answers; only queries.tsv (question text) is used downstream, and
# the decrypted file should be deleted with the host.
set -euo pipefail
REPO="${REPO:?set REPO to the repository checkout}"
B="${B:?set B to the benchdata root}"
PY="${PY:-$REPO/.venv/bin/python}"
mkdir -p "$B/splits" "$B/browsecomp-plus/repo"

cp "$REPO/benchdata/prep-scripts/splits/bcplus_dev.txt" \
   "$REPO/benchdata/prep-scripts/splits/bcplus_test.txt" "$B/splits/"

REPO_SHA=046949032b0328319cc9a02663a759ec601d9402
TAR_SHA=6ea1ca27b582bf89410bed8958c4f23fbf8da4615183f50a94a8a513f1d05b9d
curl -sSL -o "$B/bcplus-repo.tgz" \
  "https://codeload.github.com/texttron/BrowseComp-Plus/tar.gz/$REPO_SHA"
echo "$TAR_SHA  $B/bcplus-repo.tgz" | sha256sum -c -
tar -xzf "$B/bcplus-repo.tgz" -C "$B/browsecomp-plus/repo" --strip-components=1

hf download Tevatron/browsecomp-plus --repo-type dataset \
  --revision 144cff8e35b5eaef7e526346aa60774a9deb941f --local-dir "$B/hf/browsecomp-plus"
"$PY" "$REPO/benchdata/prep-scripts/decrypt_bcplus.py" \
  --repo-dir "$B/browsecomp-plus/repo" --snapshot-dir "$B/hf/browsecomp-plus" \
  --output "$B/browsecomp-plus/data/decrypted.jsonl" \
  --generate-tsv "$B/browsecomp-plus/repo/topics-qrels/queries.tsv"

hf download Tevatron/browsecomp-plus-corpus --repo-type dataset \
  --revision b27b02bc3e45511b8b82a13e6f90ce761df726f6 \
  --local-dir "$B/browsecomp-plus/corpus"
hf download Tevatron/browsecomp-plus-indexes --repo-type dataset \
  --revision b3f37f70c33829eb09d04784a54277a31871fd63 --include "bm25/*" \
  --local-dir "$B/browsecomp-plus/indexes"
echo "world ready under $B"
