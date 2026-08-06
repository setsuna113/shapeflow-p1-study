#!/usr/bin/env bash
# Fetch a GitHub repo as a SHA-pinned tarball.
#
# Why not `git clone`: on this host GitHub's git smart-HTTP endpoint
# (/info/refs?service=git-upload-pack) is filtered and hangs, while the REST API
# and codeload.github.com both respond normally. Pinning to the SHA the API
# reports gives the same reproducibility guarantee as a clone + recorded hash.
#
# Idempotent: if dest/.fetch_ok exists and records the same SHA, does nothing.
set -euo pipefail

OWNER_REPO="$1"   # e.g. texttron/BrowseComp-Plus
BRANCH="$2"       # e.g. main
DEST="$3"         # destination directory for the extracted tree

meta="${DEST%/}/../$(basename "$DEST").fetch.json"

sha="$(curl -sS --max-time 60 "https://api.github.com/repos/${OWNER_REPO}/commits/${BRANCH}" \
       | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])')"
[ -n "$sha" ] || { echo "FATAL: could not resolve SHA for ${OWNER_REPO}@${BRANCH}" >&2; exit 1; }

if [ -f "${DEST}/.fetch_ok" ] && [ "$(cat "${DEST}/.fetch_ok")" = "$sha" ]; then
    echo "  [skip] ${OWNER_REPO} already at ${sha}"
    exit 0
fi

tarball="$(mktemp /tmp/ghfetch.XXXXXX.tar.gz)"
trap 'rm -f "$tarball"' EXIT
echo "  downloading ${OWNER_REPO}@${sha:0:12} ..."
curl -sSL --max-time 900 --retry 5 --retry-delay 5 -C - \
     "https://codeload.github.com/${OWNER_REPO}/tar.gz/${sha}" -o "$tarball"

tar_sha256="$(sha256sum "$tarball" | cut -d' ' -f1)"
rm -rf "$DEST"; mkdir -p "$DEST"
tar -xzf "$tarball" -C "$DEST" --strip-components=1
echo "$sha" > "${DEST}/.fetch_ok"

python3 - "$OWNER_REPO" "$BRANCH" "$sha" "$tar_sha256" "$meta" <<'PY'
import json, sys, datetime
owner_repo, branch, sha, tar_sha256, meta = sys.argv[1:6]
json.dump({
    "source_url":     f"https://github.com/{owner_repo}",
    "branch":         branch,
    "commit":         sha,
    "tarball_url":    f"https://codeload.github.com/{owner_repo}/tar.gz/{sha}",
    "tarball_sha256": tar_sha256,
    "fetched_utc":    datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "method":         "codeload tarball (github git smart-HTTP blocked from this host)",
}, open(meta, "w"), indent=2)
print(f"  ok  commit={sha}  tar_sha256={tar_sha256[:16]}...")
PY
