"""Build a runner-readable frozen world directly, without an acquisition path.

The Week-1 tests reached a frozen world by running the real acquisition phase against a fake
search vendor: author a corpus, seal a registry, call the provider once per task, publish two
views. That path is gone, and the tests that depended on it were not testing it -- they were
testing the campaign runner, crash recovery and the provider boundary, and needed *a* world to
run against.

So this writes the two artifacts the runner actually reads, and nothing else:

``frozen_corpus_for_runner/tasks/<task>.json``   the treatment-visible record: the question.
``frozen_corpus_for_runner/pools/<task>.json``   the vendor-visible occurrences, self-verifying.

Both formats are the ones :mod:`shapeflow.world.pools` and
:func:`shapeflow.campaign.runner.available_tasks` parse, so a world built here is
indistinguishable from an acquired one at the seam that matters. That is deliberate: the same
shape is what a benchmark corpus adapter has to produce, so this fixture is the contract such an
adapter is written against rather than a stand-in for one.

Nothing here may write an evaluator record. The runner's view carries the question and nothing
else -- no facets, no acquisition spec -- and a fixture that quietly widened it would let a test
pass while the treatment identity read its own answer key.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from shapeflow.canonical import canonical_json
from shapeflow.hashing import occurrence_id, sha256_hex
from shapeflow.object_store import ObjectStore
from shapeflow.world.snapshot_store import SnapshotStore

__all__ = ["write_frozen_world"]

FETCHED_AT = "2026-07-24T01:00:00Z"


def _stem(task_id: str) -> str:
    return "".join(ch for ch in task_id if ch.isalnum()).lower() or "page"


def _page(task_id: str, n: int) -> str:
    """A page whose vocabulary the researcher's own queries will actually retrieve.

    The wording deliberately matches what the deleted Exa double produced. Retrieval here is
    real BM25 over the frozen pool, and the driving engine builds its query from the leading
    words of the research question -- so a page that shares no vocabulary with the question is
    retrieved by nothing, every search comes back empty, and the run completes having summarized
    no pages at all. That failure looks like a broken summarization path rather than a fixture
    whose corpus is simply unreachable, which is why the vocabulary is pinned rather than
    incidental. :data:`QUESTION` is written against these same words.
    """
    stem = _stem(task_id)
    return f"# {stem} {n}\n\n" + "\n\n".join(
        f"Paragraph {p} about {stem} number {n}. It records a measurement of "
        f"{p * 7 + n} units observed in 2025 by the {stem} authority."
        for p in range(6)
    )


#: Shares its leading words with :func:`_page`, so the engine's derived query retrieves.
QUESTION = "measurement units observed in 2025 for {task_id}"


def write_frozen_world(
    settings,
    *,
    task_ids,
    split: str = "FORMATIVE_SCREEN",
    pages_per_task: int = 2,
    question: str = QUESTION,
    corpus_tier: str = "FORMATIVE_MACHINE_AUTHORED",
    claim_scope: str = "FORMATIVE_ONLY",
) -> dict[str, list[str]]:
    """Write a frozen world for ``task_ids``. Returns {task_id: [content_hash, ...]}."""
    root = Path(settings.path("frozen_corpus_for_runner"))
    tasks_dir, pools_dir = root / "tasks", root / "pools"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    pools_dir.mkdir(parents=True, exist_ok=True)
    store = SnapshotStore(ObjectStore(root / "objects"))

    hashes: dict[str, list[str]] = {}
    for task_id in task_ids:
        (tasks_dir / f"{task_id}.json").write_text(
            json.dumps({
                "task_id": task_id,
                "split": split,
                "original_question": question.format(task_id=task_id),
                "corpus_tier": corpus_tier,
                "claim_scope": claim_scope,
            }, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        occurrences, snapshots = [], {}
        stem = _stem(task_id)
        for n in range(1, pages_per_task + 1):
            # The `/doc` suffix is asserted by callers distinguishing a frozen-corpus hit from
            # anything a live backend could have returned.
            url = f"https://{stem}{n}.example/doc"
            snap = store.freeze(_page(task_id, n), raw_content_format="markdown",
                                fetched_at_utc=FETCHED_AT)
            snapshots[snap.content_hash] = snap
            occurrences.append({
                # Derived the same way acquisition derived it, so ids are not free-form strings
                # that happen to be unique inside one test.
                "occurrence_id": occurrence_id(
                    query_snapshot_id=sha256_hex(f"{task_id}/q".encode()), rank=n, url=url),
                "url": url,
                "title": f"{stem.title()} {n}",
                "snippet_content": f"Snippet for {stem} {n}.",
                "content_hash": snap.content_hash,
                "vendor_visible_order": n - 1,
            })

        body = {
            "task_id": task_id,
            "occurrences": occurrences,
            "snapshots": {
                h: {
                    "object_ref": s.object_ref,
                    "byte_len": s.byte_len,
                    "raw_content_format": s.raw_content_format,
                    "normalization_version": s.normalization_version,
                    "fetched_at_utc": s.fetched_at_utc,
                }
                for h, s in sorted(snapshots.items())
            },
        }
        body["pool_sha256"] = sha256_hex(canonical_json(body))
        path = pools_dir / f"{task_id}.json"
        path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
        # The runner reads this; it never writes it. Acquisition made that structural, and a
        # fixture that left it writable would not reproduce the permission the runner runs under.
        os.chmod(path, 0o444)
        hashes[task_id] = [o["content_hash"] for o in occurrences]

    return hashes
