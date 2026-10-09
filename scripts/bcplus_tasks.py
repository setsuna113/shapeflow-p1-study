#!/usr/bin/env python3
"""Write the first N tasks of a BC+ design layer as ``{task_id, question}`` JSONL.

    python scripts/bcplus_tasks.py --benchdata /work/benchdata --n 12 --out tasks.jsonl

Uses the campaign's own ``resolve_layer_tasks``, so the split files are checked against the
prereg pins and the layer is the one the campaign would carve. Only question text is written;
the answers in the decrypted query file never reach the agent.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from shapeflow.campaign.bcplus import PILOT_LAYER, resolve_layer_tasks  # noqa: E402
from shapeflow.campaign.settings import Settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchdata", required=True)
    ap.add_argument("--layer", default=PILOT_LAYER)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as data_root:
        settings = Settings.load(REPO, data_root=data_root)
        task_ids, questions, provenance = resolve_layer_tasks(
            settings, layer=args.layer, repo=REPO, benchdata=Path(args.benchdata))
    chosen = task_ids[: args.n]
    out = Path(args.out)
    out.write_text("".join(json.dumps({"task_id": t, "question": questions[t]}) + "\n"
                           for t in chosen), encoding="utf-8")
    meta = {k: v for k, v in provenance.items() if k != "unassigned_dev_ids"}
    out.with_suffix(".provenance.json").write_text(
        json.dumps({**meta, "chosen": chosen}, indent=2) + "\n", encoding="utf-8")
    print(f"{len(chosen)} of {len(task_ids)} {args.layer} tasks -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
