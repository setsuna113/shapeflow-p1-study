"""The preflight overrun distribution across the whole campaign.

The interim report characterised the mechanism from 85 overruns. This restates it over every
committed cell of every LLM-selector arm, and reports the CPU arms alongside as the contrast.
"""
import collections, json, sqlite3, sys

sys.path.insert(0, "/storage/nvme/shapeflow-p1-study/src")
from shapeflow.object_store import ObjectStore  # noqa: E402

BUDGET = 512
per_arm = collections.defaultdict(lambda: {"defer": collections.Counter(), "tokens": []})
sample_reasons = collections.Counter()

for L in (0, 1):
    root = f"/storage/nvme/shapeflow-data/runner-lane{L}"
    store = ObjectStore(root + "/object_store")
    c = sqlite3.connect(f"file:{root}/runs/ledger.sqlite?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    q = ("select w.arm_id a, x.result_object_ref r from work_items w "
         "join attempts x on x.work_key=w.work_key "
         "where w.phase_id='campaign' and w.state='COMMITTED'")
    for row in c.execute(q):
        if not row["r"]:
            continue
        body = json.loads(store.get_bytes(row["r"]).decode())
        bucket = per_arm[row["a"]]
        for e in body.get("events") or []:
            if not isinstance(e, dict):
                continue
            kind = e.get("kind") or ""
            if kind not in ("PAGE_BATCH_DEFERRED", "CLOSE_DEFERRED_TO_VENDOR",
                            "PAGE_BATCH_REDUCED", "CLOSE_REDUCED"):
                continue
            bucket["defer"][kind] += 1
            for key in ("reason", "detail", "message", "why"):
                if e.get(key):
                    text = str(e[key])
                    sample_reasons[f"{row['a']}: {text[:90]}"] += 1
                    break
            for key in ("rendered_tokens", "rendered_token_count", "token_count"):
                if isinstance(e.get(key), (int, float)):
                    bucket["tokens"].append(int(e[key]))
                    break
        counts = body.get("counts") or {}
        mx = counts.get("max_rendered_tokens")
        if isinstance(mx, (int, float)) and mx:
            bucket["tokens"].append(int(mx))

print(f"{'arm':<18} {'deferred':>9} {'reduced':>8}   max_rendered_tokens over cells")
for arm in sorted(per_arm):
    d = per_arm[arm]["defer"]
    toks = sorted(t for t in per_arm[arm]["tokens"] if t > 0)
    dep = d["PAGE_BATCH_DEFERRED"] + d["CLOSE_DEFERRED_TO_VENDOR"]
    red = d["PAGE_BATCH_REDUCED"] + d["CLOSE_REDUCED"]
    if toks:
        n = len(toks)
        stat = (f"n={n} min={toks[0]} p50={toks[n//2]} p90={toks[int(n*0.9)]} max={toks[-1]}"
                f"  over budget: {sum(1 for t in toks if t > BUDGET)}/{n}")
    else:
        stat = "no rendered-token record"
    print(f"{arm:<18} {dep:>9} {red:>8}   {stat}")

print("\nmost common deferral reasons:")
for text, n in sample_reasons.most_common(8):
    print(f"  x{n:<5} {text}")
