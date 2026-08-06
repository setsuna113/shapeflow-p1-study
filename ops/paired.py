"""Paired per-arm work summary over a finished BC+ phase, read straight from the object store.

Deliberately does not go through the ledger's work-key path: that embeds the execution binding,
so a phase run under a superseded binding becomes unreadable the moment the tree moves. The
committed attempt rows carry everything needed.
"""
import collections, json, sqlite3, statistics, sys

sys.path.insert(0, "/storage/nvme/shapeflow-p1-study/src")
from shapeflow.object_store import ObjectStore  # noqa: E402

PHASE = sys.argv[1]
METRICS = ("prompt_tokens", "completion_tokens", "cached_prompt_tokens")

cells = []
for L in (0, 1):
    root = f"/storage/nvme/shapeflow-data/runner-lane{L}"
    store = ObjectStore(root + "/object_store")
    c = sqlite3.connect(f"file:{root}/runs/ledger.sqlite?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    for r in c.execute(
        "select w.task_id,w.arm_id,a.result_object_ref ref from work_items w "
        "join attempts a on a.work_key=w.work_key "
        "where w.phase_id=? and w.state='COMMITTED' and a.state='COMMITTED'", (PHASE,)):
        cells.append((r["task_id"], r["arm_id"], json.loads(store.get_bytes(r["ref"]).decode())))

by_arm = collections.defaultdict(dict)
for task, arm, b in cells:
    by_arm[arm][task] = b
arms = sorted(by_arm)
print(f"{len(cells)} committed cells, {len(arms)} arms\n")

def val(b, key):
    ws = b["work_summary"]
    if key in METRICS:
        return ws["tokens"][key]
    return ws[key]

KEYS = ("interval_union_seconds", "prompt_tokens", "completion_tokens",
        "e2e_latency_seconds", "energy_joules", "service_seconds")

print(f"{'arm':<20} {'n':>3} " + " ".join(f"{k[:14]:>15}" for k in KEYS))
for arm in arms:
    n = len(by_arm[arm])
    means = [statistics.mean([val(b, k) for b in by_arm[arm].values()]) for k in KEYS]
    print(f"{arm:<20} {n:>3} " + " ".join(f"{m:>15,.0f}" for m in means))

base = "P0"
print(f"\nPaired contrasts against {base} (same task, both arms committed):")
for arm in arms:
    if arm == base:
        continue
    shared = sorted(set(by_arm[arm]) & set(by_arm[base]))
    if not shared:
        print(f"  {arm}: no shared tasks"); continue
    print(f"  {arm}  (n={len(shared)} paired tasks)")
    for k in KEYS:
        d = [val(by_arm[arm][t], k) - val(by_arm[base][t], k) for t in shared]
        b0 = statistics.mean([val(by_arm[base][t], k) for t in shared])
        m = statistics.mean(d)
        pct = 100 * m / b0 if b0 else float("nan")
        sign = "+" if m >= 0 else ""
        print(f"      {k:<26} {sign}{m:>12,.0f}  ({sign}{pct:5.1f}%)  base {b0:,.0f}")

print("\nPer-op-class mean count and tokens per cell:")
for arm in arms:
    ops = collections.defaultdict(lambda: [0, 0, 0])
    for b in by_arm[arm].values():
        for op, v in b["work_summary"]["by_op"].items():
            ops[op][0] += v["count"]; ops[op][1] += v["prompt_tokens"]; ops[op][2] += v["completion_tokens"]
    n = len(by_arm[arm])
    print(f"  {arm}")
    for op, (cnt, pt, ct) in sorted(ops.items()):
        print(f"      {op:<26} count={cnt/n:6.1f}  prompt={pt/n:9,.0f}  completion={ct/n:8,.0f}")
