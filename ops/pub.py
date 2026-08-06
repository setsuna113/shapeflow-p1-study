"""Per-boundary publication and op-class mix for a running phase."""
import collections, json, sqlite3, sys

sys.path.insert(0, "/storage/nvme/shapeflow-p1-study/src")
from shapeflow.object_store import ObjectStore  # noqa: E402

PHASE = sys.argv[1] if len(sys.argv) > 1 else "campaign"
agg = collections.defaultdict(collections.Counter)
for L in (0, 1):
    root = f"/storage/nvme/shapeflow-data/runner-lane{L}"
    store = ObjectStore(root + "/object_store")
    c = sqlite3.connect(f"file:{root}/runs/ledger.sqlite?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    for r in c.execute(
        "select w.arm_id,a.result_object_ref ref from work_items w "
        "join attempts a on a.work_key=w.work_key "
        "where w.phase_id=? and w.state='COMMITTED' and a.state='COMMITTED'", (PHASE,)):
        b = json.loads(store.get_bytes(r["ref"]).decode())
        cn, a = b["counts"], agg[r["arm_id"]]
        a["cells"] += 1
        a["h_batches"] += cn.get("page_batches_reduced", 0)
        a["h_fallbacks"] += cn.get("page_fallbacks", 0)
        a["c_reduced"] += cn.get("close_reduced", 0)
        a["c_failed"] += cn.get("close_failed", 0)
        for op, v in b["work_summary"]["by_op"].items():
            a["op:" + op] += v["count"]

for arm in sorted(agg):
    a = agg[arm]
    hp = max(0, a["h_batches"] - a["h_fallbacks"])
    hd, cd = a["h_batches"], a["c_reduced"] + a["c_failed"]
    h = f"{hp}/{hd} ({hp / hd:.0%})" if hd else "--"
    cc = f"{a['c_reduced']}/{cd} ({a['c_reduced'] / cd:.0%})" if cd else "--"
    print(f"{arm:<18} cells={a['cells']:<4} H published {h:<16} C published {cc}")
    print("      ops:", {k[3:]: v for k, v in sorted(a.items()) if k.startswith("op:")})
