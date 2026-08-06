"""Per-arm liveness view of committed cells: which op classes ran, and did P1 publish."""
import json, sqlite3, sys, collections

sys.path.insert(0, "/storage/nvme/shapeflow-p1-study/src")
from shapeflow.object_store import ObjectStore  # noqa: E402

PHASE = sys.argv[1] if len(sys.argv) > 1 else "smoke5"
rows = []
for L in (0, 1):
    root = f"/storage/nvme/shapeflow-data/runner-lane{L}"
    store = ObjectStore(root + "/object_store")
    c = sqlite3.connect(f"file:{root}/runs/ledger.sqlite?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    q = ("select w.task_id,w.arm_id,a.result_object_ref as ref from work_items w "
         "join attempts a on a.work_key=w.work_key "
         "where w.phase_id=? and w.state='COMMITTED' and a.state='COMMITTED'")
    for r in c.execute(q, (PHASE,)):
        rows.append((L, r["task_id"], r["arm_id"], json.loads(store.get_bytes(r["ref"]).decode())))

if not rows:
    print("no committed cells for", PHASE); raise SystemExit(0)
print(f"{len(rows)} committed cells\n")
print("counts keys:", sorted(rows[0][3]["counts"]))
print()
per = collections.defaultdict(list)
for L, task, arm, b in rows:
    per[arm].append((L, task, b))

for arm in sorted(per):
    print(f"### {arm}")
    ops = collections.Counter()
    for L, task, b in per[arm]:
        cn, ws = b["counts"], b["work_summary"]
        ops.update({k: v["count"] for k, v in ws["by_op"].items()})
        p1 = {k: v for k, v in cn.items()
              if any(t in k for t in ("p1", "published", "select", "render", "fallback", "id_"))
              and v}
        print(f"  lane{L} t={task:<6} e2e={ws['e2e_latency_seconds']:7.1f} "
              f"union={ws['interval_union_seconds']:7.1f} "
              f"pt={ws['tokens']['prompt_tokens']:>7} ct={ws['tokens']['completion_tokens']:>6} "
              f"searches={cn.get('search_queries')} rounds={cn.get('research_rounds')}")
        print(f"           p1-ish counts: {p1}")
    print(f"  op classes: {dict(ops)}\n")
