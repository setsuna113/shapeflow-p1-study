"""Why did these cells fail terminally?"""
import json, sqlite3, sys

sys.path.insert(0, "/storage/nvme/shapeflow-p1-study/src")
from shapeflow.object_store import ObjectStore  # noqa: E402

PHASE = sys.argv[1] if len(sys.argv) > 1 else "campaign"
for L in (0, 1):
    root = f"/storage/nvme/shapeflow-data/runner-lane{L}"
    store = ObjectStore(root + "/object_store")
    c = sqlite3.connect(f"file:{root}/runs/ledger.sqlite?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    for r in c.execute(
        "select w.task_id,w.arm_id,w.state,a.result_object_ref ref,a.state astate "
        "from work_items w left join attempts a on a.work_key=w.work_key "
        "where w.phase_id=? and w.state in ('FAILED_FINAL','FAILED_UNKNOWN')", (PHASE,)):
        print(f"lane{L} task={r['task_id']} arm={r['arm_id']} {r['state']}/{r['astate']}")
        if not r["ref"]:
            print("    (no stored result)")
            continue
        try:
            b = json.loads(store.get_bytes(r["ref"]).decode())
        except Exception as e:
            print("    unreadable:", e); continue
        print("    error:", str(b.get("error"))[:600])
        cn = b.get("counts") or {}
        print("    counts:", {k: v for k, v in cn.items() if v})
