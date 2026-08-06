"""Classify every terminal failure of a phase, and show the most recent few."""
import collections, json, sqlite3, sys

sys.path.insert(0, "/storage/nvme/shapeflow-p1-study/src")
from shapeflow.object_store import ObjectStore  # noqa: E402

PHASE = sys.argv[1] if len(sys.argv) > 1 else "campaign"
rows = []
for L in (0, 1):
    root = f"/storage/nvme/shapeflow-data/runner-lane{L}"
    store = ObjectStore(root + "/object_store")
    c = sqlite3.connect(f"file:{root}/runs/ledger.sqlite?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    q = ("select w.arm_id as a, w.updated_at as t, x.result_object_ref as ref "
         "from work_items w left join attempts x on x.work_key=w.work_key "
         "where w.phase_id=? and w.state='FAILED_FINAL'")
    for r in c.execute(q, (PHASE,)):
        rows.append((r["t"], L, r["a"], r["ref"], store))
rows.sort()

kinds = collections.Counter()
for t, L, arm, ref, store in rows:
    if not ref:
        kinds["no stored result"] += 1
        continue
    b = json.loads(store.get_bytes(ref).decode())
    ws = b.get("work_summary") or {}
    err = str(b.get("error") or "")
    if ws.get("telemetry_complete") is False:
        kinds["telemetry_incomplete"] += 1
    elif ws.get("overlap_valid") is False:
        kinds["overlap_invalid"] += 1
    elif err and err != "None":
        kinds["error:" + err[:60]] += 1
    else:
        kinds["other"] += 1

print(f"total failures: {len(rows)}")
print("cause counts:", dict(kinds))
print("most recent 8:")
for t, L, arm, ref, store in rows[-8:]:
    if not ref:
        print(f"  lane{L} {arm}: no stored result")
        continue
    b = json.loads(store.get_bytes(ref).decode())
    ws = b.get("work_summary") or {}
    print("  lane{} {:<16} telem={} unavail={} unknown_after_send={} err={}".format(
        L, arm, ws.get("telemetry_complete"),
        len(ws.get("unavailable_attempt_ids") or []),
        ws.get("unknown_after_send"), str(b.get("error"))[:60]))
