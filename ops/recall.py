"""Is agent evidence recall really zero, or are the two docid namespaces different?"""
import json, sqlite3, sys
from pathlib import Path

sys.path.insert(0, "/storage/nvme/shapeflow-p1-study/src")
from shapeflow.object_store import ObjectStore  # noqa: E402
from shapeflow.bench.bcplus.qrels import load_bcplus_evaluator_queries  # noqa: E402

PHASE = sys.argv[1] if len(sys.argv) > 1 else "competence"
views = load_bcplus_evaluator_queries(
    Path("/storage/sata/shapeflow/benchdata/browsecomp-plus/data"))

for L in (0, 1):
    root = f"/storage/nvme/shapeflow-data/runner-lane{L}"
    store = ObjectStore(root + "/object_store")
    c = sqlite3.connect(f"file:{root}/runs/ledger.sqlite?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    for r in c.execute(
        "select w.task_id,w.arm_id,a.result_object_ref ref from work_items w "
        "join attempts a on a.work_key=w.work_key "
        "where w.phase_id=? and w.state='COMMITTED' and a.state='COMMITTED'", (PHASE,)):
        b = json.loads(store.get_bytes(r["ref"]).decode())
        trace = b.get("retrieval_trace") or []
        got = set()
        for t in trace:
            got.update(t.get("docids") or [])
        tid = str(r["task_id"])
        try:
            v = views.get(tid)
        except Exception as e:
            print(f"task {tid}: no evaluator view ({type(e).__name__}: {e})")
            continue
        print(f"task {tid} arm={r['arm_id']}: queries={len(trace)} retrieved={len(got)}")
        print(f"   retrieved : {sorted(got)[:10]}")
        print(f"   evidence  : {sorted(v.evidence_docids)[:10]}")
        print(f"   gold      : {sorted(v.gold_docids)[:10]}")
        print(f"   overlap   : evidence={len(got & v.evidence_docids)} "
              f"gold={len(got & v.gold_docids)}")
        if trace:
            print(f"   trace[0] keys: {sorted(trace[0])}")
            print(f"   trace[0] query: {str(trace[0].get('query'))[:110]}")
        print(f"   task question : {v.query[:110]}")
