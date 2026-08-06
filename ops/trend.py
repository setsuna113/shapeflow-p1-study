"""Running agent-level recall over a phase, so the competence floor is not a surprise at the end."""
import json, sqlite3, statistics, sys
from pathlib import Path

sys.path.insert(0, "/storage/nvme/shapeflow-p1-study/src")
from shapeflow.object_store import ObjectStore  # noqa: E402
from shapeflow.bench.bcplus.qrels import load_bcplus_evaluator_queries  # noqa: E402

PHASE = sys.argv[1] if len(sys.argv) > 1 else "competence"
views = load_bcplus_evaluator_queries(
    Path("/storage/sata/shapeflow/benchdata/browsecomp-plus/data"))

ev, gd, nq, nd, empties = [], [], [], [], 0
for L in (0, 1):
    root = f"/storage/nvme/shapeflow-data/runner-lane{L}"
    store = ObjectStore(root + "/object_store")
    c = sqlite3.connect(f"file:{root}/runs/ledger.sqlite?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    for r in c.execute(
        "select w.task_id,a.result_object_ref ref from work_items w "
        "join attempts a on a.work_key=w.work_key "
        "where w.phase_id=? and w.state='COMMITTED' and a.state='COMMITTED'", (PHASE,)):
        b = json.loads(store.get_bytes(r["ref"]).decode())
        trace = b.get("retrieval_trace") or []
        got = set()
        for t in trace:
            got.update(t.get("docids") or [])
        try:
            v = views.get(str(r["task_id"]))
        except Exception:
            continue
        nq.append(len(trace))
        nd.append(len(got))
        if not (b.get("final_report") or "").strip():
            empties += 1
        if v.evidence_docids:
            ev.append(len(got & v.evidence_docids) / len(v.evidence_docids))
        if v.gold_docids:
            gd.append(len(got & v.gold_docids) / len(v.gold_docids))

n = len(ev)
if not n:
    print("no cells yet"); raise SystemExit(0)
print(f"cells={n}  empty reports={empties}")
print(f"queries/cell   mean={statistics.mean(nq):.1f}  distinct docs/cell mean={statistics.mean(nd):.1f}")
print(f"evidence recall mean={statistics.mean(ev):.4f}   floor 0.40   "
      f"(>0 in {sum(1 for x in ev if x > 0)}/{n} cells)")
print(f"gold     recall mean={statistics.mean(gd):.4f}   "
      f"(>0 in {sum(1 for x in gd if x > 0)}/{len(gd)} cells)")
