"""Full work-item state distribution for a phase, per lane, restricted to items the lane owns."""
import collections, sqlite3, sys

PHASE = sys.argv[1] if len(sys.argv) > 1 else "campaign"
for L in (0, 1):
    root = f"/storage/nvme/shapeflow-data/runner-lane{L}"
    c = sqlite3.connect(f"file:{root}/runs/ledger.sqlite?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    q = "select state as s, count(*) as n from work_items where phase_id=? group by state"
    d = {r["s"]: r["n"] for r in c.execute(q, (PHASE,))}
    print(f"lane{L} phase={PHASE}: {d}  (sum {sum(d.values())})")
    q2 = ("select state as s, arm_id as a, count(*) as n from work_items "
          "where phase_id=? and state not in ('COMMITTED','FAILED_FINAL','PENDING') "
          "group by state, arm_id")
    for r in c.execute(q2, (PHASE,)):
        print(f"    non-standard: {r['s']:<16} {r['a']:<18} {r['n']}")
# every phase, to see FAILED_UNKNOWN globally
print("--- all phases, lane-wide ---")
for L in (0, 1):
    root = f"/storage/nvme/shapeflow-data/runner-lane{L}"
    c = sqlite3.connect(f"file:{root}/runs/ledger.sqlite?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    q = "select phase_id as p, state as s, count(*) as n from work_items group by phase_id, state"
    agg = collections.defaultdict(dict)
    for r in c.execute(q):
        agg[r["p"]][r["s"]] = r["n"]
    for p in sorted(agg):
        print(f"  lane{L} {p:<14} {agg[p]}")
