"""Per-arm terminal outcome counts for a phase, plus the remaining work per lane."""
import collections, sqlite3, sys

PHASE = sys.argv[1] if len(sys.argv) > 1 else "campaign"
per_arm = collections.defaultdict(collections.Counter)
lane_open = {}
for L in (0, 1):
    root = f"/storage/nvme/shapeflow-data/runner-lane{L}"
    c = sqlite3.connect(f"file:{root}/runs/ledger.sqlite?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    q = ("select arm_id as a, state as s, count(*) as n from work_items "
         "where phase_id=? group by arm_id, state")
    for r in c.execute(q, (PHASE,)):
        if r["s"] in ("COMMITTED", "FAILED_FINAL"):
            per_arm[r["a"]][r["s"]] += r["n"]
    # owned but not terminal, for this lane only
    q2 = ("select count(*) as n from work_items where phase_id=? "
          "and state not in ('COMMITTED','FAILED_FINAL')")
    lane_open[L] = c.execute(q2, (PHASE,)).fetchone()["n"]

print(f"{'arm':<18} {'ok':>5} {'fail':>5} {'fail%':>7}")
tot_ok = tot_bad = 0
for arm in sorted(per_arm):
    ok = per_arm[arm]["COMMITTED"]; bad = per_arm[arm]["FAILED_FINAL"]
    tot_ok += ok; tot_bad += bad
    print(f"{arm:<18} {ok:>5} {bad:>5} {100*bad/max(1,ok+bad):>6.1f}%")
print(f"{'TOTAL':<18} {tot_ok:>5} {tot_bad:>5} {100*tot_bad/max(1,tot_ok+tot_bad):>6.1f}%")
print(f"terminal {tot_ok+tot_bad} of 560; not-terminal in each lane ledger: {lane_open}")
