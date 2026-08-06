import sqlite3, sys, collections
phase = sys.argv[1] if len(sys.argv) > 1 else "campaign"
tot = collections.Counter()
for L in (0, 1):
    c = sqlite3.connect(
        f"file:/storage/nvme/shapeflow-data/runner-lane{L}/runs/ledger.sqlite?mode=ro", uri=True)
    st = dict(c.execute(
        "select state,count(*) from work_items where phase_id=? group by state", (phase,)))
    print(f"lane {L}: {st}")
    for arm, n in c.execute(
        "select arm_id,count(*) from work_items where phase_id=? and state='COMMITTED' "
        "group by arm_id", (phase,)):
        tot[arm] += n
print("committed by arm:", dict(sorted(tot.items())), "total", sum(tot.values()))
