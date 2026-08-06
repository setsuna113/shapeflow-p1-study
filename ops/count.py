import sqlite3, sys
phase = sys.argv[1] if len(sys.argv) > 1 else "campaign"
n = 0
for L in (0, 1):
    c = sqlite3.connect(
        f"file:/storage/nvme/shapeflow-data/runner-lane{L}/runs/ledger.sqlite?mode=ro", uri=True)
    n += list(c.execute(
        "select count(*) from work_items where phase_id=? and state=?",
        (phase, "COMMITTED")))[0][0]
print(n)
