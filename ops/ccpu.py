"""Where C_CPU_CONTROL's failures sit: by lane, and over time against the arm's own commits."""
import sqlite3, sys

PHASE = "campaign"
ARM = sys.argv[1] if len(sys.argv) > 1 else "C_CPU_CONTROL"
rows = []
for L in (0, 1):
    root = f"/storage/nvme/shapeflow-data/runner-lane{L}"
    c = sqlite3.connect(f"file:{root}/runs/ledger.sqlite?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    q = ("select state as s, updated_at as t from work_items "
         "where phase_id=? and arm_id=? and state in ('COMMITTED','FAILED_FINAL')")
    for r in c.execute(q, (PHASE, ARM)):
        rows.append((r["t"], L, r["s"]))
rows.sort()
for L in (0, 1):
    sub = [r for r in rows if r[1] == L]
    bad = sum(1 for r in sub if r[2] == "FAILED_FINAL")
    print(f"  lane{L}: {len(sub)} terminal, {bad} failed ({100*bad/max(1,len(sub)):.0f}%)")
q = len(rows) // 4 or 1
for i in range(0, len(rows), q):
    chunk = rows[i:i+q]
    bad = sum(1 for r in chunk if r[2] == "FAILED_FINAL")
    print(f"  quarter {i//q+1}: {len(chunk)} terminal, {bad} failed "
          f"({100*bad/max(1,len(chunk)):.0f}%)")
