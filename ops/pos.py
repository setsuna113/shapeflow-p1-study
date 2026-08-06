"""Is the censoring positional rather than arm-specific? If cells run in a fixed order within a
task block, an arm that always sits at one position inherits whatever fails at that position."""
import collections, json, sqlite3, sys

sched = "/storage/nvme/shapeflow-data/runner-lane0/runs/schedules/campaign1/b1_select.json"
import glob
cands = glob.glob("/storage/nvme/shapeflow-data/runner-lane*/runs/schedules/campaign1/*.json")
print("schedules:", cands)
body = json.loads(open(cands[0]).read())
pos_of = {}
order_seen = collections.Counter()
for block in body["blocks"]:
    arms = [c["arm"]["arm_id"] for c in block["cells"]]
    order_seen[tuple(arms)] += 1
    for i, a in enumerate(arms):
        pos_of.setdefault(a, collections.Counter())[i] += 1
print("distinct within-block arm orders:", len(order_seen))
for order, n in order_seen.most_common(3):
    print(f"  x{n}: {' > '.join(order)}")

state = {}
for L in (0, 1):
    root = f"/storage/nvme/shapeflow-data/runner-lane{L}"
    c = sqlite3.connect(f"file:{root}/runs/ledger.sqlite?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    for r in c.execute("select task_id t, arm_id a, state s from work_items "
                       "where phase_id='campaign' and state in ('COMMITTED','FAILED_FINAL')"):
        state[(str(r["t"]), r["a"])] = r["s"]

by_pos = collections.defaultdict(collections.Counter)
for block in body["blocks"]:
    for i, cell in enumerate(block["cells"]):
        key = (str(cell["task_id"]), cell["arm"]["arm_id"])
        if key in state:
            by_pos[i][state[key]] += 1
print(f"\n{'position':>8} {'ok':>5} {'fail':>5} {'fail%':>7}")
for i in sorted(by_pos):
    ok, bad = by_pos[i]["COMMITTED"], by_pos[i]["FAILED_FINAL"]
    print(f"{i:>8} {ok:>5} {bad:>5} {100*bad/max(1,ok+bad):>6.1f}%")
