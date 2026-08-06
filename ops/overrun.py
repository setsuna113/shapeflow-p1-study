"""Every preflight overrun in the campaign, by magnitude, and every non-preflight deferral."""
import collections, json, re, sqlite3, sys

sys.path.insert(0, "/storage/nvme/shapeflow-p1-study/src")
from shapeflow.object_store import ObjectStore  # noqa: E402

BUDGET = 512
PREFLIGHT = re.compile(r"PREFLIGHT: rendered (\d+) tokens exceeds selected_token_budget (\d+)")
over = collections.defaultdict(list)
other = collections.Counter()

for L in (0, 1):
    root = f"/storage/nvme/shapeflow-data/runner-lane{L}"
    store = ObjectStore(root + "/object_store")
    c = sqlite3.connect(f"file:{root}/runs/ledger.sqlite?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    q = ("select w.arm_id a, x.result_object_ref r from work_items w "
         "join attempts x on x.work_key=w.work_key "
         "where w.phase_id='campaign' and w.state='COMMITTED'")
    for row in c.execute(q):
        if not row["r"]:
            continue
        body = json.loads(store.get_bytes(row["r"]).decode())
        for e in body.get("events") or []:
            if not isinstance(e, dict):
                continue
            text = " ".join(str(e.get(k)) for k in ("reason", "detail", "message", "why")
                            if e.get(k))
            if not text:
                continue
            m = PREFLIGHT.search(text)
            if m:
                over[row["a"]].append(int(m.group(1)))
            elif "Error" in text or "error" in text:
                other[f"{row['a']}: {text.split(':')[1].strip()[:52]}"] += 1

print(f"{'arm':<18} {'overruns':>9} {'min':>6} {'p50':>6} {'p90':>6} {'max':>7} "
      f"{'median x budget':>16} {'within 2x':>10}")
allv = []
for arm in sorted(over):
    v = sorted(over[arm]); allv += v
    n = len(v)
    print(f"{arm:<18} {n:>9} {v[0]:>6} {v[n//2]:>6} {v[int(n*.9)]:>6} {v[-1]:>7} "
          f"{v[n//2]/BUDGET:>15.2f}x {100*sum(1 for x in v if x <= 2*BUDGET)/n:>9.0f}%")
v = sorted(allv); n = len(v)
print(f"{'ALL':<18} {n:>9} {v[0]:>6} {v[n//2]:>6} {v[int(n*.9)]:>6} {v[-1]:>7} "
      f"{v[n//2]/BUDGET:>15.2f}x {100*sum(1 for x in v if x <= 2*BUDGET)/n:>9.0f}%")
print(f"\noverruns that would fit if the aggregator trimmed to budget: all of them, by "
      f"construction; {sum(1 for x in v if x <= 2*BUDGET)}/{n} are within 2x of 512.")
print("\nnon-preflight deferral causes:")
for text, k in other.most_common(6):
    print(f"  x{k:<4} {text}")
