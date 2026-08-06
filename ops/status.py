import json, sys
for L in (0, 1):
    p = f"/storage/nvme/shapeflow-data/runner-lane{L}/runs/STATUS_{sys.argv[1]}.json"
    try:
        b = json.load(open(p))
    except FileNotFoundError:
        print(f"=== LANE {L}: no status ==="); continue
    print(f"=== LANE {L} ===  ok={b.get('ok')}  cells={b.get('cell_states')}")
    print(f"  blocks_frozen={b.get('blocks_frozen')} paired_valid={b.get('blocks_valid_for_paired_estimate')} owned={b.get('owned_blocks')} tasks={b.get('tasks')}")
    w = b.get("world_searched", {})
    print(f"  world_searched: {w.get('status')} | {w.get('detail')}")
    tl = b.get("treatment_live", {})
    print(f"  treatment_live: {tl.get('status')} | {tl.get('detail')}")
    for a, s in sorted((tl.get("by_arm") or {}).items()):
        print("    {:<18} cells={} published={} page_fallbacks={} close_failures={} -> {}".format(
            a, s["cells"], s["published"], s["page_fallbacks"], s["close_failures"], s["status"]))
