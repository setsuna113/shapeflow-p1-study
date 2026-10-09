"""Audit a brokered run: what the broker decided, on what, and what the agents actually sent.

Reads the broker's logs and every worker's logs and checks three claims, each against evidence
the claim's author did not write:

1. **The offers came from the real agent.** Every offer names a checkpoint the agent's own cell
   record shows being captured, and the cell record orders checkpoint -> offer -> decision.
2. **The decisions were made on live engine state.** Every tick's snapshot came from the
   engine's ``/metrics`` within the age ceiling; re-running the policy on each logged
   ``(offers, snapshot)`` reproduces the logged decision digest; and re-running it with the
   engine's waiting gauge altered shows how many decisions that gauge actually moved.
3. **Only the chosen form was sent.** Every request the provider committed is joined to an
   offer by ``prompt_sha256``; a P1 item's P0 requests appear only when its P1 failed and fell
   back (C2), a P0 item's selector request never appears, and no request reached the provider
   before the ticket that released it (C3).

Kept outside ``shapeflow/broker`` on purpose: it reconstructs snapshots from logged feature
names, and the observability check constrains what *decides*, not what audits a decision.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

from ..broker.codec import offer_digest, offer_from_content
from ..contracts.observability import ClosedFeatureMap
from ..contracts.offer import SchedulerSnapshot, SnapshotVersion

__all__ = ["audit", "load_jsonl"]

P1_OPS = {"PAGE_P1_SELECTOR_BATCH", "COMPRESSOR_P1_SELECTOR"}
P0_OPS = {"PAGE_P0_SUMMARY", "COMPRESSOR_P0"}
WAITING = "engine.num_requests_waiting"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _stats(values: list[float]) -> dict | None:
    if not values:
        return None
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "max": max(values)}


def _snapshot(row: dict) -> SchedulerSnapshot:
    snap = row["snapshot"]
    return SchedulerSnapshot(
        version=SnapshotVersion(epoch=snap["epoch"], tick=snap["tick"]),
        taken_at_ns=snap["taken_at_ns"], features=ClosedFeatureMap(snap["features"]))


def _with_waiting(snapshot: SchedulerSnapshot, waiting: float) -> SchedulerSnapshot:
    return SchedulerSnapshot(version=snapshot.version, taken_at_ns=snapshot.taken_at_ns,
                             features=ClosedFeatureMap({**dict(snapshot.features),
                                                        WAITING: waiting}))


def _replay(ticks: Iterable[dict], offers_by_id: dict, broker, max_waiting: int) -> dict:
    reproduced = mismatched = skipped = 0
    moved_low = moved_high = considered = 0
    for row in ticks:
        decision = row.get("decision")
        if not decision or row.get("fail_closed") or not row.get("snapshot"):
            skipped += 1
            continue
        offers = [offers_by_id[i] for i in row["pending"] if i in offers_by_id]
        snapshot = _snapshot(row)
        replayed = broker.decide(offers, snapshot, 0)
        if replayed.digest == row["decision_digest"]:
            reproduced += 1
        else:
            mismatched += 1
        considered += 1
        if broker.decide(offers, _with_waiting(snapshot, 0.0), 0).digest != replayed.digest:
            moved_low += 1
        if broker.decide(offers, _with_waiting(snapshot, max_waiting + 1.0), 0).digest \
                != replayed.digest:
            moved_high += 1
    return {
        "ticks_replayed": considered, "digest_reproduced": reproduced,
        "digest_mismatched": mismatched, "ticks_skipped_fail_closed_or_empty": skipped,
        "ticks_whose_decision_changes_if_queue_empty": moved_low,
        "ticks_whose_decision_changes_if_queue_long": moved_high,
    }


def audit(run_dir: Path, *, broker_factory: Callable[[], object], max_waiting: int,
          max_snapshot_age_ns: int) -> dict:
    run_dir = Path(run_dir)
    broker_dir = run_dir / "broker"
    ticks = load_jsonl(broker_dir / "ticks.jsonl")
    received = load_jsonl(broker_dir / "offers.jsonl")
    gate = load_jsonl(broker_dir / "gate.jsonl")
    snapshots = load_jsonl(broker_dir / "snapshots.jsonl")
    items = {row["item_id"]: row for row in load_jsonl(broker_dir / "items.jsonl")}

    agent_rows, requests, cells = [], [], []
    for worker in sorted(run_dir.glob("worker-*")):
        agent_rows += load_jsonl(worker / "agent_offers.jsonl")
        requests += load_jsonl(worker / "requests.jsonl")
        cells += load_jsonl(worker / "cells.jsonl")
    offered = [r for r in agent_rows if "offer" in r]
    executed = {r["item_id"]: r["executed"] for r in agent_rows if "executed" in r}

    # --- 1. offers came from the agents' own handoffs -------------------------------------------
    events_by_cell = {c["cell_token"]: c["events"] for c in cells}
    order_ok, order_bad = 0, []
    for row in offered:
        cell = row["item_id"].split(":")[0]
        kinds = [(e["kind"], e.get("checkpoint")) for e in events_by_cell.get(cell, [])]
        positions = [i for i, (k, c) in enumerate(kinds)
                     if k == "BROKER_OFFER" and c == row["checkpoint"]]
        ok = False
        for i in positions:
            before = [k for k, c in kinds[:i] if c == row["checkpoint"]]
            after = [k for k, c in kinds[i + 1:] if c == row["checkpoint"]]
            if before and before[-1] in {"H_CHECKPOINT", "C_CHECKPOINT"} \
                    and after and after[0] == "BROKER_DECISION":
                ok = True
        if ok:
            order_ok += 1
        else:
            order_bad.append(row["item_id"])
    received_digests = {r["offer"]["item_id"]: r["digest"] for r in received}
    agent_digests = {}
    for row in offered:
        agent_digests[row["item_id"]] = offer_digest(offer_from_content(row["offer"]))
    offers_section = {
        "offers_from_agents": len(offered),
        "by_boundary": dict(Counter(r["boundary"] for r in offered)),
        "p1_offered": sum(1 for r in offered if r["offer"]["p1"]),
        "p1_absent_reasons": dict(Counter(r["p1_absent_reason"] for r in offered
                                          if not r["offer"]["p1"])),
        "checkpoint_then_offer_then_decision": order_ok,
        "order_violations": order_bad,
        "broker_received_matches_agent_offer": sum(
            1 for i, d in agent_digests.items() if received_digests.get(i) == d),
        "cells": len(cells), "cells_ok": sum(1 for c in cells if c["ok"]),
        "cell_errors": [c["error"] for c in cells if not c["ok"]],
    }

    # --- 2. decisions on live engine state ------------------------------------------------------
    offers_by_id = {r["offer"]["item_id"]: offer_from_content(r["offer"]) for r in received}
    with_offers = [t for t in ticks if t.get("pending")]
    passed = [g for g in gate if g["passed"]]
    engine_section = {
        "ticks_with_offers": len(with_offers),
        "ticks_engine_unobservable": sum(1 for t in with_offers if "error" in t),
        "ticks_gate_shut": sum(1 for t in with_offers
                               if "engine gate shut" in (t.get("reason") or "")),
        "ticks_deadline_forced": sum(1 for t in with_offers
                                     if "deadline-forced" in (t.get("reason") or "")
                                     and "deadline-forced 0" not in (t.get("reason") or "")),
        "stale_retries": sum(t.get("stale_retries", 0) for t in with_offers),
        "fail_closed_ticks": sum(1 for t in with_offers if t.get("fail_closed")),
        "budget_overruns": sum(1 for t in with_offers if t.get("budget_overrun")),
        "snapshots": len(snapshots),
        "engine_epochs": sorted({s["epoch"] for s in snapshots}),
        "gate_age_ms_at_commit": _stats([g["age_ns"] / 1e6 for g in passed]),
        "gate_refusals": dict(Counter(g["why"] for g in gate if not g["passed"])),
        "commits_within_age_ceiling": all(g["age_ns"] <= max_snapshot_age_ns for g in passed),
        "waiting_gauge_seen": _stats([s["features"][WAITING] for s in snapshots]),
        "kv_usage_seen": _stats([s["features"]["engine.kv_cache_usage_perc"]
                                 for s in snapshots]),
        "replay": _replay(with_offers, offers_by_id, broker_factory(), max_waiting),
    }

    # --- 3. only the chosen form was sent -------------------------------------------------------
    committed = [e for e in requests if e.get("kind") == "INFERENCE_COMMITTED"]
    sent_by_cell: dict[str, Counter] = defaultdict(Counter)
    ingress: dict[tuple, list] = defaultdict(list)
    for e in committed:
        sent_by_cell[e["cell"]][e["prompt_sha256"]] += 1
        ingress[(e["cell"], e["prompt_sha256"])].append(e)
    fell_back = {(c["cell_token"], e["checkpoint"]) for c in cells for e in c["events"]
                 if e["kind"] == "PAGE_BATCH_REDUCED" and e.get("fell_back")}
    # Two items in one cell can price byte-identical requests (a researcher that repeats a
    # search can rebuild the same view). Requests are therefore paired with items per
    # (cell, prompt) in time order -- each ticket with the next request after it -- rather than
    # every item claiming every request that shares its hash.
    tickets = {r["item_id"]: ((items.get(r["item_id"]) or {}).get("ticket") or r.get("ticket"))
               for r in offered}
    pending_by_key: dict[tuple, list] = defaultdict(list)
    for row in offered:
        cell, offer = row["item_id"].split(":")[0], row["offer"]
        outcome = (executed.get(row["item_id"]) or {}).get("outcome")
        p1 = offer["p1"]["selector"] if offer["p1"] else None
        chosen = [p1] if row["form"] == "P1" and p1 else list(offer["p0"]["requests"])
        if row["form"] == "P1" and outcome == "P1_FAILED":
            # The failed P1 fell back: at H the adapter re-renders P0, at C vendor compresses.
            chosen += list(offer["p0"]["requests"])
        committed_ns = (tickets[row["item_id"]] or {}).get("committed_ns", 0)
        for descriptor in chosen:
            pending_by_key[(cell, descriptor["prompt_sha256"])].append(
                (committed_ns, row["item_id"], descriptor))
    claimed, c3_violations, est = set(), [], []
    paired_by_item: dict[str, Counter] = defaultdict(Counter)
    for key, wants in pending_by_key.items():
        arrivals = sorted(ingress.get(key, []), key=lambda e: e["proxy_ingress_ts"])
        used = set()
        for committed_ns, item_id, descriptor in sorted(wants, key=lambda w: w[0]):
            match = next((i for i, e in enumerate(arrivals) if i not in used
                          and e["proxy_ingress_ts"] >= committed_ns / 1e9 - 1e-3), None)
            if match is None:
                early = next((i for i, e in enumerate(arrivals) if i not in used), None)
                if early is not None:
                    used.add(early)
                    claimed.add(id(arrivals[early]))
                    c3_violations.append(item_id)
                continue
            used.add(match)
            claimed.add(id(arrivals[match]))
            paired_by_item[item_id][descriptor["prompt_sha256"]] += 1
            est.append((descriptor, arrivals[match]))
    form_rows = []
    for row in offered:
        cell, offer = row["item_id"].split(":")[0], row["offer"]
        sent = sent_by_cell[cell]
        p0 = Counter(r["prompt_sha256"] for r in offer["p0"]["requests"])
        p1 = offer["p1"]["selector"]["prompt_sha256"] if offer["p1"] else None
        mine = paired_by_item[row["item_id"]]
        outcome = (executed.get(row["item_id"]) or {}).get("outcome")
        form_rows.append({
            "item_id": row["item_id"], "boundary": offer["boundary"], "form": row["form"],
            "source": row["source"], "reason": row["reason"],
            "p0_priced": sum(p0.values()),
            "p0_sent": sum(min(mine[h], n) for h, n in p0.items()),
            "p1_sent": mine[p1] if p1 else 0,
            # Requests carrying this item's P1 prompt, from any item in the cell.
            "p1_prompt_seen_in_cell": sent[p1] if p1 else 0,
            "fell_back": (cell, row["checkpoint"]) in fell_back
            or (row["form"] == "P1" and outcome == "P1_FAILED"),
            "executed": outcome,
        })
    rows_h = [r for r in form_rows if r["boundary"] == "H"]
    exact = est
    sends_section = {
        "decided": dict(Counter(f"{r['boundary']}/{r['form']}/{r['source']}"
                                for r in form_rows).most_common()),
        "fail_closed_reasons": dict(Counter(r["reason"] for r in form_rows
                                            if r["source"] == "fail_closed")),
        "executed_outcomes": dict(Counter(r["executed"] for r in form_rows)),
        "h_p1_items_with_p1_sent": sum(1 for r in rows_h if r["form"] == "P1"
                                       and r["p1_sent"] >= 1),
        "h_p1_items": sum(1 for r in rows_h if r["form"] == "P1"),
        "h_p1_items_with_p0_sent_without_fallback": [
            r["item_id"] for r in rows_h
            if r["form"] == "P1" and r["p0_sent"] and not r["fell_back"]],
        "h_p1_items_fell_back": sum(1 for r in rows_h if r["form"] == "P1" and r["fell_back"]),
        "h_p0_items": sum(1 for r in rows_h if r["form"] == "P0"),
        "h_p0_items_all_priced_requests_sent": sum(
            1 for r in rows_h if r["form"] == "P0" and r["p0_sent"] == r["p0_priced"]),
        "p0_items_with_selector_sent": [r["item_id"] for r in form_rows
                                        if r["form"] == "P0" and r["p1_sent"]],
        "c_p1_items_fell_back": sum(1 for r in form_rows if r["boundary"] == "C"
                                    and r["form"] == "P1" and r["fell_back"]),
        "c_p1_items_with_p1_sent": sum(1 for r in form_rows if r["boundary"] == "C"
                                       and r["form"] == "P1" and r["p1_sent"] >= 1),
        "c_p1_items": sum(1 for r in form_rows if r["boundary"] == "C" and r["form"] == "P1"),
        "c_p0_items_with_exact_prompt_match": sum(
            1 for r in form_rows if r["boundary"] == "C" and r["form"] == "P0"
            and r["p0_sent"] == r["p0_priced"]),
        "c_p0_items": sum(1 for r in form_rows if r["boundary"] == "C" and r["form"] == "P0"),
        "requests_before_their_ticket": sorted(set(c3_violations)),
        "treatment_requests_committed": dict(Counter(
            e["op_class"] for e in committed if e["op_class"] in P0_OPS | P1_OPS)),
        "treatment_requests_unjoined": dict(Counter(
            e["op_class"] for e in committed
            if e["op_class"] in P0_OPS | P1_OPS and id(e) not in claimed)),
        "prefill_estimate_exact": sum(1 for d, e in exact
                                      if d["est_uncached_prefill"] == e["prompt_tokens"]),
        "prefill_estimate_joined": len(exact),
        "prefill_abs_error": _stats([abs(d["est_uncached_prefill"] - e["prompt_tokens"])
                                     for d, e in exact]),
        "decode_cap_respected": all(e["completion_tokens"] <= d["est_decode"]
                                    for d, e in exact),
        "completion_tokens_by_op": {
            op: _stats([e["completion_tokens"] for d, e in exact if e["op_class"] == op])
            for op in sorted({e["op_class"] for d, e in exact})},
    }
    return {"offers": offers_section, "engine": engine_section, "sends": sends_section,
            "items": form_rows}
