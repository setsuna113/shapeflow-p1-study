#!/usr/bin/env python3
"""Drive the broker against a live vLLM through ProxyHost, replaying page batches as offers.

Each tick: release arrivals into the host's pending set, then ``run_tick`` -- scrape a versioned
snapshot from the engine's ``/metrics``, decide admission and form jointly, and materialize only
the chosen plans as real ``/v1/completions`` requests. Everything the run did is written under
``--out``:

    run.json        configuration, versions, digests, summary counts
    offers.jsonl    every offer as submitted (both plans, with their estimates)
    ticks.jsonl     per tick: arrivals, the decision and its digest, retries, what was committed
    snapshots.jsonl every snapshot: version, scrape time, features
    gate.jsonl      every materialize attempt: snapshot age at the gate and the verdict
    requests.jsonl  every sub-request sent and what vLLM returned
    renders.jsonl   the CPU render of every P1 selection

This exercises the mechanism end to end. It is not an evaluation: there is no baseline arm and
nothing here measures whether the joint decision is better than any other.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from shapeflow.broker.driver import DEFAULT_SOLVE_BUDGET_NS, run_tick  # noqa: E402
from shapeflow.broker.hosts.proxy import (  # noqa: E402
    EngineUnobservable,
    HttpEngine,
    ProxyHost,
    RequestRecord,
)
from shapeflow.broker.reference import EngineGatedBroker, GreedyJointBroker  # noqa: E402
from shapeflow.broker.replay import (  # noqa: E402
    CHAT_FORMAT,
    P0_PROMPT_VERSION,
    SELECTOR_PROMPT_VERSION,
    ReplayBatch,
    ReplayPage,
    build_h_offer,
    render_selection,
)
from shapeflow.contracts.versions import contracts_digest  # noqa: E402
from shapeflow.evidence.model_tokenizer import FrozenModelTokenizer  # noqa: E402
from shapeflow.hashing import sha256_hex  # noqa: E402

_TITLE = re.compile(r"^= (?!=)(.+?) =$")
_SUBTITLE = re.compile(r"^(?:= )+=?\s*(.+?)\s*(?:= )*=$")
#: Batch sizes cycled in order: single-page turns and wide fan-outs both occur.
BATCH_SIZES = (1, 3, 2, 5, 1, 4, 2, 3)


def load_wikitext_pages(parquet: Path, tokenizer, page_max_tokens: int,
                        min_tokens: int) -> list[ReplayPage]:
    """Wikitext-2 articles as pages, each cut to ``page_max_tokens`` at a token boundary."""
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet, columns=["text"]).column("text").to_pylist()
    articles: list[tuple[str, list[str]]] = []
    for row in rows:
        line = row.strip()
        if not line:
            continue
        match = _TITLE.match(line)
        if match:
            articles.append((match.group(1).strip(), []))
        elif articles:
            if line.startswith("= ="):
                sub = _SUBTITLE.match(line)
                line = "## " + (sub.group(1) if sub else line.strip("= "))
            articles[-1][1].append(line)
    pages = []
    for index, (title, paragraphs) in enumerate(articles):
        text = "\n\n".join(paragraphs)
        offsets = tokenizer.encode_offsets(text)
        if len(offsets) < min_tokens:
            continue
        if len(offsets) > page_max_tokens:
            text = text[: offsets[page_max_tokens - 1][1]]
        pages.append(ReplayPage(page_id=f"wikitext2-test-{index:04d}", title=title, text=text))
    return pages


def make_batches(pages: list[ReplayPage], items: int) -> list[ReplayBatch]:
    batches, cursor = [], 0
    for i in range(items):
        size = BATCH_SIZES[i % len(BATCH_SIZES)]
        chosen = tuple(pages[(cursor + k) % len(pages)] for k in range(size))
        cursor += size
        topic = "Collect the key facts about: " + "; ".join(p.title for p in chosen)
        batches.append(ReplayBatch(batch_id=f"H{i:03d}", topic=topic, pages=chosen))
    return batches


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen3-14b-awq", help="served model name")
    ap.add_argument("--model-dir", required=True, help="holds tokenizer.json")
    ap.add_argument("--model-revision", default="")
    ap.add_argument("--corpus", required=True, help="wikitext-2-raw-v1 test parquet")
    ap.add_argument("--corpus-revision", default="")
    ap.add_argument("--source-rev", default="", help="git revision of this tree")
    ap.add_argument("--items", type=int, default=48)
    ap.add_argument("--arrivals-per-tick", type=int, default=3)
    ap.add_argument("--tick-ms", type=float, default=250.0)
    ap.add_argument("--capacity-tokens", type=int, default=8192)
    ap.add_argument("--max-waiting", type=int, default=4)
    ap.add_argument("--max-snapshot-age-ms", type=float, default=100.0)
    ap.add_argument("--page-max-tokens", type=int, default=1200)
    ap.add_argument("--p0-max-tokens", type=int, default=8192,
                    help="P0 summary cap; vendor default 8192, and est_decode prices it")
    ap.add_argument("--selector-max-tokens", type=int, default=768)
    ap.add_argument("--chunk-max-tokens", type=int, default=320)
    ap.add_argument("--min-page-tokens", type=int, default=200)
    ap.add_argument("--timeout-s", type=float, default=900.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = FrozenModelTokenizer(Path(args.model_dir) / "tokenizer.json")
    pages = load_wikitext_pages(Path(args.corpus), tokenizer, args.page_max_tokens,
                                args.min_page_tokens)
    batches = make_batches(pages, args.items)

    engine = HttpEngine(args.base_url)
    candidates_by_item: dict[str, tuple] = {}
    renders: list[dict] = []
    renders_lock = threading.Lock()

    def on_complete(record: RequestRecord) -> None:
        if record.form != "P1" or record.error:
            return
        outcome = render_selection(record.text, candidates_by_item[record.item_id])
        with renders_lock:
            renders.append({"item_id": record.item_id, **outcome.__dict__})

    host = ProxyHost(scrape=engine.scrape, send=engine.send, model=args.model,
                     sampling={"temperature": 0.0, "seed": 0},
                     max_snapshot_age_ns=int(args.max_snapshot_age_ms * 1e6),
                     on_complete=on_complete)
    broker = EngineGatedBroker(inner=GreedyJointBroker(capacity_tokens=args.capacity_tokens),
                               max_waiting=args.max_waiting)

    import httpx

    vllm_version = httpx.get(args.base_url + "/version", timeout=10).json().get("version")
    offers_log, ticks = [], []
    started = time.time()
    next_batch = 0
    tick_index = 0
    print(f"{len(pages)} pages, {len(batches)} offers; vLLM {vllm_version} at {args.base_url}",
          flush=True)

    while True:
        tick_started = time.perf_counter()
        arrivals = []
        for _ in range(args.arrivals_per_tick):
            if next_batch >= len(batches):
                break
            built = build_h_offer(batches[next_batch], tokenizer=tokenizer,
                                  submitted_at_ns=time.time_ns(),
                                  p0_max_tokens=args.p0_max_tokens,
                                  selector_max_tokens=args.selector_max_tokens,
                                  chunk_max_tokens=args.chunk_max_tokens)
            next_batch += 1
            candidates_by_item[built.offer.item_id] = built.candidates
            host.submit(built.offer, built.bodies)
            arrivals.append(built.offer.item_id)
            offers_log.append({**built.offer.content(),
                               "page_ids": [p.page_id for p in batches[next_batch - 1].pages]})
        pending_before = [o.item_id for o in host.pending()]

        row = {"tick": tick_index, "wall_ns": time.time_ns(), "arrivals": arrivals,
               "pending_before": pending_before, "inflight_before": host.inflight()}
        try:
            outcome = run_tick(host, broker, budget_ns=DEFAULT_SOLVE_BUDGET_NS)
        except EngineUnobservable as exc:
            row["error"] = f"EngineUnobservable: {exc}"
        else:
            row.update({
                "attempts": outcome.attempts, "stale_retries": outcome.stale_retries,
                "budget_overrun": outcome.budget_overrun, "fail_closed": outcome.fail_closed,
                "committed": outcome.committed, "reason": outcome.reason,
                "decision": outcome.decision.content() if outcome.decision else None,
                "decision_digest": outcome.decision.digest if outcome.decision else None,
                "result": outcome.result.content() if outcome.result else None,
            })
        ticks.append(row)
        tick_index += 1

        done = next_batch >= len(batches) and not host.pending() and host.inflight() == 0
        if done or time.time() - started > args.timeout_s:
            break
        remaining = args.tick_ms / 1000 - (time.perf_counter() - tick_started)
        if remaining > 0:
            time.sleep(remaining)

    drained = host.drain(timeout_s=60)
    host.close()
    finished = time.time()

    records = sorted((r.to_json() for r in host.records),
                     key=lambda r: (r["committed_ns"], r["item_id"], r["ordinal"]))
    write_jsonl(out / "offers.jsonl", offers_log)
    write_jsonl(out / "ticks.jsonl", ticks)
    write_jsonl(out / "snapshots.jsonl", (s.__dict__ for s in host.snapshots))
    write_jsonl(out / "gate.jsonl", host.gate_log)
    write_jsonl(out / "requests.jsonl", records)
    write_jsonl(out / "renders.jsonl", sorted(renders, key=lambda r: r["item_id"]))

    forms = {}
    for decision in host.committed:
        forms.update({k: v.value for k, v in decision.form.items()})
    passed = [g["age_ns"] for g in host.gate_log if g["passed"]]
    scrapes = [s.scrape_ns for s in host.snapshots]
    matched = [r for r in records if r["prompt_tokens"] is not None]
    summary = {
        "ticks": len(ticks),
        "ticks_with_engine_error": sum(1 for t in ticks if "error" in t),
        "ticks_gate_shut": sum(1 for t in ticks
                               if (t.get("reason") or "").startswith("engine gate shut")),
        "stale_retries": sum(t.get("stale_retries", 0) for t in ticks),
        "fail_closed_ticks": sum(1 for t in ticks if t.get("fail_closed")),
        "budget_overruns": sum(1 for t in ticks if t.get("budget_overrun")),
        "offers": len(offers_log),
        "offers_admitted": len(forms),
        "offers_never_admitted": sorted(set(o["item_id"] for o in offers_log) - set(forms)),
        "admitted_as": {f: sum(1 for v in forms.values() if v == f) for f in ("P0", "P1")},
        "requests_sent": len(records),
        "requests_failed": sum(1 for r in records if r["error"]),
        "prompt_tokens_estimate_exact": sum(
            1 for r in matched if r["prompt_tokens"] == r["est_prompt_tokens"]),
        "prompt_tokens_compared": len(matched),
        "p1_renders_ok": sum(1 for r in renders if r["ok"]),
        "p1_renders": len(renders),
        "gate_age_ms": ({"min": min(passed) / 1e6, "median": statistics.median(passed) / 1e6,
                         "max": max(passed) / 1e6} if passed else None),
        "scrape_ms": ({"median": statistics.median(scrapes) / 1e6, "max": max(scrapes) / 1e6}
                      if scrapes else None),
        "drained": drained,
        "wall_s": round(finished - started, 3),
    }
    run = {
        "what": "ProxyHost live replay: broker outside an unmodified vLLM",
        "args": vars(args),
        "vllm_version": vllm_version,
        "tokenizer_sha256": tokenizer.identity_sha256,
        "corpus_sha256": sha256_hex(Path(args.corpus).read_bytes()),
        "contracts_sha": contracts_digest(),
        "chat_format": CHAT_FORMAT,
        "p0_prompt_version": P0_PROMPT_VERSION,
        "selector_prompt_version": SELECTOR_PROMPT_VERSION,
        "broker": {"type": "EngineGatedBroker(GreedyJointBroker)",
                   "capacity_tokens": args.capacity_tokens, "max_waiting": args.max_waiting},
        "max_snapshot_age_ns": host.max_snapshot_age_ns,
        "started_unix": started, "finished_unix": finished,
        "summary": summary,
    }
    (out / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if drained else 1


if __name__ == "__main__":
    raise SystemExit(main())
