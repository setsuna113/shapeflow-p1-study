#!/usr/bin/env python3
"""Run N real ODR agents against one live vLLM, with one broker choosing the form at each handoff.

    python scripts/broker_agents_live.py --model-dir /models/Qwen3-14B-AWQ \
        --bm25-url http://127.0.0.1:8720 --tasks tasks.jsonl --workers 4 --out runs/X

1. Preflight: the installed graph is the pinned patched tree, vLLM answers, the BM25 world
   answers, and the contract digests are recorded.
2. The broker service starts in this process: an ``AgentHost`` reading the engine's ``/metrics``
   and ``DeadlineBroker(EngineGatedBroker(GreedyJointBroker))``.
3. N worker processes (``shapeflow.campaign.brokered``) run disjoint task shards. Each worker's
   agent suspends at every handoff until the broker decides.
4. When every worker has exited the broker stops and the run is audited
   (``shapeflow.campaign.broker_audit``).

Writes ``run.json`` (configuration, versions, digests, audit) next to the broker's and the
workers' logs. A mechanism run: no baseline arm, and nothing here measures whether the decisions
were good.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from shapeflow.broker.hosts.agent import AgentHost  # noqa: E402
from shapeflow.broker.hosts.proxy import HttpEngine  # noqa: E402
from shapeflow.broker.reference import (  # noqa: E402
    DeadlineBroker,
    EngineGatedBroker,
    GreedyJointBroker,
)
from shapeflow.broker.service import BrokerService  # noqa: E402
from shapeflow.campaign.broker_audit import audit  # noqa: E402
from shapeflow.contracts.versions import contracts_digest  # noqa: E402
from shapeflow.evidence.model_tokenizer import FrozenModelTokenizer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-revision", default="")
    ap.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    ap.add_argument("--bm25-url", required=True)
    ap.add_argument("--tasks", required=True, help="JSONL of {task_id, question}")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-rev", default="")
    ap.add_argument("--page-variant", default="LLM-PROMPTVIEW")
    ap.add_argument("--close-variant", default="C01")
    ap.add_argument("--tick-ms", type=float, default=250.0)
    ap.add_argument("--capacity-tokens", type=int, default=262_144)
    ap.add_argument("--max-waiting", type=int, default=4)
    ap.add_argument("--max-snapshot-age-ms", type=float, default=100.0)
    ap.add_argument("--deadline-s", type=float, default=60.0)
    ap.add_argument("--grace-s", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    import httpx

    from shapeflow.doctor import check_installed_graph

    graph = check_installed_graph(REPO)
    vllm_version = httpx.get(args.vllm_url + "/version", timeout=10).json().get("version")
    bm25_health = httpx.get(args.bm25_url + "/healthz", timeout=10).json()
    tokenizer = FrozenModelTokenizer(Path(args.model_dir) / "tokenizer.json")
    tasks = [json.loads(line) for line in Path(args.tasks).read_text(encoding="utf-8")
             .splitlines() if line.strip()]

    def policy():
        return DeadlineBroker(EngineGatedBroker(
            GreedyJointBroker(capacity_tokens=args.capacity_tokens),
            max_waiting=args.max_waiting))

    engine = HttpEngine(args.vllm_url, timeout_s=10.0)
    host = AgentHost(scrape=engine.scrape,
                     max_snapshot_age_ns=int(args.max_snapshot_age_ms * 1e6))
    service = BrokerService(host, policy(), tick_ms=args.tick_ms, log_dir=out / "broker")
    broker_url = f"http://127.0.0.1:{service.start()}"
    print(f"broker at {broker_url}; {len(tasks)} tasks over {args.workers} workers; "
          f"vLLM {vllm_version}", flush=True)

    started = time.time()
    procs = []
    try:
        for k in range(args.workers):
            shard = tasks[k::args.workers]
            if not shard:
                continue
            wdir = out / f"worker-{k}"
            wdir.mkdir()
            (wdir / "tasks.jsonl").write_text(
                "".join(json.dumps(t) + "\n" for t in shard), encoding="utf-8")
            cmd = [sys.executable, "-m", "shapeflow.campaign.brokered",
                   "--repo", str(REPO), "--model-dir", args.model_dir,
                   "--vllm-url", args.vllm_url, "--broker-url", broker_url,
                   "--bm25-url", args.bm25_url, "--tasks", str(wdir / "tasks.jsonl"),
                   "--out", str(wdir), "--worker", str(k), "--run-id", out.name,
                   "--page-variant", args.page_variant, "--close-variant", args.close_variant,
                   "--deadline-s", str(args.deadline_s), "--grace-s", str(args.grace_s),
                   "--seed", str(args.seed)]
            log = (wdir / "stdout.log").open("w", encoding="utf-8")
            env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            procs.append((k, proc, log))
        codes = {}
        for k, proc, log in procs:
            codes[k] = proc.wait()
            log.close()
            print(f"worker {k} exited {codes[k]}", flush=True)
    finally:
        service.stop()
    finished = time.time()

    report = audit(out, broker_factory=policy, max_waiting=args.max_waiting,
                   max_snapshot_age_ns=int(args.max_snapshot_age_ms * 1e6))
    run = {
        "what": "Real ODR agents offering both forms at every handoff; a broker outside an "
                "unmodified vLLM chooses",
        "args": vars(args),
        "vllm_version": vllm_version,
        "installed_graph": {"status": graph.status, "detail": graph.detail},
        "bm25": bm25_health,
        "tokenizer_sha256": tokenizer.identity_sha256,
        "contracts_sha": contracts_digest(),
        "broker": {"type": "DeadlineBroker(EngineGatedBroker(GreedyJointBroker))",
                   "capacity_tokens": args.capacity_tokens, "max_waiting": args.max_waiting,
                   "max_snapshot_age_ns": int(args.max_snapshot_age_ms * 1e6),
                   "deadline_s": args.deadline_s, "grace_s": args.grace_s,
                   "tick_ms": args.tick_ms},
        "worker_exit_codes": codes,
        "started_unix": started, "finished_unix": finished,
        "audit": {k: v for k, v in report.items() if k != "items"},
    }
    (out / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True, default=str) + "\n",
                                  encoding="utf-8")
    (out / "items.json").write_text(json.dumps(report["items"], indent=1) + "\n",
                                    encoding="utf-8")
    print(json.dumps(run["audit"], indent=2, default=str), flush=True)
    return 0 if all(c == 0 for c in codes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
