"""Brokered cells: the real agent offers both forms at every handoff and a broker chooses.

Two pieces:

- :func:`build_brokered_bundle` turns a page arm and a close arm into the strategies that price,
  offer, wait and execute (:mod:`shapeflow.strategies.brokered`). Only arms whose P1 is one
  selector request can be offered, because an offer's P1 plan *is* one selector request: at H a
  ``whole_batch`` LLM arm, at C a ``C_VISIBLE`` LLM arm.
- :func:`main` is one agent worker. It runs its share of tasks one cell at a time -- the graph
  driver refuses two in one process -- against a live vLLM through an in-process provider that
  does not serialize upstream calls, so P0's sibling summaries reach the engine independently.
  Several workers against one broker service give the broker several offers to decide jointly.

A brokered cell is never a registered arm: its variant id is ``BROKERED[<page>+<close>]`` and
it is run outside the campaign ledger, so nothing here changes the frozen design.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import secrets
import sys
import time
from collections.abc import Callable
from pathlib import Path

from ..hashing import sha256_hex
from ..odr.hooks import StrategyBundle
from ..p1.prompt_pack import PromptBudget
from ..strategies.brokered import (
    BrokeredCloseStrategy,
    BrokeredPageStrategy,
    BrokerSettings,
)
from ..strategies.offer_pricing import VendorPrompts
from .runner import strategy_factory_for

__all__ = ["build_brokered_bundle", "BrokeredArmError", "main"]


class BrokeredArmError(ValueError):
    """An arm whose P1 cannot be offered as a single selector request."""


def _check_eligible(registry, page_variant: str, close_variant: str) -> None:
    page = registry.get(page_variant)
    close = registry.get(close_variant)
    if page is None or close is None:
        raise BrokeredArmError(f"unknown variant(s): {page_variant!r}, {close_variant!r}")
    if not (page.node == "WEBPAGE_P1" and page.scope == "whole_batch"
            and page.selector_backend == "LLM" and page.contract.startswith("P1_")):
        raise BrokeredArmError(
            f"{page_variant}: an H offer's P1 plan is one selector request, so the page arm must "
            "be a whole_batch LLM selector")
    if not (close.node == "C_VISIBLE" and close.selector_backend == "LLM"
            and close.contract.startswith("P1_")):
        raise BrokeredArmError(
            f"{close_variant}: a C offer's P1 plan is one selector request, so the close arm "
            "must be a C_VISIBLE LLM selector")


def build_brokered_bundle(settings, registry, *, page_variant: str, close_variant: str,
                          model_call, tokenizer, pages, client, cell_token: str,
                          deadline_s: float, prompts: VendorPrompts | None = None,
                          log: Callable[[dict], None] | None = None) -> StrategyBundle:
    _check_eligible(registry, page_variant, close_variant)
    max_model_len = int(settings.get("stack", "engine", "max_model_len"))
    budget = PromptBudget(
        max_model_len=max_model_len,
        completion_cap=int(settings.get("week1", "measurement",
                                        "selector_max_completion_tokens")),
        overhead_tokens=1024,
    )
    factory = strategy_factory_for(settings, registry, model_call=model_call,
                                   tokenizer=tokenizer, pages=pages, prompt_budget=budget)
    shared = BrokerSettings(
        client=client, cell_token=cell_token, tokenizer=tokenizer,
        prompts=prompts or VendorPrompts.load(),
        p0_page_max_tokens=int(settings.get("week1", "odr", "summarization_model_max_tokens")),
        p0_close_max_tokens=int(settings.get("week1", "odr", "compression_model_max_tokens")),
        max_model_len=max_model_len, deadline_ns=int(deadline_s * 1e9), log=log,
    )
    return StrategyBundle(
        variant_id=f"BROKERED[{page_variant}+{close_variant}]",
        page=BrokeredPageStrategy(
            settings=shared, executor=factory.build(page_variant).page,
            probe_strategy_for=lambda probe: factory.rebind(model_call=probe)
            .build(page_variant).page,
            body_for=model_call.request_body),
        close=BrokeredCloseStrategy(
            settings=shared, executor=factory.build(close_variant).close,
            probe_strategy_for=lambda probe: factory.rebind(model_call=probe)
            .build(close_variant).close,
            body_for=model_call.request_body),
    )


# --- worker ---------------------------------------------------------------------------------


class _Jsonl:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("a", encoding="utf-8", buffering=1)

    def __call__(self, row: dict) -> None:
        self._f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")

    def close(self) -> None:
        self._f.close()


async def _run_tasks(args) -> int:
    from ..broker.client import BrokerClient
    from ..evidence.model_tokenizer import FrozenModelTokenizer
    from ..evidence.shared_view import budget_from_settings
    from ..experiment.budget import Budget
    from ..experiment.ledger import Ledger
    from ..object_store import ObjectStore
    from ..providers.provider_client import ProviderClient
    from ..retrieval.bm25 import Bm25Client
    from ..runtime.provider_main import httpx_upstream
    from ..runtime.provider_server import ProviderService, RoleTokens, serve_forever
    from ..secrets import SecretRedactor
    from ..strategies.factory import load_registry
    from .graph_driver import CellSpec, run_cell
    from .pages import PageRegistry
    from .selector_client import SelectorModelCall
    from .settings import Settings

    repo = Path(args.repo)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = Settings.load(repo, data_root=out / "data")
    tokenizer = FrozenModelTokenizer(Path(args.model_dir) / "tokenizer.json")
    registry = load_registry(repo / "configs")

    # In-process provider, wired from the campaign config, pointed at the live engine, and
    # never serializing upstream calls: a queue here would gang P0's sibling summaries.
    tokens = {role: f"{role}-{secrets.token_urlsafe(18)}"
              for role in ("runner", "steward", "evaluator")}
    config = dataclasses.replace(
        settings.provider_config(), vllm_base_url=args.vllm_url.rstrip("/") + "/v1",
        bind_port=0, unix_socket=None, max_upstream_inflight=0, serves_paid_upstreams=False)
    if config.max_upstream_inflight != 0:
        raise RuntimeError("a brokered worker's provider must not serialize upstream calls")
    ledger = Ledger(str(out / "provider.sqlite"))
    budget = Budget(ledger)
    for resource, cap in settings.budget_caps().items():
        budget.ensure_account(resource, cap)
    redactor = SecretRedactor()
    service = ProviderService(
        config, ledger=ledger, budget=budget, store=ObjectStore(out / "provider-objects"),
        redactor=redactor, tokens=RoleTokens(tokens), upstream=httpx_upstream(),
        tavily_key=None, deepseek_key=None)
    service.reconcile_on_start()
    tcp, _ = serve_forever(service, dataclasses.replace(config, bind_port=0), redactor)
    base = f"http://127.0.0.1:{tcp.server_address[1]}"
    provider = ProviderClient(base_url=base, token=tokens["runner"])

    world = Bm25Client(args.bm25_url)
    broker = BrokerClient(args.broker_url, grace_ns=int(args.grace_s * 1e9))
    offers_log = _Jsonl(out / "agent_offers.jsonl")
    requests_log = _Jsonl(out / "requests.jsonl")
    cells_log = _Jsonl(out / "cells.jsonl")
    prompts = VendorPrompts.load()
    tasks = [json.loads(line) for line in Path(args.tasks).read_text(encoding="utf-8")
             .splitlines() if line.strip()]
    failures = 0
    event_cursor = 0
    try:
        for task in tasks:
            task_id, question = str(task["task_id"]), str(task["question"])
            token = "cell-" + sha256_hex(f"{args.run_id}:{args.worker}:{task_id}".encode())[:24]
            cell = CellSpec(
                run_id=args.run_id, task_id=task_id, arm_id="BROKERED",
                page_variant=args.page_variant, close_variant=args.close_variant,
                replicate_id="0", seed=args.seed, work_key=f"WK-{token}", question=question,
                cell_token=token, execution_binding_sha256="e" * 64,
                protocol_document_sha256="d" * 64, form_policy="brokered")
            await provider.register_cell(
                cell_token=token, run_id=cell.run_id, task_id=task_id, arm_id=cell.arm_id,
                variant_id=cell.variant_id, replicate_id=cell.replicate_id,
                work_key=cell.work_key)
            model_call = SelectorModelCall(
                provider, cell_token=token, repo=repo,
                temperature=float(settings.get("stack", "sampling", "temperature")),
                top_p=float(settings.get("stack", "sampling", "top_p")),
                max_completion_tokens=int(settings.get(
                    "week1", "measurement", "selector_max_completion_tokens")),
                seed=args.seed,
                guided_decoding=bool(settings.get("week1", "measurement", "guided_decoding")))
            pages = PageRegistry()
            bundle = build_brokered_bundle(
                settings, registry, page_variant=args.page_variant,
                close_variant=args.close_variant, model_call=model_call, tokenizer=tokenizer,
                pages=pages, client=broker, cell_token=token, deadline_s=args.deadline_s,
                prompts=prompts, log=lambda row, t=task_id: offers_log({"task_id": t, **row}))
            trace_start = len(world.trace)
            started = time.time()
            result = await run_cell(
                settings, cell, backend=world, pages=pages, bundle=bundle,
                provider_base_url=base, runner_token=tokens["runner"],
                content_budget=budget_from_settings(settings), tokenizer=tokenizer)
            elapsed = time.time() - started
            events = list(service.events)
            for event in events[event_cursor:]:
                if event.get("kind", "").startswith("INFERENCE"):
                    requests_log(event)
            event_cursor = len(events)
            failures += 0 if result.error is None else 1
            cells_log({
                "task_id": task_id, "cell_token": token, "variant_id": cell.variant_id,
                "ok": result.error is None, "error": result.error, "wall_s": round(elapsed, 3),
                "final_report_sha256": sha256_hex(result.final_report.encode("utf-8")),
                "final_report_chars": len(result.final_report),
                "checkpoints": result.checkpoints, "events": result.events,
                "search_trace": world.trace[trace_start:],
            })
            print(f"[worker {args.worker}] {task_id}: ok={result.error is None} "
                  f"{elapsed:.0f}s", flush=True)
    finally:
        tcp.shutdown()
        ledger.close()
        for log in (offers_log, requests_log, cells_log):
            log.close()
    return 1 if failures else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="One brokered agent worker.")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    ap.add_argument("--broker-url", required=True)
    ap.add_argument("--bm25-url", required=True)
    ap.add_argument("--tasks", required=True, help="JSONL of {task_id, question}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default="BROKERED-LIVE")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--page-variant", default="LLM-PROMPTVIEW")
    ap.add_argument("--close-variant", default="C01")
    ap.add_argument("--deadline-s", type=float, default=60.0)
    ap.add_argument("--grace-s", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    return asyncio.run(_run_tasks(args))


if __name__ == "__main__":
    sys.exit(main())
