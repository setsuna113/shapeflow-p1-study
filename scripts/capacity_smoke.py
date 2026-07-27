#!/usr/bin/env python3
"""Does the engine actually serve the workload the graph will send it?

Three questions, none of which an idle health check can answer:

**Does anything get refused?** vLLM rejects when ``prompt + max_tokens`` exceeds the window, and
vendor turns that rejection into "return the raw page" -- which only P0 can suffer, on the
largest pages, the stratum P1 is meant to win. So the acceptance criterion is not "mostly fine":
it is **zero** refusals across the real length distribution.

**Does the KV pool hold?** The pool here is ~63k tokens. At a 32,768-token window that is under
two full-length sequences, so eight concurrent page summaries cannot all be resident at their
maximum -- they will queue and may be preempted. That is fine and expected; what is not fine is
discovering it as a timeout in the middle of a campaign.

**Is it the real distribution?** The prompts are drawn from the frozen corpus itself, bounded by
the same shared budget the run will apply, in the same concurrency the graph uses (one tool call
returns ``num_results_per_query`` pages, summarised together by ``asyncio.gather``). A smoke
built from synthetic short prompts would pass and prove nothing about the pages that break.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import random
import sys
import time
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


def _load_pages(corpus: pathlib.Path, budget, tokenizer, limit: int) -> list[str]:
    """The largest real pages, bounded exactly as the run will bound them.

    Largest rather than random: the failure being tested is a length failure, and a random draw
    from a corpus whose median page is 3,840 tokens would almost never sample one.
    """
    import zstandard as zstd

    from shapeflow_p1.evidence.shared_view import apply_shared_budget

    decompressor = zstd.ZstdDecompressor()
    sized: list[tuple[int, str]] = []
    for path in corpus.rglob("*"):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        if path.suffix == ".zst":
            try:
                raw = decompressor.decompress(raw)
            except Exception:  # noqa: BLE001 - a non-zstd blob is simply not a page
                continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if text.lstrip()[:1] in "{[":
            continue
        bounded = apply_shared_budget(text, budget, tokenizer)
        sized.append((len(tokenizer.encode_offsets(bounded.text)), bounded.text))
    sized.sort(key=lambda item: -item[0])
    return [text for _tokens, text in sized[:limit]]


async def _one(session, url: str, model: str, content: str, cap: int) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": cap,
        "temperature": 0.0,
    }
    started = time.monotonic()
    try:
        async with session.post(url, json=body) as response:
            payload = await response.text()
            return {
                "status": response.status,
                "seconds": time.monotonic() - started,
                "error": "" if response.status == 200 else payload[:300],
            }
    except Exception as exc:  # noqa: BLE001 - a transport failure is a capacity result
        return {
            "status": 0,
            "seconds": time.monotonic() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def _run(pages: list[str], *, url: str, model: str, cap: int, waves: int,
               concurrency: int) -> list[dict[str, Any]]:
    import aiohttp

    results: list[dict[str, Any]] = []
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for wave in range(waves):
            batch = [pages[(wave * concurrency + i) % len(pages)] for i in range(concurrency)]
            wave_results = await asyncio.gather(
                *(_one(session, url, model, content, cap) for content in batch)
            )
            for result in wave_results:
                result["wave"] = wave
            results.extend(wave_results)
            print(
                f"  wave {wave}: "
                + ", ".join(f"{r['status']}/{r['seconds']:.0f}s" for r in wave_results),
                flush=True,
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/storage/nvme/shapeflow-p1-study")
    parser.add_argument("--data-root", default="/storage/nvme/shapeflow-data")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--waves", type=int, default=3)
    args = parser.parse_args()

    import yaml

    from shapeflow_p1.evidence.model_tokenizer import FrozenModelTokenizer
    from shapeflow_p1.evidence.shared_view import SharedContentBudget

    repo = pathlib.Path(args.repo)
    stack = yaml.safe_load((repo / "configs" / "stack.yaml").read_text(encoding="utf-8"))
    week1 = yaml.safe_load((repo / "configs" / "week1.yaml").read_text(encoding="utf-8"))
    retrieval = yaml.safe_load(
        (repo / "configs" / "retrieval.yaml").read_text(encoding="utf-8"))

    cap = int(week1["odr"]["summarization_model_max_tokens"])
    budget = SharedContentBudget.derive(
        max_chars=int(week1["odr"]["max_content_length"]),
        max_model_len=int(stack["engine"]["max_model_len"]),
        completion_cap=cap,
        prompt_overhead_tokens=int(week1["odr"]["summarization_prompt_overhead_tokens"]),
    )
    # The graph's own concurrency: one tool call returns this many pages and summarises them
    # together. Testing more would measure a system nobody runs; testing fewer would pass.
    concurrency = int(retrieval["frozen_corpus"]["top_k"])
    tokenizer = FrozenModelTokenizer(
        pathlib.Path(str(stack["model"]["path"])) / str(stack["model"]["tokenizer_file"]))

    corpus = pathlib.Path(args.data_root) / "runner" / "frozen_corpus" / "objects"
    pages = _load_pages(corpus, budget, tokenizer, concurrency * args.waves)
    if not pages:
        print("no frozen pages found; cannot smoke the real distribution", file=sys.stderr)
        return 1

    lengths = [len(tokenizer.encode_offsets(p)) for p in pages]
    print(json.dumps({
        "max_model_len": int(stack["engine"]["max_model_len"]),
        "completion_cap": cap,
        "content_token_budget": budget.max_tokens,
        "concurrency": concurrency,
        "waves": args.waves,
        "prompt_tokens": {"max": max(lengths), "min": min(lengths)},
    }, indent=2, sort_keys=True), flush=True)
    random.seed(0)

    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    results = asyncio.run(_run(
        pages, url=url, model=str(stack["model"]["repo"]).split("/")[-1],
        cap=cap, waves=args.waves, concurrency=concurrency,
    ))

    rejected = [r for r in results if r["status"] == 400]
    failed = [r for r in results if r["status"] not in (200, 400)]
    latencies = sorted(r["seconds"] for r in results if r["status"] == 200)
    summary = {
        "requests": len(results),
        "ok": len(latencies),
        "context_rejections_400": len(rejected),
        "other_failures": len(failed),
        "latency_seconds": {
            "max": round(latencies[-1], 1) if latencies else None,
            "median": round(latencies[len(latencies) // 2], 1) if latencies else None,
        },
        "first_rejection": rejected[0]["error"] if rejected else "",
        "first_other_failure": failed[0]["error"] if failed else "",
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    # Zero, not "few". A single refusal is a page where P0 silently publishes the raw text and
    # P1 does not, and that asymmetry lands on the largest pages by construction.
    if rejected or failed:
        print(
            "CAPACITY SMOKE FAILED: the engine refused or dropped requests the graph will send",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
