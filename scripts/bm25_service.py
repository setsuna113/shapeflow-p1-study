#!/usr/bin/env python3
"""Serve BM25 over the BrowseComp-Plus corpus on loopback, for ``shapeflow.retrieval.bm25``.

Runs in its own environment (Java 21 JDK, pyjnius, pyarrow, and the Anserini fat jar that ships
inside pyserini), not the project's frozen one. Anserini's ``SimpleSearcher`` is called directly
through pyjnius -- the same class pyserini's ``LuceneSearcher`` wraps, with the same BM25
defaults -- because importing pyserini's Python package also imports its dense-retrieval stack,
which needs a torch/transformers pairing this host need not have. The index is the benchmark's
prebuilt Lucene BM25 index; documents come from the corpus parquet
shards through the project's own ``load_corpus``, and titles and snippets from the dense
backend's helpers, so a result looks exactly like a dense-world result apart from its ranking.

    PYTHONPATH=src python scripts/bm25_service.py --jar anserini-fatjar.jar \
        --index DIR --corpus DIR --port 8720

One searcher, one lock: Anserini's batch search aborts on this index (see
benchdata/prep-scripts/bm25_full_run.py), and sequential search is fast enough.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jar", required=True, help="Anserini fat jar (pyserini/resources/jars)")
    ap.add_argument("--index", required=True)
    ap.add_argument("--corpus", required=True, help="directory of corpus parquet shards")
    ap.add_argument("--port", type=int, default=8720)
    args = ap.parse_args()

    import jnius_config

    jnius_config.set_classpath(args.jar)
    from jnius import autoclass

    from shapeflow.retrieval.backend import _snippet_of, _title_of
    from shapeflow.retrieval.corpus import load_corpus

    corpus = load_corpus(Path(args.corpus))
    searcher = autoclass("io.anserini.search.SimpleSearcher")(args.index)
    lock = threading.Lock()
    health = {"ok": True, "jar": Path(args.jar).name, "corpus_docs": len(corpus),
              "corpus_shard_sha256": list(corpus.shard_sha256), "index": str(args.index)}
    print(json.dumps(health), flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002
            return

        def _reply(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path == "/healthz":
                return self._reply(200, health)
            return self._reply(404, {"error": "no route"})

        def do_POST(self):  # noqa: N802
            if self.path != "/v1/search":
                return self._reply(404, {"error": "no route"})
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                query, top_k = str(body["query"]), int(body["top_k"])
                with lock:
                    hits = [(h.docid, float(h.score)) for h in searcher.search(query, top_k)]
                results = []
                for rank, (docid, score) in enumerate(hits, start=1):
                    doc = corpus.get(docid)
                    results.append({
                        "docid": docid, "rank": rank, "score": score,
                        "url": doc.url, "title": _title_of(doc.text, doc.url),
                        "snippet": _snippet_of(doc.text), "text": doc.text,
                    })
                return self._reply(200, {"query": query, "top_k": top_k, "results": results})
            except Exception as exc:  # noqa: BLE001
                return self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
