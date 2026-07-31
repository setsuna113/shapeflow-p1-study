"""The questions, and which layer of the design each one belongs to.

Treatment-visible, and only just: a BrowseComp-Plus query is the task, so the agent must read
it, but *nothing else* about the query may cross. The answer, the gold documents, the evidence
set and the hard negatives live in the evaluator tree behind the leakage firewall
(:mod:`shapeflow.bench.bcplus.qrels`), and this module reads a different file from a different
directory so that no import edge exists between the two.

``queries.tsv`` is the benchmark's own public topic file -- two columns, id and text, the same
file its BM25 and dense baselines are scored on. Reading the decrypted record set instead would
be the same questions and the entire answer key, which is exactly the mistake the split between
these two modules exists to make hard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ...hashing import sha256_hex

__all__ = ["load_questions", "read_split_ids", "TaskSourceError", "questions_digest"]


class TaskSourceError(RuntimeError):
    """The task source cannot be read as what it claims to be."""


def load_questions(path: Path) -> dict[str, str]:
    """query_id -> question text, from the benchmark's ``topics-qrels/queries.tsv``.

    Tab-separated with exactly two fields. A row that splits into three is not a wider schema to
    be tolerated -- it means the question itself contains a tab, and taking the second field
    would hand the agent a truncated question while everything downstream reported the whole one.
    """
    path = Path(path)
    if not path.exists():
        raise TaskSourceError(f"no query file at {path}")
    questions: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise TaskSourceError(
                f"{path}:{number} has {len(fields)} tab-separated fields, not 2; a question "
                "carrying a tab would be silently truncated by taking the second one")
        query_id, text = fields[0].strip(), fields[1].strip()
        if not query_id or not text:
            raise TaskSourceError(f"{path}:{number} has an empty id or an empty question")
        if query_id in questions:
            raise TaskSourceError(
                f"{path}:{number} repeats query id {query_id!r}; two questions under one id "
                "means every result recorded against it is ambiguous")
        questions[query_id] = text
    if not questions:
        raise TaskSourceError(f"{path} holds no questions")
    return questions


def read_split_ids(path: Path) -> list[str]:
    """The frozen dev or test id list, one per line, in file order."""
    path = Path(path)
    if not path.exists():
        raise TaskSourceError(f"no split file at {path}")
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
           if line.strip()]
    if len(set(ids)) != len(ids):
        raise TaskSourceError(f"{path} repeats a query id; the split is not a partition")
    return ids


def questions_digest(questions: Mapping[str, str], task_ids: Sequence[str]) -> str:
    """Digest of exactly the questions a run was given, so the run record pins its own inputs."""
    return sha256_hex(
        "\n".join(f"{task_id}\t{questions[task_id]}" for task_id in sorted(task_ids))
        .encode("utf-8"))
