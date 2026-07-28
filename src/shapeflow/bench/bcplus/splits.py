"""Carving the frozen dev/test splits into the layers the plan spends them on.

Freeze-1 §3.1 fixes the sizes and, more importantly, fixes that the layers are **pairwise
task-disjoint**. That is the property doing the work: B1 selects a champion form, B2 measures a
dose-response, B3 evaluates a gated policy, and if any task appeared in two of those the later
result would be partly a memory of the earlier one.

Two layers are load-bearing beyond disjointness:

- **FV-B** is the only post-iteration retest allowance in the whole program. It is carved here
  and then not touched, so that when a gate fails and the single repair iteration is spent, an
  untouched split still exists to re-measure on.
- **The sealed confirmatory 300** is the test split, opened once by S1. Nothing in the science
  track may read it, which is why it is carved by a different function that records that it was
  called.

The carve is deterministic from the frozen seed and the split file contents, and the manifest is
write-once: two runs either produce the same assignment or the second refuses. An assignment that
could be silently recomputed is not a pre-registration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ...canonical import canonical_json
from ...hashing import sha256_hex

__all__ = ["SplitPlan", "LAYER_SIZES", "carve", "write_manifest", "load_manifest", "SplitError"]

#: Freeze-1 §3.1. ITD 380 + FV 150 = the 530 dev queries; the 300 test queries are sealed.
LAYER_SIZES: Mapping[str, int] = {
    "b1_select": 80,
    "b1_confirm": 170,
    "b2": 110,
    "fv_a": 75,
    "fv_b": 75,
}
SEALED_LAYER = "sealed_confirmatory"


class SplitError(RuntimeError):
    """The split cannot be carved as the plan specifies."""


@dataclass(frozen=True)
class SplitPlan:
    seed: int
    layers: Mapping[str, tuple[str, ...]]
    dev_sha256: str
    test_sha256: str

    def layer_of(self, query_id: str) -> str:
        for name, ids in self.layers.items():
            if query_id in ids:
                return name
        raise KeyError(f"{query_id!r} is in no layer")

    def content(self) -> dict:
        return {
            "seed": self.seed,
            "dev_sha256": self.dev_sha256,
            "test_sha256": self.test_sha256,
            "layers": {name: list(ids) for name, ids in sorted(self.layers.items())},
            "sizes": {name: len(ids) for name, ids in sorted(self.layers.items())},
        }

    @property
    def digest(self) -> str:
        return sha256_hex(canonical_json(self.content()))


def _ranked(query_ids: Sequence[str], seed: int, salt: str) -> list[str]:
    """Deterministic shuffle by digest.

    Not ``random.shuffle`` with a seeded generator: that depends on the interpreter's PRNG
    implementation, so the same seed could carve differently under a different Python. A digest
    of (seed, salt, id) depends on nothing but its inputs.
    """
    return sorted(query_ids, key=lambda q: sha256_hex(f"{seed}:{salt}:{q}".encode()))


def carve(dev_ids: Sequence[str], test_ids: Sequence[str], *, seed: int) -> SplitPlan:
    """Assign every dev id to exactly one ITD/FV layer, and every test id to the sealed layer."""
    dev_ids, test_ids = list(dev_ids), list(test_ids)
    if len(set(dev_ids)) != len(dev_ids) or len(set(test_ids)) != len(test_ids):
        raise SplitError("duplicate query ids in a split file")
    overlap = set(dev_ids) & set(test_ids)
    if overlap:
        raise SplitError(
            f"{len(overlap)} query ids are in both dev and test, e.g. {sorted(overlap)[:5]}; "
            "the confirmatory split would then contain tasks the design was tuned on")

    needed = sum(LAYER_SIZES.values())
    if len(dev_ids) < needed:
        raise SplitError(
            f"dev has {len(dev_ids)} queries, the layers need {needed}. Shrinking a layer to fit "
            "is a design change and must be an amendment, not a runtime accommodation.")

    ordered = _ranked(dev_ids, seed, "itd-fv")
    layers: dict[str, tuple[str, ...]] = {}
    cursor = 0
    for name, size in LAYER_SIZES.items():
        layers[name] = tuple(sorted(ordered[cursor:cursor + size]))
        cursor += size
    layers[SEALED_LAYER] = tuple(sorted(test_ids))

    assigned = [q for ids in layers.values() for q in ids]
    if len(assigned) != len(set(assigned)):
        raise SplitError("a query id landed in two layers")
    return SplitPlan(
        seed=seed,
        layers=layers,
        dev_sha256=sha256_hex("\n".join(dev_ids).encode()),
        test_sha256=sha256_hex("\n".join(test_ids).encode()),
    )


def write_manifest(plan: SplitPlan, path: Path) -> dict:
    """Write once. An existing identical manifest is fine; a different one is an error.

    Exclusive creation rather than exists-then-write: two resumptions carving at once is a
    truncation race, and a split assignment that can be silently rewritten is not frozen.
    """
    body = plan.content()
    body["digest"] = plan.digest
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
        return body
    except FileExistsError:
        pass

    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing.get("digest") != body["digest"]:
        raise SplitError(
            f"{path} already holds a different split assignment "
            f"({existing.get('digest', '')[:12]} vs {body['digest'][:12]}). Two assignments under "
            "one manifest cannot both be the frozen one; results computed under the old one would "
            "silently mean something else.")
    return existing


def load_manifest(path: Path) -> SplitPlan:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    plan = SplitPlan(
        seed=int(body["seed"]),
        layers={name: tuple(ids) for name, ids in body["layers"].items()},
        dev_sha256=body["dev_sha256"],
        test_sha256=body["test_sha256"],
    )
    if plan.digest != body.get("digest"):
        raise SplitError(f"{path} has been edited: it records {body.get('digest', '')[:12]} "
                         f"but hashes to {plan.digest[:12]}")
    return plan
