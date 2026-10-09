"""Engine state for a host outside the engine: versioned snapshots and the commit-time gate.

Shared by every host that reads vLLM from the outside -- ``ProxyHost``, which dispatches the
chosen plan itself, and ``AgentHost``, which releases an agent to dispatch it. Both must judge
staleness identically, or the two would be two different contracts that happen to share a name.

**Snapshot.** A scrape of the engine's Prometheus ``/metrics``. The scrape has a measurable age,
so ``taken_at_ns`` is the wall-clock time the scrape was *sent* -- the oldest the reported state
can be.

**Epoch.** The API server's ``process_start_time_seconds``. A restarted engine reports a new
one, so a snapshot taken before a restart cannot alias a tick taken after it.

**Gate.** A commit is refused when its snapshot has been superseded by a newer one or is older
than ``max_snapshot_age_ns``. No ceiling is pre-registered in Freeze-1, so the default here is a
placeholder in the same sense as the reference broker's ``capacity_tokens``.

Not locked: a host calls :meth:`EngineSnapshotGate.stamp` and :meth:`EngineSnapshotGate.check`
under its own lock, so the version it hands out and the pending set it decides over move
together. :meth:`EngineSnapshotGate.observe` touches no shared state and runs outside it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ...contracts.observability import ClosedFeatureMap
from ...contracts.offer import SchedulerSnapshot, SnapshotVersion, StaleSnapshot
from ...contracts.versions import assert_contract

CONTRACT = "OFFER_INTERFACE_v1.md"
CONTRACT_SHA256 = "f9a400457f00ed7cf918fd202ac0307a9cf9c196c4ce7f22a94f3ce575c93084"

__all__ = [
    "EngineSnapshotGate", "EngineObservation", "SnapshotRecord", "EngineUnobservable",
    "parse_prometheus", "ENGINE_GAUGES", "EPOCH_METRIC", "DEFAULT_MAX_SNAPSHOT_AGE_NS",
]

#: vLLM gauge -> registered observable feature. Names read off vLLM 0.24.0's ``/metrics``.
ENGINE_GAUGES: Mapping[str, str] = {
    "vllm:num_requests_running": "engine.num_requests_running",
    "vllm:num_requests_waiting": "engine.num_requests_waiting",
    "vllm:kv_cache_usage_perc": "engine.kv_cache_usage_perc",
}
EPOCH_METRIC = "process_start_time_seconds"

#: Placeholder ceiling on snapshot age at commit. Not pre-registered; recorded with every run.
DEFAULT_MAX_SNAPSHOT_AGE_NS = 100_000_000


class EngineUnobservable(RuntimeError):
    """The engine's metrics did not carry what a snapshot needs; no snapshot was produced."""


def parse_prometheus(text: str, names) -> dict[str, float]:
    """Sum each wanted sample name over its label sets.

    Exact names only: ``vllm:prefix_cache_hits_total`` and ``vllm:prefix_cache_hits`` are
    different series, and a prefix match would silently fold a counter into a gauge.
    """
    wanted = set(names)
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        brace = line.find("{")
        space = line.find(" ")
        if brace != -1 and (space == -1 or brace < space):
            name, rest = line[:brace], line[line.rfind("}") + 1:]
        else:
            name, rest = line[:space], line[space:]
        if name not in wanted:
            continue
        fields = rest.split()
        if not fields:
            continue
        out[name] = out.get(name, 0.0) + float(fields[0])
    return out


@dataclass(frozen=True)
class SnapshotRecord:
    epoch: str
    tick: int
    taken_at_ns: int
    scrape_ns: int
    features: dict


@dataclass(frozen=True)
class EngineObservation:
    """One scrape, parsed, before it has been given a place in the tick sequence."""

    epoch: str
    gauges: Mapping[str, float]
    sent_ns: int
    scrape_ns: int


class EngineSnapshotGate:
    def __init__(
        self,
        *,
        clock_ns: Callable[[], int] = time.time_ns,
        max_snapshot_age_ns: int = DEFAULT_MAX_SNAPSHOT_AGE_NS,
    ) -> None:
        self.clock_ns = clock_ns
        self.max_snapshot_age_ns = max_snapshot_age_ns
        self._tick = 0
        self._latest: SnapshotVersion | None = None
        self.snapshots: list[SnapshotRecord] = []
        #: One entry per materialize attempt: the snapshot's age at the gate and the verdict.
        self.gate_log: list[dict] = []

    def observe(self, scrape: Callable[[], str]) -> EngineObservation:
        sent_ns = self.clock_ns()
        text = scrape()
        received_ns = self.clock_ns()
        values = parse_prometheus(text, set(ENGINE_GAUGES) | {EPOCH_METRIC})
        missing = sorted((set(ENGINE_GAUGES) | {EPOCH_METRIC}) - set(values))
        if missing:
            raise EngineUnobservable(f"/metrics carried none of {missing}")
        return EngineObservation(
            epoch=f"vllm-start-{values[EPOCH_METRIC]!r}",
            gauges={feature: values[metric] for metric, feature in ENGINE_GAUGES.items()},
            sent_ns=sent_ns,
            scrape_ns=received_ns - sent_ns,
        )

    def stamp(self, observation: EngineObservation,
              local: Mapping[str, Any] | None = None) -> SchedulerSnapshot:
        """Give an observation the next tick. The caller holds its host lock."""
        self._tick += 1
        version = SnapshotVersion(epoch=observation.epoch, tick=self._tick)
        features = {**observation.gauges, **dict(local or {})}
        self._latest = version
        self.snapshots.append(SnapshotRecord(
            epoch=observation.epoch, tick=self._tick, taken_at_ns=observation.sent_ns,
            scrape_ns=observation.scrape_ns, features=dict(features)))
        return SchedulerSnapshot(version=version, taken_at_ns=observation.sent_ns,
                                 features=ClosedFeatureMap(features))

    def check(self, snapshot: SchedulerSnapshot, admitted: Sequence[str]) -> dict:
        """Raise :class:`StaleSnapshot` unless a commit may proceed on ``snapshot``.

        Logs a refusal here. A pass is logged by :meth:`passed` once the host has actually
        committed, so a gate entry marked passed always names a commit that happened.
        """
        entry = {"epoch": snapshot.version.epoch, "tick": snapshot.version.tick,
                 "age_ns": snapshot.age_ns(self.clock_ns()), "admitted": list(admitted)}
        if snapshot.version != self._latest:
            self.gate_log.append({**entry, "passed": False, "why": "superseded"})
            raise StaleSnapshot(snapshot.version)
        if entry["age_ns"] > self.max_snapshot_age_ns:
            self.gate_log.append({**entry, "passed": False, "why": "too old"})
            raise StaleSnapshot(snapshot.version)
        return entry

    def passed(self, entry: dict) -> None:
        self.gate_log.append({**entry, "passed": True, "why": ""})


assert_contract(CONTRACT, CONTRACT_SHA256)
