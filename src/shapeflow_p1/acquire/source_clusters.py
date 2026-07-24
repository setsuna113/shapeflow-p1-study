"""Source clusters and strata, derived from the world that was actually acquired.

Two things were taken on trust and are measured here instead.

**Clusters.** The split unit is supposed to be "tasks that share sources", because two tasks
drawing on the same pages are one observation, not two -- and if they land in different
splits, the holdout is contaminated by the screen. What existed was the topic label the
authoring model invented, assigned to tasks by integer division before any page had been
fetched, and an audit that checked no *label* spanned two splits. That check was
structurally satisfied by the splitter that assigned per label; it could not fail. Real
overlap is computed here from the URLs and content hashes the acquisition actually returned.

**Strata.** ``evidence_volume``, ``citation_dense``, ``raw_content_missing`` and the rest
were stamped onto the spec from the plan the model was asked to satisfy, regardless of what
it returned, and then verified by counting those same stamps. They are recomputed from the
frozen pool, and a declared label the world does not support is a finding rather than a row
in a coverage table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence
from urllib.parse import urlsplit

__all__ = [
    "TaskWorld",
    "SourceCluster",
    "build_source_clusters",
    "cross_split_overlap",
    "realized_strata",
    "strata_disagreements",
]


@dataclass(frozen=True)
class TaskWorld:
    """One task's acquired world, reduced to what clustering and strata need."""

    task_id: str
    split: str
    urls: frozenset[str]
    content_hashes: frozenset[str]
    occurrence_count: int
    missing_content: int
    declared_strata: tuple[str, ...] = ()

    @property
    def registrable_domains(self) -> frozenset[str]:
        return frozenset(_registrable_domain(u) for u in self.urls if u)


@dataclass
class SourceCluster:
    """A set of tasks connected by shared sources."""

    cluster_id: str
    task_ids: list[str] = field(default_factory=list)
    splits: set[str] = field(default_factory=set)
    shared_urls: set[str] = field(default_factory=set)
    shared_content: set[str] = field(default_factory=set)


def _registrable_domain(url: str) -> str:
    """Host without a leading ``www``. Deliberately not a public-suffix lookup.

    A shared host is the signal that matters -- two tasks both citing the same site are not
    independent -- and pulling in a suffix list would make the split unit depend on a
    dataset that is not pinned by the protocol.
    """
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def build_source_clusters(
    worlds: Sequence[TaskWorld], *, min_shared_sources: int = 1
) -> list[SourceCluster]:
    """Union-find over shared sources: two tasks are linked when they share content.

    Content hash first, URL second. The same page reachable at two URLs is one source, and
    two tasks that both depend on it are one observation however differently they cite it.
    """
    parent: dict[str, str] = {w.task_id: w.task_id for w in worlds}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    links: dict[tuple[str, str], tuple[set[str], set[str]]] = {}
    for i, left in enumerate(worlds):
        for right in worlds[i + 1:]:
            shared_content = left.content_hashes & right.content_hashes
            shared_urls = left.urls & right.urls
            if len(shared_content) + len(shared_urls) >= min_shared_sources:
                union(left.task_id, right.task_id)
                links[(left.task_id, right.task_id)] = (shared_urls, shared_content)

    grouped: dict[str, SourceCluster] = {}
    by_id = {w.task_id: w for w in worlds}
    for world in worlds:
        root = find(world.task_id)
        cluster = grouped.setdefault(root, SourceCluster(cluster_id=f"SC-{root}"))
        cluster.task_ids.append(world.task_id)
        cluster.splits.add(world.split)
    for (a, b), (urls, content) in links.items():
        cluster = grouped[find(a)]
        cluster.shared_urls |= urls
        cluster.shared_content |= content
        assert by_id[b].task_id in cluster.task_ids or find(b) == find(a)
    for cluster in grouped.values():
        cluster.task_ids.sort()
    return [grouped[key] for key in sorted(grouped)]


def cross_split_overlap(clusters: Iterable[SourceCluster]) -> list[SourceCluster]:
    """Clusters whose tasks are spread across more than one split.

    Each one is a leak: the holdout would be scored on sources the screen already saw.
    """
    return [c for c in clusters if len(c.splits) > 1]


#: How a realized world maps onto the declared stratum vocabulary. The thresholds are the
#: measurement; they belong in one place rather than in whichever module last needed them.
EVIDENCE_VOLUME_BANDS = ((0, 6, "evidence_volume_low"),
                         (6, 15, "evidence_volume_medium"),
                         (15, 10 ** 9, "evidence_volume_high"))


def realized_strata(
    world: TaskWorld, *, redundancy_ratio: float = 0.3, dense_sources: int = 8
) -> set[str]:
    """The strata this task's world actually exhibits."""
    found: set[str] = set()
    for low, high, label in EVIDENCE_VOLUME_BANDS:
        if low <= world.occurrence_count < high:
            found.add(label)
            break
    if world.missing_content:
        found.add("raw_content_missing")
    if world.occurrence_count and len(world.content_hashes) < world.occurrence_count:
        duplicated = 1.0 - (len(world.content_hashes) / world.occurrence_count)
        if duplicated >= redundancy_ratio:
            found.add("high_redundancy")
    if len(world.urls) >= dense_sources:
        found.add("citation_dense")
    if len(world.registrable_domains) > 1:
        found.add("multi_source_synthesis")
    elif world.urls:
        found.add("single_source_fact")
    return found


def strata_disagreements(
    worlds: Sequence[TaskWorld], *, checked: Optional[Iterable[str]] = None
) -> dict[str, list[str]]:
    """Declared strata the acquired world does not support, per task.

    Only the measurable ones are checked: ``source_conflict`` and the close-reason strata are
    properties of the content or of a run, not of a pool, and claiming to have verified them
    from a URL count would be the same substitution this function exists to catch.
    """
    measurable = set(checked) if checked is not None else {
        label for _, _, label in EVIDENCE_VOLUME_BANDS
    } | {"raw_content_missing", "high_redundancy", "citation_dense",
         "multi_source_synthesis", "single_source_fact"}
    out: dict[str, list[str]] = {}
    for world in worlds:
        realized = realized_strata(world)
        unsupported = sorted(
            label for label in world.declared_strata
            if label in measurable and label not in realized
        )
        if unsupported:
            out[world.task_id] = unsupported
    return out


def worlds_from_pools(
    pools: Mapping[str, dict], splits: Mapping[str, str],
    declared: Optional[Mapping[str, Sequence[str]]] = None,
) -> list[TaskWorld]:
    """Build the reduced worlds from published runner pools."""
    declared = declared or {}
    worlds: list[TaskWorld] = []
    for task_id, body in sorted(pools.items()):
        occurrences = body.get("occurrences") or []
        urls = {str(o["url"]) for o in occurrences if o.get("url")}
        hashes = {str(o["content_hash"]) for o in occurrences if o.get("content_hash")}
        worlds.append(TaskWorld(
            task_id=task_id,
            split=str(splits.get(task_id, "")),
            urls=frozenset(urls),
            content_hashes=frozenset(hashes),
            occurrence_count=len(occurrences),
            missing_content=sum(1 for o in occurrences if not o.get("content_hash")),
            declared_strata=tuple(declared.get(task_id) or ()),
        ))
    return worlds
