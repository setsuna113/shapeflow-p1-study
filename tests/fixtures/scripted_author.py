"""A scripted DeepSeek author, shaped exactly like the provider transport.

Shared between the unit and integration suites so both exercise the same double. A double that
behaves differently in the test guarding a boundary from the one exercising it is worse than
none: the guard passes against behaviour the real path never sees.
"""

from __future__ import annotations

import json
from typing import Optional


class ScriptedAuthor:
    """Replays cluster and task payloads, recording every request it received."""

    def __init__(
        self,
        clusters: int,
        per_cluster: int,
        *,
        skip_profile: Optional[str] = None,
        duplicate_question: bool = False,
    ) -> None:
        self.clusters = clusters
        self.per_cluster = per_cluster
        self.skip_profile = skip_profile
        self.duplicate_question = duplicate_question
        self.requests: list[dict] = []
        self._cluster_index = 0

    async def __call__(self, body: dict) -> tuple[int, dict]:
        self.requests.append(body)
        prompt = body["messages"][-1]["content"]
        if "TOPIC CLUSTERS" in prompt:
            payload = {"clusters": [
                {"cluster_id": f"cluster-{i:02d}", "title": f"Cluster {i}",
                 "scope": f"Everything about subject area number {i}."}
                for i in range(self.clusters)
            ]}
        else:
            index = self._cluster_index
            self._cluster_index += 1
            profile_ids = [
                line.split(":")[0].strip("- ").strip()
                for line in prompt.splitlines() if line.startswith("- P")
            ]
            tasks = []
            for n, profile_id in enumerate(profile_ids):
                if profile_id == self.skip_profile:
                    continue
                tasks.append(self._task(profile_id, index, n))
            payload = {"tasks": tasks}
        return 200, {
            "id": f"req-{len(self.requests)}", "model": "deepseek-chat",
            "system_fingerprint": "fp_test",
            "choices": [{"message": {"content": json.dumps(payload)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

    def _task(self, profile_id: str, cluster_index: int, n: int) -> dict:
        """Build a task whose vocabulary is mostly its own.

        Real authored questions differ in substance; a double that emitted one template with a
        swapped noun would trip the near-duplicate audit for a reason the real corpus would not
        have, and the test would then be measuring the double rather than the audit.
        """
        if self.duplicate_question:
            terms = ["alpha"] * 6
        else:
            terms = [f"{_STEM[(cluster_index * 7 + n * 3 + k) % len(_STEM)]}"
                     f"{cluster_index}{n}{k}" for k in range(6)]
        a, b, c, d, e, f = terms
        return {
            "profile_id": profile_id,
            "question": (
                f"Which {a} bodies published {b} figures covering {c} between 2023 and 2025, "
                f"and where do the {d} totals they report disagree with {e} records held by "
                f"{f} authorities?"
            ),
            "required_facets": [f"{a} publications", f"{d} discrepancies"],
            "fixed_queries": [
                f"{a} {b} figures {c} 2025",
                f"{d} totals disagreement {e} records",
                f"{f} authorities {c} published data",
            ],
            "conflict_probe": f"{d} totals disputed {f}",
            "negative_probe": f"{e} records unpublished {a}",
        }


#: Distinct stems so two generated questions share only their frame, not their vocabulary.
_STEM = (
    "quarry", "harbour", "lantern", "meadow", "cinder", "trellis", "basalt", "furrow",
    "gantry", "kelpbed", "moraine", "pylon", "sextant", "tundra", "vellum", "windrow",
    "alcove", "bramble", "citadel", "dovecot", "estuary", "fjord", "gable", "hollow",
)
