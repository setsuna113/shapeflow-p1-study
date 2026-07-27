"""One resolved view of the campaign's configuration, and where its data lives.

Two things this module keeps apart on purpose.

**What is hashed and what is not.** Every config file is loaded through
:func:`~shapeflow_p1.config.load_config`, so its bytes reach the approval binding. The *data
root* is not: it comes from ``SHAPEFLOW_DATA_ROOT`` and differs between the authoring machine
and the run host. Putting an absolute path in a hashed config would give the same protocol two
different SHAs depending on where it was read, and the approval would never verify anywhere.

**Who owns which directory.** The steward writes the corpus, the runner writes treatment
outputs, the evaluator holds the answer key. They are separate trees with separate owners so
"the treatment cannot read the truth" is a filesystem fact rather than a rule in a document.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..config import ConfigError, load_config

__all__ = ["Settings", "DATA_ROOT_ENV", "LANE_ENV"]


def _resolve_lane(configs: dict[str, dict]) -> Optional[int]:
    """Read this process's lane from the environment the unit sets.

    Never inferred from a visible device: a lane that guessed its own identity from whichever
    GPU it could see could write another lane's ledger, and the two would look equally valid
    until the merge.
    """
    raw = os.environ.get(LANE_ENV, "").strip()
    if not raw:
        return None
    if not raw.isdigit():
        raise ValueError(f"{LANE_ENV} must be a non-negative integer, got {raw!r}")
    lane = int(raw)
    count = int(configs["week1"]["measurement"]["shards"]["lane_count"])
    if lane >= count:
        raise ValueError(f"{LANE_ENV} {lane} is outside the frozen {count}-lane campaign")
    return lane


def _with_port(url: str, port: int) -> str:
    """Repoint a base URL at this lane's engine, preserving scheme, host and path."""
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    return urlunsplit(parts._replace(netloc=f"{parts.hostname or '127.0.0.1'}:{port}"))

DATA_ROOT_ENV = "SHAPEFLOW_DATA_ROOT"
LANE_ENV = "SHAPEFLOW_LANE"

_CONFIG_FILES = {
    "week1": "week1.yaml",
    "acquisition": "acquisition.yaml",
    "task_source": "task_source.yaml",
    "judge": "judge.yaml",
    "variants": "variants.yaml",
    "decision": "decision.yaml",
    "budget": "budget_v1.yaml",
    "stack": "stack.yaml",
}


@dataclass(frozen=True)
class Settings:
    repo: Path
    data_root: Path
    configs: dict[str, dict]
    shas: dict[str, str]
    #: Which execution lane this process serves, or None for an unsharded campaign. Resolved
    #: once at load rather than read from the environment on each access: a settings object
    #: whose paths change underneath it would let one lane write another lane's ledger and both
    #: would look correct until the merge.
    lane_id: Optional[int] = None

    # --- loading ---------------------------------------------------------------------

    @classmethod
    def load(cls, repo: Path | str, *, data_root: Optional[Path | str] = None) -> "Settings":
        repo = Path(repo)
        configs: dict[str, dict] = {}
        shas: dict[str, str] = {}
        for name, filename in _CONFIG_FILES.items():
            data, sha = load_config(repo / "configs" / filename)
            configs[name] = data
            shas[name] = sha
        root = data_root or os.environ.get(DATA_ROOT_ENV) or (repo / "data")
        return cls(
            repo=repo, data_root=Path(root), configs=configs, shas=shas,
            lane_id=_resolve_lane(configs),
        )

    # --- accessors -------------------------------------------------------------------

    def get(self, config: str, *keys: str) -> Any:
        """Fetch a nested key. A missing key is an error, never a silent default: a threshold
        that defaulted itself is exactly what the config hash exists to catch."""
        node: Any = self.configs[config]
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                raise ConfigError(f"missing required key {config}:{'.'.join(keys)}")
            node = node[key]
        return node

    #: Paths that belong to one execution lane rather than to the campaign. The runner ledger
    #: is single-writer by design, so four concurrent lanes need four of them; the corpus, the
    #: evaluator tree and the provider are shared and must NOT be duplicated.
    _LANE_SCOPED_PATHS = frozenset({
        "runner_root", "runs", "object_store", "checkpoints",
    })

    def path(self, name: str) -> Path:
        """An absolute path for one of the declared relative locations.

        Lane-scoped names are rehomed under this lane's runner root. Four runners sharing one
        ledger would be four writers on a store designed for one; four runners sharing one
        object store would make "which lane produced this artifact" unanswerable at merge time.
        """
        relative = str(self.get("week1", "paths", name))
        lane = self.lane_id
        if lane is not None and name in self._LANE_SCOPED_PATHS:
            template = str(self.get("week1", "measurement", "shards", "runner_root_template"))
            base = str(self.get("week1", "paths", "runner_root"))
            relative = template.format(lane=lane) + relative[len(base):]
        return self.data_root / relative

    def ensure_paths(self, *names: str) -> None:
        for name in names:
            self.path(name).mkdir(parents=True, exist_ok=True)

    # --- convenience blocks ------------------------------------------------------------

    @property
    def claim_scope(self) -> str:
        return str(self.get("week1", "campaign", "claim_scope"))

    @property
    def corpus_tier(self) -> str:
        return str(self.get("week1", "campaign", "corpus_tier"))

    @property
    def campaign_id(self) -> str:
        return str(self.get("week1", "campaign", "id"))

    @property
    def measurement_layer(self) -> str:
        return str(self.get("week1", "measurement", "layer"))

    @property
    def layer_is_serialized(self) -> bool:
        """Does the active layer admit one upstream request at a time?

        Read from the layer's own declaration rather than from its *name*. Several checks used
        to ask ``layer == "causal"`` and act on the answer as though it meant "serialized", but
        those are different properties: what makes a layer causal is prefix caching off, so one
        arm's prefill cannot subsidise another's. Serialization was only ever a precondition of
        the summed-service metric, and conflating the two is how a metric's requirement came to
        reshape the system under test.
        """
        return int(
            self.get("stack", "isolation", self.measurement_layer,
                     "gateway_max_upstream_inflight")
        ) == 1

    @property
    def layer_is_causal(self) -> bool:
        """Is the active layer free of cross-arm prefix-cache carry-over?"""
        return self.get(
            "stack", "isolation", self.measurement_layer, "enable_prefix_caching"
        ) is False

    @property
    def provider_port(self) -> int:
        """The port THIS lane's provider binds, and the one its runner must dial.

        Derived once here because it is needed in two places -- the server's own config and the
        client that connects to it -- and a lane whose runner dialled the base port would send
        every request to lane 0's provider while believing it was talking to its own. The two
        would agree on nothing afterwards: work keys, budgets and the ledger would all land on
        the wrong lane.
        """
        lane = self.lane_id
        if lane is None:
            return int(self.get("week1", "provider", "bind_port"))
        return int(self.get("week1", "measurement", "shards", "provider_base_port")) + lane

    @property
    def provider_unix_socket(self) -> Optional[str]:
        """This lane's socket path, or None. Four providers cannot share one socket."""
        declared = self.get("week1", "provider", "unix_socket")
        if not declared:
            return None
        lane = self.lane_id
        if lane is None:
            return str(declared)
        path = Path(str(declared))
        return str(path.with_name(f"{path.stem}-lane{lane}{path.suffix}"))

    def provider_config(self):
        """Build the provider's frozen configuration from the campaign config."""
        from ..runtime.provider_server import ProviderConfig

        block = dict(self.get("week1", "provider"))
        # Passed as strings: ProviderConfig.from_mapping is what turns them into OpClass, and
        # an alias naming an op class that does not exist must fail there, once.
        block["model_aliases"] = dict(self.get("week1", "model_aliases"))
        pricing = self.admission_pricing()
        block["deepseek_usd_per_1m_input"] = pricing["usd_per_1m_input_tokens"]
        block["deepseek_usd_per_1m_input_cached"] = pricing["usd_per_1m_input_tokens_cached"]
        block["deepseek_usd_per_1m_output"] = pricing["usd_per_1m_output_tokens"]
        block["deepseek_usd_worst_case"] = self.deepseek_usd_worst_case()
        block.setdefault(
            "max_consecutive_failures",
            int(self.get("budget", "retry_policy", "max_consecutive_failures")),
        )
        layer = str(self.get("week1", "measurement", "layer"))
        block["max_upstream_inflight"] = int(
            self.get("stack", "isolation", layer, "gateway_max_upstream_inflight")
        )
        # One lane holds the campaign's budget and serves the paid upstreams; the others serve
        # only local inference. Four providers each admitting against the full DeepSeek cap
        # would be a four-fold budget, and a cap that another process can multiply is not
        # admission control. The provider refuses those routes rather than trusting a runbook.
        lane = self.lane_id
        shards = self.get("week1", "measurement", "shards")
        if lane is not None:
            block["lane_id"] = lane
            block["bind_port"] = self.provider_port
            block["unix_socket"] = self.provider_unix_socket
            block["vllm_base_url"] = _with_port(
                str(block["vllm_base_url"]), int(shards["vllm_base_port"]) + lane
            )
        block["serves_paid_upstreams"] = (
            lane is None or lane == int(shards["paid_upstream_lane"])
        )
        return ProviderConfig.from_mapping(block)

    def admission_pricing(self) -> dict[str, float]:
        """The prices admission control reserves against.

        Defaults to the official snapshot. A separate ``admission_pricing`` block may only
        raise them: reserving against a cheaper price than the vendor charges is not
        conservatism, it is an under-estimate wearing the word "worst case".
        """
        official = self.get("judge", "pricing")
        rates = {
            "usd_per_1m_input_tokens": float(official["usd_per_1m_input_tokens"]),
            # Cache-hit prompt tokens bill ~120x below cache-miss. Reservations ignore the
            # discount (they must bound the worst case, where nothing is cached); settlement
            # applies it, which is why it travels with the other rates rather than as a default.
            "usd_per_1m_input_tokens_cached": float(official["usd_per_1m_input_tokens_cached"]),
            "usd_per_1m_output_tokens": float(official["usd_per_1m_output_tokens"]),
        }
        override = self.configs["judge"].get("admission_pricing") or {}
        for key, value in rates.items():
            if key in override:
                bound = float(override[key])
                if bound < value:
                    raise ConfigError(
                        f"admission_pricing.{key} ({bound}) is below the official snapshot "
                        f"({value}); admission may only be more conservative, never less"
                    )
                rates[key] = bound
        return rates

    def deepseek_usd_worst_case(self) -> float:
        """The money reservation, derived from the token ceilings rather than declared.

        A standalone USD constant is not a worst case -- it is a number that happens to sit
        next to one. The configured 0.05 was under half of what its own token ceilings imply
        at its own prices, so every authoring call was admitted against a bound it could
        exceed, and settlement then clamped the overspend out of sight.
        """
        provider = self.get("week1", "provider")
        pricing = self.admission_pricing()
        derived = (
            float(provider["deepseek_input_tokens_worst_case"]) / 1e6
            * pricing["usd_per_1m_input_tokens"]
            + float(provider["deepseek_output_tokens_worst_case"]) / 1e6
            * pricing["usd_per_1m_output_tokens"]
        )
        declared = provider.get("deepseek_usd_worst_case")
        if declared is not None and float(declared) < derived:
            raise ConfigError(
                f"provider.deepseek_usd_worst_case ({declared}) does not cover the "
                f"{derived:.4f} USD its own token ceilings imply at the admission prices; "
                "a reservation that is not an upper bound is not admission control"
            )
        return max(derived, float(declared or 0.0))

    def budget_caps(self) -> dict[str, float]:
        """The hash-locked caps, expressed as the resources the ledger accounts for.

        These are ceilings from ``configs/budget_v1.yaml``. The effective cap may only ever be
        *tightened* by the provider against a real balance; nothing may raise one, and nothing
        may raise one after the fact.
        """
        api = self.get("budget", "api_budget")
        campaign = self.get("budget", "campaign_budget")
        return {
            "exa_requests": float(api["exa_max_requests"]),
            "exa_usd": float(api["exa_max_usd"]),
            "tavily_requests": float(api["tavily_max_requests"]),
            "tavily_credits": float(api["tavily_max_credits"]),
            "deepseek_requests": float(api["deepseek_max_requests"]),
            "deepseek_input_tokens": float(api["deepseek_max_input_tokens"]),
            "deepseek_output_tokens": float(api["deepseek_max_output_tokens"]),
            "deepseek_usd": float(api["deepseek_max_usd"]),
            "remote_calls": float(api["max_remote_calls"]),
            "gpu_seconds": float(campaign["max_gpu_hours"]) * 3600.0,
        }

    def authoring_sampling(self):
        """The decoding policy the authoring calls must actually carry.

        Read from the frozen config rather than defaulted in the client. Every one of these
        keys existed in configs/task_source.yaml and none of them was ever read, so the
        corpus fingerprint described a policy that never left the machine.
        """
        from ..bench.grading.judge_client import SamplingEnvelope

        block = self.get("task_source", "authoring")
        return SamplingEnvelope(
            temperature=float(block["temperature"]),
            top_p=float(block["top_p"]),
            seed=int(block["seed"]),
            max_tokens=int(block["max_tokens"]),
            enable_thinking=bool(block.get("enable_thinking", False)),
            # DeepSeek has no thinking switch: reasoning is a property of the model id. The
            # attempts record the reasoning tokens that actually came back instead.
            send_thinking_switch=bool(block.get("send_thinking_switch", False)),
        )

    def judge_sampling(self):
        """Return the exact evaluator decoding envelope frozen in ``judge.yaml``.

        Evaluation previously constructed :class:`DeepSeekJudge` with client defaults.  The
        defaults happened to match temperature/top-p, but the frozen file did not control the
        request body and therefore could drift without changing the measurement.  Keep the
        actual wire policy behind the same Settings boundary as corpus authoring.
        """
        from ..bench.grading.judge_client import SamplingEnvelope

        block = self.get("judge", "model")
        return SamplingEnvelope(
            temperature=float(block["temperature"]),
            top_p=float(block["top_p"]),
            seed=int(block["seed"]),
            max_tokens=int(block["max_tokens"]),
            enable_thinking=bool(block["enable_thinking"]),
            send_thinking_switch=bool(block["send_thinking_switch"]),
        )

    def judge_max_retries(self) -> int:
        """The retry envelope is protocol, not a client-library default."""
        value = int(self.get("judge", "retry", "max_retries"))
        if value < 0:
            raise ConfigError("judge.retry.max_retries must be non-negative")
        return value

    def judge_model(self) -> str:
        """Resolve the judge model from the environment, and refuse a forbidden one.

        Never hardcoded (plan §3.4). ``deepseek-reasoner`` is refused because JSON mode is
        mandatory for judging and it does not support ``response_format: json_object`` -- an
        unvalidatable judgment scored as if it had been validated is a wrong number, not a
        missing one.
        """
        env_var = str(self.get("judge", "model", "env_var"))
        model = (os.environ.get(env_var) or "").strip()
        if not model:
            model = str(self.get("judge", "model", "expected"))
        forbidden = [str(f) for f in self.get("judge", "model", "forbid")]
        if model in forbidden:
            raise ConfigError(
                f"judge model {model!r} is forbidden by configs/judge.yaml: JSON mode is "
                "mandatory for judging and this model cannot honour it"
            )
        return model
