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

__all__ = ["Settings", "DATA_ROOT_ENV"]

DATA_ROOT_ENV = "SHAPEFLOW_DATA_ROOT"

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
        return cls(repo=repo, data_root=Path(root), configs=configs, shas=shas)

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

    def path(self, name: str) -> Path:
        """An absolute path for one of the declared relative locations."""
        return self.data_root / str(self.get("week1", "paths", name))

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

    def provider_config(self):
        """Build the provider's frozen configuration from the campaign config."""
        from ..runtime.provider_server import ProviderConfig

        block = dict(self.get("week1", "provider"))
        # Passed as strings: ProviderConfig.from_mapping is what turns them into OpClass, and
        # an alias naming an op class that does not exist must fail there, once.
        block["model_aliases"] = dict(self.get("week1", "model_aliases"))
        pricing = self.admission_pricing()
        block["deepseek_usd_per_1m_input"] = pricing["usd_per_1m_input_tokens"]
        block["deepseek_usd_per_1m_output"] = pricing["usd_per_1m_output_tokens"]
        block["deepseek_usd_worst_case"] = self.deepseek_usd_worst_case()
        block.setdefault(
            "max_consecutive_failures",
            int(self.get("budget", "retry_policy", "max_consecutive_failures")),
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
        from ..evaluation.judge_client import SamplingEnvelope

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
