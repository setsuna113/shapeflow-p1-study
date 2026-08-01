"""Loading configuration as hash-locked protocol constants, not tunable defaults.

Every config file is loaded, canonicalized and hashed. Those hashes are what the external
append-only approval store pins, so the decision thresholds and budgets that were approved
cannot drift after approval without minting a new execution binding and invalidating the
approval. The loader therefore never fills in a missing value with a default -- a missing key
is an error, because a silently-defaulted threshold is exactly the kind of change the hash lock
exists to catch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .canonical import canonical_json
from .hashing import sha256_hex

__all__ = ["ConfigError", "load_config", "config_sha", "ConfigBundle"]


class ConfigError(ValueError):
    pass


def config_sha(data: Any) -> str:
    """The canonical hash of a config value -- the identity pinned by the approval file."""
    return sha256_hex(canonical_json(data))


def load_config(path: str | Path) -> tuple[dict, str]:
    """Load one YAML config, returning (data, config_sha). Rejects a non-mapping top level."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"config {path} must be a mapping at top level")
    return data, config_sha(data)


def require(data: dict, *keys: str) -> Any:
    """Fetch a nested key, raising ConfigError (never returning a default) if absent."""
    node: Any = data
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"missing required config key: {'.'.join(keys)}")
        node = node[key]
    return node


@dataclass(frozen=True)
class ConfigBundle:
    """A loaded set of protocol configs with their individual hashes and a combined protocol SHA."""

    configs: dict[str, dict]
    shas: dict[str, str]

    @property
    def protocol_sha(self) -> str:
        """A hash over the per-config hashes, so any change to any config changes it."""
        return sha256_hex(canonical_json({name: self.shas[name] for name in sorted(self.shas)}))

    @classmethod
    def load(cls, paths: dict[str, str | Path]) -> "ConfigBundle":
        configs: dict[str, dict] = {}
        shas: dict[str, str] = {}
        for name, path in paths.items():
            data, sha = load_config(path)
            configs[name] = data
            shas[name] = sha
        return cls(configs=configs, shas=shas)
