"""Prefix-cache hygiene: reuse within an arm, never across arms. Freeze-1 §5.2(7).

The failure this guards against leaves no trace in any artifact. The researcher's early ReAct
turns are byte-identical across arms up to the point P1 first intervenes, so without isolation
whichever arm runs second finds those prefixes already cached and completes faster -- for reasons
that have nothing to do with its compression form. Every number would look plausible, and the
measured advantage would be an artifact of the run order.
"""

from __future__ import annotations

import pytest

from shapeflow.experiment.budget import Budget
from shapeflow.experiment.ledger import Ledger
from shapeflow.object_store import ObjectStore
from shapeflow.runtime.provider_server import ProviderConfig, ProviderService, RoleTokens
from shapeflow.secrets import SecretRedactor

TOKENS = {"runner": "runner-token-000000000000", "steward": "steward-token-00000000000",
          "evaluator": "evaluator-token-0000000000"}


def _service(tmp_path, **config):
    ledger = Ledger(str(tmp_path / f"ledger-{len(config)}-{sorted(config)}.sqlite"))
    return ProviderService(
        ProviderConfig(served_model="Qwen3-14B-AWQ", **config),
        ledger=ledger, budget=Budget(ledger), store=ObjectStore(tmp_path / "objects"),
        redactor=SecretRedactor(), tokens=RoleTokens(TOKENS),
        upstream=lambda *a, **k: (200, {}, 0.0),
    )


@pytest.fixture()
def isolated(tmp_path):
    return _service(tmp_path, cache_isolation_enabled=True, cache_namespace="engine-epoch-1")


def test_the_causal_layer_sends_no_salt_at_all(tmp_path):
    """Isolation off means byte-identical requests, exactly as before this existed.

    The causal layer runs with prefix caching disabled outright, so a salt there would change the
    request bytes for no benefit -- and changing the measured system to implement a safeguard it
    does not need is its own defect.
    """
    assert _service(tmp_path).cache_salt_for("P0") == ""


def test_reuse_within_one_arm_is_allowed(isolated):
    """R1, exact reuse: the same arm in the same epoch must get the same namespace."""
    assert isolated.cache_salt_for("P0") == isolated.cache_salt_for("P0")
    assert isolated.cache_salt_for("P0")


def test_reuse_across_arms_is_impossible(isolated):
    salts = {arm: isolated.cache_salt_for(arm) for arm in ("P0", "P1", "H_ID", "C_ID")}
    assert len(set(salts.values())) == len(salts), (
        f"two arms share a prefix-cache namespace: {salts}. Whichever ran second would inherit "
        "the other's warm cache and look faster for reasons unrelated to its form.")


def test_a_restart_invalidates_reuse(tmp_path):
    """A new engine epoch must produce a new namespace.

    The blocks are gone after a restart, so a salt that survived it would claim a hit rate the
    new process cannot deliver -- and the APC telemetry would disagree with the timings.
    """
    before = _service(tmp_path, cache_isolation_enabled=True, cache_namespace="epoch-1")
    after = _service(tmp_path, cache_isolation_enabled=True, cache_namespace="epoch-2")
    assert before.cache_salt_for("P0") != after.cache_salt_for("P0")


def test_an_unattributed_request_gets_no_salt(isolated):
    """No cell means no arm, and something that belongs to no arm has nothing to be isolated
    from. Inventing a namespace for it would partition the cache on nothing."""
    assert isolated.cache_salt_for(None) == ""
    assert isolated.cache_salt_for("") == ""


def test_the_salt_is_derived_not_configured(isolated, tmp_path):
    """Two services with the same namespace agree; the value is a function of its inputs.

    Configured-per-arm salts would let an editing mistake give two arms the same string, and
    nothing downstream would notice.
    """
    twin = _service(tmp_path, cache_isolation_enabled=True, cache_namespace="engine-epoch-1")
    assert twin.cache_salt_for("P0") == isolated.cache_salt_for("P0")


def test_the_salt_does_not_leak_the_namespace_in_clear(isolated):
    """The salt is a digest, so an arm id or epoch is not readable off a request body."""
    salt = isolated.cache_salt_for("P0")
    assert "P0" not in salt and "engine-epoch-1" not in salt
    assert len(salt) == 32 and all(c in "0123456789abcdef" for c in salt)
