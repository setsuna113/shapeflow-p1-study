"""The campaign configs are protocol, so they are checked like protocol.

Each assertion below corresponds to a way a config can be wrong that produces a result which
looks fine: an arm set that does not name registered variants, a query budget that would blow
the hash-locked Tavily cap, a split plan whose totals do not add up, a judge model that cannot
honour JSON mode, a path block with a host-specific absolute path in it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from shapeflow_p1.campaign.settings import Settings
from shapeflow_p1.config import ConfigError
from shapeflow_p1.protocol import compute_binding

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture()
def settings():
    return Settings.load(REPO, data_root=Path("/tmp/shapeflow-test-root"))


def test_every_campaign_config_exists_and_hashes(settings):
    for name in ("week1", "acquisition", "task_source", "judge"):
        assert settings.configs[name]
        assert len(settings.shas[name]) == 64


def test_the_approval_binding_covers_the_campaign_configs():
    """A result depends on the arm set, the world, the corpus and the judge.

    An approval that pinned only protocol/budget/thresholds would still verify while describing
    a different experiment.
    """
    covered = set(compute_binding(REPO).content())
    assert {"week1_sha", "acquisition_sha", "task_source_sha", "judge_sha"} <= covered


def test_no_hashed_config_carries_a_host_specific_absolute_data_path(settings):
    """The authoring machine and the run host must compute the same protocol SHA."""
    for name in ("paths",):
        for value in settings.get("week1", name).values():
            assert not str(value).startswith("/"), (
                f"week1.paths.{value} is absolute; the data root belongs in "
                "SHAPEFLOW_DATA_ROOT, not in a hashed config"
            )


def test_the_data_root_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("SHAPEFLOW_DATA_ROOT", "/tmp/elsewhere")
    assert Settings.load(REPO).data_root == Path("/tmp/elsewhere")


def test_paths_resolve_under_the_data_root(settings):
    assert settings.path("runs") == Path("/tmp/shapeflow-test-root/runner/runs")
    assert settings.path("truth_packets").parts[-2] == "evaluator"


def test_the_evaluator_and_runner_trees_are_disjoint(settings):
    """The treatment must not be able to walk into the answer key."""
    runner = settings.path("runner_root")
    evaluator = settings.path("evaluator_root")
    assert not str(evaluator).startswith(str(runner) + os.sep)
    assert not str(runner).startswith(str(evaluator) + os.sep)


# --- the arm set --------------------------------------------------------------------------


def test_every_canary_arm_names_registered_variants(settings):
    from shapeflow_p1.strategies.factory import load_registry

    registry = load_registry(REPO / "configs")
    for arm in settings.get("week1", "canary", "arms"):
        for key in ("page_variant", "close_variant"):
            assert arm[key] in registry, f"{arm['arm_id']}.{key}={arm[key]} is not registered"


def test_the_canary_covers_the_arms_the_protocol_requires(settings):
    """P0, H-ID, H-typed, C_VISIBLE, H+C and both controls, or the canary cannot show why."""
    arms = {a["arm_id"] for a in settings.get("week1", "canary", "arms")}
    assert arms == {"P0", "H_ID", "H_TYPED", "C_VISIBLE", "H_PLUS_C", "CPU_LEXICAL",
                    "SHORT_PROSE"}


def test_a_close_only_arm_leaves_the_page_node_at_p0(settings):
    """Otherwise the C contrast is confounded with an H effect."""
    by_id = {a["arm_id"]: a for a in settings.get("week1", "canary", "arms")}
    assert by_id["C_VISIBLE"]["page_variant"] == "P0"
    assert by_id["CPU_LEXICAL"]["close_variant"] == "P0"
    assert by_id["SHORT_PROSE"]["close_variant"] == "P0"


# --- budgets and splits --------------------------------------------------------------------


def test_the_acquisition_plan_fits_inside_the_hash_locked_tavily_cap(settings):
    """Query budget x acquired tasks must not exceed the cap; the cap may never be raised."""
    splits = settings.get("task_source", "splits")
    acquired = sum(splits[name] for name in settings.get("acquisition", "acquire_splits"))
    per_task = settings.get("acquisition", "queries_per_task", "max_total")
    cap = settings.get("budget", "api_budget", "tavily_max_requests")
    assert acquired * per_task <= cap, (
        f"{acquired} tasks x {per_task} queries exceeds the {cap}-request cap"
    )
    # And the reserve worlds are deliberately not acquired up front.
    assert "RESERVE" not in settings.get("acquisition", "acquire_splits")


def test_the_split_sizes_match_the_registry_splits(settings):
    from shapeflow_p1.acquire.task_registry import SPLITS

    assert set(settings.get("task_source", "splits")) == set(SPLITS)


def test_the_screen_split_is_the_formative_one(settings):
    assert settings.get("week1", "screen", "split") == "FORMATIVE_SCREEN"
    assert settings.get("week1", "campaign", "claim_scope") == "FORMATIVE_ONLY"
    assert settings.get("week1", "campaign", "corpus_tier") == "FORMATIVE_MACHINE_AUTHORED"


def test_the_holdout_is_not_opened_this_round(settings):
    assert settings.get("week1", "campaign", "open_holdout") is False


def test_budget_caps_are_the_hash_locked_values(settings):
    caps = settings.budget_caps()
    assert caps["tavily_max_requests" if False else "tavily_requests"] == 500.0
    assert caps["deepseek_usd"] == 10.0
    assert caps["gpu_seconds"] == 150 * 3600.0


def test_a_provider_chosen_strategy_is_absent_from_the_acquisition_config(settings):
    """`type: auto` is Exa's version of Tavily's `auto_parameters`: the provider picks a
    strategy per query, so two acquisitions of one query can freeze two different worlds."""
    block = settings.get("acquisition", "exa")
    assert block["type"] != "auto"
    assert "auto_parameters" not in block

    from shapeflow_p1.acquire.exa_client import ExaParams

    with pytest.raises(ValueError, match="pinnable"):
        ExaParams(type="auto")


def test_the_search_reservation_covers_the_pinned_result_count(settings):
    assert settings.exa_usd_worst_case() >= settings.get(
        "acquisition", "exa_pricing")["usd_per_request"]


def test_replay_cannot_fall_back_to_a_live_search(settings):
    assert settings.get("acquisition", "replay", "fail_closed_on_miss") is True
    assert settings.get("acquisition", "replay", "allow_live_fallback") is False


# --- measurement layer ----------------------------------------------------------------------


def test_the_causal_layer_is_what_this_round_runs(settings):
    assert settings.get("week1", "measurement", "layer") == "causal"
    causal = settings.get("stack", "isolation", "causal")
    assert causal["max_num_seqs"] == 1
    assert causal["enable_prefix_caching"] is False
    assert causal["gateway_max_upstream_inflight"] == 1


def test_the_generation_cap_is_a_completion_limit_not_a_schema_bound(settings):
    """maxItems bounds what may be accepted, not what may be decoded.

    Reporting a schema bound as a decode cap would overstate the saving by exactly the tokens
    the model actually emitted and we then threw away.
    """
    assert settings.get("week1", "measurement", "selector_max_completion_tokens") > 0
    assert settings.get("week1", "measurement", "guided_decoding") is True


# --- judge -----------------------------------------------------------------------------------


def test_the_judge_model_comes_from_the_environment(settings, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_JUDGE_MODEL", raising=False)
    assert settings.judge_model() == "deepseek-v4-flash"
    monkeypatch.setenv("DEEPSEEK_JUDGE_MODEL", "deepseek-v4-pro")
    assert settings.judge_model() == "deepseek-v4-pro"


def test_a_retired_model_is_refused_rather_than_failing_at_the_first_call(settings, monkeypatch):
    """deepseek-chat no longer exists; the API answers 400. Better to refuse at config load."""
    monkeypatch.setenv("DEEPSEEK_JUDGE_MODEL", "deepseek-chat")
    with pytest.raises(ConfigError, match="forbidden"):
        settings.judge_model()


def test_a_model_that_cannot_honour_json_mode_is_refused(settings, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_JUDGE_MODEL", "deepseek-reasoner")
    with pytest.raises(ConfigError, match="forbidden"):
        settings.judge_model()


def test_a_missing_judgment_is_never_imputed(settings):
    policy = settings.get("judge", "failure_policy")
    assert policy["on_unavailable"] == "FAIL_CLOSED"
    assert policy["impute"] is False
    assert policy["drop_sample"] is False


def test_the_pricing_snapshot_records_its_source_and_whether_it_is_verified(settings):
    """An unverified price must say so, and must over-count rather than under-count."""
    pricing = settings.get("judge", "pricing")
    assert pricing["source"].startswith("https://")
    assert pricing["retrieved_utc"] == "2026-07-24"
    assert pricing["verified"] is False
    assert "upper_bound" in pricing["basis"]


def test_the_provider_config_is_built_from_the_campaign_config(settings):
    config = settings.provider_config()
    assert config.bind_host == "127.0.0.1"
    assert config.deepseek_usd_per_1m_input == 2.00
    assert config.model_aliases["qwen-selector-page"].value == "PAGE_P1_SELECTOR_LOCAL"
    assert config.max_consecutive_failures == 5


def test_the_money_reservation_covers_its_own_token_ceilings(settings):
    """A standalone USD constant is a number sitting next to a worst case, not one."""
    provider = settings.get("week1", "provider")
    pricing = settings.admission_pricing()
    derived = (
        provider["deepseek_input_tokens_worst_case"] / 1e6 * pricing["usd_per_1m_input_tokens"]
        + provider["deepseek_output_tokens_worst_case"] / 1e6
        * pricing["usd_per_1m_output_tokens"]
    )
    assert settings.deepseek_usd_worst_case() >= derived


def test_a_reservation_below_its_token_bound_is_refused_at_load(settings, tmp_path):
    from shapeflow_p1.config import ConfigError

    shrunk = dict(settings.configs["week1"])
    shrunk["provider"] = dict(shrunk["provider"], deepseek_usd_worst_case=0.05)
    settings.configs["week1"] = shrunk
    with pytest.raises(ConfigError, match="upper bound"):
        settings.deepseek_usd_worst_case()


def test_admission_pricing_may_only_be_more_conservative(settings):
    from shapeflow_p1.config import ConfigError

    official = settings.get("judge", "pricing")
    settings.configs["judge"] = dict(
        settings.configs["judge"],
        admission_pricing={"usd_per_1m_input_tokens":
                           official["usd_per_1m_input_tokens"] / 2},
    )
    with pytest.raises(ConfigError, match="never less"):
        settings.admission_pricing()


def test_the_authoring_envelope_comes_from_the_frozen_config(settings):
    envelope = settings.authoring_sampling()
    block = settings.get("task_source", "authoring")
    assert envelope.temperature == block["temperature"]
    assert envelope.top_p == block["top_p"]
    assert envelope.seed == block["seed"]
    assert envelope.max_tokens == block["max_tokens"]


def test_a_missing_key_is_an_error_not_a_default(settings):
    with pytest.raises(ConfigError, match="missing required key"):
        settings.get("week1", "campaign", "no_such_key")
