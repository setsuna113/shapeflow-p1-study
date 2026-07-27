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
    """Every expensive screen arm is exercised before the remaining 700+ cells are admitted."""
    canary = settings.get("week1", "canary", "arms")
    screen = settings.get("week1", "screen_arms", "arms")
    assert canary == screen
    assert len(canary) == 19


def test_a_close_only_arm_leaves_the_page_node_at_p0(settings):
    """Otherwise the C contrast is confounded with an H effect."""
    by_id = {a["arm_id"]: a for a in settings.get("week1", "canary", "arms")}
    assert by_id["C_ID"]["page_variant"] == "P0"
    assert by_id["H_CPU_CONTROL"]["close_variant"] == "P0"
    assert by_id["H_PROSE_CONTROL"]["close_variant"] == "P0"


def test_the_weeklong_screen_exercises_every_runnable_p1_design_axis(settings):
    from shapeflow_p1.strategies.factory import load_registry

    arms = settings.get("week1", "screen_arms", "arms")
    registry = load_registry(REPO / "configs")
    variants = {
        variant
        for arm in arms
        for variant in (arm["page_variant"], arm["close_variant"])
    }
    assert settings.get("week1", "screen", "arms_block") == "screen_arms"
    assert {
        "P0",
        "H00-CPU", "H00-PROSE", "H01", "H02", "H_TYPED_STABLE", "H03",
        "H_HIER_COVERAGE", "H04", "H05",
        "C00-CPU", "C00-PROSE", "C01", "C_TYPED_STABLE", "C02", "C03",
    } <= variants
    assert any(
        arm["page_variant"] == "H02" and arm["close_variant"] == "C01"
        for arm in arms
    )
    assert len({arm["arm_id"] for arm in arms}) == len(arms)
    for arm in arms:
        for variant_id in (arm["page_variant"], arm["close_variant"]):
            assert variant_id in registry
            assert registry[variant_id].runnable


def test_matched_contrasts_change_exactly_the_declared_factor(settings):
    """Resolve arm IDs and prove every named ablation is one-factor-at-a-time."""
    from shapeflow_p1.analysis.matched import resolve_arm_semantics

    week1 = settings.configs["week1"]
    variant_data = settings.configs["variants"]["variants"]
    variants = {variant["variant_id"]: variant for variant in variant_data}
    arms = {
        arm["arm_id"]: arm
        for arm in settings.get("week1", "screen_arms", "arms")
    }
    matched = settings.get("week1", "matched_contrasts")
    executable_fields = matched["executable_variant_fields"]
    fields = set(executable_fields)
    required_kinds = {
        "ID_vs_TYPED",
        "stable_vs_coverage",
        "per_page_vs_hierarchical",
        "typed_vs_bridge",
        "LLM_vs_CPU",
        "structured_selection_vs_prose",
    }

    seen_ids = set()
    seen_kinds = set()
    for pair in matched["pairs"]:
        assert pair["contrast_id"] not in seen_ids
        seen_ids.add(pair["contrast_id"])
        seen_kinds.add(pair["kind"])
        left = resolve_arm_semantics(
            pair["left_arm_id"],
            arms[pair["left_arm_id"]],
            variants,
            executable_fields,
        )
        right = resolve_arm_semantics(
            pair["right_arm_id"],
            arms[pair["right_arm_id"]],
            variants,
            executable_fields,
        )
        target = pair["target_factor"]
        factor_fields = set(pair["factor_fields"])
        assert target in factor_fields
        assert factor_fields <= fields
        assert left.get(target) == pair["left_level"]
        assert right.get(target) == pair["right_level"]
        observed_differences = {
            field
            for field in fields
            if left.get(field) != right.get(field)
        }
        assert observed_differences == factor_fields, (
            f"{pair['contrast_id']} is confounded: expected differences "
            f"{sorted(factor_fields)}, observed {sorted(observed_differences)}"
        )
        if isinstance(pair["left_level"], dict):
            assert pair["factor_scope"] == "JOINT_E2E_POLICY"
            assert pair["affected_nodes"] == ["WEBPAGE_P1", "C_VISIBLE"]
            assert pair["first_treatment_boundary"] == "WEBPAGE_P1"

    assert required_kinds <= seen_kinds
    # The block is part of the already approval-bound week1 config, not an unhashed sidecar.
    from shapeflow_p1.config import config_sha

    without_contrasts = dict(week1)
    without_contrasts.pop("matched_contrasts")
    assert config_sha(without_contrasts) != settings.shas["week1"]


def test_the_matched_screen_freezes_an_honest_gpu_break_even(settings):
    """Freeze the cell count and honest break-even budget; do not call a request a cell."""
    arms = settings.get("week1", "screen_arms", "arms")
    screen_tasks = settings.get("task_source", "splits", "FORMATIVE_SCREEN")
    canary_tasks = settings.get("week1", "canary", "tasks")
    canary_arms = settings.get("week1", "canary", "arms")
    replicated_screen_tasks = round(
        screen_tasks * settings.get("week1", "screen", "second_seed_fraction")
    )
    offered_cells = (
        len(arms) * (screen_tasks + replicated_screen_tasks)
        + len(canary_arms) * canary_tasks
    )
    assert offered_cells == 836
    assert settings.get("week1", "screen", "second_seed_fraction") == 0.25
    assert settings.get("week1", "screen", "seeds") == [1, 2]
    # This is the maximum *observed campaign mean* that can complete under the hard cap.
    # provider.gpu_seconds_worst_case is a per-inference-request reservation and must never
    # be multiplied by cells as if every complete graph issued exactly one request.
    break_even_seconds_per_cell = settings.budget_caps()["gpu_seconds"] / offered_cells
    assert break_even_seconds_per_cell == pytest.approx(645.933014, rel=1e-6)


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
    """Pins the caps so a change has to be deliberate, not a default that drifted.

    Re-derived 2026-07-26 at verified prices. The earlier set was denominated in an invented
    2.00/8.00 price snapshot that over-stated spend ~19x, so its $200 was about $10 of real
    authority -- and its 20,000-request cap would have halted the campaign well before the
    dollar cap was ever approached.
    """
    caps = settings.budget_caps()
    assert caps["tavily_requests"] == 500.0
    assert caps["deepseek_usd"] == 400.0
    assert caps["deepseek_requests"] == 150000.0
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


def test_the_primary_layer_is_causal_without_serializing_the_graph(settings):
    """Causal means no cross-arm prefix-cache carry-over. It never meant one request at a time.

    Serialization was a precondition of the summed-service metric, not of isolation, and
    imposing it on a graph that summarises a result set with ``asyncio.gather`` produced 212
    vendor timeouts -- each of which publishes the whole raw page instead of a summary, worst on
    the largest pages, which is the stratum the study exists to measure.
    """
    assert settings.measurement_layer == "causal_native"
    active = settings.get("stack", "isolation", "causal_native")
    assert active["enable_prefix_caching"] is False
    assert settings.layer_is_causal is True
    assert settings.layer_is_serialized is False
    # Large enough that the engine is never the limiter: the graph's own ceiling is eight
    # concurrent page summaries.
    assert active["max_num_seqs"] > 8
    assert active["gateway_max_upstream_inflight"] == 0


def test_the_serialized_layer_survives_as_a_mechanism_control_on_the_core_2x2(settings):
    """Kept, demoted, and still serialized -- otherwise its own metric means nothing."""
    mechanism = settings.get("stack", "isolation", "causal")
    assert mechanism["max_num_seqs"] == 1
    assert mechanism["enable_prefix_caching"] is False
    assert mechanism["gateway_max_upstream_inflight"] == 1
    assert settings.get("week1", "measurement", "mechanism_layer") == "causal"
    arms = settings.get("week1", "measurement", "mechanism_layer_arms")
    assert set(arms) == {"P0", "H_MARKDOWN_ID", "C_ID", "H_PLUS_C"}


def test_the_work_endpoints_are_pre_registered_and_tokens_are_never_summed(settings):
    """Prompt and completion tokens move in opposite directions between P0 and P1.

    P1 buys a shorter decode with a longer selector prefill. One combined "tokens" number with a
    threshold on it would hide the entire trade the study is about, so they are co-primary and
    separately reported.
    """
    from shapeflow_p1.analysis.e2e_effects import PRIMARY_WORK_ENDPOINT

    endpoints = settings.get("week1", "measurement", "endpoints")
    assert endpoints["primary_work"] == PRIMARY_WORK_ENDPOINT
    assert endpoints["co_primary"] == ["prompt_tokens", "completion_tokens"]
    # The summed-service metric is defined only where intervals do not overlap.
    assert endpoints["mechanism_only"] == ["service_work_seconds"]
    assert "energy_joules" in endpoints["secondary"]


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
    """A verified price must name its source and date; an unverified one must over-count.

    Settlement is now computed at the published rates rather than an invented ceiling. The
    invented one made the ledger read $31.40 against a real bill near $1.60, which also made
    every USD cap meaningless. Reservations remain worst-case via the token ceilings.
    """
    pricing = settings.get("judge", "pricing")
    assert pricing["source"].startswith("https://")
    assert pricing["retrieved_utc"] == "2026-07-26"
    assert pricing["verified"] is True
    # Quoted for the dearer candidate model, so it still bounds whichever judge is selected.
    assert "upper_bound" in pricing["basis"]
    assert pricing["usd_per_1m_input_tokens_cached"] < pricing["usd_per_1m_input_tokens"]


def test_the_provider_config_is_built_from_the_campaign_config(settings):
    config = settings.provider_config()
    assert config.bind_host == "127.0.0.1"
    assert config.deepseek_usd_per_1m_input == 0.435
    assert config.deepseek_usd_per_1m_input_cached == 0.003625
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

    # Derived from the live config rather than hard-coded: a literal 0.05 silently stopped
    # discriminating the moment verified prices dropped the real bound below it, and a test
    # that cannot fail is worse than no test.
    derived = settings.deepseek_usd_worst_case()
    shrunk = dict(settings.configs["week1"])
    shrunk["provider"] = dict(shrunk["provider"], deepseek_usd_worst_case=derived / 2)
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


def test_the_judge_request_and_retry_envelopes_come_from_frozen_config(settings):
    envelope = settings.judge_sampling()
    block = settings.get("judge", "model")
    assert envelope.request_fields() == {
        "temperature": block["temperature"],
        "top_p": block["top_p"],
        "seed": block["seed"],
        "max_tokens": block["max_tokens"],
    }
    assert envelope.enable_thinking is block["enable_thinking"]
    assert envelope.send_thinking_switch is block["send_thinking_switch"]
    assert settings.judge_max_retries() == settings.get(
        "judge", "retry", "max_retries")
    assert settings.get("judge", "prompts", "truth_version") == (
        "judge_truth_v2_full_world")


def test_judge_config_versions_must_match_the_code_that_is_actually_run(settings):
    settings.validate_judge_policy_versions()
    drifted = dict(settings.configs["judge"])
    drifted["prompts"] = dict(
        drifted["prompts"], truth_version="stale_truth_prompt")
    settings.configs["judge"] = drifted
    with pytest.raises(ConfigError, match="policy drift"):
        settings.judge_sampling()


def test_atomizer_truth_and_report_measurements_have_distinct_code_identities(settings):
    from shapeflow_p1.campaign.evaluate import relation_prompt_sha256
    from shapeflow_p1.campaign.truth import truth_prompt_sha256
    from shapeflow_p1.evaluation.atomizer import atomize_protocol_sha256

    assert len({
        atomize_protocol_sha256(), truth_prompt_sha256(), relation_prompt_sha256(),
    }) == 3
    declared = settings.get("stack", "evaluation")
    assert set(declared) == {
        "atomize_prompt_sha256", "truth_prompt_sha256", "report_prompt_sha256",
    }
    assert all(value == "@STEWARD_FREEZES@" for value in declared.values())


def test_a_missing_key_is_an_error_not_a_default(settings):
    with pytest.raises(ConfigError, match="missing required key"):
        settings.get("week1", "campaign", "no_such_key")


def test_the_summarization_timeout_is_protocol_and_defaults_to_vendor(settings):
    """Vendor's 60s cap silently turns P0 into a no-op on this engine.

    On timeout `summarize_webpage` returns the RAW page instead of a summary. Only P0 can reach
    it -- under P1 the page hook returns from defer_page_batch before vendor's summarizer is
    awaited -- so the degradation is asymmetric, and worst on the largest pages, which is the
    stratum where P1 is hypothesised to help. That is a manufactured headline result, so the
    value is protocol rather than tuning.

    The patch reads it from the environment and falls back to vendor's own 60.0, so a graph
    nobody configured stays byte-for-byte vendor and the parity gate is unaffected.
    """
    seconds = settings.get("week1", "odr", "summarization_timeout_seconds")
    assert float(seconds) > 60.0, "a value at or below vendor's default fixes nothing"

    patch = (REPO / "patches" / "odr_p1_hooks.patch").read_text(encoding="utf-8")
    assert 'os.environ.get("SHAPEFLOW_SUMMARIZE_TIMEOUT_S") or 60.0' in patch, (
        "the patch must fall back to vendor's own default when the variable is unset"
    )
