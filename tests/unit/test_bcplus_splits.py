"""The split carve: disjoint, deterministic, write-once, and refusing to quietly shrink."""

from __future__ import annotations

import json

import pytest

from shapeflow.bench.bcplus.splits import (
    LAYER_SIZES,
    SEALED_LAYER,
    SplitError,
    carve,
    load_manifest,
    write_manifest,
)

DEV = [str(i) for i in range(1000, 1530)]     # 530, as the frozen dev split
TEST = [str(i) for i in range(2000, 2300)]    # 300, as the sealed confirmatory split
SEED = 20260727


def test_the_layers_are_the_sizes_the_plan_fixes():
    plan = carve(DEV, TEST, seed=SEED)
    assert {n: len(ids) for n, ids in plan.layers.items() if n != SEALED_LAYER} == dict(LAYER_SIZES)
    assert len(plan.layers[SEALED_LAYER]) == 300


def test_every_layer_is_pairwise_disjoint():
    """The property the whole design rests on: a task selected on cannot be measured on."""
    plan = carve(DEV, TEST, seed=SEED)
    names = list(plan.layers)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(plan.layers[a]) & set(plan.layers[b])
            assert not shared, f"{a} and {b} share {len(shared)} tasks, e.g. {sorted(shared)[:3]}"


def test_the_carve_is_deterministic():
    assert carve(DEV, TEST, seed=SEED).digest == carve(DEV, TEST, seed=SEED).digest


def test_a_different_seed_carves_differently():
    assert carve(DEV, TEST, seed=SEED).digest != carve(DEV, TEST, seed=SEED + 1).digest


def test_the_carve_does_not_depend_on_input_order():
    """A split file listed in a different order is the same split."""
    a = carve(DEV, TEST, seed=SEED)
    b = carve(list(reversed(DEV)), list(reversed(TEST)), seed=SEED)
    assert {n: set(v) for n, v in a.layers.items()} == {n: set(v) for n, v in b.layers.items()}


def test_dev_and_test_overlap_is_refused():
    """A confirmatory split containing tuned-on tasks is the failure sealing exists to prevent."""
    with pytest.raises(SplitError, match="both dev and test"):
        carve(DEV, TEST[:-1] + [DEV[0]], seed=SEED)


def test_too_few_queries_is_an_error_not_a_smaller_layer():
    with pytest.raises(SplitError, match="must be an amendment"):
        carve(DEV[:100], TEST, seed=SEED)


def test_duplicate_ids_are_refused():
    with pytest.raises(SplitError, match="duplicate query ids"):
        carve(DEV + [DEV[0]], TEST, seed=SEED)


def test_the_manifest_is_write_once(tmp_path):
    path = tmp_path / "splits.json"
    first = write_manifest(carve(DEV, TEST, seed=SEED), path)
    again = write_manifest(carve(DEV, TEST, seed=SEED), path)
    assert first["digest"] == again["digest"], "an identical re-carve must be accepted"

    with pytest.raises(SplitError, match="cannot both be the frozen one"):
        write_manifest(carve(DEV, TEST, seed=SEED + 1), path)


def test_an_edited_manifest_is_detected(tmp_path):
    path = tmp_path / "splits.json"
    write_manifest(carve(DEV, TEST, seed=SEED), path)
    body = json.loads(path.read_text())
    # Move one task from the retest reserve into the layer B3 first evaluates on -- the exact
    # edit that would quietly spend the single post-iteration allowance.
    body["layers"]["fv_a"].append(body["layers"]["fv_b"].pop())
    path.write_text(json.dumps(body))
    with pytest.raises(SplitError, match="has been edited"):
        load_manifest(path)


def test_a_manifest_round_trips(tmp_path):
    path = tmp_path / "splits.json"
    plan = carve(DEV, TEST, seed=SEED)
    write_manifest(plan, path)
    assert load_manifest(path).digest == plan.digest


def test_layer_lookup_names_the_layer_a_task_belongs_to():
    plan = carve(DEV, TEST, seed=SEED)
    assert plan.layer_of(plan.layers["b2"][0]) == "b2"
    assert plan.layer_of(plan.layers[SEALED_LAYER][0]) == SEALED_LAYER
    with pytest.raises(KeyError):
        plan.layer_of("not-a-query")
