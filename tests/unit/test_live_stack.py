"""Freezing the stack once, and comparing against it read-only ever after.

The failure this guards is subtle: a doctor that wrote the manifest itself would take whatever
it happened to observe -- an already-drifted driver, an engine started without --max-num-seqs 1,
a different model revision -- and launder it into a legitimate baseline. So the writer and the
checker are separate, and the checker never fills anything in.
"""

from __future__ import annotations

import json

import pytest

from shapeflow.ops.live_stack import (
    PLACEHOLDER,
    StackError,
    StackObservation,
    attention_backend_from_log,
    compare,
    freeze_stack,
    model_merkle_root,
)

DECLARED = {
    "model": {"repo": "Qwen/Qwen3-14B-AWQ", "revision": "a" * 40,
              "artifact_merkle_root": PLACEHOLDER},
    "engine": {"vllm_dunder_version": "0.24.0", "vllm_package_tree_sha256": PLACEHOLDER,
               "attention_backend": PLACEHOLDER},
    "host": {"gpu_uuid": "GPU-1234", "python": "3.12"},
}


def _observation(**overrides) -> StackObservation:
    values = {
        "model_artifact_merkle_root": "m" * 64,
        "vllm_package_tree_sha256": "v" * 64,
        "attention_backend": "Using Flash Attention backend",
        "model_revision": "a" * 40,
        "vllm_dunder_version": "0.24.0",
        "gpu_uuid": "GPU-1234",
        "python": "3.12",
    }
    values.update(overrides)
    return StackObservation(values=values)


def test_freezing_resolves_every_placeholder(tmp_path):
    (tmp_path / "protocol").mkdir()
    body = freeze_stack(tmp_path, DECLARED, _observation(), frozen_at_utc="2026-07-24T00:00:00Z")
    assert set(body["resolved"]) == {
        "model.artifact_merkle_root", "engine.vllm_package_tree_sha256",
        "engine.attention_backend",
    }
    assert len(body["manifest_sha256"]) == 64
    assert (tmp_path / "protocol" / "stack_manifest.json").exists()


def test_a_field_that_cannot_be_observed_refuses_the_freeze(tmp_path):
    """A manifest with an unresolved field is not a baseline, and doctor must not fill it later."""
    (tmp_path / "protocol").mkdir()
    observation = _observation()
    observation.values.pop("attention_backend")
    with pytest.raises(StackError, match="not observable"):
        freeze_stack(tmp_path, DECLARED, observation, frozen_at_utc="2026-07-24T00:00:00Z")


def test_freezing_a_different_stack_over_an_existing_manifest_is_refused(tmp_path):
    (tmp_path / "protocol").mkdir()
    freeze_stack(tmp_path, DECLARED, _observation(), frozen_at_utc="2026-07-24T00:00:00Z")
    with pytest.raises(StackError, match="new protocol version"):
        freeze_stack(tmp_path, DECLARED,
                     _observation(model_artifact_merkle_root="z" * 64),
                     frozen_at_utc="2026-07-24T01:00:00Z")


def test_comparison_passes_on_the_stack_that_was_frozen(tmp_path):
    (tmp_path / "protocol").mkdir()
    manifest = freeze_stack(tmp_path, DECLARED, _observation(),
                            frozen_at_utc="2026-07-24T00:00:00Z")
    assert compare(manifest, DECLARED, _observation()) == []


def test_a_swapped_model_is_caught_even_at_an_unchanged_revision(tmp_path):
    (tmp_path / "protocol").mkdir()
    manifest = freeze_stack(tmp_path, DECLARED, _observation(),
                            frozen_at_utc="2026-07-24T00:00:00Z")
    problems = compare(manifest, DECLARED,
                       _observation(model_artifact_merkle_root="z" * 64))
    assert any("artifact_merkle_root" in p for p in problems)


def test_an_unobservable_field_is_a_mismatch_not_a_pass(tmp_path):
    """A check that could not run has not been satisfied."""
    (tmp_path / "protocol").mkdir()
    manifest = freeze_stack(tmp_path, DECLARED, _observation(),
                            frozen_at_utc="2026-07-24T00:00:00Z")
    partial = _observation()
    partial.values.pop("vllm_package_tree_sha256")
    problems = compare(manifest, DECLARED, partial)
    assert any("not observable" in p for p in problems)


def test_an_edited_stack_config_invalidates_the_manifest(tmp_path):
    (tmp_path / "protocol").mkdir()
    manifest = freeze_stack(tmp_path, DECLARED, _observation(),
                            frozen_at_utc="2026-07-24T00:00:00Z")
    edited = json.loads(json.dumps(DECLARED))
    edited["engine"]["vllm_dunder_version"] = "0.25.0"
    problems = compare(manifest, edited, _observation())
    assert any("stack.yaml changed" in p for p in problems)


def test_the_engine_flags_are_read_from_the_serving_process(tmp_path):
    """A config saying max_num_seqs=1 proves nothing about the process that is serving."""
    (tmp_path / "protocol").mkdir()
    manifest = freeze_stack(tmp_path, DECLARED, _observation(),
                            frozen_at_utc="2026-07-24T00:00:00Z")
    running = _observation(engine_flags="python -m vllm ... --max-num-seqs 256")
    problems = compare(manifest, DECLARED, running,
                       expected_engine_flags=["--max-num-seqs", "1",
                                              "--no-enable-prefix-caching"])
    assert any("--no-enable-prefix-caching" in p for p in problems)

    correct = _observation(
        engine_flags="python -m vllm ... --max-num-seqs 1 --no-enable-prefix-caching")
    assert compare(manifest, DECLARED, correct,
                   expected_engine_flags=["--max-num-seqs", "1",
                                          "--no-enable-prefix-caching"]) == []


def test_missing_engine_flags_are_a_mismatch_when_they_were_expected(tmp_path):
    (tmp_path / "protocol").mkdir()
    manifest = freeze_stack(tmp_path, DECLARED, _observation(),
                            frozen_at_utc="2026-07-24T00:00:00Z")
    problems = compare(manifest, DECLARED, _observation(),
                       expected_engine_flags=["--max-num-seqs", "1"])
    assert any("not observable" in p for p in problems)


def test_the_model_merkle_root_follows_the_files_that_load(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "README.md").write_text("docs")
    before = model_merkle_root(model)

    (model / "README.md").write_text("different docs")
    assert model_merkle_root(model) == before, "a README change invalidated the stack"

    (model / "model.safetensors").write_bytes(b"other weights")
    assert model_merkle_root(model) != before, "a weight change was invisible"


def test_the_hf_revision_reads_the_commit_not_the_blob_etag(tmp_path):
    """Each metadata file's first line is the commit; the second is that file's own etag."""
    from shapeflow.ops.live_stack import _hf_revision

    meta = tmp_path / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    for name, etag in (("config.json", "b" * 40), ("model.safetensors", "c" * 40)):
        (meta / f"{name}.metadata").write_text(f"{'a' * 40}\n{etag}\n1781600732.28\n")
    assert _hf_revision(tmp_path) == "a" * 40

    # Files from two different commits are not one revision.
    (meta / "extra.metadata").write_text(f"{'d' * 40}\n{'e' * 40}\n1.0\n")
    assert _hf_revision(tmp_path) == ""


def test_the_attention_backend_comes_from_the_engines_own_log(tmp_path):
    log = tmp_path / "vllm.log"
    log.write_text("(EngineCore pid=42) INFO 07-24 19:54 [cuda.py:480] Using FLASH_ATTN "
                   "attention backend out of potential backends: [...]\nINFO ready\n")
    # Normalized to the backend NAME, not the raw line: the line carries a pid and a timestamp
    # that change every restart, and a frozen field that moved every restart could never match.
    assert attention_backend_from_log(log) == "FLASH_ATTN"
    assert attention_backend_from_log(tmp_path / "missing.log") == ""


def test_the_attention_backend_is_the_current_engine_not_the_first_launch(tmp_path):
    """The supervisor appends to one log across restarts, so a stale first line must not win.

    Reading the first "Using X" line would pin the answer to the oldest launch and make the
    doctor check vacuous -- it would always agree with the freeze. The backend that matters is
    the one the engine running now chose, i.e. the last such line.
    """
    log = tmp_path / "vllm.log"
    log.write_text(
        "(EngineCore pid=1) INFO 07-24 10:00 [cuda.py:480] Using FLASHINFER attention backend "
        "out of potential backends: [...]\n"
        "... the engine ran, exited, and the supervisor restarted it ...\n"
        "(EngineCore pid=2) INFO 07-24 12:00 [cuda.py:480] Using FLASH_ATTN attention backend "
        "out of potential backends: [...]\n"
    )
    assert attention_backend_from_log(log) == "FLASH_ATTN"
