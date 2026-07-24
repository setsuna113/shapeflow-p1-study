"""Every schema must be a valid 2020-12 schema and must be closed.

An unclosed schema is a launch blocker: if an object silently accepts unknown keys, a
renamed or misspelled field stops being an error and starts being ignored, and an
artifact that no longer carries the field it claims to carry still validates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"
SCHEMA_FILES = sorted(SCHEMA_DIR.glob("*.schema.json"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _object_nodes(node, path="<root>"):
    """Yield (path, subschema) for every subschema that describes an object."""
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            yield path, node
        for key, sub in node.items():
            # 'properties' keys are field names, not schema keywords: descend one extra
            # level so a field literally named "type" isn't mistaken for a keyword.
            if key == "properties" and isinstance(sub, dict):
                for field, field_schema in sub.items():
                    yield from _object_nodes(field_schema, f"{path}.{field}")
            elif isinstance(sub, (dict, list)):
                yield from _object_nodes(sub, f"{path}.{key}")
    elif isinstance(node, list):
        for i, sub in enumerate(node):
            yield from _object_nodes(sub, f"{path}[{i}]")


def test_schema_directory_is_not_empty():
    assert SCHEMA_FILES, f"no schemas found in {SCHEMA_DIR}"


@pytest.mark.parametrize("path", SCHEMA_FILES, ids=lambda p: p.name)
def test_schema_is_valid_2020_12(path: Path):
    Draft202012Validator.check_schema(_load(path))


@pytest.mark.parametrize("path", SCHEMA_FILES, ids=lambda p: p.name)
def test_every_object_is_closed(path: Path):
    unclosed = [
        where
        for where, node in _object_nodes(_load(path))
        if "properties" in node and node.get("additionalProperties") is not False
    ]
    assert not unclosed, f"{path.name}: unclosed object(s) at {unclosed}"


@pytest.mark.parametrize("path", SCHEMA_FILES, ids=lambda p: p.name)
def test_schema_documents_itself(path: Path):
    schema = _load(path)
    assert schema.get("title"), f"{path.name}: missing title"
    assert schema.get("description"), f"{path.name}: missing description"


def test_the_two_span_namespaces_cannot_be_mixed():
    """A raw-source span must not validate as a visible-message span, or vice versa.

    This is the schema-level enforcement of the C_VISIBLE boundary: if the namespaces
    were interchangeable, a compressor-only arm could select raw page text that vendor
    compress_research never saw, and the result would be C_REGISTRY mislabelled.
    """
    schema = _load(SCHEMA_DIR / "evidence_span.schema.json")
    validator = Draft202012Validator(schema)
    h = "a" * 64

    raw_span = {
        "namespace": "RAW_SOURCE",
        "span_id": h,
        "content_hash": h,
        "source_occurrence_ids": [h],
        "char_start": 0,
        "char_end": 10,
        "text_sha256": h,
        "kind": "paragraph",
        "token_len": 4,
        "chunker_version": "markdown_structure_v1",
    }
    visible_span = {
        "namespace": "VISIBLE_MESSAGE",
        "visible_span_id": h,
        "message_id": "m1",
        "message_role": "tool",
        "byte_start": 0,
        "byte_end": 10,
        "exact_text_sha256": h,
        "kind": "TOOL_EVIDENCE",
        "visible_compressor_view_hash": h,
    }

    validator.validate(raw_span)
    validator.validate(visible_span)

    # Swapping the discriminator must fail: the ID field names differ, so neither
    # branch of the oneOf accepts the hybrid.
    with pytest.raises(Exception):
        validator.validate({**raw_span, "namespace": "VISIBLE_MESSAGE"})
    with pytest.raises(Exception):
        validator.validate({**visible_span, "namespace": "RAW_SOURCE"})


def test_p1_id_contract_cannot_carry_free_text():
    """P1-ID exists to isolate the pointer mechanism; if prose fit through it, the
    contract comparison would be measuring nothing."""
    schema = _load(SCHEMA_DIR / "selector_output.schema.json")
    validator = Draft202012Validator(schema)

    validator.validate({"contract": "P1_ID", "selected_ids": ["E1", "E12"]})

    with pytest.raises(Exception):
        validator.validate(
            {"contract": "P1_ID", "selected_ids": ["E1"], "summary": "some prose"}
        )


def test_bridge_requires_evidence_binding():
    schema = _load(SCHEMA_DIR / "selector_output.schema.json")
    validator = Draft202012Validator(schema)

    validator.validate(
        {
            "contract": "P1_BRIDGE",
            "selections": [{"span_id": "E1", "role": "support"}],
            "bridges": [{"text": "A follows B.", "evidence_ids": ["E1"]}],
        }
    )

    # An unsourced bridge is an assertion, not connective tissue.
    with pytest.raises(Exception):
        validator.validate(
            {
                "contract": "P1_BRIDGE",
                "selections": [{"span_id": "E1", "role": "support"}],
                "bridges": [{"text": "A follows B.", "evidence_ids": []}],
            }
        )
