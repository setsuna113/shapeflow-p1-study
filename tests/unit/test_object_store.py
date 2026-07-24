"""Object store: dedup, verified reads, crash-atomic writes."""

from __future__ import annotations

import pytest

from shapeflow_p1.object_store import CorruptObject, ObjectStore


def test_roundtrip(tmp_path):
    store = ObjectStore(tmp_path)
    ref = store.put_bytes(b"hello frozen page")
    assert store.get_bytes(ref.key) == b"hello frozen page"


def test_identical_bytes_dedup_to_one_blob(tmp_path):
    store = ObjectStore(tmp_path)
    a = store.put_bytes(b"same content")
    b = store.put_bytes(b"same content")
    assert a.key == b.key
    # one physical file
    blobs = list(tmp_path.rglob("*.zst"))
    assert len(blobs) == 1


def test_get_verifies_and_raises_on_corruption(tmp_path):
    store = ObjectStore(tmp_path)
    ref = store.put_bytes(b"trustworthy bytes")
    # Corrupt the stored blob on disk.
    blob = next(tmp_path.rglob("*.zst"))
    blob.write_bytes(b"garbage that will not decompress to the same hash")
    with pytest.raises(CorruptObject):
        store.get_bytes(ref.key)
    # verify() must report False rather than raising -- resume relies on that.
    assert store.verify(ref.key) is False


def test_missing_key_raises_keyerror(tmp_path):
    store = ObjectStore(tmp_path)
    with pytest.raises(KeyError):
        store.get_bytes("0" * 64)
    assert store.verify("0" * 64) is False


def test_no_tmp_files_survive_a_successful_write(tmp_path):
    store = ObjectStore(tmp_path)
    store.put_bytes(b"x" * 10000)
    assert not list(tmp_path.rglob("*.tmp"))


def test_rejects_non_hex_key(tmp_path):
    store = ObjectStore(tmp_path)
    with pytest.raises(ValueError):
        store.get_bytes("not-a-hash")
