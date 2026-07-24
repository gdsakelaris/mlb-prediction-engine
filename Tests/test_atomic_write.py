"""atomic_write (Scrapers/seasons.py) — crash-safe rewrite contract.

(a) a normal write lands with the utf-8-sig BOM and exact content and
    leaves no .tmp behind;
(b) an exception inside the context leaves a pre-existing target
    byte-identical and cleans up the .tmp;
(c) mode/kwargs pass through unchanged (plain utf-8 json, str path,
    no BOM), matching the gamelogs cache-write call shape.
"""
import json
import sys
from pathlib import Path

import pytest

_SCRAPERS = Path(__file__).resolve().parent.parent / "Scrapers"
if str(_SCRAPERS) not in sys.path:
    sys.path.insert(0, str(_SCRAPERS))

from seasons import atomic_write  # noqa: E402

BOM = b"\xef\xbb\xbf"


def test_normal_write_lands_with_bom_and_no_tmp(tmp_path):
    target = tmp_path / "out.csv"
    with atomic_write(target, "w", newline="", encoding="utf-8-sig") as f:
        f.write("A,B\n1,2\n")
    assert target.read_bytes() == BOM + b"A,B\n1,2\n"
    assert list(tmp_path.glob("*.tmp")) == []


def test_exception_preserves_target_and_removes_tmp(tmp_path):
    target = tmp_path / "out.csv"
    original = BOM + b"A,B\nkeep,me\n"
    target.write_bytes(original)
    with pytest.raises(RuntimeError, match="boom"):
        with atomic_write(target, "w", newline="",
                          encoding="utf-8-sig") as f:
            f.write("A,B\npartial")
            raise RuntimeError("boom")
    assert target.read_bytes() == original          # byte-identical
    assert list(tmp_path.glob("*.tmp")) == []       # no litter
    assert list(tmp_path.iterdir()) == [target]


def test_kwargs_passthrough_plain_utf8_json(tmp_path):
    target = tmp_path / "cache.json"
    data = {"games": [1, 2, 3], "name": "José"}
    # str path + plain utf-8, exactly the gamelogs cache-write shape
    with atomic_write(str(target), "w", encoding="utf-8") as f:
        json.dump(data, f)
    raw = target.read_bytes()
    assert not raw.startswith(BOM)                  # no sig in plain utf-8
    with open(target, encoding="utf-8") as f:
        assert json.load(f) == data
    assert list(tmp_path.glob("*.tmp")) == []
