"""Ops durability: backup-generation rotation in update_all.py (the
2026-07-23 audit found a single incoherent backup generation — one
schema-valid-but-wrong morning could poison the only restore point).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Scrapers"))
import update_all as U  # noqa: E402


def _seed(d, tag):
    d.mkdir(parents=True)
    (d / "a.csv").write_text(f"a,{tag}\n")
    (d / "b.csv").write_text(f"b,{tag}\n")


def test_rotate_backups_three_generations(tmp_path, monkeypatch):
    live = tmp_path / "backups"
    _seed(live, "live1")
    monkeypatch.setattr(U, "BACKUP_DIR", live)

    U.rotate_backups(n=3)
    assert (tmp_path / "backups.1" / "a.csv").read_text() == "a,live1\n"
    assert live.exists()                       # live dir copied, not moved
    assert not (tmp_path / "backups.2").exists()

    # second morning: yesterday's snapshot shifts to .2
    (live / "a.csv").write_text("a,live2\n")
    U.rotate_backups(n=3)
    assert (tmp_path / "backups.1" / "a.csv").read_text() == "a,live2\n"
    assert (tmp_path / "backups.2" / "a.csv").read_text() == "a,live1\n"

    # third morning: the oldest generation is dropped, not accumulated
    (live / "a.csv").write_text("a,live3\n")
    U.rotate_backups(n=3)
    assert (tmp_path / "backups.1" / "a.csv").read_text() == "a,live3\n"
    assert (tmp_path / "backups.2" / "a.csv").read_text() == "a,live2\n"
    assert not (tmp_path / "backups.3").exists()


def test_rotate_backups_missing_dir_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(U, "BACKUP_DIR", tmp_path / "backups")
    U.rotate_backups(n=3)                      # must not raise or create
    assert not (tmp_path / "backups.1").exists()
