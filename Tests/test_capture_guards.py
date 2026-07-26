"""Noon-chain capture guards: Tools/2 capture_verdict (the odds run's
total/partial-failure decision) and Tools/1 slate_floor_verdict (the
zero-game schedule cross-check). Both are the pure seams main() wires to
sys.exit / the log, so everything here runs on plain ints — no network.

The marker substrings are load-bearing contracts with the watchdog:
check_health.ps1 greps the noon log for the literal "ODDS CAPTURE
DEGRADED", and "ODDS CAPTURE FAILED" rides exit 1 through the noon
chain's fail alert. Tests assert the exact substrings, not just tone.
"""
from conftest import load_tool

T1 = load_tool("1) Get Todays Games")
T2 = load_tool("2) Scrape Odds")


# ------------------------------------------------- Tools/2 capture_verdict

def test_capture_verdict_total_failure_all_events_failed():
    """Mid-slate outage: every per-event fetch raised, nothing captured.
    Before the guard this wrote an empty run and exited 0 — the day's ONLY
    capture (quota policy) lost with zero alert."""
    marker, fatal = T2.capture_verdict(15, 15, 0)
    assert fatal
    assert "ODDS CAPTURE FAILED" in marker


def test_capture_verdict_zero_rows_clean_fetches_degrades_not_fatal():
    # every call returned 200 but nothing parsed into rows: either no
    # book posted a requested market yet, or the response format
    # drifted. Nothing was LOST, so this must alert (DEGRADED marker
    # -> check_health) without exit 1 killing the noon serve floor.
    marker, fatal = T2.capture_verdict(15, 0, 0)
    assert not fatal
    assert "ODDS CAPTURE DEGRADED" in marker


def test_capture_verdict_zero_rows_with_failures_is_fatal():
    # zero rows AND real fetch failures = the outage class: the day's
    # only capture is gone, alert hard
    marker, fatal = T2.capture_verdict(15, 3, 0)
    assert fatal and "ODDS CAPTURE FAILED" in marker


def test_capture_verdict_total_failure_even_with_bulk_rows():
    # zero events successfully captured is fatal even when the bulk
    # game-markets call salvaged some rows (main() still writes them
    # before exiting 1 — the alert must fire either way)
    marker, fatal = T2.capture_verdict(15, 15, 40)
    assert fatal and "ODDS CAPTURE FAILED" in marker


def test_capture_verdict_partial_failure_exact_degraded_marker():
    marker, fatal = T2.capture_verdict(15, 4, 1200)
    assert not fatal
    # exact substring contract: "ODDS CAPTURE DEGRADED <k>/<n> events failed"
    assert "ODDS CAPTURE DEGRADED 4/15 events failed" in marker


def test_capture_verdict_clean_run_no_marker():
    assert T2.capture_verdict(15, 0, 1200) == (None, False)


def test_capture_verdict_nothing_attempted_is_clean():
    # off-day (no events) or a rerun where every game already started
    # (pending empty): a no-op run, never a failure
    assert T2.capture_verdict(0, 0, 0) == (None, False)


# ---------------------------------------------- Tools/1 slate_floor_verdict

def test_slate_floor_selector_break_blocks_write():
    """0 scraped while StatsAPI schedules a slate = the silent
    selector-break mode; before the guard main() overwrote
    todays_games.json with {"games": []} and exited 0."""
    ok, msg = T1.slate_floor_verdict(0, 12)
    assert not ok
    assert "12" in msg and "todays_games.json" in msg


def test_slate_floor_offday_writes_empty():
    # schedule agrees there is nothing today: current behavior stands
    assert T1.slate_floor_verdict(0, 0) == (True, None)


def test_slate_floor_schedule_fetch_failure_fails_safe():
    # can't tell an off-day from a break -> do not overwrite, exit 1
    ok, msg = T1.slate_floor_verdict(0, None)
    assert not ok and msg


def test_slate_floor_short_slate_warns_nonfatal():
    ok, msg = T1.slate_floor_verdict(11, 15)
    assert ok
    assert "SLATE SHORT 11/15" in msg


def test_slate_floor_full_or_unverifiable_slate_is_clean():
    assert T1.slate_floor_verdict(15, 15) == (True, None)
    # scraped > scheduled (schedule undercount) never warns
    assert T1.slate_floor_verdict(15, 14) == (True, None)
    # a nonzero scrape stands on its own when the count is unknown
    assert T1.slate_floor_verdict(15, None) == (True, None)
