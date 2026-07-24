"""Degraded-serve banner (predict._status_sheet): a missing-calibrator
serve must be unmissable — red sheet at index 0, opened active; a clean
serve must add nothing.
"""
from openpyxl import Workbook

import predict as PR


def test_clean_serve_adds_no_sheet():
    wb = Workbook()
    wb.active.title = "Batter Props"
    PR._status_sheet(wb, [], "2026-07-24")
    assert wb.sheetnames == ["Batter Props"]


def test_degraded_serve_banner_first_and_active():
    wb = Workbook()
    wb.active.title = "Batter Props"
    wb.create_sheet("Bets")
    PR._status_sheet(wb, ["output_calibrators.joblib MISSING — serving "
                          "RAW probabilities"], "2026-07-24")
    assert wb.sheetnames[0] == "!! STATUS"
    assert wb.active.title == "!! STATUS"
    v = wb["!! STATUS"].cell(row=1, column=1).value
    assert "DEGRADED SERVE 2026-07-24" in v and "RAW probabilities" in v
    # existing sheets untouched
    assert "Batter Props" in wb.sheetnames and "Bets" in wb.sheetnames
