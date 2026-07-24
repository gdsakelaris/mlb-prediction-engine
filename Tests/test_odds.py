"""Characterization tests for Model/odds.py: american-odds conversion,
proportional de-vig, and the Pinnacle-preferring sharp_fair consensus."""
import math

import odds as O


def _row(book, over, under):
    """A store-schema row (only OverPrice/UnderPrice/Book are read)."""
    return dict(Date="2026-07-22", GamePk=777, Team="NYY",
                PlayerId=592450, PlayerName="Aaron Judge",
                Market="batter_hits", Line=0.5, OverPrice=over,
                UnderPrice=under, Book=book,
                CapturedAt="2026-07-22T15:00:00Z")


# ------------------------------------------------------ american_to_prob

def test_american_to_prob_table():
    assert math.isclose(O.american_to_prob(-110), 110 / 210)   # ~0.52381
    assert math.isclose(O.american_to_prob(150), 0.40)
    assert math.isclose(O.american_to_prob(100), 0.50)
    assert math.isclose(O.american_to_prob(-200), 2 / 3)
    # numeric strings coerce (store round-trips through csv)
    assert math.isclose(O.american_to_prob("-110"), 110 / 210)


def test_american_to_prob_edges_all_none():
    for bad in (None, "", "abc", 0, 0.0, "0",
                float("nan"), float("inf"), float("-inf")):
        assert O.american_to_prob(bad) is None


# --------------------------------------------------------------- no_vig

def test_no_vig_even_prices():
    fo, fu = O.no_vig(-110, -110)
    assert math.isclose(fo, 0.5) and math.isclose(fu, 0.5)


def test_no_vig_proportional_and_sums_to_one():
    fo, fu = O.no_vig(100, -120)
    po, pu = 0.5, 120 / 220
    assert math.isclose(fo, po / (po + pu))
    assert math.isclose(fo + fu, 1.0)


def test_no_vig_one_sided_none():
    assert O.no_vig(-110, None) is None
    assert O.no_vig(None, -110) is None
    assert O.no_vig("", "") is None


# ----------------------------------------------------------- sharp_fair

def test_sharp_fair_pinnacle_wins_outright():
    rows = [_row("draftkings", -150, 130),    # fair over ~0.5798
            _row("pinnacle", -110, -110),     # fair over 0.5
            _row("fanduel", -200, 170)]
    assert math.isclose(O.sharp_fair(rows), 0.5)


def test_sharp_fair_pinnacle_book_case_insensitive():
    rows = [_row("Pinnacle", -110, -110), _row("draftkings", -150, 130)]
    assert math.isclose(O.sharp_fair(rows), 0.5)


def test_sharp_fair_median_even_book_count():
    # no pinnacle -> mean of the two middle (here: only) fair probs
    rows = [_row("draftkings", -110, -110),   # 0.5
            _row("fanduel", 100, -120)]       # 0.5/(0.5+120/220)
    f2 = 0.5 / (0.5 + 120 / 220)
    assert math.isclose(O.sharp_fair(rows), (0.5 + f2) / 2)


def test_sharp_fair_median_odd_book_count():
    rows = [_row("draftkings", -110, -110),   # 0.5
            _row("fanduel", 100, -120),       # ~0.4783
            _row("betmgm", -150, 130)]        # ~0.5798
    assert math.isclose(O.sharp_fair(rows), 0.5)


def test_sharp_fair_one_sided_pinnacle_falls_back_to_books():
    rows = [_row("pinnacle", -110, None),     # one-sided: excluded
            _row("draftkings", -110, -110)]
    assert math.isclose(O.sharp_fair(rows), 0.5)


def test_sharp_fair_all_one_sided_none():
    rows = [_row("pinnacle", -110, None), _row("draftkings", None, -110),
            _row("fanduel", "", "")]
    assert O.sharp_fair(rows) is None
