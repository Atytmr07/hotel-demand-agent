import pandas as pd

from hotel_agent.data import daily_occupancy


def _bookings():
    return pd.DataFrame(
        {
            "hotel": ["Resort Hotel"] * 3,
            "is_canceled": [0, 0, 1],
            "arrival_date": pd.to_datetime(["2016-01-01", "2016-01-02", "2016-01-01"]),
            "nights": [2, 2, 5],
        }
    )


def test_occupancy_counts_overlapping_stays_and_drops_cancellations():
    occ = daily_occupancy(_bookings(), "Resort Hotel", trim_edges=False)
    # 1 Oca: A; 2 Oca: A+B; 3 Oca: B; iptal edilen kayıt sayılmaz
    assert occ.loc["2016-01-01"] == 1
    assert occ.loc["2016-01-02"] == 2
    assert occ.loc["2016-01-03"] == 1
    assert len(occ) == 3


def test_include_canceled_adds_canceled_stay():
    occ = daily_occupancy(_bookings(), "Resort Hotel", include_canceled=True, trim_edges=False)
    assert occ.loc["2016-01-01"] == 2


def test_trim_edges_drops_warmup_days():
    b = _bookings()
    b["arrival_date"] = pd.to_datetime(["2016-01-01", "2016-02-01", "2016-03-01"])
    occ = daily_occupancy(b, "Resort Hotel")
    assert occ.index.min() == pd.Timestamp("2016-01-15")
    # üçüncü kayıt iptal; son dolu gün ikinci konaklamanın son gecesi
    assert occ.index.max() == pd.Timestamp("2016-02-02")
