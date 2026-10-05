import numpy as np
import pandas as pd

from hotel_agent.optimize import (
    Params,
    Scenarios,
    optimize_limits,
    realized_revenue,
    rule_limits,
)

P = Params(capacity=100, price=100.0, walk_cost=150.0)


def _idx(n=1):
    return pd.date_range("2017-08-01", periods=n)


def test_certain_show_rate_overbooks_to_capacity_over_q():
    # gelme oranı kesin 0,5 ve talep bol: limit tam olarak C/q = 200 olmalı
    sc = Scenarios(np.zeros(5), np.full(5, 0.5))
    lim = optimize_limits(pd.Series([10_000.0], index=_idx()), P, sc)
    assert lim.iloc[0] == 200


def test_no_overbooking_when_everyone_shows():
    sc = Scenarios(np.zeros(5), np.ones(5))
    lim = optimize_limits(pd.Series([10_000.0], index=_idx()), P, sc)
    assert lim.iloc[0] == 100


def test_uncertain_show_rate_overbooks_less_than_certain_mean():
    # q ∈ {0,4, 0,6}: ortalama 0,5 ama walk cezası yüzünden limit 200'ün altında kalmalı
    sc = Scenarios(np.zeros(4), np.array([0.4, 0.6, 0.4, 0.6]))
    lim = optimize_limits(pd.Series([10_000.0], index=_idx()), P, sc)
    assert 100 < lim.iloc[0] < 200


def test_realized_revenue_walks_guests_above_capacity():
    idx = _idx(2)
    limits = pd.Series([150, 150], index=idx)
    gross = pd.Series([200, 80], index=idx)
    show = pd.Series([1.0, 1.0], index=idx)
    r = realized_revenue(limits, gross, show, P)
    # gün1: kabul 150, gelen 150 -> 100 satış, 50 walk
    assert r.loc[idx[0], "revenue"] == 100 * 100 - 150 * 50
    # gün2: talep 80 < limit -> 80 satış
    assert r.loc[idx[1], "revenue"] == 80 * 100


def test_rule_limits_adds_ten_percent():
    assert rule_limits(_idx(), P).iloc[0] == 110
