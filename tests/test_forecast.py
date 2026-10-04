import numpy as np
import pandas as pd

from hotel_agent.forecast import forecast, make_features, select_model, CANDIDATES


def _series(n=300):
    idx = pd.date_range("2016-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    return pd.Series(100 + 20 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(0, 2, n), index=idx)


def test_features_use_no_data_newer_than_horizon():
    y = _series()
    X = make_features(y, h=28)
    t = y.index[100]
    assert X.loc[t, "lag_h"] == y.loc[t - pd.Timedelta(days=28)]
    # y'nin son 27 günü değişse bile özellikler değişmemeli
    y2 = y.copy()
    y2.loc[t - pd.Timedelta(days=27) : t] += 1000
    assert make_features(y2, h=28).loc[t].equals(X.loc[t])


def test_select_model_prefers_seasonal_models_on_weekly_pattern():
    y = _series()
    cands = {k: CANDIDATES[k] for k in ("seasonal_naive", "XGBoost")}
    sel = select_model(y, horizon=28, n_folds=2, candidates=cands)
    assert sel.best in cands
    assert sel.scores.loc[sel.best, "MAPE"] < 10


def test_forecast_returns_horizon_days_after_series_end():
    y = _series()
    f = forecast(y, "seasonal_naive", horizon=14)
    assert len(f) == 14
    assert f.index[0] == y.index[-1] + pd.Timedelta(days=1)
