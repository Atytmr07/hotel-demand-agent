"""Tahmin ajanının çekirdeği: aday modelleri zaman bölmeli validasyonla kıyaslar, en iyisini seçer.

Model seçimini LLM değil bu kod yapar; LLM yalnızca `ModelSelection.summary()` çıktısını yorumlar.
Tüm modeller aynı h günlük ufuk için doğrudan (recursive olmayan) tahmin üretir.
"""
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

HORIZON = 28
N_FOLDS = 4


def mape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs((y - yhat) / np.maximum(y, 1))) * 100)


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)))


# ---- özellikler: h günden yeni bilgi kullanılmaz, böylece ufuk boyunca sızıntı olmaz ----
def make_features(y: pd.Series, h: int = HORIZON) -> pd.DataFrame:
    lagged = y.shift(h)
    X = pd.DataFrame(index=y.index)
    X["lag_h"] = lagged
    X["lag_h7"] = y.shift(h + 7)
    X["lag_h14"] = y.shift(h + 14)
    X["roll7_h"] = lagged.rolling(7).mean()
    X["roll28_h"] = lagged.rolling(28).mean()
    X["dow"] = y.index.dayofweek
    doy = y.index.dayofyear
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return X


# ---- aday modeller: (train serisi, test indeksi) -> tahmin dizisi ----
def _seasonal_naive(train: pd.Series, test_idx: pd.DatetimeIndex) -> np.ndarray:
    # h günlük ufukta 7 günlük geri adım: son 7 günün tekrarı
    last_week = train.iloc[-7:].to_numpy()
    return np.resize(last_week, len(test_idx)).astype(float)


def _sarimax(order, seasonal):
    def fit_predict(train: pd.Series, test_idx: pd.DatetimeIndex) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SARIMAX(
                train.asfreq("D"),
                order=order,
                seasonal_order=seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=100)
            return res.forecast(len(test_idx)).to_numpy()

    return fit_predict


def _tabular(make_model):
    def fit_predict(train: pd.Series, test_idx: pd.DatetimeIndex) -> np.ndarray:
        # train + test günlerini içeren özellikler; test satırları yalnızca eski y'yi görür
        full_idx = train.index.append(test_idx)
        y_full = train.reindex(full_idx)  # test günleri NaN, lag_h ile en fazla h gün ileri gidilir
        X = make_features(y_full)
        Xtr = X.loc[train.index].dropna()
        model = make_model()
        model.fit(Xtr, train.loc[Xtr.index])
        return model.predict(X.loc[test_idx].ffill())

    return fit_predict


CANDIDATES = {
    "seasonal_naive": _seasonal_naive,
    "ARIMA(2,1,2)": _sarimax((2, 1, 2), (0, 0, 0, 0)),
    "SARIMA(1,0,1)x(1,1,1,7)": _sarimax((1, 0, 1), (1, 1, 1, 7)),
    "RandomForest": _tabular(
        lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1)
    ),
    "XGBoost": _tabular(
        lambda: XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0)
    ),
}


@dataclass
class ModelSelection:
    scores: pd.DataFrame  # index=model, kolonlar: MAPE, RMSE (fold ortalaması)
    best: str
    horizon: int
    n_folds: int
    errors: dict = field(default_factory=dict)

    def summary(self) -> str:
        t = self.scores.round(2).to_string()
        return (
            f"Ufuk {self.horizon} gün, {self.n_folds} katlı ileri-yönlü validasyon.\n{t}\n"
            f"Seçilen model (en düşük MAPE): {self.best}"
        )


def select_model(
    y: pd.Series, horizon: int = HORIZON, n_folds: int = N_FOLDS, candidates=None
) -> ModelSelection:
    """Genişleyen pencere: her fold'da kesim noktasına kadar eğit, sonraki `horizon` günü tahmin et."""
    candidates = candidates or CANDIDATES
    cutoffs = [len(y) - horizon * k for k in range(n_folds, 0, -1)]
    rows, errors = {}, {}
    for name, fn in candidates.items():
        m, r = [], []
        for c in cutoffs:
            train, test = y.iloc[:c], y.iloc[c : c + horizon]
            try:
                pred = fn(train, test.index)
            except Exception as e:  # tek model hatası seçimi durdurmamalı
                errors[name] = repr(e)
                break
            m.append(mape(test, pred))
            r.append(rmse(test, pred))
        else:
            rows[name] = {"MAPE": np.mean(m), "RMSE": np.mean(r)}
    scores = pd.DataFrame(rows).T.sort_values("MAPE")
    return ModelSelection(scores, scores.index[0], horizon, n_folds, errors)


def validation_errors(
    y: pd.Series, model_name: str, horizon: int = HORIZON, n_folds: int = N_FOLDS
) -> np.ndarray:
    """Seçilen modelin ileri-yönlü validasyondaki göreli hataları (gerçek/tahmin - 1).

    Optimizasyon ajanı bunları talep belirsizliği senaryoları olarak kullanır.
    """
    fn = CANDIDATES[model_name]
    errs = []
    for c in [len(y) - horizon * k for k in range(n_folds, 0, -1)]:
        train, test = y.iloc[:c], y.iloc[c : c + horizon]
        pred = np.maximum(fn(train, test.index), 1)
        errs.append(test.to_numpy() / pred - 1)
    return np.concatenate(errs)


def forecast(y: pd.Series, model_name: str, horizon: int = HORIZON) -> pd.Series:
    """Seçilen modeli tüm seriyle eğitip y'nin bitişinden sonraki `horizon` günü tahmin eder."""
    future = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    pred = CANDIDATES[model_name](y, future)
    return pd.Series(pred, index=future, name="forecast")
