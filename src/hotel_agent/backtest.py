"""Geriye dönük kıyas: MILP limitleri, sabit %10 kural ve overbooking'siz politika, gerçekleşen gelire göre.

Her fold'da yalnızca kesim noktasına kadarki veri kullanılır (tahmin, hata dağılımı, gelme oranı).
Model seçimi, kıyas penceresinden önceki veriyle yapılır (sızıntıyı önlemek için).
"""
import numpy as np
import pandas as pd

from hotel_agent.data import daily_occupancy, load_bookings
from hotel_agent.forecast import HORIZON, forecast, select_model, validation_errors
from hotel_agent.optimize import (
    Params,
    make_scenarios,
    no_overbooking_limits,
    optimize_limits,
    realized_revenue,
    rule_limits,
    show_rate_limits,
)

CAPACITY_QUANTILE = 0.9  # varsayım: kapasite, net doluluğun %90'lık dilimi


def default_params(bookings: pd.DataFrame, hotel: str, net: pd.Series, walk_multiplier: float = 1.5) -> Params:
    stays = bookings[(bookings["hotel"] == hotel) & (bookings["is_canceled"] == 0) & (bookings["adr"] > 0)]
    price = float(stays["adr"].mean())
    return Params(
        capacity=int(np.ceil(net.quantile(CAPACITY_QUANTILE))),
        price=price,
        walk_cost=walk_multiplier * price,
    )


def run_backtest(hotel: str, n_folds: int = 4, horizon: int = HORIZON, params: Params | None = None):
    bookings = load_bookings()
    net = daily_occupancy(bookings, hotel)
    gross = daily_occupancy(bookings, hotel, include_canceled=True).reindex(net.index)
    show_rate = (net / gross.clip(lower=1)).clip(upper=1)
    params = params or default_params(bookings, hotel, net)

    first_cut = len(gross) - horizon * n_folds
    model = select_model(gross.iloc[:first_cut], horizon, n_folds=2).best

    rows = []
    for k in range(n_folds):
        cut = first_cut + k * horizon
        train = gross.iloc[:cut]
        test_idx = gross.index[cut : cut + horizon]
        pred = forecast(train, model, horizon).clip(lower=0)
        errs = validation_errors(train, model, horizon, n_folds=3)
        scen = make_scenarios(errs, show_rate.iloc[:cut].to_numpy(), seed=k)
        limits = {
            "optimized": optimize_limits(pred, params, scen),
            "rule_show_rate": show_rate_limits(test_idx, params, float(show_rate.iloc[:cut].mean())),
            "rule_10pct": rule_limits(test_idx, params),
            "no_overbooking": no_overbooking_limits(test_idx, params),
        }
        for name, lim in limits.items():
            r = realized_revenue(lim, gross, show_rate, params)
            r["policy"], r["fold"], r["limit"] = name, k, lim
            rows.append(r)
    out = pd.concat(rows)
    summary = out.groupby("policy").agg(
        revenue=("revenue", "sum"), walked=("walked", "sum"), accepted=("accepted", "sum")
    )
    summary["revenue_vs_no_overbooking_%"] = (
        100 * (summary["revenue"] / summary.loc["no_overbooking", "revenue"] - 1)
    )
    return model, params, out, summary.sort_values("revenue", ascending=False)


if __name__ == "__main__":
    for hotel in ("Resort Hotel", "City Hotel"):
        model, params, _, summary = run_backtest(hotel)
        print(f"\n== {hotel} | model={model} | {params}")
        print(summary.round(1).to_string())
