"""Duyarlılık analizi: varsayımlar (walk maliyeti, kapasite) değişince MILP, gelme oranı kuralına göre nasıl kalıyor?

Çıktı: reports/sensitivity.csv ve ekrana özet tablo. Çalıştır: python -m hotel_agent.sensitivity
"""
from pathlib import Path

import pandas as pd

from hotel_agent.backtest import default_params, prepare, run_backtest
from hotel_agent.optimize import Params

WALK_MULTIPLIERS = (0.5, 1.0, 1.5, 3.0, 5.0, 10.0)  # walk maliyeti = çarpan x ADR
CAPACITY_QUANTILES = (0.8, 0.9, 0.95)
OUT = Path(__file__).resolve().parents[2] / "reports" / "sensitivity.csv"


def run(hotels=("Resort Hotel", "City Hotel")) -> pd.DataFrame:
    rows = []
    for hotel in hotels:
        prep = prepare(hotel)
        base = default_params(prep["bookings"], hotel, prep["net"])
        for q in CAPACITY_QUANTILES:
            cap = int(prep["net"].quantile(q).__ceil__())
            for m in WALK_MULTIPLIERS:
                params = Params(capacity=cap, price=base.price, walk_cost=m * base.price)
                _, _, _, s = run_backtest(hotel, params=params, prepared=prep)
                rev = s["revenue"]
                rows.append(
                    {
                        "hotel": hotel,
                        "capacity_q": q,
                        "capacity": cap,
                        "walk_x_adr": m,
                        "rev_milp": rev["optimized"],
                        "rev_rule_show_rate": rev["rule_show_rate"],
                        "milp_vs_rule_%": 100 * (rev["optimized"] / rev["rule_show_rate"] - 1),
                        "walked_milp": s.loc["optimized", "walked"],
                        "walked_rule": s.loc["rule_show_rate", "walked"],
                    }
                )
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    return df


if __name__ == "__main__":
    df = run()
    for hotel, g in df.groupby("hotel"):
        print(f"\n== {hotel}: MILP geliri, gelme oranı kuralına göre (%)")
        print(g.pivot(index="walk_x_adr", columns="capacity_q", values="milp_vs_rule_%").round(1).to_string())
        print("walk sayısı (MILP / kural), kapasite q=0.9:")
        print(g[g.capacity_q == 0.9][["walk_x_adr", "walked_milp", "walked_rule"]].round(0).to_string(index=False))
