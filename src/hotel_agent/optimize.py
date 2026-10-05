"""Optimizasyon ajanının çekirdeği: günlük rezervasyon limitini (overbooking) MILP ile seçer.

Her gün t için tamsayı limit L_t seçilir. Senaryo s'de (talep hatası e_s, gelme oranı q_s):
  kabul edilen   a_s = min(L, G_t * (1 + e_s))      (G_t: tahmin edilen brüt rezervasyon talebi)
  gelen misafir  S_s = a_s * q_s
  satılan oda    sold_s = min(S_s, C),  yürütülen (walk) walked_s = max(S_s - C, 0)
Amaç: ortalama (price * sold - walk_cost * walked) enbüyükleme.
min() ifadesi senaryo başına ikili değişkenle (big-M) tam olarak modellenir.

Varsayımlar (kapasite, fiyat, walk maliyeti) veri setinde yoktur, `Params` ile açıkça verilir.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pulp


@dataclass(frozen=True)
class Params:
    capacity: int  # oda sayısı (varsayım)
    price: float  # oda-gecesi geliri (ADR'den)
    walk_cost: float  # overbooking'de misafiri başka otele gönderme maliyeti (oda-gecesi başına)


@dataclass
class Scenarios:
    demand_err: np.ndarray  # göreli talep hatası e_s
    show_rate: np.ndarray  # gelme oranı q_s (1 - iptal oranı)


def make_scenarios(
    demand_errors: np.ndarray, show_rates: np.ndarray, n: int = 40, seed: int = 0
) -> Scenarios:
    """Geçmiş hata ve gelme oranı gözlemlerinden n eşleşmiş senaryo çeker."""
    rng = np.random.default_rng(seed)
    e = rng.choice(np.asarray(demand_errors, float), size=n)
    q = rng.choice(np.asarray(show_rates, float), size=n)
    return Scenarios(np.clip(e, -0.9, 2.0), np.clip(q, 0.05, 1.0))


def optimize_limits(
    gross_forecast: pd.Series, params: Params, scenarios: Scenarios, time_limit: int = 60
) -> pd.Series:
    """Her gün için rezervasyon limitini döndürür (tamsayı)."""
    C = params.capacity
    S = len(scenarios.demand_err)
    M = 2 * C  # L <= 2C olduğundan talebi 2C'de kırpmak min(L, G) sonucunu değiştirmez
    prob = pulp.LpProblem("overbooking", pulp.LpMaximize)

    L = {t: pulp.LpVariable(f"L_{i}", 0, 2 * C, cat="Integer") for i, t in enumerate(gross_forecast.index)}
    obj = []
    for i, (t, g) in enumerate(gross_forecast.items()):
        for s in range(S):
            G_s = min(float(g) * (1 + scenarios.demand_err[s]), M)
            q = scenarios.show_rate[s]
            a = pulp.LpVariable(f"a_{i}_{s}", 0, M)
            z = pulp.LpVariable(f"z_{i}_{s}", cat="Binary")  # 1: limit bağlayıcı (a = L)
            sold = pulp.LpVariable(f"sold_{i}_{s}", 0, C)
            walked = pulp.LpVariable(f"walk_{i}_{s}", 0)
            # a = min(L, G_s)
            prob += a <= L[t]
            prob += a <= G_s
            prob += a >= L[t] - M * (1 - z)
            prob += a >= G_s - M * z
            # sold = min(a*q, C), walked = max(a*q - C, 0)
            prob += sold <= a * q
            prob += walked >= a * q - C
            obj.append(params.price * sold - params.walk_cost * walked)
    prob += pulp.lpSum(obj) / S
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"MILP optimal değil: {pulp.LpStatus[prob.status]}")
    return pd.Series({t: int(round(L[t].value())) for t in gross_forecast.index}, name="limit")


# ---- karşılaştırma politikaları ----
def no_overbooking_limits(index: pd.DatetimeIndex, params: Params) -> pd.Series:
    return pd.Series(params.capacity, index=index, name="limit")


def rule_limits(index: pd.DatetimeIndex, params: Params, pct: float = 0.10) -> pd.Series:
    return pd.Series(int(round(params.capacity * (1 + pct))), index=index, name="limit")


def show_rate_limits(index: pd.DatetimeIndex, params: Params, mean_show_rate: float) -> pd.Series:
    """Sektörde yaygın kural: limit = kapasite / ortalama gelme oranı (her gün aynı)."""
    return pd.Series(int(round(params.capacity / mean_show_rate)), index=index, name="limit")


def realized_revenue(
    limits: pd.Series, gross_actual: pd.Series, show_rate_actual: pd.Series, params: Params
) -> pd.DataFrame:
    """Gerçekleşen günlük gelir: kabul = min(limit, gerçek talep), gelen = kabul * gerçek gelme oranı."""
    accepted = np.minimum(limits, gross_actual.loc[limits.index])
    shows = accepted * show_rate_actual.loc[limits.index]
    sold = np.minimum(shows, params.capacity)
    walked = np.maximum(shows - params.capacity, 0)
    return pd.DataFrame(
        {
            "accepted": accepted,
            "shows": shows,
            "walked": walked,
            "revenue": params.price * sold - params.walk_cost * walked,
        }
    )
