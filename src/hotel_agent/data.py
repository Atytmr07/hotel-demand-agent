"""Veri ajanının çekirdeği: rezervasyon kayıtlarını günlük doluluk serisine çevirir."""
from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "hotel_bookings.csv"
HOTELS = ("Resort Hotel", "City Hotel")


def load_bookings(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str)
        + "-"
        + df["arrival_date_month"]
        + "-"
        + df["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d",
    )
    df["nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    return df


EDGE_WARMUP_DAYS = 14  # veri başlangıcından önce varan konaklamalar eksik


def daily_occupancy(
    bookings: pd.DataFrame,
    hotel: str,
    include_canceled: bool = False,
    trim_edges: bool = True,
) -> pd.Series:
    """Günlük dolu oda sayısı (oda-gecesi). Varsayılan: iptaller ayıklanır.

    Her rezervasyon varış gününden itibaren `nights` gün boyunca bir oda tutar.
    Veri setinde kapasite bilgisi yok, bu yüzden çıktı oran değil mutlak sayıdır.
    `trim_edges`: başta ilk 14 gün (öncesi eksik) ve sonda son varış gününden
    sonrası (sonradan gelecek rezervasyonlar yok) kırpılır.
    """
    if hotel not in HOTELS:
        raise ValueError(f"hotel {HOTELS} içinden biri olmalı: {hotel!r}")
    df = bookings[bookings["hotel"] == hotel]
    if not include_canceled:
        df = df[df["is_canceled"] == 0]
    df = df[df["nights"] > 0]

    starts = df["arrival_date"]
    departures = starts + pd.to_timedelta(df["nights"], unit="D")
    idx = pd.date_range(starts.min(), departures.max() - pd.Timedelta(days=1), freq="D")

    # fark dizisi: varışta +1, ayrılışta -1, kümülatif toplam = o gün dolu oda
    arrivals = starts.value_counts()
    departs = departures.value_counts()
    delta = arrivals.sub(departs, fill_value=0).sort_index()
    occ = delta.cumsum().reindex(idx, method="ffill")
    if trim_edges:
        first = starts.min() + pd.Timedelta(days=EDGE_WARMUP_DAYS)
        occ = occ.loc[first : bookings[bookings["hotel"] == hotel]["arrival_date"].max()]
    occ.name = f"occupied_rooms[{hotel}]"
    return occ.astype(int)


def to_weekly(daily: pd.Series) -> pd.Series:
    return daily.resample("W").mean()
