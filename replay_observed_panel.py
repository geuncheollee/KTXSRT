"""Replay only the revised observed panel from archived public sources.

Run from the root of PPTE45089_reproducibility_20260929.zip after extraction.
No old SRT source, source comparison, or imputation is used. The archived
panel is checked first; --write writes a separate replay CSV, never the input.
"""
from __future__ import annotations
import argparse
import calendar
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(relative: str, encoding: str = "utf-8-sig") -> list[dict[str, str]]:
    with (ROOT / relative).open(encoding=encoding, newline="") as handle:
        return list(csv.DictReader(handle))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    expected = read("revision/data/monthly_panel_revision_v1.csv")
    ktx = {
        r["yyyymm"] + "-01": sum(int(r[c]) for c in
            ("KTX_pax", "KTX-산천_pax", "KTX-호남_pax", "KTX-이음_pax", "KTX-청룡_pax"))
        for r in read("data/01_passenger_rail/kric_carkind_raw.csv")
    }
    sr = read("revision/data/srt_2025_20260924/srt_monthly_15071484_20251231.csv", "cp949")
    srt = {}
    for r in sr:
        month = int(r["운행월"])
        for year in range(2021, 2026):
            total = int(r[f"{year}년 전체"])
            routes = [int(v) for k, v in r.items()
                      if k.startswith(f"{year}년 ") and not k.endswith("전체") and v.strip()]
            assert sum(routes) == total
            srt[f"{year}-{month:02d}-01"] = total
    def keyed(relative):
        return {r["date"]: r for r in read(relative)}
    cal = keyed("data/04_holiday_calendar/holiday_monthly.csv")
    fares = keyed("data/05_fare_index/fare_monthly.csv")
    oil = keyed("data/02_oil_price/dubai_monthly.csv")
    deduplicate = {"2023-10-01", "2024-10-01", "2025-01-01"}
    replay = []
    for year in range(2021, 2026):
        for month in range(1, 13):
            date = f"{year}-{month:02d}-01"
            lag_date = f"{year-1}-{month:02d}-01"
            c = cal[date]
            assert int(c["month_days"]) == calendar.monthrange(year, month)[1]
            lag = ktx[lag_date] + srt[lag_date] if year >= 2022 else ""
            correction = int(date in deduplicate)
            r = {
                "date": date, "year": year, "month": month,
                "ktx_pax_total": ktx[date], "srt_pax_observed": srt[date],
                "hsr_pax_total": ktx[date] + srt[date], "y_lag12": lag,
                "lag12_date": lag_date if lag != "" else "",
                "month_days": int(c["month_days"]),
                "weekday_holidays": int(c["weekday_holidays"]) - correction,
                "weekday_holidays_archived": int(c["weekday_holidays"]),
                "holiday_duplicate_correction": correction,
                "weekday_count": int(c["weekday_count"]),
                "weekend_count": int(c["weekend_count"]),
                "holiday_lunar_ny_days": int(c["holiday_lunar_ny_days"]),
                "holiday_chuseok_days": int(c["holiday_chuseok_days"]),
                "bus_fare_idx": fares[date]["bus_fare_idx"],
                "air_fare_idx": fares[date]["air_fare_idx"],
                "rail_fare_idx": fares[date]["rail_fare_idx"],
                "oil_dubai_usd": oil[date]["oil_dubai_usd"],
                "ktx_source": "KRIC_carkind_five_class_sum",
                "srt_source_dataset_id": "15071484", "srt_source_version": "20251231",
                "srt_observation_status": "observed_published_total",
            }
            replay.append(r)
    assert len(replay) == len(expected) == 60
    for a, b in zip(replay, expected):
        assert a.keys() == b.keys()
        for key in a:
            assert str(a[key]) == b[key], (a["date"], key, a[key], b[key])
    assert all(r["y_lag12"] != "" for r in replay[12:])
    assert all(r["srt_observation_status"] == "observed_published_total" for r in replay)
    if args.write:
        with (ROOT / "monthly_panel_replayed.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(replay[0]))
            writer.writeheader()
            writer.writerows(replay)
    print("PASS: 60 observed source months; 48 observed analysis targets and lags; no SRT imputation")
    print("Archived panel SHA-256:", hashlib.sha256((ROOT / "revision/data/monthly_panel_revision_v1.csv").read_bytes()).hexdigest())

if __name__ == "__main__":
    main()
