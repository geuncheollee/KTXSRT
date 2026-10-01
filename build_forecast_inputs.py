"""Reconstruct and verify the fixed-origin input matrices using the original rule."""
from __future__ import annotations
import argparse
import calendar
import csv
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def make_inputs(panel: dict[str, dict[str, str]], ledger: list[dict[str, str]],
                target_year: int, rolling: bool) -> list[dict]:
    rows = []
    for month in range(1, 13):
        target = f"{target_year}-{month:02d}-01"
        if rolling:
            origin = date(target_year, month, 1) - timedelta(days=1)
        else:
            origin = date(target_year - 1, 12, 31)
        origin_iso = origin.isoformat()
        prior = f"{target_year - 1}-{month:02d}-01"
        assert prior in panel and target in panel
        assert date.fromisoformat(prior) <= origin
        realised_wh = int(panel[target]["weekday_holidays"])
        omitted = [e for e in ledger if e["holiday_date"].startswith(target[:7])
                   and e["announcement_date"] > origin_iso]
        for event in omitted:
            assert date.fromisoformat(event["holiday_date"]).weekday() < 5
        available_wh = realised_wh - len(omitted)
        assert 0 <= available_wh <= int(panel[target]["weekday_count"])
        rows.append({
            "origin": origin_iso,
            "target_month": target,
            "horizon": 1 if rolling else month,
            "y_lag12": int(panel[prior]["hsr_pax_total"]),
            "lag12_source_month": prior,
            "month_days": calendar.monthrange(target_year, month)[1],
            "weekday_holidays_asof": available_wh,
            "later_announced_holiday_dates": "|".join(e["holiday_date"] for e in omitted),
            "lag_data_vintage": "retrospective_2026_final",
        })
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Also write to outputs/inputs/.")
    args = parser.parse_args()
    panel = {r["date"]: r for r in read(ROOT / "revision/data/monthly_panel_revision_v1.csv")}
    ledger = read(ROOT / "revision/forecast_protocol/holiday_announcement_ledger.csv")
    for year in (2023, 2024, 2025):
        rows = make_inputs(panel, ledger, year, False)
        relative = (f"revision/forecast_2024/origin_inputs_{year}_fixed_h1_12.csv" if year == 2023
                    else f"revision/forecast_protocol/origin_inputs_{year}_fixed_h1_12.csv")
        expected = read(ROOT / relative)
        assert [{k: str(v) for k, v in row.items()} for row in rows] == expected, year
        if args.write:
            output = ROOT / "outputs/inputs"
            output.mkdir(parents=True, exist_ok=True)
            with (output / f"origin_inputs_{year}_fixed_h1_12.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        print(f"PASS: {year} fixed-origin matrix, 12 months; announcement rule reproduced.", flush=True)

if __name__ == "__main__":
    main()
