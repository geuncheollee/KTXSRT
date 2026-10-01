"""Recalculate the current manuscript Tables 2-4 from 72 monthly forecast files.

The calculations match the actual 1 October aggregate_and_verify.py experiment.
Only file discovery, table export and a portable CLI have been added.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS = {
    "M1_seasonal_naive": "Seasonal-naïve",
    "M2_holt_winters": "Holt–Winters",
    "M3_sarima": "SARIMA",
    "M4_sarimax_3": "SARIMAX",
    "M5_prophet": "Prophet",
    "M6_xgboost": "XGBoost",
    "M7_lstm": "LSTM",
    "M8_transformer": "Transformer",
    "M9_sarimax_lstm_residual": "SARIMAX–LSTM",
}
NEURAL = set(list(MODELS)[6:])

def calculate(predictions: Path) -> pd.DataFrame:
    panel = pd.read_csv(ROOT / "revision/data/monthly_panel_revision_v1.csv")
    truth = panel.set_index("date").hsr_pax_total
    rows = []
    count = 0
    for model in MODELS:
        period_scores = {p: [] for p in ("2024", "2025", "overall")}
        seeds = range(42, 52) if model in NEURAL else [42]
        for seed in seeds:
            pieces = []
            for year in (2024, 2025):
                path = predictions / str(year) / f"{model}_seed{seed}.csv"
                frame = pd.read_csv(path)
                expected_dates = [f"{year}-{month:02d}-01" for month in range(1, 13)]
                assert len(frame) == 12 and list(frame.target_month) == expected_dates, path
                np.testing.assert_array_equal(frame.y_true, truth.loc[frame.target_month])
                assert np.isfinite(frame.y_pred).all(), path
                pieces.append(frame)
                count += 1
            both = pd.concat(pieces, ignore_index=True)
            for period, frame in (("2024", pieces[0]), ("2025", pieces[1]), ("overall", both)):
                error = frame.y_true.to_numpy() - frame.y_pred.to_numpy()
                period_scores[period].append([
                    float(np.mean(np.abs(error) / frame.y_true.to_numpy()) * 100),
                    float(np.mean(np.abs(error))),
                    float(np.sqrt(np.mean(error**2))),
                ])
        row = {"model": model, "seeds": len(seeds)}
        for period, values in period_scores.items():
            mean = np.asarray(values).mean(axis=0)
            row.update({f"{period}_{metric}": float(mean[k]) for k, metric in enumerate(("MAPE", "MAE", "RMSE"))})
        rows.append(row)
    assert count == 72
    return pd.DataFrame(rows)

def export_tables(scores: pd.DataFrame, output: Path, verify: bool = False) -> int:
    output.mkdir(parents=True, exist_ok=True)
    verified = 0
    for number, metric in ((2, "MAPE"), (3, "MAE"), (4, "RMSE")):
        ordered = scores.sort_values(f"overall_{metric}", kind="stable")
        table = pd.DataFrame({"Model": [MODELS[m] for m in ordered.model]})
        for period, label in (("2024", "2024"), ("2025", "2025"), ("overall", "All 24 months")):
            fmt = ".4f" if metric == "MAPE" else ",.0f"
            table[label] = [format(v, fmt) for v in ordered[f"{period}_{metric}"]]
        if verify:
            expected = pd.read_csv(ROOT / f"reference/table{number}_{metric}.csv", dtype=str, keep_default_na=False)
            pd.testing.assert_frame_equal(table, expected)
            verified += 27
        table.to_csv(output / f"table{number}_{metric}.csv", index=False, encoding="utf-8-sig")
        print(f"Table {number}: {metric}, sorted by overall error", flush=True)
        print(table.to_string(index=False), flush=True)
    scores.to_csv(output / "all_scores_unrounded.csv", index=False, encoding="utf-8-sig")
    return verified

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "refit_outputs/predictions")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--reference-dir", type=Path, help="Optional private QA tables; not required for reproduction.")
    args = parser.parse_args()
    scores = calculate(args.predictions)
    if args.reference_dir:
        expected = pd.read_csv(args.reference_dir / "model_scores.csv")
        keys = [f"{p}_{m}" for p in ("2024", "2025", "overall") for m in ("MAPE", "MAE", "RMSE")]
        np.testing.assert_allclose(scores.set_index("model")[keys], expected.set_index("model").loc[scores.model, keys], rtol=0, atol=1e-7)
    count = export_tables(scores, args.output, verify=False)
    if args.reference_dir:
        for number, metric in ((2, "MAPE"), (3, "MAE"), (4, "RMSE")):
            actual = pd.read_csv(args.output / f"table{number}_{metric}.csv", dtype=str, keep_default_na=False)
            expected = pd.read_csv(args.reference_dir / f"table{number}_{metric}.csv", dtype=str, keep_default_na=False)
            pd.testing.assert_frame_equal(actual, expected)
            count += 27
    summary = {"models": 9, "forecast_files": 72, "target_months": 24, "displayed_values_verified": count, "neural_aggregation": "Mean of ten seed-specific period scores, not metrics of the mean forecast path", "panel_sha256": hashlib.sha256((ROOT / "revision/data/monthly_panel_revision_v1.csv").read_bytes()).hexdigest()}
    (args.output / "verification.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Computed Tables 2-4 from 72 forecast files. Optional reference checks: {count}/81.", flush=True)

if __name__ == "__main__":
    main()
