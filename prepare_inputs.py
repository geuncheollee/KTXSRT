"""Prepare local reproduction inputs from the two journal supplementary ZIPs.

The public repository contains code only. No input panels, selected settings or
forecast results are downloaded from GitHub or embedded in this script.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
MODELS = {
    "M1_seasonal_naive": "Seasonal-naïve", "M2_holt_winters": "Holt–Winters",
    "M3_sarima": "SARIMA", "M4_sarimax_3": "SARIMAX", "M5_prophet": "Prophet",
    "M6_xgboost": "XGBoost", "M7_lstm": "LSTM", "M8_transformer": "Transformer",
    "M9_sarimax_lstm_residual": "SARIMAX–LSTM",
}
NEURAL = set(list(MODELS)[6:])

def write(relative, data):
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-archive", type=Path, required=True, help="PPTE45089_reproducibility_20260929.zip supplied with the submission")
    parser.add_argument("--aic-archive", type=Path, required=True, help="PPTE45089_all_AIC_update_20261001.zip supplied with the submission")
    args = parser.parse_args()
    with zipfile.ZipFile(args.original_archive) as original, zipfile.ZipFile(args.aic_archive) as current:
        def original_read(path):
            return original.read(path)
        def current_read(path):
            # Only the explicitly named current files supersede original materials.
            return current.read(path) if path in current.namelist() else original.read(path)
        def record(data):
            return json.loads(data.decode("utf-8-sig"))
        for relative in ["data/01_passenger_rail/kric_carkind_raw.csv", "data/04_holiday_calendar/holiday_monthly.csv", "data/05_fare_index/fare_monthly.csv", "data/02_oil_price/dubai_monthly.csv", "revision/data/srt_2025_20260924/srt_monthly_15071484_20251231.csv", "revision/data/monthly_panel_revision_v1.csv", "revision/forecast_protocol/protocol_v1.json", "revision/forecast_protocol/holiday_announcement_ledger.csv", "revision/forecast_2024/origin_inputs_2023_fixed_h1_12.csv", "revision/forecast_protocol/origin_inputs_2024_fixed_h1_12.csv", "revision/forecast_protocol/origin_inputs_2025_fixed_h1_12.csv"]:
            write(relative, original_read(relative))
        settings = {"scope": "Tables 2-4; current eligible minimum-AIC orders", "years": {}}
        for year in (2024, 2025):
            yearly = {}
            for model in MODELS:
                if model in ("M3_sarima", "M4_sarimax_3"):
                    directory = f"revision/pooled_ic144/outputs/{year}/{model}"
                    selected_bytes = current_read(directory + "/selected_aic.json")
                    selected = record(selected_bytes)
                    yearly[model] = {"candidate": {"order": selected["order"], "seasonal_order": selected["seasonal_order"]}, "candidate_index": selected["candidate_index"], "selection": "eligible_minimum_AIC"}
                    write(f"reference/order_search/{year}/{model}.csv", current_read(directory + "/candidates.csv"))
                    write(f"reference/order_search/{year}/{model}_selected.json", selected_bytes)
                elif model == "M9_sarimax_lstm_residual":
                    directory = f"revision/aic_consistent_20261001/hybrid_outputs/{year}"
                    yearly[model] = record(current_read(directory + "/selected.json"))
                    write(f"reference/validation/{year}/{model}.csv", current_read(directory + "/selection.csv"))
                else:
                    directory = f"revision/forecast_2024/outputs" if year == 2024 else "revision/outputs_protocol_v1"
                    yearly[model] = record(original_read(directory + f"/selected/{model}.json"))
                    write(f"reference/validation/{year}/{model}.csv", original_read(directory + f"/selection/{model}.csv"))
                for seed in (range(42, 52) if model in NEURAL else [42]):
                    if model in ("M3_sarima", "M4_sarimax_3"):
                        data = current_read(directory + "/prediction_aic.csv")
                    elif model == "M9_sarimax_lstm_residual":
                        data = current_read(directory + f"/prediction_seed{seed}.csv")
                    else:
                        data = original_read(directory + f"/predictions/{model}_seed{seed}.csv")
                    write(f"reference/predictions/{year}/{model}_seed{seed}.csv", data)
            settings["years"][str(year)] = yearly
        validation = "revision/aic_consistent_20261001/hybrid_outputs/validation_base_2023"
        settings["hybrid_validation_base_2023"] = record(current_read(validation + "/decision.json"))
        write("reference/order_search/2023/M4_sarimax_3.csv", current_read(validation + "/candidates.csv"))
        write("settings.json", (json.dumps(settings, indent=2)+"\n").encode())
        score_bytes = current_read("revision/aic_consistent_20261001/main9_AIC_2024_2025_overall.csv")
        write("reference/model_scores.csv", score_bytes)
        scores = list(csv.DictReader(io.StringIO(score_bytes.decode("utf-8-sig"))))
        for number, metric in ((2, "MAPE"), (3, "MAE"), (4, "RMSE")):
            output = io.StringIO(newline="")
            writer = csv.writer(output)
            writer.writerow(["Model", "2024", "2025", "All 24 months"])
            fmt = ".4f" if metric == "MAPE" else ",.0f"
            for row in sorted(scores, key=lambda r: float(r[f"overall_{metric}"])):
                writer.writerow([MODELS[row["model"]]]+[format(float(row[f"{period}_{metric}"]), fmt) for period in ("2024", "2025", "overall")])
            write(f"reference/table{number}_{metric}.csv", output.getvalue().encode("utf-8-sig"))
        function_sources = {"model_core": (original, "revision/modeling/model_core.py"), "grid": (original, "revision/split_sarimax/full_grid_144.py"), "statistical_fit": (original, "revision/pooled_ic144/run_pooled_ic144.py"), "residual_lstm": (current, "revision/aic_consistent_20261001/run_hybrid_aic.py"), "make_inputs": (original, "revision/forecast_protocol/build_origin_inputs.py")}
        provenance = {name: {"supplement_path": path, "source_file_sha256": hashlib.sha256(archive.read(path)).hexdigest()} for name, (archive, path) in function_sources.items()}
        write("provenance/source_functions.json", (json.dumps(provenance, indent=2)+"\n").encode())
        provenance["original_archive_sha256"] = hashlib.sha256(args.original_archive.read_bytes()).hexdigest()
        provenance["aic_archive_sha256"] = hashlib.sha256(args.aic_archive.read_bytes()).hexdigest()
        write("provenance/input_preparation.json", (json.dumps(provenance, indent=2)+"\n").encode())
    print("Prepared local inputs from the two submission supplements: 72 forecasts and current settings.", flush=True)
    print("These data/settings/results are ignored by Git and are not part of the public repository.", flush=True)

if __name__ == "__main__":
    main()
