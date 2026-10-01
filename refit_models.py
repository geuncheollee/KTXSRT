"""Select and fit all nine models from public-source observations.

Every run repeats training-only AIC selection and prior-year tuning.
No saved settings or forecasts are required.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "revision/modeling"))
import model_core as core
from aic_support import fit, grid
from residual_hybrid import residual_lstm
from reproduce_tables import MODELS, NEURAL

STATISTICAL = {"M3_sarima", "M4_sarimax_3"}
HYBRID = "M9_sarimax_lstm_residual"

def future(year):
    if year == 2023:
        return pd.read_csv(ROOT / "revision/forecast_2024/origin_inputs_2023_fixed_h1_12.csv")
    return core.read_future(year)

def eligible(result, train, prediction):
    return (bool(result.mle_retvals.get("converged", False))
            and np.isfinite([result.llf, result.aic, result.bic]).all()
            and int(result.nobs_effective) == len(train)
            and len(train) - int(result.nobs_diffuse) > len(result.params)
            and len(prediction) == 12 and np.isfinite(prediction).all())

def search_aic(panel, year, model, output):
    train, inputs = core.train_frame(panel, year-1), future(year)
    records = []
    options = grid()
    for index, candidate in enumerate(options):
        row = {"candidate_index": index, "candidate_json": json.dumps(candidate)}
        try:
            result, messages = fit(train, candidate, model == "M4_sarimax_3")
            prediction = core._stat_forecast(result, inputs, model == "M4_sarimax_3")
            valid = eligible(result, train, prediction)
            row.update(status="eligible" if valid else "ineligible", aic=float(result.aic), bic=float(result.bic), k_params=len(result.params), nobs_effective=int(result.nobs_effective), nobs_diffuse=int(result.nobs_diffuse), converged=bool(result.mle_retvals.get("converged", False)), warning_count=len(messages))
        except Exception as exc:
            row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        records.append(row)
        if (index+1) % 12 == 0:
            pd.DataFrame(records).to_csv(output / f"AIC_{year}_{model}.csv", index=False)
            print(f"AIC search {year} {MODELS[model]}: {index+1}/144", flush=True)
    valid = pd.DataFrame(records).query("status == 'eligible'")
    if valid.empty:
        raise RuntimeError(f"No eligible AIC candidate: {year}, {model}")
    winner = valid.sort_values(["aic", "k_params", "candidate_index"]).iloc[0]
    candidate = options[int(winner.candidate_index)]
    print(f"Selected minimum AIC: {year} {MODELS[model]} {candidate}", flush=True)
    return candidate

def selected_stat_candidate(year, model, choices):
    return choices[(year, model)]


def tune(panel, year, model, choices, output):
    # No 2024/2025 target-year truth enters order selection or prior-year tuning.
    validation_year = year-1
    train = core.train_frame(panel, year-2)
    inputs = future(validation_year)
    if model in ("M1_seasonal_naive", "M2_holt_winters"):
        return {}
    if model == HYBRID:
        options = [{"lookback": l, "epochs": e} for l in (6, 12) for e in (50, 100, 200)]
        base_candidate = selected_stat_candidate(validation_year, "M4_sarimax_3", choices)
        base_result, _ = fit(train, base_candidate, True)
        if not base_result.mle_retvals.get("converged", False):
            raise RuntimeError("Hybrid validation base did not converge")
        base_prediction = core._stat_forecast(base_result, inputs, True)
    else:
        options = core.candidates(model)
    rows = []
    for index, candidate in enumerate(options):
        row = {"candidate_index": index, "candidate_json": json.dumps(candidate), "complexity": core.complexity(model, candidate)}
        try:
            if model == HYBRID:
                prediction = residual_lstm(train, inputs, base_result, base_prediction, candidate, 42)
            else:
                prediction = core.forecast(model, candidate, panel, train, inputs, seed=42)
            truth = panel.loc[panel.date.str.startswith(str(validation_year)), "hsr_pax_total"].to_numpy(float)
            row.update(status="ok", **core.scores(truth, prediction))
        except Exception as exc:
            row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        print(f"Validation {year} {MODELS[model]}: {index+1}/{len(options)} {row['status']}", flush=True)
    pd.DataFrame(rows).to_csv(output / f"validation_{year}_{model}.csv", index=False)
    valid = [r for r in rows if r["status"] == "ok"]
    if not valid:
        raise RuntimeError(f"No valid candidate: {year} {model}")
    best_mape = min(r["MAPE"] for r in valid)
    close = [r for r in valid if r["MAPE"] <= best_mape + 0.01]
    winner = min(close, key=lambda r: (r["complexity"], r["candidate_index"]))
    return options[winner["candidate_index"]]

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["all"]+list(MODELS), default=["all"])
    parser.add_argument("--years", nargs="+", type=int, choices=(2024, 2025), default=[2024, 2025])
    parser.add_argument("--output", type=Path, default=ROOT / "refit_outputs")
    parser.add_argument("--retune", action="store_true", help="Compatibility option: tuning is always performed.")
    parser.add_argument("--search-orders", action="store_true", help="Compatibility option: AIC search is always performed.")
    args = parser.parse_args()
    models = list(MODELS) if "all" in args.models else args.models
    if args.output.resolve().is_relative_to((ROOT / "reference").resolve()):
        parser.error("Choose an output outside reference/.")
    logs = args.output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    panel = core.read_panel()
    choices = {}
    # Always repeat selection; no preselected settings are distributed.
    if models:
        required = {(year, model) for year in args.years for model in STATISTICAL if model in models or (model == "M4_sarimax_3" and HYBRID in models)}
        if HYBRID in models:
            required.update((year-1, "M4_sarimax_3") for year in args.years)
        for year, model in sorted(required):
            choices[(year, model)] = search_aic(panel, year, model, logs)
    selections = {}
    for year in args.years:
        train, inputs = core.train_frame(panel, year-1), future(year)
        for model in models:
            candidate = (selected_stat_candidate(year, model, choices) if model in STATISTICAL
                         else tune(panel, year, model, choices, logs))
            selections[f"{year}:{model}"] = candidate
            if model in STATISTICAL or model == HYBRID:
                base_candidate = candidate if model in STATISTICAL else selected_stat_candidate(year, "M4_sarimax_3", choices)
                result, _ = fit(train, base_candidate, model != "M3_sarima")
                if not result.mle_retvals.get("converged", False):
                    raise RuntimeError(f"Selected statistical fit did not converge: {year} {model}")
                baseline = core._stat_forecast(result, inputs, model != "M3_sarima")
            seeds = range(42, 52) if model in NEURAL else [42]
            for seed in seeds:
                if model in STATISTICAL:
                    prediction = baseline
                elif model == HYBRID:
                    prediction = residual_lstm(train, inputs, result, baseline, candidate, seed)
                else:
                    prediction = core.forecast(model, candidate, panel, train, inputs, seed)
                prediction = core.validate_forecast(prediction, inputs)
                truth = panel.loc[panel.date.str.startswith(str(year)), "hsr_pax_total"].to_numpy(float)
                directory = args.output / "predictions" / str(year)
                directory.mkdir(parents=True, exist_ok=True)
                name = f"{model}_seed{seed}.csv"
                pd.DataFrame({"target_month": inputs.target_month, "y_true": truth, "y_pred": prediction}).to_csv(directory / name, index=False)
                print(f"Computed {year} {MODELS[model]} seed {seed}", flush=True)
    (logs / "selections.json").write_text(json.dumps(selections, indent=2)+"\n", encoding="utf-8")
    print("Completed selected refits. Recalculate tables with reproduce_tables.py --predictions <output>/predictions.", flush=True)

if __name__ == "__main__":
    main()
