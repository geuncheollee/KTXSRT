# KTX/SRT forecasting code — PPTE manuscript 45089

This public repository provides the code used to obtain **Table 2 (MAPE), Table 3 (MAE) and Table 4 (RMSE)** in the manuscript revised on 1 October 2026. It contains Python code, requirements and execution instructions. **Input data, selected settings and prediction results are supplied with the journal submission as supplementary files and are not published in this repository.**

## Files

| Code | Purpose |
| --- | --- |
| `prepare_inputs.py` | Read the required local inputs/settings/results from the two submission supplements |
| `replay_observed_panel.py` | Reconstruct the observed monthly panel from archived source tables |
| `build_forecast_inputs.py` | Reconstruct the fixed-origin predictors and dated holiday exclusions |
| `refit_models.py` | Refit the nine models, with optional prior-year tuning and AIC order search |
| `reproduce_tables.py` | Recalculate and export Tables 2–4, preserving metric aggregation, ranking and rounding |
| `revision/modeling/model_core.py` | Original six benchmark model functions, retained byte for byte |
| `revision/modeling/aic_support.py` | Original 144-order grid and exact-diffuse SARIMA/SARIMAX fitting function |
| `revision/modeling/residual_hybrid.py` | Original residual-LSTM training and recursive correction function |

The function bodies of the grid, statistical fit, forecast-input construction and residual LSTM are unchanged from the research scripts. The portable runners adapt input discovery and output directories and route the statistical models through the current AIC rule. The two supplementary ZIPs retain the original scripts, source metadata and recorded selection logs for inspection.

## Prepare local inputs

Obtain both files supplied with the manuscript:

- `PPTE45089_reproducibility_20260929.zip`
- `PPTE45089_all_AIC_update_20261001.zip`

They are submission supplements, not files downloadable from this GitHub repository. They provide the public-source input tables, 60-month observed panel, fixed-origin input matrices, current settings, candidate logs and 72 forecast paths. The current AIC update supersedes the earlier SARIMA/SARIMAX and hybrid results. No API key is required to replay these archived inputs. Place the two ZIPs in a local directory; do not add them to Git.

Run from the repository root, replacing the paths below with their local locations:

```bash
python prepare_inputs.py --original-archive /path/to/PPTE45089_reproducibility_20260929.zip --aic-archive /path/to/PPTE45089_all_AIC_update_20261001.zip
```

On Windows, quote paths containing spaces. The preparation script writes only the required local inputs; `.gitignore` excludes all CSV, JSON, ZIP and document files and all data/output folders from version control.

## Reproduce Tables 2–4

The recorded reference environment is **Python 3.14.3 on Windows, CPU**. Create and activate an isolated environment:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on Linux/macOS. Then:

```bash
python -m pip install -r requirements-verify.txt
python replay_observed_panel.py
python build_forecast_inputs.py
python reproduce_tables.py
```

Panel and forecast-input replay use only the Python standard library. Table calculation uses NumPy and pandas, reads the saved monthly predictions and checks **81 displayed values** against the current supplementary score file. It writes `outputs/table2_MAPE.csv`, `outputs/table3_MAE.csv`, `outputs/table4_RMSE.csv`, unrounded scores and a verification record. Each table is sorted by its own overall metric.

The LSTM, Transformer and residual-hybrid metrics are means of ten seed-specific period scores, using seeds 42–51. They are not metrics of the mean forecast path. Overall RMSE uses all 24 monthly errors within each seed and is not the average of two annual RMSEs. MAPE is in percent; MAE and RMSE are in passengers.

## Refit and repeat selection

```bash
python -m pip install -r requirements.txt
python refit_models.py
python reproduce_tables.py --predictions refit_outputs/predictions --output refit_outputs/tables

# Repeat prior-year hyperparameter tuning using the recorded AIC orders.
python refit_models.py --retune

# Repeat statistical order search and prior-year tuning; this takes longer.
python refit_models.py --search-orders --retune

# Refit only the two proposed models.
python refit_models.py --models M4_sarimax_3 M9_sarimax_lstm_residual
```

Default refitting uses the recorded selections from the supplements and writes forecasts and logs to `refit_outputs/`; it never changes the reference inputs. For SARIMA and SARIMAX, `--search-orders` chooses the eligible training fit with the smallest AIC from 144 combinations: p,q ∈ {0,1,2}; d,P,D,Q ∈ {0,1}; seasonal period 12; no trend. Eligibility requires convergence, finite likelihood/AIC/BIC, the common full likelihood window, more post-diffuse observations than fitted parameters, and finite 12-month forecasts. Ties prefer fewer parameters then the candidate index. BIC does not select the current orders. Stationarity and invertibility are not enforced, so the manuscript's SARIMA root warnings remain relevant. The hybrid shares the AIC-selected SARIMAX order.

Prophet, XGBoost, LSTM, Transformer and residual-LSTM settings use 2023 validation for the 2024 path and 2024 validation for the 2025 path. The selection seed is 42; candidates within 0.01 percentage points of the minimum validation MAPE prefer the simpler configuration then candidate index. Seasonal-naïve and Holt–Winters have fixed specifications. Final neural evaluation uses all ten seeds. The hybrid uses all fitted training residuals, including initialization values, as in the original experiment.

The frozen `protocol_v1.json` prepared locally from the original supplement contains earlier statistical candidate definitions. The public runner supersedes those branches using `aic_support.py` and the current AIC settings prepared from the update supplement. Internal model IDs in filenames map to the manuscript model names in the generated tables.

## Verification and interpretation

The published code was checked with the supplied supplements in the recorded research environment. Source-panel replay and all three forecast-input matrices matched; recorded orders were checked against the four eligible minimum-AIC logs; all 72 selected-model refits matched the archived forecasts within numerical precision; prior-year hyperparameter reselection matched the recorded settings; and both saved and refitted predictions reproduced all 81 displayed values. No newly installed clean-environment run or fresh execution of every 144-candidate grid is claimed. Different package versions or platforms may produce different fitted paths; the archived predictions support exact table verification.

The panel covers January 2021–December 2025. The 2021 observations provide lag/sequence context; training targets start in 2022. The 2024 and 2025 fixed-origin forecasts use 24 and 36 training targets and all 12 observed evaluation months each. Future holidays announced after each origin are excluded. These are retrospective evaluations using final data vintages; real-time source availability is not established. The 2024 evaluation outcomes also tune the 2025 non-statistical models, and the common AIC rule was adopted after forecast inspection. Public code does not remove those limitations or establish statistical superiority from model rankings.

Source providers and dated announcements are documented in the manuscript and submission supplements: KRIC train-type statistics, SR public dataset 15071484 and the calendar/source records. The fare/oil columns retained in the original panel are not predictors in Tables 2–4. Additional coefficient, residual-diagnostic and exploratory paired-comparison outputs remain in the submission supplements.
