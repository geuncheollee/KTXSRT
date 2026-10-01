# KTX/SRT forecasting code - PPTE manuscript 45089

This repository publishes the Python code used for Table 2 (MAPE), Table 3 (MAE) and Table 4 (RMSE), with code for Table 5 coefficients and Table 6 exploratory comparisons. **No input data, selected model settings, saved predictions or manuscript files are distributed here. No journal supplementary files or ZIP archives are required or supplied.** Obtain the data directly from the public sources cited in the manuscript; the preparation script downloads the forecasting inputs and saves them locally.

Current release: **ppte45089-public-data-20261001**. This release replaces the earlier ZIP-dependent preparation workflow.

## Public data sources

1. **KRIC train-type monthly passenger statistics:** https://www.kric.go.kr/jsp/industry/rss/railcarkindpassList.jsp . Select each year 2021-2025. The script sends `q_fdate` and `fdate` with the year and reads the monthly passenger table. Sum KTX, KTX-Sancheon, KTX-Honam, KTX-Eum and KTX-Cheongryong passenger counts. This excludes SRT, which is collected separately.
2. **SR dataset 15071484:** https://www.data.go.kr/data/15071484/fileData.do . Download **(주)에스알_연도별월별수송실적_20251231**, the release registered on **23 February 2026**. The 12-row CP949 CSV contains each year's monthly totals. Use the `2021년 전체` through `2025년 전체` fields; check totals against all available route columns. No login or API key is needed for this file download. The script discovers the download link on the source page rather than using GitHub-hosted data.
3. **Calendar:** `holidays==0.95` generates Korean public and substitute holidays. Public dates and announcement links in `public_inputs.py` document 2 October 2023, 1 October 2024, 27 January 2025 and 3 June 2025. Count distinct holiday dates on Monday-Friday, without duplicate temporary holidays. At each December forecast origin, exclude later-announced holidays. Actual training calendars retain the realized holidays. Government announcement links are written to a local ledger.
4. **Bank of Korea ECOS, for the descriptive fare indices discussed in the manuscript:** https://ecos.bok.or.kr . In consumer-price table `901Y009`, monthly frequency, select item `G03202` (bus fares) and `G03301` (air fares), January 2021-December 2025. These are descriptive candidate variables and **are not forecasting inputs for Tables 2-6**; the forecasting commands do not require an ECOS key or download fare indices. ECOS web downloads are available; API use requires the reader's own ECOS key. Never commit an API key.

The study panel has 60 observed months, January 2021-December 2025. Combined demand equals the five KTX-class sum plus the published SRT total. No missing target month is imputed. The 2021 observations supply lag/sequence context; training targets begin in January 2022. Monthly totals are in passengers.

**Version matters:** the study used the SR release above and the KRIC values available in the study period. Providers may update data or remove old releases. The code records source URLs, download time and source SHA256 hashes locally. A different source vintage may produce different results; this repository does not mirror historical source data. Retrospective use of final vintages does not establish real-time data availability.

## Run from public sources

The verified experiment environment is Python 3.14.3 on Windows, CPU. Create an isolated environment and install the requirements:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare_inputs.py
python refit_models.py
python reproduce_tables.py
python diagnostics.py
```

`prepare_inputs.py` downloads the public source tables, checks monthly coverage and SR route totals, builds the observed panel and the three fixed-origin input matrices. It never downloads supplementary materials or stored predictions. Downloaded data stay under ignored local folders.

If the public sites change their download interfaces, download the same source release manually and use:

```bash
python prepare_inputs.py --kric-csv /path/to/kric_carkind_raw.csv --srt-csv /path/to/srt_20251231.csv
```

Either option can be supplied independently. The KRIC CSV must have `yyyymm,total_pax,total_pkm`, then passenger columns `<train_type>_pax` and distance columns `<train_type>_pkm`, using these twelve train-type names in the original site order: KTX, 새마을, 무궁화, 통근열차, 누리로, KTX-산천, KTX-호남, KTX-이음, KTX-청룡, ITX-새마을, ITX-청춘열차, ITX-마음. Use the first monthly table values, not annual totals. `replay_observed_panel.py` can subsequently rebuild the panel from the downloaded source files without network access; `build_forecast_inputs.py` rebuilds the predictor matrices.

## Selection, fitting and outputs

`refit_models.py` repeats all order searches and prior-year tuning rather than reading stored selected settings. For SARIMA and SARIMAX it evaluates 144 combinations: p,q in {0,1,2}; d,P,D,Q in {0,1}; seasonal period 12; no trend. The fitting function uses exact diffuse initialization and at most 500 optimizer iterations. Eligibility requires convergence, finite likelihood/AIC/BIC, the full common likelihood window, more post-diffuse observations than parameters and a finite 12-month forecast. The eligible minimum AIC selects the order; ties prefer fewer parameters then candidate index. Stationarity/invertibility are not enforced. The hybrid uses the same selected SARIMAX order.

Prophet, XGBoost, LSTM, Transformer and residual-LSTM settings use prior-year validation: 2023 for the 2024 forecast and 2024 for 2025. Candidate grids and optimizer specifications are defined in `experiment_config.py`; they contain method definitions, not preselected results or observed data. Candidates within 0.01 percentage points of the smallest validation MAPE prefer the simpler configuration then candidate index. Selection uses seed 42. Final neural evaluation uses seeds 42-51. Seasonal-naive and Holt-Winters specifications are fixed. The hybrid trains on all fitted residuals, including initialization values, as in the original experiment.

The original model core, extracted full-order grid, exact-diffuse fitting function and residual-LSTM function are retained. Portable runners change input acquisition and output discovery. Older statistical branches in `model_core.py` are not called by the current runner: the current runner routes SARIMA/SARIMAX and the hybrid base through `aic_support.fit`.

Generated local files include:

- `refit_outputs/logs/`: complete AIC candidate/failure logs, validation logs and selections.
- `refit_outputs/predictions/`: 72 forecast files, covering nine models, two evaluation years and ten seeds for each neural model.
- `outputs/table2_MAPE.csv`, `table3_MAE.csv`, `table4_RMSE.csv`: tables sorted by their own overall metric, plus unrounded scores.
- `outputs/diagnostics/`: coefficient estimates/SE/95% intervals/p-values, standardized residuals, Ljung-Box diagnostics, locally generated residual plots, monthly absolute percentage errors and paired tests with Holm correction over eight comparisons.

Neural scores are means of ten seed-specific period scores, not errors of the mean forecast. Overall RMSE uses all 24 errors within each seed and is not the mean of annual RMSEs. MAPE is percent; MAE/RMSE are passengers. The paired tests use seed-mean monthly APE differences over 24 months and are exploratory.

Internal filename IDs map to model names in the exported tables. To fit only the proposed models, run `python refit_models.py --models M4_sarimax_3 M9_sarimax_lstm_residual`; generating all Tables 2-6 requires the full run.

## Verification and limits

The standalone public-source pipeline was checked in the recorded research environment: freshly downloaded KRIC/SR values matched every forecasting input; all three origin matrices matched; full AIC searches and prior-year tuning reproduced the selected settings; all 72 forecast paths and all 81 displayed values in Tables 2-4 matched the manuscript, including table order and rounding. Coefficient/residual and paired-test calculations were also checked. This is not a newly installed clean-environment test. Package versions, operating system and subsequently revised source data may affect numerical results.

The 2024 outcomes tune the 2025 non-statistical models. The common AIC rule was adopted after forecast inspection. SARIMA root warnings and adverse outcomes are retained; the public code does not establish confirmatory superiority. Residual inference uses only 21/12 usable observations for 2024/2025, and the 2025 Ljung-Box result at lag 11 indicates remaining autocorrelation. Coefficient uncertainty is conditional on the selected order and excludes order-selection uncertainty. Longer evaluations are needed.

`.gitignore` excludes source data, JSON settings, CSV outputs, ZIP archives, manuscripts, credentials and generated figures. Downloaded data and generated results are not published by running these commands.
