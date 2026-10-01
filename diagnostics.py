"""Generate Table 5 coefficients, residual diagnostics and Table 6 paired tests.

Uses orders and forecasts computed locally by refit_models.py. Generated CSVs
and plots are local outputs, not journal supplementary files or public data.
Inference is conditional on selected orders and the very short evaluation.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, t as student_t
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import acf
from refit_models import core, fit
from reproduce_tables import MODELS, NEURAL
ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refits", type=Path, default=ROOT/"refit_outputs")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/diagnostics")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = json.loads((args.refits/"logs/selections.json").read_text())
    panel = core.read_panel()
    coefficients, checks, residual_rows = [], [], []
    for year in (2024, 2025):
        train = core.train_frame(panel, year-1)
        result, warnings = fit(train, selected[f"{year}:M4_sarimax_3"], True)
        ci = np.asarray(result.conf_int())
        for i, name in enumerate(result.param_names):
            coefficients.append({"forecast_year": year, "parameter": name, "estimate": float(result.params[i]),
                                 "se": float(result.bse[i]), "ci_low": float(ci[i,0]),
                                 "ci_high": float(ci[i,1]), "p_value": float(result.pvalues[i])})
        burn = int(result.nobs_diffuse)
        residual = np.asarray(result.filter_results.standardized_forecasts_error[0], float)[burn:]
        dates = train.date.iloc[burn:].tolist()
        lag = min(12, len(residual)-1)
        lb = acorr_ljungbox(residual, lags=[6,lag], return_df=True)
        checks.append({"forecast_year":year,"train_months":len(train),"excluded_diffuse_months":burn,
                       "residual_n":len(residual),"converged":bool(result.mle_retvals.get("converged")),
                       "min_ar_root_modulus":float(np.min(np.abs(result.arroots))),
                       "min_ma_root_modulus":float(np.min(np.abs(result.maroots))),
                       "ljung_box_lag6_p":float(lb.loc[6,"lb_pvalue"]),
                       "ljung_box_final_lag":lag,"ljung_box_final_p":float(lb.loc[lag,"lb_pvalue"]),
                       "warnings":" | ".join(warnings)})
        residual_rows.extend({"forecast_year":year,"fit_month":d,"standardized_residual":float(r)} for d,r in zip(dates,residual))
        fig, axes = plt.subplots(3,1,figsize=(7.2,6.7),constrained_layout=True)
        axes[0].plot(range(len(residual)),residual,marker="o")
        ticks=np.unique(np.linspace(0,len(residual)-1,min(7,len(residual)),dtype=int))
        axes[0].set_xticks(ticks,[dates[t][:7] for t in ticks],rotation=35,ha="right")
        axes[0].set_ylabel("Standardized residual")
        axes[1].hist(residual,bins=min(7,max(5,int(np.sqrt(len(residual))))))
        axes[1].set(xlabel="Standardized residual",ylabel="Frequency")
        correlations=acf(residual,nlags=lag,fft=False)
        axes[2].stem(range(1,lag+1),correlations[1:])
        band=1.96/np.sqrt(len(residual))
        axes[2].axhline(band,linestyle="--",color="gray")
        axes[2].axhline(-band,linestyle="--",color="gray")
        axes[2].set(xlabel="Lag (months)",ylabel="Residual ACF",ylim=(-1,1))
        fig.savefig(args.output/f"residual_diagnostics_{year}.jpg",dpi=300)
        plt.close(fig)
        print(f"Computed {year} coefficients and diagnostics: {len(residual)} residuals",flush=True)
    pd.DataFrame(coefficients).to_csv(args.output/"table5_coefficients.csv",index=False)
    pd.DataFrame(checks).to_csv(args.output/"fit_diagnostics.csv",index=False)
    pd.DataFrame(residual_rows).to_csv(args.output/"standardized_residuals.csv",index=False)
    monthly={}
    for model in MODELS:
        paths=[]
        for seed in (range(42,52) if model in NEURAL else [42]):
            frame=pd.concat([pd.read_csv(args.refits/"predictions"/str(year)/f"{model}_seed{seed}.csv") for year in (2024,2025)])
            paths.append(np.abs(frame.y_true.to_numpy()-frame.y_pred.to_numpy())/frame.y_true.to_numpy()*100)
        monthly[model]=np.mean(paths,axis=0)
    tests=[]
    for model in MODELS:
        if model=="M4_sarimax_3":
            continue
        difference=monthly[model]-monthly["M4_sarimax_3"]
        t,p=ttest_1samp(difference,0)
        margin=student_t.ppf(0.975,len(difference)-1)*np.std(difference,ddof=1)/np.sqrt(len(difference))
        tests.append({"Model":MODELS[model],"n_months":len(difference),"mean_APE_difference_pp":float(difference.mean()),
                      "ci95_lower_pp":float(difference.mean()-margin),"ci95_upper_pp":float(difference.mean()+margin),
                      "t_statistic":float(t),"df":len(difference)-1,"raw_p":float(p)})
    adjusted=multipletests([r["raw_p"] for r in tests],method="holm")[1]
    for row,p in zip(tests,adjusted):
        row["holm_p"]=float(p)
    pd.DataFrame(tests).to_csv(args.output/"table6_paired_comparisons.csv",index=False)
    pd.DataFrame(monthly).to_csv(args.output/"monthly_APE.csv",index=False)
    print("Computed Table 6: exploratory paired tests and Holm correction for eight comparisons.",flush=True)

if __name__=="__main__":
    main()
