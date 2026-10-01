"""Original full-grid construction and exact-diffuse SARIMA/SARIMAX fitting functions.

Extracted without algorithm changes; source hashes are in provenance/source_functions.json.
"""
from __future__ import annotations
import itertools
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import model_core as core

def grid() -> list[dict]:
    ans = []
    for p, d, q, P, D, Q in itertools.product(range(3), range(2), range(3),
                                               range(2), range(2), range(2)):
        ans.append({"order": (p, d, q), "seasonal_order": (P, D, Q, 12)})
    assert len(ans) == 144 and len({str(x) for x in ans}) == 144
    return ans

def fit(train: pd.DataFrame, candidate: dict, use_exog: bool):
    y = train.hsr_pax_total.to_numpy(float) / core.SCALE
    x = None
    if use_exog:
        x = train.loc[:, core.EXOG].to_numpy(float)
        x[:, 0] /= core.SCALE
        assert np.linalg.matrix_rank(x) == 3
    model = SARIMAX(y, exog=x, order=candidate["order"],
                    seasonal_order=candidate["seasonal_order"], trend="n",
                    enforce_stationarity=False, enforce_invertibility=False,
                    use_exact_diffuse=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(disp=False, maxiter=500)
    messages = sorted({type(w.message).__name__ + ": " + str(w.message) for w in caught})
    return result, messages
