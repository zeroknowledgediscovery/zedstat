"""Approximate pointwise log-ratio CIs and separate display-only smoothing.

Statistical scope:
- Large-sample log-delta intervals at fixed score thresholds, conditional on
  the validation sample's positive/negative totals and a fixed target prevalence.
- No simultaneous confidence-band or data-driven frontier selection guarantee.
- When rates are read from a smoothed ROC, inference is approximate; for a
  rigorous coverage claim compute the estimates on empirical thresholds first.
- *_display columns are *cosmetic*, not separately calibrated 95% intervals.

No extrapolation, clipping of large widths, or empirical-CI shrinkage is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit, logit
from scipy.stats import norm


def _smooth_log_widths(nominal, lower, upper, *, sigma=0.0, scale="log"):
    """Smooth lower/upper distances around nominal, respecting NaN gaps.

    log for positive LR, logit for probability. Never changes nominal, never
    extrapolates or bridges a missing interval. The outputs are display-only.
    """
    y = np.asarray(nominal, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    d_lo = np.full(y.shape, np.nan)
    d_hi = np.full(y.shape, np.nan)
    if scale == "logit":
        transform, inv = logit, expit
        valid = (np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
                 & (y > 0) & (y < 1) & (lo > 0) & (lo < 1)
                 & (hi > 0) & (hi < 1) & (lo <= y) & (y <= hi))
    elif scale == "log":
        transform, inv = np.log, np.exp
        valid = (np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
                 & (y > 0) & (lo > 0) & (hi > 0)
                 & (lo <= y) & (y <= hi))
    else:
        raise ValueError("scale must be 'log' or 'logit'")
    indices = np.flatnonzero(valid)
    if not len(indices):
        return d_lo, d_hi
    for seg in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
        center = transform(y[seg])
        lower_width = np.maximum(center - transform(lo[seg]), 0)
        upper_width = np.maximum(transform(hi[seg]) - center, 0)
        if sigma > 0 and len(seg) >= 3:
            lower_width = gaussian_filter1d(lower_width, sigma=float(sigma), mode="reflect")
            upper_width = gaussian_filter1d(upper_width, sigma=float(sigma), mode="reflect")
        d_lo[seg] = inv(center - lower_width)
        d_hi[seg] = inv(center + upper_width)
    return d_lo, d_hi


def ppv_logratio_ci(
    roc: pd.DataFrame,
    *,
    positive_samples: int,
    negative_samples: int,
    prevalence: float,
    confidence: float = 0.95,
    min_tp: int = 10,
    min_fp: int = 10,
    smooth_sigma: float = 0.0,
) -> pd.DataFrame:
    """Return approximate pointwise PPV CIs on a ROC-derived data frame.

    Under fixed-prevalence transport, logit(PPV) = logit(pi) + log(TPR/FPR).
    Assuming independent binomial TPR/FPR sampling, delta-method variance is
      Var(log(TPR/FPR)) ≈ 1/TP - 1/n_pos + 1/FP - 1/n_neg.

    The nominal PPV column, if supplied, is used as the center; when it has
    been isotonic-corrected, CIs are *approximate display-aligned* intervals,
    not a coverage assertion for a constrained estimator.
    """
    if not 0 < prevalence < 1 or not 0 < confidence < 1:
        raise ValueError("prevalence and confidence must be in (0,1)")
    if min(positive_samples, negative_samples) <= 0 or min_tp < 1 or min_fp < 1:
        raise ValueError("Sample sizes must be positive and min counts >=1")
    if smooth_sigma < 0:
        raise ValueError("smooth_sigma must be >= 0")
    work = roc.copy()
    fpr = (pd.to_numeric(work["fpr"], errors="coerce").to_numpy(dtype=float)
           if "fpr" in work.columns else work.index.to_numpy(dtype=float))
    tpr = pd.to_numeric(work["tpr"], errors="coerce").to_numpy(dtype=float)
    tp = positive_samples * tpr
    fp = negative_samples * fpr
    with np.errstate(invalid="ignore", divide="ignore"):
        ppv_raw = prevalence*tpr / (prevalence*tpr + (1-prevalence)*fpr)
    y = (pd.to_numeric(work["ppv"], errors="coerce").to_numpy(dtype=float)
         if "ppv" in work.columns else ppv_raw)
    mask = (np.isfinite(y) & (y > 0) & (y < 1) & (tpr > 0) & (fpr > 0)
            & (fpr <= 1) & (tpr <= 1) & (tp >= min_tp) & (fp >= min_fp))
    se = np.full(y.shape, np.nan, dtype=float)
    se[mask] = np.sqrt(np.maximum(
        1/tp[mask] - 1/positive_samples + 1/fp[mask] - 1/negative_samples, 0))
    z = norm.ppf(0.5 + confidence/2)
    lower = np.full(y.shape, np.nan)
    upper = np.full(y.shape, np.nan)
    lower[mask] = expit(logit(y[mask]) - z*se[mask])
    upper[mask] = expit(logit(y[mask]) + z*se[mask])
    work["ppv_logratio_lower"] = lower
    work["ppv_logratio_upper"] = upper
    work["ppv_logratio_se_logit"] = se
    display_lo, display_hi = _smooth_log_widths(y, lower, upper, sigma=smooth_sigma, scale="logit")
    work["ppv_logratio_lower_display"] = display_lo
    work["ppv_logratio_upper_display"] = display_hi
    work.attrs["ppv_logratio_ci"] = {
        "confidence": confidence, "method": "fixed_threshold_log_delta_approximation",
        "min_tp": min_tp, "min_fp": min_fp,
        "display_sigma": smooth_sigma,
        "warning": "Display smoothing and ROC isotonic correction can alter nominal coverage"
    }
    return work


def lr_logratio_ci(
    curve: pd.DataFrame,
    *,
    confidence: float = 0.95,
    min_count: int = 10,
    smooth_sigma: float = 0.0,
) -> pd.DataFrame:
    """Approximate log-delta 95% LR+ and LR- CIs from observed cell counts.

    SE(log LR+)² = 1/TP - 1/(TP+FN) + 1/FP - 1/(FP+TN)
    SE(log LR-)² = 1/FN - 1/(TP+FN) + 1/TN - 1/(FP+TN)
    with nonzero support. Raw and display-smoothed limits are separate.
    """
    if not 0 < confidence < 1 or min_count < 1 or smooth_sigma < 0:
        raise ValueError("Invalid confidence, min_count or smooth_sigma")
    out = curve.copy()
    tp, fp, fn, tn = (pd.to_numeric(out[k], errors="coerce").to_numpy(dtype=float)
                      for k in ("TP", "FP", "FN", "TN"))
    n_pos, n_neg = tp+fn, fp+tn
    support = (np.isfinite(tp) & np.isfinite(fp) & np.isfinite(fn) & np.isfinite(tn)
               & (np.minimum.reduce([tp,fp,fn,tn]) >= min_count)
               & (n_pos > 0) & (n_neg > 0))
    z = norm.ppf(0.5 + confidence/2)
    for metric, a,b in (("LR+",tp,fp),("LR-",fn,tn)):
        nominal = pd.to_numeric(out[metric], errors="coerce").to_numpy(dtype=float)
        valid = support & np.isfinite(nominal) & (nominal > 0)
        se = np.full(nominal.shape, np.nan)
        se[valid] = np.sqrt(np.maximum(1/a[valid] - 1/n_pos[valid]
                                       + 1/b[valid] - 1/n_neg[valid], 0))
        lo, hi = np.full(nominal.shape, np.nan), np.full(nominal.shape, np.nan)
        lo[valid] = nominal[valid] * np.exp(-z*se[valid])
        hi[valid] = nominal[valid] * np.exp(z*se[valid])
        out[f"{metric}_logratio_lower"] = lo
        out[f"{metric}_logratio_upper"] = hi
        out[f"{metric}_logratio_se_log"] = se
        dlo, dhi = _smooth_log_widths(nominal, lo, hi, sigma=smooth_sigma, scale="log")
        out[f"{metric}_logratio_lower_display"] = dlo
        out[f"{metric}_logratio_upper_display"] = dhi
    out.attrs["lr_logratio_ci"] = {
        "confidence": confidence, "method": "fixed_threshold_log_delta_approximation",
        "min_count": min_count, "display_sigma": smooth_sigma,
        "warning": "Not simultaneous or post-frontier-selection adjusted; display smoothing not calibrated"
    }
    return out
