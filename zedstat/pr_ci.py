"""Score-level precision--recall curves and approximate fixed-threshold CIs.

The PR curve is computed from actual score thresholds, never by regridding a
ROC in FPR or by inserting the (recall=0, precision=1) plotting convention.
Unsupported endpoints are omitted rather than imputed or extrapolated.

The pointwise intervals use a log-ratio delta approximation. Their coverage
is not simultaneous over the curve, does not account for threshold selection,
and the optional smoothed display boundaries are purely graphical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit, logit
from scipy.stats import norm


_COLUMNS = ("threshold", "tpr", "fpr", "ppv", "TP", "FP", "FN", "TN", "flags",
            "ppv_lower", "ppv_upper", "ppv_se_logit",
            "ppv_lower_display", "ppv_upper_display")


def precision_recall_ci_from_scores(
    data,
    *,
    score_col="predicted_risk",
    label_col="target",
    prevalence=None,
    prevalence_sample_size=None,
    confidence=0.95,
    min_tp=10,
    min_fp=5,
    min_flags=20,
    max_points=1000,
    display_sigma=0.0,
):
    """Return a supported PR curve with optional 95% pointwise PPV intervals.

    Rates are computed from *observed score thresholds*.  ``prevalence`` is
    the target/deployment prevalence; when omitted, the sample prevalence is
    used, and its binomial estimation variance is included by default.

    When a target prevalence is supplied, it is treated as fixed unless
    ``prevalence_sample_size`` is provided. If target prevalence was estimated
    from an independent representative cohort, provide its sample size to
    include approximate prevalence uncertainty. For weights, clustering, or
    dependent data, these simple iid delta-method intervals are insufficient.

    The nominal PR curve starts at the first real threshold meeting the
    minimum cell-count support. It never inserts the (0,1) plot convention.
    For identical recalls the last threshold (largest FPR) is retained.

    The raw CI limits are in ``ppv_lower`` and ``ppv_upper``; smoothed limits
    are separate, and are not guaranteed 95% intervals. The supplied target
    prevalence is not modified by smoothing or regularization.
    """
    if not (0 < float(confidence) < 1):
        raise ValueError("confidence must lie in (0,1)")
    if min_tp < 1 or min_fp < 1 or min_flags < 1:
        raise ValueError("min_tp, min_fp, min_flags must all be >= 1")
    if max_points is not None and int(max_points) < 2:
        raise ValueError("max_points must be >= 2 or None")
    if float(display_sigma) < 0:
        raise ValueError("display_sigma must be nonnegative")

    labels = pd.to_numeric(data[label_col], errors="coerce").to_numpy(dtype=float)
    scores = pd.to_numeric(data[score_col], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(labels) & np.isfinite(scores) & np.isin(labels, [0, 1])
    labels = labels[good].astype(np.int8)
    scores = scores[good]
    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = int(n - n_pos)
    if not n_pos or not n_neg:
        raise ValueError("At least one valid positive and negative label is required")

    if prevalence is None:
        pi = n_pos / n
        prevalence_n = n if prevalence_sample_size is None else int(prevalence_sample_size)
    else:
        pi = float(prevalence)
        prevalence_n = None if prevalence_sample_size is None else int(prevalence_sample_size)
    if not 0 < pi < 1:
        raise ValueError("prevalence must be in (0,1)")
    if prevalence_n is not None and prevalence_n <= 0:
        raise ValueError("prevalence_sample_size must be positive")

    # Sort once and form confusion-matrix counts only at real, unique scores.
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    last = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
    tp = np.cumsum(sorted_labels, dtype=np.int64)[last]
    fp = (last + 1) - tp
    fn = n_pos - tp
    tn = n_neg - fp
    flags = last + 1
    tpr = tp / n_pos
    fpr = fp / n_neg
    denom = pi * tpr + (1 - pi) * fpr
    with np.errstate(divide="ignore", invalid="ignore"):
        ppv = pi * tpr / denom

    # Suppress low-count regimes, not merely x=0.  Those operating points are
    # precisely where the legacy ROC-grid origin produces spurious PRC spikes.
    supported = ((tp >= min_tp) & (fp >= min_fp) & (flags >= min_flags) &
                 np.isfinite(ppv) & (denom > 0))
    df = pd.DataFrame({
        "threshold": sorted_scores[last][supported],
        "tpr": tpr[supported], "fpr": fpr[supported], "ppv": ppv[supported],
        "TP": tp[supported], "FP": fp[supported],
        "FN": fn[supported], "TN": tn[supported], "flags": flags[supported],
    })

    # A PRC is a graph against recall: a unique recall coordinate gives a
    # non-self-intersecting polyline. The last empirical threshold at that
    # recall is retained, never an interpolated or fictitious threshold.
    df = df.drop_duplicates(subset="tpr", keep="last").reset_index(drop=True)
    if max_points is not None and len(df) > int(max_points):
        r = df["tpr"].to_numpy(dtype=float)
        wanted = np.linspace(r[0], r[-1], int(max_points))
        keep = np.unique(np.r_[0, np.searchsorted(r, wanted, side="left"), len(r)-1])
        df = df.iloc[keep].reset_index(drop=True)

    tp_v = df["TP"].to_numpy(dtype=float)
    fp_v = df["FP"].to_numpy(dtype=float)
    center = df["ppv"].to_numpy(dtype=float)
    # Standard fixed-threshold delta variance on logit(PPV) = logit(pi) + log(LR+).
    if len(df):
        variance = (1.0 / tp_v - 1.0 / n_pos +
                    1.0 / fp_v - 1.0 / n_neg)
        if prevalence_n is not None:
            variance += 1.0 / (prevalence_n * pi * (1 - pi))
        se = np.sqrt(np.maximum(variance, 0))
        z = norm.ppf(0.5 + float(confidence) / 2.0)
        # With min counts, 0<PPV<1. Do not infer limits at degenerate points.
        lo = expit(logit(center) - z * se)
        hi = expit(logit(center) + z * se)
        if display_sigma > 0 and len(df) > 2:
            # Display-only: smooth uncertainty widths, NEVER nominal precision.
            dlo = np.maximum(logit(center) - logit(lo), 0.0)
            dhi = np.maximum(logit(hi) - logit(center), 0.0)
            dlo = gaussian_filter1d(dlo, float(display_sigma), mode="nearest")
            dhi = gaussian_filter1d(dhi, float(display_sigma), mode="nearest")
            lo_display = expit(logit(center) - dlo)
            hi_display = expit(logit(center) + dhi)
        else:
            lo_display, hi_display = lo.copy(), hi.copy()
    else:
        se = lo = hi = lo_display = hi_display = np.empty(0, dtype=float)
    df["ppv_lower"] = lo
    df["ppv_upper"] = hi
    df["ppv_se_logit"] = se
    df["ppv_lower_display"] = lo_display
    df["ppv_upper_display"] = hi_display
    df = df.loc[:, _COLUMNS]
    df.attrs["precision_recall_ci"] = {
        "confidence": float(confidence),
        "method": "score_level_fixed_threshold_log_ratio_delta",
        "prevalence": pi,
        "prevalence_sample_size": prevalence_n,
        "sample_n_pos": n_pos,
        "sample_n_neg": n_neg,
        "min_tp": min_tp, "min_fp": min_fp, "min_flags": min_flags,
        "display_sigma": display_sigma,
        "warning": "Pointwise approximate intervals only; not simultaneous or threshold-selection adjusted."
    }
    return df


def plot_precision_recall_ci_fill(ax, curve, *, color=None, alpha=0.2,
                                  use_display=False, zorder=1, **kwargs):
    """Plot only finite score-level CI support; never invent endpoint values."""
    suffix = "_display" if use_display else ""
    x = curve["tpr"].to_numpy(dtype=float)
    y = curve["ppv"].to_numpy(dtype=float)
    lo = curve[f"ppv_lower{suffix}"].to_numpy(dtype=float)
    hi = curve[f"ppv_upper{suffix}"].to_numpy(dtype=float)
    ok = (np.isfinite(x) & np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
          & (lo <= y) & (y <= hi))
    return ax.fill_between(x, lo, hi, where=ok, color=color, alpha=alpha,
                           linewidth=0, interpolate=False, zorder=zorder, **kwargs)
