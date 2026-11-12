"""Pointwise likelihood-ratio confidence intervals from raw scores.

This module deliberately does not use interpolated ROC or getBounds() tables.
CIs are estimated at fixed score thresholds using case/control binomial
resampling; they do not account for frontier selection and are not simultaneous
confidence bands. Display smoothing is separated from raw inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_METRICS = ("LR+", "LR-")
_CI_COLS = tuple(f"{m}_{lim}" for m in _METRICS for lim in ("lower", "upper"))


def _smooth_lr_widths(curve: pd.DataFrame, sigma: float) -> pd.DataFrame:
    """Add *_lower_display/*_upper_display (not inferential CIs).

    Smooth log-relative widths only within consecutive fully finite segments.
    Never bridge invalid thresholds, extrapolate, or modify nominal values.
    """
    out = curve.copy()
    if sigma <= 0:
        for m in _METRICS:
            for end in ("lower", "upper"):
                out[f"{m}_{end}_display"] = out[f"{m}_{end}"]
        return out

    from scipy.ndimage import gaussian_filter1d

    for metric in _METRICS:
        nominal = out[metric].to_numpy(dtype=float)
        lo = out[f"{metric}_lower"].to_numpy(dtype=float)
        hi = out[f"{metric}_upper"].to_numpy(dtype=float)
        lo_display = np.full(len(out), np.nan, dtype=float)
        hi_display = np.full(len(out), np.nan, dtype=float)
        valid = (np.isfinite(nominal) & (nominal > 0) &
                 np.isfinite(lo) & (lo > 0) &
                 np.isfinite(hi) & (hi > 0) &
                 (lo <= nominal) & (nominal <= hi))
        indices = np.flatnonzero(valid)
        if len(indices):
            for segment in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
                low_gap = np.log(nominal[segment] / lo[segment])
                high_gap = np.log(hi[segment] / nominal[segment])
                if len(segment) >= 4:
                    low_gap = gaussian_filter1d(low_gap, sigma=sigma, mode="nearest")
                    high_gap = gaussian_filter1d(high_gap, sigma=sigma, mode="nearest")
                lo_display[segment] = nominal[segment] * np.exp(-np.maximum(low_gap, 0))
                hi_display[segment] = nominal[segment] * np.exp(np.maximum(high_gap, 0))
        out[f"{metric}_lower_display"] = lo_display
        out[f"{metric}_upper_display"] = hi_display
    return out


def lr_curve_ci_from_scores(
    data: pd.DataFrame,
    score_col: str = "predicted_risk",
    label_col: str = "target",
    *,
    min_count: int = 10,
    fpr_min: float = 0.005,
    fpr_max: float = 0.95,
    min_lrplus: float = 1.001,
    confidence: float = 0.95,
    n_boot: int = 3000,
    random_state: int = 1769,
    chunk_size: int = 48,
    smooth_sigma: float = 0.0,
) -> pd.DataFrame:
    """Return pointwise LR+/LR- bootstrap CIs along a non-dominated frontier.

    TP* and FP* are independently drawn from Binomial(n_pos, empirical TPR)
    and Binomial(n_neg, empirical FPR), respectively, AT EACH FIXED THRESHOLD;
    the same simulated TP*/FP* are used for both likelihood ratios. Thus the
    columns give marginal, pointwise CIs. They are not joint CIs, simultaneous
    curve bands, or valid post-selection coverage guarantees for the frontier.

    The returned nominal curve is sorted by LR- and restricted to non-dominated
    operating points with all four confusion-matrix cells >= min_count.
    Both raw *_lower/*_upper and optional smoothed *_display columns are
    returned. Smoothing is cosmetic: only RAW columns are inferential outputs.

    Score direction: a larger score denotes greater positive-class risk.
    """
    if not 0 < float(fpr_min) < float(fpr_max) < 1:
        raise ValueError("Require 0 < fpr_min < fpr_max < 1")
    if not 0 < float(confidence) < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    if min_count < 1 or n_boot < 100 or chunk_size < 1 or smooth_sigma < 0:
        raise ValueError("min_count>=1, n_boot>=100, chunk_size>=1, smooth_sigma>=0 required")

    labels = pd.to_numeric(data[label_col], errors="coerce").to_numpy(dtype=float)
    scores = pd.to_numeric(data[score_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(labels) & np.isfinite(scores) & np.isin(labels, (0, 1))
    labels, scores = labels[ok], scores[ok]
    positive_scores = np.sort(scores[labels == 1])
    negative_scores = np.sort(scores[labels == 0])
    n_pos, n_neg = len(positive_scores), len(negative_scores)
    if min(n_pos, n_neg) < 2 * min_count:
        raise ValueError(f"Too few positives or negatives: n_pos={n_pos}, n_neg={n_neg}")

    lo_fpr = max(float(fpr_min), 1 / n_neg)
    if lo_fpr >= fpr_max:
        raise ValueError("No FPR search range supports the specified sample size")
    knee = min(max(0.10, lo_fpr), fpr_max)
    levels = np.unique(np.r_[np.geomspace(lo_fpr, knee, 90),
                              np.linspace(knee, fpr_max, 170)])
    idx = np.clip(np.floor((1 - levels) * (n_neg - 1)).astype(int), 0, n_neg - 1)
    thresholds = np.unique(negative_scores[idx])[::-1]
    tp = n_pos - np.searchsorted(positive_scores, thresholds, side="left")
    fp = n_neg - np.searchsorted(negative_scores, thresholds, side="left")
    fn, tn = n_pos - tp, n_neg - fp
    tpr, fpr = tp / n_pos, fp / n_neg

    valid = ((tp >= min_count) & (fp >= min_count) &
             (fn >= min_count) & (tn >= min_count) &
             (fpr >= fpr_min) & (fpr <= fpr_max))
    with np.errstate(divide="ignore", invalid="ignore"):
        lr_plus = tpr / fpr
        lr_minus = (1 - tpr) / (1 - fpr)
    valid &= (np.isfinite(lr_plus) & np.isfinite(lr_minus) &
              (lr_plus > min_lrplus) & (lr_minus > 0) & (lr_minus < 1))
    out = pd.DataFrame({
        "LR-": lr_minus[valid], "LR+": lr_plus[valid],
        "TP": tp[valid], "FP": fp[valid], "FN": fn[valid], "TN": tn[valid],
        "TPR": tpr[valid], "FPR": fpr[valid], "threshold": thresholds[valid],
    }).sort_values(["LR-", "LR+"], ascending=[True, False]).reset_index(drop=True)

    if len(out):
        values = out["LR+"].to_numpy(dtype=float)
        prev = np.r_[-np.inf, np.maximum.accumulate(values)[:-1]]
        out = out.loc[values > prev + 1e-10].reset_index(drop=True)

    out["finite_bootstrap_fraction"] = np.nan
    for column in _CI_COLS:
        out[column] = np.nan

    rng = np.random.default_rng(random_state)
    qlow, qhigh = (1 - confidence) / 2, (1 + confidence) / 2
    for start in range(0, len(out), chunk_size):
        stop = min(start + chunk_size, len(out))
        part = out.iloc[start:stop]
        bs_tp = rng.binomial(n_pos, part["TPR"].to_numpy(), size=(n_boot, len(part)))
        bs_fp = rng.binomial(n_neg, part["FPR"].to_numpy(), size=(n_boot, len(part)))
        bs_fn, bs_tn = n_pos - bs_tp, n_neg - bs_fp
        bootstrap_valid = (bs_tp > 0) & (bs_fp > 0) & (bs_fn > 0) & (bs_tn > 0)
        finite_fraction = bootstrap_valid.mean(axis=0)
        out.loc[start:stop-1, "finite_bootstrap_fraction"] = finite_fraction
        columns_to_compute = np.flatnonzero(finite_fraction >= 0.98)
        if not len(columns_to_compute):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            bs_plus = (bs_tp / n_pos) / (bs_fp / n_neg)
            bs_minus = (bs_fn / n_pos) / (bs_tn / n_neg)
        for metric, sample in (("LR+", bs_plus), ("LR-", bs_minus)):
            s = sample[:, columns_to_compute].astype(float)
            s[~bootstrap_valid[:, columns_to_compute]] = np.nan
            lo, hi = np.nanquantile(s, [qlow, qhigh], axis=0)
            index = np.arange(start, stop)[columns_to_compute]
            nominal = out.loc[index, metric].to_numpy(dtype=float)
            out.loc[index, f"{metric}_lower"] = np.minimum(lo, nominal)
            out.loc[index, f"{metric}_upper"] = np.maximum(hi, nominal)

    out = _smooth_lr_widths(out, sigma=float(smooth_sigma))
    out.attrs.update(n_positive=n_pos, n_negative=n_neg,
                     confidence=confidence, method="pointwise_binomial_bootstrap",
                     frontier_selected=True, valid_bootstrap_min=0.98,
                     display_smoothed=bool(smooth_sigma > 0),
                     warning="Pointwise marginal CIs; no joint, simultaneous, or frontier post-selection coverage")
    return out


def plot_lr_ci_fill(ax, curve: pd.DataFrame, *, color: str, alpha: float = 0.2,
                    use_display: bool = False, **kwargs):
    """Draw an LR+ pointwise marginal CI fill against nominal LR- (no edges).

    LR- uncertainty is available in the returned curve table but is not shown
    in this vertical ribbon. This helper is visual, NOT a joint LR CI region.
    """
    suffix = "_display" if use_display else ""
    x = curve["LR-"].to_numpy(dtype=float)
    lo = curve[f"LR+_lower{suffix}"].to_numpy(dtype=float)
    hi = curve[f"LR+_upper{suffix}"].to_numpy(dtype=float)
    good = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi) & (lo <= hi)
    return ax.fill_between(x, lo, hi, where=good, color=color,
                           alpha=alpha, linewidth=0, interpolate=False, **kwargs)
