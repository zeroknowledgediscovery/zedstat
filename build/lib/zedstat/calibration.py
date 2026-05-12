"""
Standalone calibration submodule for zedstat.

Place this file at:
    zedstat/calibration.py

Then import with:
    from zedstat import calibration
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm

__all__ = [
    "ensure_binary_01",
    "make_prevalence_weights",
    "weighted_mean",
    "nice_axis_limit",
    "wilson_interval",
    "logit_clip",
    "calibration_intercept_slope",
    "calibration_table",
    "bootstrap_test_metrics",
    "plot_reliability_diagram",
    "heldout_isotonic_calibration_with_bootstrap",
    "interpret_calibration_results",
]


def ensure_binary_01(y):
    y = pd.to_numeric(y, errors="coerce")
    if y.isna().any():
        raise ValueError("Label column contains NaN or non-numeric values.")
    vals = set(pd.Series(y).unique().tolist())
    if not vals.issubset({0, 1}):
        raise ValueError("Label column must be coded as 0/1.")
    return y.astype(int)


def make_prevalence_weights(y, target_prevalence=None):
    y = np.asarray(y, dtype=int)

    if target_prevalence is None:
        return np.ones_like(y, dtype=float)

    p_sample = y.mean()
    p_target = float(target_prevalence)

    if not (0 < p_sample < 1):
        raise ValueError("Sample must contain both classes.")
    if not (0 < p_target < 1):
        raise ValueError("target_prevalence must be between 0 and 1.")

    w = np.where(
        y == 1,
        p_target / p_sample,
        (1.0 - p_target) / (1.0 - p_sample)
    )
    return w.astype(float)


def weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    return float(np.sum(x * w) / np.sum(w))


def nice_axis_limit(max_val):
    candidates = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00])
    target = max(0.05, min(1.0, float(max_val) * 1.08))
    idx = np.searchsorted(candidates, target, side="left")
    return 1.0 if idx >= len(candidates) else float(candidates[idx])


def wilson_interval(k, n, alpha=0.05):
    if n <= 0:
        return np.nan, np.nan

    z = norm.ppf(1.0 - alpha / 2.0)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    radius = z * np.sqrt((phat * (1.0 - phat) / n) + (z * z / (4.0 * n * n))) / denom

    lo = max(0.0, center - radius)
    hi = min(1.0, center + radius)
    return lo, hi


def logit_clip(p, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _as_path_or_none(value):
    if value is None or value is False:
        return None
    if isinstance(value, (str, Path)):
        return Path(value)
    return None


def calibration_intercept_slope(y_true, p_pred, eps=1e-12):
    """
    Fit:
        logit(E[Y]) = a + b * logit(p_pred)

    Perfect calibration:
        intercept ~= 0
        slope ~= 1
    """
    y_true = np.asarray(y_true, dtype=int)
    p_pred = np.asarray(p_pred, dtype=float)

    x = logit_clip(p_pred, eps=eps).reshape(-1, 1)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*Setting penalty=None will ignore the C and l1_ratio parameters.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*'penalty' was deprecated.*",
            category=FutureWarning,
        )
        try:
            lr = LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=1000
            )
            lr.fit(x, y_true)
        except Exception:
            lr = LogisticRegression(
                penalty="none",
                solver="lbfgs",
                max_iter=1000
            )
            lr.fit(x, y_true)

    intercept = float(lr.intercept_[0])
    slope = float(lr.coef_[0][0])
    return intercept, slope


def calibration_table(prob, y, n_bins=10, alpha=0.05):
    d = pd.DataFrame({
        "prob": pd.to_numeric(prob, errors="coerce"),
        "y": ensure_binary_01(pd.Series(y))
    }).dropna().copy()

    if len(d) < 2:
        raise ValueError("Not enough rows to build calibration table.")

    n_bins = min(int(n_bins), len(d))
    if n_bins < 2:
        raise ValueError("Need at least 2 bins.")

    d["bin"] = pd.qcut(
        d["prob"].rank(method="first"),
        q=n_bins,
        labels=False,
        duplicates="drop"
    )

    rows = []
    for b, g in d.groupby("bin", sort=True):
        n = int(len(g))
        k = int(g["y"].sum())
        obs = float(k / n)
        lo, hi = wilson_interval(k, n, alpha=alpha)

        lo = float(min(lo, obs))
        hi = float(max(hi, obs))

        rows.append({
            "bin": int(b),
            "n": n,
            "cases": k,
            "mean_pred": float(g["prob"].mean()),
            "obs_rate": obs,
            "obs_rate_lo": lo,
            "obs_rate_hi": hi,
            "prob_min": float(g["prob"].min()),
            "prob_max": float(g["prob"].max())
        })

    out = pd.DataFrame(rows).sort_values("mean_pred").reset_index(drop=True)
    return out

def bootstrap_test_metrics(y_true, raw_score, calibrated_prob, n_boot=1000, random_state=42):
    y_true = np.asarray(y_true, dtype=int)
    raw_score = np.asarray(raw_score, dtype=float)
    calibrated_prob = np.asarray(calibrated_prob, dtype=float)

    n = len(y_true)
    rng = np.random.default_rng(random_state)

    raw_is_probability = np.all((raw_score >= 0.0) & (raw_score <= 1.0))
    boot_rows = []

    for _ in tqdm(range(int(n_boot))):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        rb = raw_score[idx]
        pb = calibrated_prob[idx]

        if np.unique(yb).size < 2:
            continue

        row = {}

        try:
            row["auc_raw"] = float(roc_auc_score(yb, rb))
        except Exception:
            row["auc_raw"] = np.nan

        try:
            row["auc_calibrated"] = float(roc_auc_score(yb, pb))
        except Exception:
            row["auc_calibrated"] = np.nan

        if raw_is_probability:
            try:
                row["brier_raw"] = float(brier_score_loss(yb, rb))
            except Exception:
                row["brier_raw"] = np.nan
        else:
            row["brier_raw"] = np.nan

        try:
            row["brier_calibrated"] = float(brier_score_loss(yb, pb))
        except Exception:
            row["brier_calibrated"] = np.nan

        try:
            intercept, slope = calibration_intercept_slope(yb, pb)
            row["calibration_intercept"] = float(intercept)
            row["calibration_slope"] = float(slope)
        except Exception:
            row["calibration_intercept"] = np.nan
            row["calibration_slope"] = np.nan

        boot_rows.append(row)

    boot_df = pd.DataFrame(boot_rows)

    def ci(series, alpha=0.05):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) == 0:
            return (np.nan, np.nan)
        return (
            float(np.quantile(s, alpha / 2.0)),
            float(np.quantile(s, 1.0 - alpha / 2.0))
        )

    ci_summary = {
        "auc_raw_ci": ci(boot_df["auc_raw"]),
        "auc_calibrated_ci": ci(boot_df["auc_calibrated"]),
        "brier_raw_ci": ci(boot_df["brier_raw"]),
        "brier_calibrated_ci": ci(boot_df["brier_calibrated"]),
        "calibration_intercept_ci": ci(boot_df["calibration_intercept"]),
        "calibration_slope_ci": ci(boot_df["calibration_slope"])
    }

    return boot_df, ci_summary


def plot_reliability_diagram(cal_table, title="Held-out test calibration curve", outfile=None, show=True):
    lim = nice_axis_limit(
        max(
            cal_table["mean_pred"].max(),
            cal_table["obs_rate_hi"].max()
        )
    )

    x = pd.to_numeric(cal_table["mean_pred"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(cal_table["obs_rate"], errors="coerce").to_numpy(dtype=float)
    lo = pd.to_numeric(cal_table["obs_rate_lo"], errors="coerce").to_numpy(dtype=float)
    hi = pd.to_numeric(cal_table["obs_rate_hi"], errors="coerce").to_numpy(dtype=float)

    lo = np.minimum(lo, y)
    hi = np.maximum(hi, y)
    yerr_low = np.maximum(0.0, y - lo)
    yerr_high = np.maximum(0.0, hi - y)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr_low) & np.isfinite(yerr_high)
    x = x[finite]
    y = y[finite]
    yerr_low = yerr_low[finite]
    yerr_high = yerr_high[finite]

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.plot([0, lim], [0, lim], "--", linewidth=1)
    ax.errorbar(
        x,
        y,
        yerr=np.vstack([yerr_low, yerr_high]),
        fmt="o-",
        linewidth=1.5,
        markersize=4,
        capsize=3
    )
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title(title)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if outfile is not None:
        fig.savefig(outfile, bbox_inches="tight", dpi=300, transparent=True)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax



def format_calibration_summary_df(summary):
    """
    Convert the summary Series into a 2-column dataframe:
      variable | value_ci

    The right column is a string with 3 decimals:
      value (ci_low, ci_high)

    If no CI exists for a metric, it returns just:
      value
    """
    summary = pd.Series(summary)

    rows = []

    ci_map = {
        "auc_raw_test": ("auc_raw_ci_low", "auc_raw_ci_high"),
        "auc_calibrated_test": ("auc_calibrated_ci_low", "auc_calibrated_ci_high"),
        "brier_raw_test": ("brier_raw_ci_low", "brier_raw_ci_high"),
        "brier_calibrated_test": ("brier_calibrated_ci_low", "brier_calibrated_ci_high"),
        "calibration_intercept_test": ("calibration_intercept_ci_low", "calibration_intercept_ci_high"),
        "calibration_slope_test": ("calibration_slope_ci_low", "calibration_slope_ci_high"),
    }

    skip_keys = {
        "auc_raw_ci_low", "auc_raw_ci_high",
        "auc_calibrated_ci_low", "auc_calibrated_ci_high",
        "brier_raw_ci_low", "brier_raw_ci_high",
        "brier_calibrated_ci_low", "brier_calibrated_ci_high",
        "calibration_intercept_ci_low", "calibration_intercept_ci_high",
        "calibration_slope_ci_low", "calibration_slope_ci_high",
    }

    for key, val in summary.items():
        if key in skip_keys:
            continue

        if pd.isna(val):
            value_str = ""
        else:
            value_str = f"{float(val):.3f}"

        if key in ci_map:
            lo_key, hi_key = ci_map[key]
            lo = summary.get(lo_key, np.nan)
            hi = summary.get(hi_key, np.nan)

            if pd.notna(lo) and pd.notna(hi):
                value_str = f"{float(val):.3f} ({float(lo):.3f}, {float(hi):.3f})"

        rows.append({
            "variable": str(key),
            "value": value_str,
        })

    return pd.DataFrame(rows, columns=["variable", "value"])



def heldout_isotonic_calibration_with_bootstrap(
    df,
    score_col="predicted_risk",
    label_col="target",
    test_size=0.25,
    random_state=42,
    lower_score_is_risk=False,
    target_prevalence=None,
    n_bins=10,
    n_boot=1000,
    plot=True,
    calibration_df_path=None,
    calibration_df_index=False,
):
    work = df[[score_col, label_col]].dropna().copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work[np.isfinite(work[score_col])].copy()
    work[label_col] = ensure_binary_01(work[label_col])

    if work[label_col].nunique() < 2:
        raise ValueError("Need both classes present.")

    train_df, test_df = train_test_split(
        work,
        test_size=test_size,
        stratify=work[label_col],
        random_state=random_state
    )

    x_train = train_df[score_col].to_numpy(dtype=float)
    x_test = test_df[score_col].to_numpy(dtype=float)
    y_train = train_df[label_col].to_numpy(dtype=int)
    y_test = test_df[label_col].to_numpy(dtype=int)

    if lower_score_is_risk:
        x_train_fit = -x_train
        x_test_fit = -x_test
    else:
        x_train_fit = x_train
        x_test_fit = x_test

    w_train = make_prevalence_weights(y_train, target_prevalence=target_prevalence)

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(x_train_fit, y_train, sample_weight=w_train)

    test_df = test_df.copy()
    test_df["calibrated_prob"] = iso.predict(x_test_fit)

    cal_tbl = calibration_table(
        prob=test_df["calibrated_prob"].to_numpy(dtype=float),
        y=test_df[label_col].to_numpy(dtype=int),
        n_bins=n_bins
    )

    calibration_df_path = _as_path_or_none(calibration_df_path)
    if calibration_df_path is not None:
        calibration_df_path.parent.mkdir(parents=True, exist_ok=True)
        # cal_tbl already includes obs_rate_lo and obs_rate_hi CI bounds
        cal_tbl.to_csv(calibration_df_path, index=calibration_df_index)

    auc_raw = float(roc_auc_score(y_test, x_test_fit))
    auc_calibrated = float(roc_auc_score(y_test, test_df["calibrated_prob"]))

    raw_score_is_probability = np.all((x_test >= 0.0) & (x_test <= 1.0))
    if raw_score_is_probability:
        brier_raw = float(brier_score_loss(y_test, x_test))
    else:
        brier_raw = np.nan

    brier_calibrated = float(brier_score_loss(y_test, test_df["calibrated_prob"]))

    cal_intercept, cal_slope = calibration_intercept_slope(
        y_test,
        test_df["calibrated_prob"].to_numpy(dtype=float)
    )

    boot_df, ci_summary = bootstrap_test_metrics(
        y_true=y_test,
        raw_score=x_test,
        calibrated_prob=test_df["calibrated_prob"].to_numpy(dtype=float),
        n_boot=n_boot,
        random_state=random_state
    )

    summary = pd.Series({
        "n_total": int(len(work)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "train_prevalence": float(y_train.mean()),
        "test_prevalence": float(y_test.mean()),
        "auc_raw_test": auc_raw,
        "auc_calibrated_test": auc_calibrated,
        "brier_raw_test": brier_raw,
        "brier_calibrated_test": brier_calibrated,
        "calibration_intercept_test": cal_intercept,
        "calibration_slope_test": cal_slope,
        "auc_raw_ci_low": ci_summary["auc_raw_ci"][0],
        "auc_raw_ci_high": ci_summary["auc_raw_ci"][1],
        "auc_calibrated_ci_low": ci_summary["auc_calibrated_ci"][0],
        "auc_calibrated_ci_high": ci_summary["auc_calibrated_ci"][1],
        "brier_raw_ci_low": ci_summary["brier_raw_ci"][0],
        "brier_raw_ci_high": ci_summary["brier_raw_ci"][1],
        "brier_calibrated_ci_low": ci_summary["brier_calibrated_ci"][0],
        "brier_calibrated_ci_high": ci_summary["brier_calibrated_ci"][1],
        "calibration_intercept_ci_low": ci_summary["calibration_intercept_ci"][0],
        "calibration_intercept_ci_high": ci_summary["calibration_intercept_ci"][1],
        "calibration_slope_ci_low": ci_summary["calibration_slope_ci"][0],
        "calibration_slope_ci_high": ci_summary["calibration_slope_ci"][1],
    })

    plot_path = _as_path_or_none(plot)
    if plot is not False:
        plot_reliability_diagram(
            cal_tbl,
            title="Held-out test calibration curve",
            outfile=plot_path,
            show=True,
        )

    return {
        "train_df": train_df,
        "test_df": test_df,
        "calibration_table": cal_tbl,
        "summary": format_calibration_summary_df(summary),
        "bootstrap_samples": boot_df,
        "iso_model": iso,
        "plot_file": str(plot_path) if plot_path is not None else None,
        "calibration_table_file": str(calibration_df_path) if calibration_df_path is not None else None,
    }


