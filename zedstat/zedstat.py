import numpy as np
import pandas as pd

from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as sk_auc
from sklearn.model_selection import StratifiedKFold


__all__ = [
    'processRoc',
    'genroc',
    'pipeline',
    'score_to_probability',
    'score_to_threshold_ppv',
    'crossfit_isotonic_probabilities',
    'calibration_curve_table',
]


# -----------------------------------------------------------------------------
# basic helpers
# -----------------------------------------------------------------------------

def _safe_divide(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast(num, den).shape, np.nan, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.isfinite(num) & np.isfinite(den) & (den != 0)
        out[mask] = num[mask] / den[mask]
    return out


def _clip01(arr):
    return np.clip(np.asarray(arr, dtype=float), 0.0, 1.0)


def _wilson_interval_from_proportion(p, n, alpha=0.05):
    p = _clip01(p)
    if n is None or n <= 0:
        nan = np.full_like(p, np.nan, dtype=float)
        return nan, nan

    z = norm.ppf(1 - alpha / 2.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    radius = z * np.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n))) / denom
    return _clip01(center - radius), _clip01(center + radius)


def _wilson_interval_from_proportion_variable_n(p, n, alpha=0.05):
    """Vectorized Wilson interval allowing a different effective n per point.

    Parameters
    ----------
    p : array-like
        Proportion estimate(s).
    n : array-like
        Effective denominator(s). Can be non-integer for prevalence-standardized
        operating-characteristic displays.
    alpha : float
        Significance level.
    """
    p = _clip01(p)
    n = np.asarray(n, dtype=float)
    p, n = np.broadcast_arrays(p, n)

    lo = np.full(p.shape, np.nan, dtype=float)
    hi = np.full(p.shape, np.nan, dtype=float)

    mask = np.isfinite(p) & np.isfinite(n) & (n > 0)
    if not np.any(mask):
        return lo, hi

    z = norm.ppf(1 - alpha / 2.0)
    z2 = z * z
    denom = 1.0 + z2 / n[mask]
    center = (p[mask] + z2 / (2.0 * n[mask])) / denom
    radius = z * np.sqrt((p[mask] * (1.0 - p[mask]) / n[mask]) + (z2 / (4.0 * n[mask] * n[mask]))) / denom

    lo[mask] = center - radius
    hi[mask] = center + radius
    return _clip01(lo), _clip01(hi)


def _direct_ppv_bounds_from_expected_flags(fpr, tpr, prevalence, total_samples, alpha=0.05, min_expected_flags=1.0):
    """Direct Wilson interval for PPV among the expected flagged set.

    The older propagated interval combines independent Wilson intervals for TPR
    and specificity. At the degenerate origin, this can produce ppv_upper=1
    even when the classifier flags no one. This function treats PPV directly as
    the event fraction among flagged subjects under the target prevalence.

    flagged fraction = prevalence * TPR + (1 - prevalence) * FPR
    PPV = prevalence * TPR / flagged fraction

    If total_samples is available, the effective denominator is
    total_samples * flagged fraction. Rows with fewer than min_expected_flags
    expected flags are undefined for PPV and are returned as NaN.
    """
    fpr = _clip01(fpr)
    tpr = _clip01(tpr)
    p = float(prevalence)
    flagged_fraction = p * tpr + (1.0 - p) * fpr
    ppv = _safe_divide(p * tpr, flagged_fraction)

    if total_samples is None or not np.isfinite(total_samples) or total_samples <= 0:
        nan = np.full_like(ppv, np.nan, dtype=float)
        return nan, nan

    n_eff = float(total_samples) * flagged_fraction
    lo, hi = _wilson_interval_from_proportion_variable_n(ppv, n_eff, alpha=alpha)

    invalid = (~np.isfinite(ppv)) | (~np.isfinite(n_eff)) | (n_eff < float(min_expected_flags)) | (flagged_fraction <= 0)
    lo[invalid] = np.nan
    hi[invalid] = np.nan
    return lo, hi


def _enforce_bounds_around_nominal(nominal, lower, upper, cols=None, clip01_cols=('tpr', 'ppv', 'acc', 'npv')):
    """Ensure lower <= nominal <= upper for displayed pointwise bounds.

    This is a display/post-processing consistency step. It is especially useful
    when the nominal curve is monotone-corrected but the pointwise analytic
    bounds are computed before that correction.
    """
    nominal = nominal.copy()
    lower = lower.copy()
    upper = upper.copy()

    if cols is None:
        cols = [c for c in nominal.columns if c in lower.columns and c in upper.columns]

    for col in cols:
        n = pd.to_numeric(nominal[col], errors='coerce').to_numpy(dtype=float)
        lo = pd.to_numeric(lower[col], errors='coerce').to_numpy(dtype=float)
        hi = pd.to_numeric(upper[col], errors='coerce').to_numpy(dtype=float)

        finite_n = np.isfinite(n)
        finite_all = finite_n & np.isfinite(lo) & np.isfinite(hi)

        lo2 = lo.copy()
        hi2 = hi.copy()
        lo2[finite_all] = np.minimum.reduce([lo[finite_all], hi[finite_all], n[finite_all]])
        hi2[finite_all] = np.maximum.reduce([lo[finite_all], hi[finite_all], n[finite_all]])

        # If the nominal value is undefined, the corresponding bounds should not
        # be plotted or interpreted. This fixes rows such as fpr=0,tpr=0 where
        # PPV is undefined but a propagated upper bound may otherwise be 1.
        lo2[~finite_n] = np.nan
        hi2[~finite_n] = np.nan

        if col in clip01_cols:
            lo2 = _clip01(lo2)
            hi2 = _clip01(hi2)

        lower[col] = lo2
        upper[col] = hi2

    return lower, upper


def _hanley_mcneil_auc_ci(auc_value, n_pos, n_neg, alpha=0.05):
    if n_pos is None or n_neg is None or n_pos <= 0 or n_neg <= 0:
        return np.nan, np.nan

    auc_value = float(np.clip(auc_value, 0.0, 1.0))
    q1 = auc_value / (2.0 - auc_value) if auc_value < 1.0 else 1.0
    q2 = (2.0 * auc_value * auc_value) / (1.0 + auc_value) if auc_value > 0.0 else 0.0
    var_auc = (
        auc_value * (1.0 - auc_value)
        + (n_pos - 1.0) * (q1 - auc_value * auc_value)
        + (n_neg - 1.0) * (q2 - auc_value * auc_value)
    ) / (n_pos * n_neg)
    var_auc = max(var_auc, 0.0)
    se_auc = np.sqrt(var_auc)
    z = norm.ppf(1 - alpha / 2.0)
    return max(0.0, auc_value - z * se_auc), min(1.0, auc_value + z * se_auc)


def _ensure_binary_labels(y):
    if y.dtype == bool:
        return y.astype(int)
    if pd.api.types.is_numeric_dtype(y):
        vals = pd.Series(y).dropna().unique()
        if set(vals.tolist()).issubset({0, 1}):
            return y.astype(int)
    raise ValueError('Label column must be binary and coded as 0/1 or bool.')


def _make_prevalence_weights(y, target_prev=None):
    y = np.asarray(y).astype(int)
    if target_prev is None:
        return np.ones_like(y, dtype=float)
    sample_prev = y.mean()
    if sample_prev <= 0 or sample_prev >= 1:
        raise ValueError('Sample must contain both cases and controls.')
    if not (0 < target_prev < 1):
        raise ValueError('target_prevalence must be between 0 and 1.')
    return np.where(y == 1, target_prev / sample_prev, (1.0 - target_prev) / (1.0 - sample_prev)).astype(float)


def _weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    return float(np.sum(w * x) / np.sum(w))


def _choose_n_splits(y, requested):
    y = np.asarray(y).astype(int)
    min_class = min((y == 0).sum(), (y == 1).sum())
    n_splits = min(int(requested), min_class)
    if n_splits < 2:
        raise ValueError('Need at least 2 samples in each class for cross-fitted calibration.')
    return n_splits

# -----------------------------------------------------------------------------
# roc helpers
# -----------------------------------------------------------------------------

def _prepare_roc_dataframe(df, fprcol='fpr', tprcol='tpr', thresholdcol='threshold'):
    if df is None:
        raise ValueError('df must not be None')

    if df.index.name == fprcol:
        work = df.reset_index().copy()
    elif fprcol in df.columns:
        work = df.copy()
    else:
        raise ValueError(f'{fprcol} not in columns or index')

    keep = [fprcol, tprcol]
    if thresholdcol in work.columns:
        keep.append(thresholdcol)
    work = work[keep].copy()
    work[fprcol] = pd.to_numeric(work[fprcol], errors='coerce')
    work[tprcol] = pd.to_numeric(work[tprcol], errors='coerce')
    if thresholdcol in work.columns:
        work[thresholdcol] = pd.to_numeric(work[thresholdcol], errors='coerce')
    work = work.dropna(subset=[fprcol, tprcol])
    work[fprcol] = _clip01(work[fprcol].values)
    work[tprcol] = _clip01(work[tprcol].values)

    work = (
        work.sort_values([fprcol, tprcol], ascending=[True, False])
        .groupby(fprcol, as_index=False)
        .first()
        .sort_values(fprcol)
        .reset_index(drop=True)
    )
    if thresholdcol not in work.columns:
        thresholdcol = None
    return work, thresholdcol


def _add_endpoints(df, fprcol='fpr', tprcol='tpr', thresholdcol=None):
    work = df.copy()
    extras = []
    has_origin = (np.isclose(work[fprcol], 0.0)).any() and np.isclose(work.loc[np.isclose(work[fprcol], 0.0), tprcol].max(), 0.0)
    has_one = (np.isclose(work[fprcol], 1.0)).any() and np.isclose(work.loc[np.isclose(work[fprcol], 1.0), tprcol].max(), 1.0)
    if not has_origin:
        row = {fprcol: 0.0, tprcol: 0.0}
        if thresholdcol is not None:
            row[thresholdcol] = np.nan
        extras.append(row)
    if not has_one:
        row = {fprcol: 1.0, tprcol: 1.0}
        if thresholdcol is not None:
            row[thresholdcol] = np.nan
        extras.append(row)
    if extras:
        work = pd.concat([work, pd.DataFrame(extras)], ignore_index=True)
    work = (
        work.sort_values([fprcol, tprcol], ascending=[True, False])
        .groupby(fprcol, as_index=False)
        .first()
        .sort_values(fprcol)
        .reset_index(drop=True)
    )
    return work


def _upper_roc_hull(df, fprcol='fpr', tprcol='tpr', thresholdcol=None):
    work = _add_endpoints(df, fprcol, tprcol, thresholdcol)
    pts = work[[fprcol, tprcol]].to_numpy(dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    hull_idx = []
    for i in range(len(pts)):
        while len(hull_idx) >= 2 and cross(pts[hull_idx[-2]], pts[hull_idx[-1]], pts[i]) >= 0:
            hull_idx.pop()
        hull_idx.append(i)
    hull = work.iloc[hull_idx].copy().reset_index(drop=True)
    hull[tprcol] = _clip01(np.maximum.accumulate(hull[tprcol].values))
    return hull.sort_values(fprcol).reset_index(drop=True)


def _resample_on_fpr_grid(df, grid, fprcol='fpr', tprcol='tpr', thresholdcol=None):
    work = df.sort_values(fprcol).reset_index(drop=True)
    x = work[fprcol].to_numpy(dtype=float)
    y = work[tprcol].to_numpy(dtype=float)
    grid = np.asarray(grid, dtype=float)
    out = pd.DataFrame({fprcol: grid})
    out[tprcol] = _clip01(np.maximum.accumulate(np.interp(grid, x, y)))
    if thresholdcol is not None and thresholdcol in work.columns:
        th = pd.to_numeric(work[thresholdcol], errors='coerce').to_numpy(dtype=float)
        mask = np.isfinite(th)
        if mask.sum() >= 2:
            out[thresholdcol] = np.interp(grid, x[mask], th[mask])
        elif mask.sum() == 1:
            out[thresholdcol] = th[mask][0]
        else:
            out[thresholdcol] = np.nan
    return out


def _numeric_interp_frame_to_target_index(df, target_index, index_name='fpr'):
    out = pd.DataFrame(index=np.asarray(target_index, dtype=float))
    out.index.name = index_name
    work = df.copy()
    if work.index.name != index_name:
        work = work.set_index(index_name)
    work = work.sort_index()
    x = work.index.to_numpy(dtype=float)
    for col in work.columns:
        vals = pd.to_numeric(work[col], errors='coerce').to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(vals)
        if mask.sum() >= 2:
            out[col] = np.interp(out.index.values, x[mask], vals[mask])
        elif mask.sum() == 1:
            out[col] = vals[mask][0]
        else:
            out[col] = np.nan
    return out


def _apply_lr_floor(df, lr_fpr_floor, fprcol='fpr', lr_sp_floor=None):
    """
    Mask numerically unstable likelihood-ratio regions.

    LR+ = TPR / FPR is unstable when FPR is close to zero.
    LR- = (1 - TPR) / specificity is unstable when specificity = 1-FPR
    is close to zero.

    The old implementation masked both LR+ and LR- at low FPR. That created
    odd LR- behavior and did not remove the high-FPR endpoint where LR- is
    undefined or dominated by numerical extrapolation.
    """
    work = df.copy()
    if lr_sp_floor is None:
        lr_sp_floor = lr_fpr_floor

    if work.index.name == fprcol:
        fpr_vals = work.index.to_numpy(dtype=float)
    else:
        fpr_vals = pd.to_numeric(work[fprcol], errors='coerce').to_numpy(dtype=float)

    fpr_vals = np.asarray(fpr_vals, dtype=float)
    sp_vals = 1.0 - fpr_vals

    bad_lr_plus = (~np.isfinite(fpr_vals)) | (fpr_vals <= float(lr_fpr_floor))
    bad_lr_minus = (~np.isfinite(sp_vals)) | (sp_vals <= float(lr_sp_floor))

    for col in ['LR+', 'LR+_raw']:
        if col in work.columns:
            work.loc[bad_lr_plus, col] = np.nan
    for col in ['LR-', 'LR-_raw']:
        if col in work.columns:
            work.loc[bad_lr_minus, col] = np.nan
    return work


def _compute_measures_from_arrays(fpr, tpr, prevalence, lr_fpr_floor=0.001, lr_sp_floor=None):
    fpr = _clip01(fpr)
    tpr = _clip01(tpr)
    sp = 1.0 - fpr
    p = float(prevalence)

    ppv = _safe_divide(tpr * p, tpr * p + fpr * (1.0 - p))
    acc = p * tpr + (1.0 - p) * sp
    npv = _safe_divide(sp * (1.0 - p), sp * (1.0 - p) + (1.0 - tpr) * p)
    lr_plus = _safe_divide(tpr, fpr)
    lr_minus = _safe_divide(1.0 - tpr, sp)

    frame = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        'ppv': ppv,
        'acc': acc,
        'npv': npv,
        'LR+': lr_plus,
        'LR-': lr_minus,
    })
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = _apply_lr_floor(frame, lr_fpr_floor=lr_fpr_floor, lr_sp_floor=lr_sp_floor, fprcol='fpr')
    return frame


def _compute_pointwise_measure_frames(df, prevalence, alpha, total_samples, positive_samples, thresholdcol='threshold', fprcol='fpr', tprcol='tpr', lr_fpr_floor=0.001, lr_sp_floor=None, ppv_ci_method='direct', min_expected_flags=1.0):
    work, thresholdcol = _prepare_roc_dataframe(df, fprcol=fprcol, tprcol=tprcol, thresholdcol=thresholdcol)
    work = _add_endpoints(work, fprcol=fprcol, tprcol=tprcol, thresholdcol=thresholdcol)

    fpr = work[fprcol].to_numpy(dtype=float)
    tpr = work[tprcol].to_numpy(dtype=float)

    n_pos = positive_samples
    n_neg = None if total_samples is None or positive_samples is None else int(total_samples - positive_samples)

    tpr_low, tpr_high = _wilson_interval_from_proportion(tpr, n_pos, alpha=alpha)
    sp = 1.0 - fpr
    sp_low, sp_high = _wilson_interval_from_proportion(sp, n_neg, alpha=alpha)
    fpr_low = 1.0 - sp_high
    fpr_high = 1.0 - sp_low

    nominal = _compute_measures_from_arrays(fpr, tpr, prevalence=prevalence, lr_fpr_floor=lr_fpr_floor, lr_sp_floor=lr_sp_floor)
    lower = pd.DataFrame({'fpr': fpr})
    upper = pd.DataFrame({'fpr': fpr})

    lower['tpr'] = tpr_low
    upper['tpr'] = tpr_high

    if ppv_ci_method == 'direct':
        # Direct PPV interval among the expected flagged set. This avoids the
        # pathological ppv_upper=1 at thresholds where no one is flagged.
        ppv_low, ppv_high = _direct_ppv_bounds_from_expected_flags(
            fpr=fpr,
            tpr=tpr,
            prevalence=prevalence,
            total_samples=total_samples,
            alpha=alpha,
            min_expected_flags=min_expected_flags,
        )
        lower['ppv'] = ppv_low
        upper['ppv'] = ppv_high
    elif ppv_ci_method == 'propagated':
        # Conservative propagated interval from independent TPR and specificity
        # Wilson intervals. This is retained for backward compatibility but can
        # be visually unstable near fpr=0.
        lower['ppv'] = _safe_divide(tpr_low * prevalence, tpr_low * prevalence + fpr_high * (1.0 - prevalence))
        upper['ppv'] = _safe_divide(tpr_high * prevalence, tpr_high * prevalence + fpr_low * (1.0 - prevalence))
    else:
        raise ValueError("ppv_ci_method must be 'direct' or 'propagated'")

    lower['acc'] = prevalence * tpr_low + (1.0 - prevalence) * (1.0 - fpr_high)
    upper['acc'] = prevalence * tpr_high + (1.0 - prevalence) * (1.0 - fpr_low)

    lower['npv'] = _safe_divide((1.0 - fpr_low) * (1.0 - prevalence), (1.0 - fpr_low) * (1.0 - prevalence) + (1.0 - tpr_high) * prevalence)
    upper['npv'] = _safe_divide((1.0 - fpr_high) * (1.0 - prevalence), (1.0 - fpr_high) * (1.0 - prevalence) + (1.0 - tpr_low) * prevalence)

    lower['LR+'] = _safe_divide(tpr_low, fpr_high)
    upper['LR+'] = _safe_divide(tpr_high, fpr_low)

    lower['LR-'] = _safe_divide(1.0 - tpr_high, 1.0 - fpr_low)
    upper['LR-'] = _safe_divide(1.0 - tpr_low, 1.0 - fpr_high)

    for frame in [nominal, lower, upper]:
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        if thresholdcol is not None and thresholdcol in work.columns:
            frame[thresholdcol] = work[thresholdcol].to_numpy(dtype=float)
        frame.index = work[fprcol].to_numpy(dtype=float)
        frame.index.name = fprcol

    # Do this after the threshold/index is attached.
    nominal = _apply_lr_floor(nominal, lr_fpr_floor=lr_fpr_floor, lr_sp_floor=lr_sp_floor, fprcol=fprcol)
    lower = _apply_lr_floor(lower, lr_fpr_floor=lr_fpr_floor, lr_sp_floor=lr_sp_floor, fprcol=fprcol)
    upper = _apply_lr_floor(upper, lr_fpr_floor=lr_fpr_floor, lr_sp_floor=lr_sp_floor, fprcol=fprcol)

    lower, upper = _enforce_bounds_around_nominal(
        nominal,
        lower,
        upper,
        cols=['tpr', 'ppv', 'acc', 'npv', 'LR+', 'LR-'],
    )
    return nominal, lower, upper


def _bootstrap_auc(scores, labels, alpha=0.05, n_boot=2000, random_state=42, stratified=True):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    valid = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[valid]
    labels = labels[valid]
    if len(scores) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(random_state)
    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return np.nan, np.nan, np.nan

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    nominal = float(sk_auc(fpr, tpr))

    aucs = []
    for _ in range(int(n_boot)):
        if stratified:
            boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            boot_idx = np.concatenate([boot_pos, boot_neg])
        else:
            boot_idx = rng.choice(np.arange(len(scores)), size=len(scores), replace=True)
        yb = labels[boot_idx]
        sb = scores[boot_idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        fpr_b, tpr_b, _ = roc_curve(yb, sb, pos_label=1, drop_intermediate=False)
        aucs.append(float(sk_auc(fpr_b, tpr_b)))
    if len(aucs) == 0:
        return nominal, np.nan, np.nan
    lower = float(np.quantile(aucs, alpha / 2.0))
    upper = float(np.quantile(aucs, 1.0 - alpha / 2.0))
    return nominal, upper, lower


# -----------------------------------------------------------------------------
# main class
# -----------------------------------------------------------------------------

class processRoc(object):
    def __init__(self,
                 df=None,
                 fprcol='fpr',
                 tprcol='tpr',
                 thresholdcol='threshold',
                 prevalence=None,
                 order=2,
                 total_samples=None,
                 positive_samples=None,
                 alpha=0.05,
                 lr_fpr_floor=0.001):

        self.fprcol = fprcol
        self.tprcol = tprcol
        self.prevalence = prevalence
        self.order = order
        self.total_samples = total_samples
        self.positive_samples = positive_samples
        self.alpha = alpha
        self.lr_fpr_floor = float(lr_fpr_floor)
        self.lr_sp_floor = float(lr_fpr_floor)

        work, thresholdcol = _prepare_roc_dataframe(df=df, fprcol=fprcol, tprcol=tprcol, thresholdcol=thresholdcol)
        self.thresholdcol = thresholdcol
        self.raw_df = work.copy().set_index(self.fprcol)
        self.df = self.raw_df.copy()
        self.df_lim = {}
        self.delta_ = None
        self._auc = {'U': np.nan, 'L': np.nan}
        self._operating_zone = pd.DataFrame()
        self.df_measure_bounds_ = {}
        self._calibration = None
        self._auc_bootstrap_data = None

    def set_lr_fpr_floor(self, lr_fpr_floor):
        self.lr_fpr_floor = float(lr_fpr_floor)
        return self

    def set_lr_sp_floor(self, lr_sp_floor):
        self.lr_sp_floor = float(lr_sp_floor)
        return self

    def set_auc_bootstrap_data(self, df, score_col, label_col, lower_score_is_risk=False):
        work = df[[score_col, label_col]].dropna().copy()
        work[label_col] = _ensure_binary_labels(work[label_col])
        scores = pd.to_numeric(work[score_col], errors='coerce').to_numpy(dtype=float)
        labels = work[label_col].to_numpy(dtype=int)
        valid = np.isfinite(scores)
        scores = scores[valid]
        labels = labels[valid]
        if lower_score_is_risk:
            scores = -scores
        self._auc_bootstrap_data = {'scores': scores, 'labels': labels}
        return self

    def get(self):
        return self.df.copy()

    def _current_df(self):
        if self.df.index.name == self.fprcol:
            work = self.df.reset_index().copy()
        else:
            work = self.df.copy()
        return work.sort_values(self.fprcol).reset_index(drop=True)

    def nominal_auc(self):
        work = self._current_df()
        self._auc['nominal'] = float(sk_auc(work[self.fprcol].values, work[self.tprcol].values))
        return self._auc['nominal']

    def auc(self, total_samples=None, positive_samples=None, alpha=None, bootstrap=False, n_boot=2000, random_state=42, stratified=True):
        nominal = self.nominal_auc()
        if bootstrap:
            if self._auc_bootstrap_data is None:
                raise ValueError('Bootstrap AUC requested, but no raw score/label data are attached. Use fit_score_calibration(...) or set_auc_bootstrap_data(...).')
            nominal_b, upper_b, lower_b = _bootstrap_auc(
                self._auc_bootstrap_data['scores'],
                self._auc_bootstrap_data['labels'],
                alpha=self.alpha if alpha is None else alpha,
                n_boot=n_boot,
                random_state=random_state,
                stratified=stratified,
            )
            self._auc['nominal'] = nominal_b
            self._auc['U'] = upper_b
            self._auc['L'] = lower_b
            return nominal_b, upper_b, lower_b

        if total_samples is None:
            total_samples = self.total_samples
        if positive_samples is None:
            positive_samples = self.positive_samples
        if alpha is None:
            alpha = self.alpha
        n_neg = None if total_samples is None or positive_samples is None else int(total_samples - positive_samples)
        lower, upper = _hanley_mcneil_auc_ci(nominal, positive_samples, n_neg, alpha=alpha)
        self._auc['U'] = upper
        self._auc['L'] = lower
        return nominal, upper, lower

    def __convexify(self):
        work = self._current_df()
        hull = _upper_roc_hull(work, fprcol=self.fprcol, tprcol=self.tprcol, thresholdcol=self.thresholdcol)
        self.df = hull.set_index(self.fprcol)
        return

    def smooth(self, STEP=0.0001, interpolate=True, convexify=True):
        work = self.raw_df.reset_index().copy()
        if convexify:
            work = _upper_roc_hull(work, fprcol=self.fprcol, tprcol=self.tprcol, thresholdcol=self.thresholdcol)
        else:
            work = _add_endpoints(work, fprcol=self.fprcol, tprcol=self.tprcol, thresholdcol=self.thresholdcol)
            work[self.tprcol] = np.maximum.accumulate(work[self.tprcol].values)
        if interpolate:
            step = float(STEP)
            if step <= 0:
                raise ValueError('STEP must be positive')
            grid = np.arange(0.0, 1.0 + step / 2.0, step)
            grid[-1] = 1.0
            work = _resample_on_fpr_grid(work, grid=grid, fprcol=self.fprcol, tprcol=self.tprcol, thresholdcol=self.thresholdcol)
        self.df = work.set_index(self.fprcol).sort_index()
        return









    def __correctPPV(self, df=None):
        """
        Enforce monotone PPV using weighted isotonic regression.

        Robust to frames that already contain self.fprcol both as index name
        and as a regular column.
        """
        if df is None:
            work = self.df.copy()
            reset_self = True
        else:
            work = df.copy()
            reset_self = False

        if 'ppv' not in work.columns:
            return work

        # Normalize so we have exactly one FPR field to sort on.
        if work.index.name == self.fprcol:
            if self.fprcol in work.columns:
                # Keep the existing column, discard the duplicate index identity.
                work = work.copy()
                work.index.name = None
                work = work.reset_index(drop=True)
            else:
                work = work.reset_index()
        elif self.fprcol not in work.columns:
            raise ValueError(f"{self.fprcol} must be present either as index or column")

        work = work.sort_values(self.fprcol).reset_index(drop=True)

        ppv = pd.to_numeric(work['ppv'], errors='coerce').to_numpy(dtype=float)
        valid = np.isfinite(ppv)

        if valid.sum() >= 2:
            if self.prevalence is not None and self.tprcol in work.columns and self.fprcol in work.columns:
                flagged_fraction = (
                    self.prevalence * pd.to_numeric(work[self.tprcol], errors='coerce').to_numpy(dtype=float)
                    + (1.0 - self.prevalence) * pd.to_numeric(work[self.fprcol], errors='coerce').to_numpy(dtype=float)
                )
                if self.total_samples is not None and self.total_samples > 0:
                    weights = np.maximum(flagged_fraction * self.total_samples, 1.0)
                else:
                    weights = np.maximum(flagged_fraction, 1e-8)
            else:
                weights = np.ones_like(ppv, dtype=float)

            iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
            fitted = iso.fit_transform(
                np.flatnonzero(valid),
                ppv[valid],
                sample_weight=weights[valid],
            )

            ppv_corrected = ppv.copy()
            ppv_corrected[valid] = fitted
            work['ppv'] = np.clip(ppv_corrected, 0.0, 1.0)

        if reset_self:
            self.df = work.set_index(self.fprcol)
            return self.df

        return work.set_index(self.fprcol)





  







    
    def _compute_measures(self, df, prevalence=None, apply_ppv_isotonic=True):
        if prevalence is None:
            prevalence = self.prevalence
        if prevalence is None:
            raise ValueError('prevalence undefined')
        work = df.copy()
        if work.index.name == self.fprcol:
            work = work.reset_index()
        work = work.sort_values(self.fprcol).reset_index(drop=True)
        fpr = _clip01(work[self.fprcol].to_numpy(dtype=float))
        tpr = _clip01(work[self.tprcol].to_numpy(dtype=float))
        out = _compute_measures_from_arrays(fpr, tpr, prevalence=prevalence, lr_fpr_floor=self.lr_fpr_floor)
        if self.thresholdcol is not None and self.thresholdcol in work.columns:
            out[self.thresholdcol] = pd.to_numeric(work[self.thresholdcol], errors='coerce').to_numpy(dtype=float)
        out = out.set_index('fpr')
        out.index.name = self.fprcol
        if apply_ppv_isotonic:
            out = self.__correctPPV(out)
        return out

    def allmeasures(self, prevalence=None, interpolate=False):
        if prevalence is not None:
            self.prevalence = prevalence
        work = self.df.copy()
        if interpolate:
            if work.index.name != self.fprcol:
                work = work.set_index(self.fprcol)
            work = work.sort_index().interpolate(limit_direction='both')
        self.df = self._compute_measures(work, prevalence=self.prevalence, apply_ppv_isotonic=True)
        return

    def usample(self, df=None, precision=3, recompute_measures=True):
        step = 10 ** (-precision)
        grid = np.round(np.arange(0.0, 1.0 + step, step), precision)
        grid[-1] = 1.0
        source = self.df.copy() if df is None else df.copy()
        if source.index.name == self.fprcol:
            source = source.reset_index()
        source = source.sort_values(self.fprcol).reset_index(drop=True)

        out = pd.DataFrame({self.fprcol: grid})
        x = pd.to_numeric(source[self.fprcol], errors='coerce').to_numpy(dtype=float)
        for col in source.columns:
            if col == self.fprcol:
                continue
            vals = pd.to_numeric(source[col], errors='coerce').to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(vals)
            if mask.sum() >= 2:
                out[col] = np.interp(grid, x[mask], vals[mask])
            elif mask.sum() == 1:
                out[col] = vals[mask][0]
            else:
                out[col] = np.nan

        if self.tprcol in out.columns:
            out[self.tprcol] = _clip01(np.maximum.accumulate(out[self.tprcol].values))

        # Important: do not interpolate derived ratios such as LR+ and LR-.
        # Interpolating them extrapolates finite values into endpoint rows
        # where the ratio is undefined, producing the spurious LR curve branch.
        # Instead, interpolate only the ROC coordinates and recompute all
        # prevalence-derived measures from the resampled FPR/TPR curve.
        if recompute_measures and self.prevalence is not None and self.tprcol in out.columns:
            recomputed = _compute_measures_from_arrays(
                out[self.fprcol].to_numpy(dtype=float),
                out[self.tprcol].to_numpy(dtype=float),
                prevalence=self.prevalence,
                lr_fpr_floor=self.lr_fpr_floor,
                lr_sp_floor=getattr(self, 'lr_sp_floor', self.lr_fpr_floor),
            )
            for col in ['ppv', 'acc', 'npv', 'LR+', 'LR-']:
                out[col] = recomputed[col].to_numpy(dtype=float)

        out = out.set_index(self.fprcol)
        if 'ppv' in out.columns:
            out = self.__correctPPV(out)
        out = _apply_lr_floor(
            out,
            lr_fpr_floor=self.lr_fpr_floor,
            lr_sp_floor=getattr(self, 'lr_sp_floor', self.lr_fpr_floor),
            fprcol=self.fprcol,
        )
        if df is None:
            self.df = out
        return out

    def getBounds(self, total_samples=None, positive_samples=None, alpha=None, prevalence=None, ppv_ci_method='direct', min_expected_flags=1.0, enforce_bounds=True):
        if total_samples is None:
            total_samples = self.total_samples
        if positive_samples is None:
            positive_samples = self.positive_samples
        if alpha is None:
            alpha = self.alpha
        if prevalence is None:
            prevalence = self.prevalence
        if prevalence is None:
            raise ValueError('prevalence undefined')

        empirical = self.raw_df.reset_index().copy()
        nominal_emp, lower_emp, upper_emp = _compute_pointwise_measure_frames(
            empirical,
            prevalence=float(prevalence),
            alpha=alpha,
            total_samples=total_samples,
            positive_samples=positive_samples,
            thresholdcol=self.thresholdcol,
            fprcol=self.fprcol,
            tprcol=self.tprcol,
            lr_fpr_floor=self.lr_fpr_floor,
            lr_sp_floor=getattr(self, 'lr_sp_floor', self.lr_fpr_floor),
            ppv_ci_method=ppv_ci_method,
            min_expected_flags=min_expected_flags,
        )
        # Only the displayed nominal PPV is monotone-corrected.
        # Bounds are not isotonic-corrected independently; doing so can make
        # the nominal curve and bounds inconsistent. Bounds are enforced below.
        nominal_emp = self.__correctPPV(nominal_emp)

        target_index = self.df.index.values.astype(float) if self.df.index.name == self.fprcol else self.df[self.fprcol].values.astype(float)
        nominal = _numeric_interp_frame_to_target_index(nominal_emp, target_index, index_name=self.fprcol)
        lower = _numeric_interp_frame_to_target_index(lower_emp, target_index, index_name=self.fprcol)
        upper = _numeric_interp_frame_to_target_index(upper_emp, target_index, index_name=self.fprcol)
        nominal = self.__correctPPV(nominal)
        if enforce_bounds:
            lower, upper = _enforce_bounds_around_nominal(
                nominal,
                lower,
                upper,
                cols=['tpr', 'ppv', 'acc', 'npv', 'LR+', 'LR-'],
            )
        nominal = _apply_lr_floor(nominal, lr_fpr_floor=self.lr_fpr_floor, lr_sp_floor=getattr(self, 'lr_sp_floor', self.lr_fpr_floor), fprcol=self.fprcol)
        lower = _apply_lr_floor(lower, lr_fpr_floor=self.lr_fpr_floor, lr_sp_floor=getattr(self, 'lr_sp_floor', self.lr_fpr_floor), fprcol=self.fprcol)
        upper = _apply_lr_floor(upper, lr_fpr_floor=self.lr_fpr_floor, lr_sp_floor=getattr(self, 'lr_sp_floor', self.lr_fpr_floor), fprcol=self.fprcol)

        self.df_lim['L'] = lower
        self.df_lim['U'] = upper
        self.df_measure_bounds_ = {
            'nominal_empirical': nominal_emp,
            'L_empirical': lower_emp,
            'U_empirical': upper_emp,
            'nominal_display': nominal,
            'L_display': lower,
            'U_display': upper,
            'kind': 'analytic_pointwise',
            'ppv_ci_method': ppv_ci_method,
            'min_expected_flags': min_expected_flags,
            'enforce_bounds': enforce_bounds,
            'lr_fpr_floor': self.lr_fpr_floor,
        }
        return

    def fit_score_calibration(self, df, score_col, label_col, target_prevalence=None, lower_score_is_risk=False, n_splits=5, random_state=42, n_bins=10):
        work = df[[score_col, label_col]].dropna().copy()
        work[label_col] = _ensure_binary_labels(work[label_col])
        scores = pd.to_numeric(work[score_col], errors='coerce')
        work = work.loc[np.isfinite(scores)].copy()
        work[score_col] = scores.loc[work.index].astype(float)
        work['_score_for_calibration'] = work[score_col] if not lower_score_is_risk else -work[score_col]
        oof = crossfit_isotonic_probabilities(
            work.rename(columns={'_score_for_calibration': '__score__'}),
            score_col='__score__',
            label_col=label_col,
            target_prevalence=target_prevalence,
            n_splits=n_splits,
            random_state=random_state,
        )
        work['calibrated_prob_oof'] = oof
        weights = _make_prevalence_weights(work[label_col].to_numpy(dtype=int), target_prevalence)
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        iso.fit(work['_score_for_calibration'].to_numpy(dtype=float), work[label_col].to_numpy(dtype=int), sample_weight=weights)
        cal_df = calibration_curve_table(work, prob_col='calibrated_prob_oof', label_col=label_col, target_prevalence=target_prevalence, n_bins=n_bins)
        self._calibration = {
            'iso': iso,
            'score_col': score_col,
            'label_col': label_col,
            'target_prevalence': target_prevalence,
            'lower_score_is_risk': bool(lower_score_is_risk),
            'work': work,
            'curve': cal_df,
        }
        self.set_auc_bootstrap_data(work, score_col=score_col, label_col=label_col, lower_score_is_risk=lower_score_is_risk)
        return self

    def calibration_curve(self):
        if self._calibration is None:
            raise ValueError('No fitted calibration found. Run fit_score_calibration(...) first.')
        return self._calibration['curve'].copy()

    def plot_calibration_curve(self, out_png=None):
        if self._calibration is None:
            raise ValueError('No fitted calibration found. Run fit_score_calibration(...) first.')
        import matplotlib.pyplot as plt
        cal_df = self._calibration['curve']
        lim = _nice_axis_limit(max(cal_df['mean_pred'].max(), cal_df['obs_rate'].max()))
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.plot([0, lim], [0, lim], linestyle='--', linewidth=1)
        ax.plot(cal_df['mean_pred'], cal_df['obs_rate'], marker='o', linewidth=1.5, markersize=4)
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Observed event rate')
        ax.set_title('Calibration curve')
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        if out_png is not None:
            fig.savefig(out_png, dpi=200)
            plt.close(fig)
            return out_png
        return fig, ax

    def _lookup_measure_at_score(self, scores, value_cols):
        df = self.get()
        if self.thresholdcol is None or self.thresholdcol not in df.columns:
            raise ValueError('Threshold not in columns or index')
        work = df.reset_index() if df.index.name == self.fprcol else df.copy()
        work = work.dropna(subset=[self.thresholdcol]).sort_values(self.thresholdcol)
        work = work.drop_duplicates(subset=[self.thresholdcol], keep='last')
        if work.empty:
            raise ValueError('No finite thresholds available')
        x = work[self.thresholdcol].to_numpy(dtype=float)
        arrays = []
        for col in value_cols:
            if col not in work.columns:
                raise ValueError(f'{col} not in columns or index')
            arrays.append(pd.to_numeric(work[col], errors='coerce').to_numpy(dtype=float))
        def one_score(s):
            s = float(np.clip(float(s), x.min(), x.max()))
            vals = [float(np.interp(s, x, y)) for y in arrays]
            return vals[0] if len(vals) == 1 else vals
        if isinstance(scores, (list, tuple, np.ndarray, pd.Series)):
            return np.array([one_score(s) for s in scores])
        return one_score(scores)

    def score_to_threshold_ppv(self, score, regen=True, **kwargs):
        if score is None:
            return None
        if regen:
            step = kwargs.get('STEP', 0.01)
            precision = kwargs.get('precision', 3)
            interpolate = kwargs.get('interpolate', True)
            convexify = kwargs.get('convexify', False)
            self.smooth(STEP=step, convexify=convexify)
            self.allmeasures(interpolate=interpolate)
            self.usample(precision=precision)
        elif 'ppv' not in self.get().columns:
            self.allmeasures(interpolate=kwargs.get('interpolate', False))
        return self._lookup_measure_at_score(score, ['ppv'])

    def scoretoprobability(self, score, regen=True, **kwargs):
        if self._calibration is None:
            raise ValueError('No fitted calibration found. Run fit_score_calibration(...) first. For threshold PPV use score_to_threshold_ppv(...).')
        if score is None:
            return None
        iso = self._calibration['iso']
        lower_score_is_risk = self._calibration['lower_score_is_risk']
        def one_score(s):
            x = -float(s) if lower_score_is_risk else float(s)
            return float(iso.predict([x])[0])
        if isinstance(score, (list, tuple, np.ndarray, pd.Series)):
            return np.array([one_score(s) for s in score])
        return one_score(score)

    def operating_zone(self, n=1, LRplus=10, LRminus=0.6):
        wf = self.df.copy()
        mask = (wf['LR+'] > LRplus) & (wf['LR-'] < LRminus)
        opf = pd.concat([
            wf[mask].sort_values('ppv', ascending=False).head(n),
            wf[mask].sort_values(self.tprcol, ascending=False).head(n),
        ])
        if opf.empty:
            self._operating_zone = opf.copy()
            return
        opf = opf.reset_index()
        labels = ['high precision'] * min(n, len(opf)) + ['high sensitivity'] * max(0, len(opf) - min(n, len(opf)))
        opf.index = labels[:len(opf)]
        self._operating_zone = opf
        return

    def samplesize(self, delta_auc=0.1, target_auc=None, alpha=None):
        if alpha is None:
            alpha = self.alpha
        if target_auc is None:
            target_auc = self.nominal_auc()
        z = norm.ppf(1 - alpha / 2.0)
        a = float(np.clip(target_auc, 1e-6, 1 - 1e-6))
        q1 = a / (2.0 - a)
        q2 = 2.0 * a * a / (1.0 + a)
        c = a * (1.0 - a) - a * a + q1 + q2
        c = max(c, 1e-12)
        return float((z * z * c) / (delta_auc * delta_auc))


    def samplesize(self, delta_auc=0.1, target_auc=None, alpha=None, prevalence=None):
        """
        Estimate required sample size for achieving an AUC tolerance delta_auc.

        Behavior
        --------
        - If prevalence is None and self.prevalence is None:
            returns the old balanced-design approximation, interpreted as
            required samples per class.

        - If prevalence is provided, or self.prevalence is set:
            returns the required total sample size under that expected
            positive fraction.

        Parameters
        ----------
        delta_auc : float
            Desired AUC tolerance / half-width target.
        target_auc : float or None
            AUC around which planning is done. If None, uses current nominal AUC.
        alpha : float or None
            Significance level. If None, uses self.alpha.
        prevalence : float or None
            Expected positive fraction in the planned study.
            If None, falls back to self.prevalence.
        """
        if alpha is None:
            alpha = self.alpha

        if target_auc is None:
            target_auc = self.nominal_auc()

        a = float(np.clip(target_auc, 1e-6, 1 - 1e-6))
        d = float(delta_auc)
        if d <= 0:
            raise ValueError("delta_auc must be positive")

        z = norm.ppf(1 - alpha / 2.0)
        q1 = a / (2.0 - a)
        q2 = 2.0 * a * a / (1.0 + a)

        # Use provided prevalence, otherwise fall back to self.prevalence
        if prevalence is None:
            prevalence = self.prevalence

        # Old balanced-design behavior
        if prevalence is None:
            c = a * (1.0 - a) - a * a + q1 + q2
            c = max(c, 1e-12)
            required_per_class = (z * z * c) / (d * d)
            return float(required_per_class)

        # Prevalence-aware total sample size
        p = float(prevalence)
        if not (0 < p < 1):
            raise ValueError("prevalence must be between 0 and 1")

        # Target variance corresponding to desired half-width delta_auc
        v_target = (d / z) ** 2

        # Hanley-McNeil variance with n_pos = pN, n_neg = (1-p)N
        c0 = a * (1.0 - a)
        c1 = q1 - a * a
        c2 = q2 - a * a

        # Solve:
        # v_target * p(1-p) * N^2 - (p*c1 + (1-p)*c2) * N - c0 = 0
        A = v_target * p * (1.0 - p)
        B = -(p * c1 + (1.0 - p) * c2)
        C = -c0

        disc = max(B * B - 4.0 * A * C, 0.0)
        N_total = (-B + np.sqrt(disc)) / (2.0 * A)

        return float(N_total)














    

    def pvalue(self, delta_auc=0.1, twosided=True):
        nominal, upper, lower = self.auc()
        se = (upper - lower) / (2.0 * norm.ppf(1 - self.alpha / 2.0))
        if not np.isfinite(se) or se <= 0:
            return np.nan
        z = delta_auc / se
        pvalue = norm.sf(abs(z))
        if twosided:
            pvalue *= 2.0
        return float(pvalue)













    

    def interpret(self, fpr=0.01, number_of_positives=10, five_yr_survival=None, factor=1):
        wf = self.df.copy()
        if wf.index.name != self.fprcol:
            wf = wf.set_index(self.fprcol)
        wf = wf.sort_index()
        grid = wf.index.values.astype(float)
        row = {}
        for col in wf.columns:
            vals = pd.to_numeric(wf[col], errors='coerce').values.astype(float)
            mask = np.isfinite(vals)
            if mask.sum() >= 2:
                row[col] = float(np.interp(fpr, grid[mask], vals[mask]))
            elif mask.sum() == 1:
                row[col] = float(vals[mask][0])
            else:
                row[col] = np.nan
        POS = float(number_of_positives)
        NEG = POS * (1.0 - self.prevalence) / self.prevalence
        TP = POS * row[self.tprcol]
        TOTALFLAGS = TP / row['ppv'] if np.isfinite(row.get('ppv', np.nan)) and row['ppv'] > 0 else np.nan
        FP = TOTALFLAGS - TP if np.isfinite(TOTALFLAGS) else np.nan
        FN = POS - TP
        TN = NEG - FP if np.isfinite(FP) else np.nan
        NNS = TOTALFLAGS / (TP * factor * (1 - five_yr_survival)) if five_yr_survival is not None and TP > 0 and np.isfinite(TOTALFLAGS) else np.nan
        resdf = pd.DataFrame.from_dict({
            'POS': np.round(POS),
            'TP': np.round(TP),
            'FP': np.round(FP),
            'NEG': np.round(NEG),
            'FLAGS': np.round(TOTALFLAGS),
            'FN': np.round(FN),
            'TN': np.round(TN),
            'NNS': np.round(NNS),
            'FLAGGED_FRACTION': np.round(TOTALFLAGS / (POS + NEG), 2) if np.isfinite(TOTALFLAGS) else np.nan,
        }, orient='index', columns=['estimates'])
        rf = pd.DataFrame({'pos': np.round(POS), 'flags': np.round(TOTALFLAGS), 'tp': np.round(TP), 'fp': np.round(FP), 'fn': np.round(FN), 'tn': np.round(TN)}, index=['numbers'])
        txt = [
            f'For every {int(np.round(POS))} positive instances',
            f'we raise {int(np.round(TOTALFLAGS)) if np.isfinite(TOTALFLAGS) else np.nan} flags,',
            f'out of which {int(np.round(TP))} are true positives',
            f'{int(np.round(FP)) if np.isfinite(FP) else np.nan} are false alarms',
            f'{int(np.round(FN))} cases are missed',
        ]
        if five_yr_survival is not None:
            txt.append(f'Number needed to screen is {NNS}')
        return rf, txt, resdf


# -----------------------------------------------------------------------------
# public top-level functions
# -----------------------------------------------------------------------------

def genroc(df, risk='predicted_risk', target='target', steps=1000, TARGET=[1], outfile=None):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f'genroc expected a pandas DataFrame, got {type(df).__name__}')
    missing = [c for c in [risk, target] if c not in df.columns]
    if missing:
        raise KeyError(f'Missing required columns: {missing}. Available columns: {list(df.columns)}')
    d = df[[risk, target]].rename(columns={risk: 'risk', target: 'target'}).dropna().copy()
    y_true = d['target'].isin(TARGET).astype(int).to_numpy()
    y_score = pd.to_numeric(d['risk'], errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]
    if y_true.size == 0:
        raise ValueError('No valid rows after filtering risk/target columns')
    if y_true.sum() == 0 or y_true.sum() == y_true.size:
        raise ValueError('ROC is undefined unless both positive and negative samples are present')
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)
    xf = pd.DataFrame({'threshold': thresholds, 'fpr': fpr, 'tpr': tpr})
    xf = xf[np.isfinite(xf['threshold'])].copy().reset_index(drop=True)
    if outfile is not None:
        xf.to_csv(outfile, index=False)
    return xf, int(y_true.size), int(y_true.sum())


def pipeline(df, risk='predicted_risk', target='target', steps=1000, TARGET=[1], order=3, alpha=0.05, prevalence=.002, precision=3, convexify=False, lr_fpr_floor=0.001, outfile=None):
    rf, total_samples, positive_samples = genroc(df, risk=risk, target=target, TARGET=TARGET, steps=steps)
    zt = processRoc(rf, order=order, total_samples=total_samples, positive_samples=positive_samples, alpha=alpha, prevalence=prevalence, lr_fpr_floor=lr_fpr_floor)
    zt.smooth(STEP=0.001, convexify=convexify)
    zt.allmeasures(interpolate=True)
    zt.usample(precision=precision)
    zt.getBounds()
    out = zt.get().join(zt.df_lim['U'], rsuffix='_upper').join(zt.df_lim['L'], rsuffix='_lower')
    if outfile is not None:
        out.to_csv(outfile)
    return out, zt.auc()


def score_to_threshold_ppv(scores, df, prevalence, total_samples, positive_samples, alpha=0.05):
    work = df.copy()
    required = {'threshold', 'tpr', 'fpr'}
    missing = required.difference(work.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    work = work.dropna(subset=['threshold', 'tpr', 'fpr']).copy()
    work = work.sort_values('threshold').drop_duplicates(subset=['threshold'], keep='last')
    se = _clip01(work['tpr'].to_numpy(dtype=float))
    sp = 1.0 - _clip01(work['fpr'].to_numpy(dtype=float))
    p = float(prevalence)
    n_pos = int(positive_samples)
    n_neg = int(total_samples - positive_samples)
    se_low, se_high = _wilson_interval_from_proportion(se, n_pos, alpha=alpha)
    sp_low, sp_high = _wilson_interval_from_proportion(sp, n_neg, alpha=alpha)
    ppv = _safe_divide(se * p, se * p + (1.0 - sp) * (1.0 - p))
    ppv_lower = _safe_divide(se_low * p, se_low * p + (1.0 - sp_low) * (1.0 - p))
    ppv_upper = _safe_divide(se_high * p, se_high * p + (1.0 - sp_high) * (1.0 - p))
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
    x = work['threshold'].to_numpy(dtype=float)
    if len(work) >= 2:
        ppv = iso.fit_transform(x, ppv)
        ppv_lower = iso.fit_transform(x, ppv_lower)
        ppv_upper = iso.fit_transform(x, ppv_upper)
    def one_score(score):
        s = float(np.clip(score, x.min(), x.max()))
        return [float(np.interp(s, x, ppv)), float(np.interp(s, x, ppv_upper)), float(np.interp(s, x, ppv_lower))]
    return [one_score(score) for score in scores]


def score_to_probability(scores, df, score_col, label_col, target_prevalence=None, lower_score_is_risk=False, n_splits=5, random_state=42, return_curve=False, n_bins=10):
    work = df[[score_col, label_col]].dropna().copy()
    work[label_col] = _ensure_binary_labels(work[label_col])
    raw_scores = pd.to_numeric(work[score_col], errors='coerce')
    work = work.loc[np.isfinite(raw_scores)].copy()
    work[score_col] = raw_scores.loc[work.index].astype(float)
    greater_is_risk = not bool(lower_score_is_risk)
    work['_score_for_calibration'] = work[score_col] if greater_is_risk else -work[score_col]
    oof = crossfit_isotonic_probabilities(
        work.rename(columns={'_score_for_calibration': '__score__'}),
        score_col='__score__',
        label_col=label_col,
        target_prevalence=target_prevalence,
        n_splits=n_splits,
        random_state=random_state,
    )
    work['calibrated_prob_oof'] = oof
    w_full = _make_prevalence_weights(work[label_col].to_numpy(), target_prevalence)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    iso.fit(work['_score_for_calibration'].to_numpy(dtype=float), work[label_col].to_numpy(dtype=int), sample_weight=w_full)
    def one_score(score):
        x = -float(score) if lower_score_is_risk else float(score)
        return float(iso.predict([x])[0])
    probs = [one_score(score) for score in scores]
    if not return_curve:
        return probs
    cal_df = calibration_curve_table(work, prob_col='calibrated_prob_oof', label_col=label_col, target_prevalence=target_prevalence, n_bins=n_bins)
    return probs, cal_df


# =============================================================================
# Additional fixed bounds/calculation layer
# =============================================================================
# This block intentionally overrides processRoc.usample and processRoc.getBounds
# without changing the public construction/API pattern.  It fixes three issues:
#   1. derived ratios are recomputed from resampled FPR/TPR rather than
#      interpolated directly;
#   2. PPV confidence bounds are direct Wilson bounds on the expected flagged
#      population, optionally monotone-smoothed and forced to bracket the point
#      estimate;
#   3. likelihood-ratio quantities are masked in unstable/dominated endpoint
#      regions, with a helper that returns the nondominated LR frontier.


def _isotonic_monotone_array(y, increasing=False, sample_weight=None, clip01=False):
    y = np.asarray(y, dtype=float)
    out = y.copy()
    valid = np.isfinite(y)
    if valid.sum() < 2:
        return _clip01(out) if clip01 else out

    x = np.arange(valid.sum(), dtype=float)
    if sample_weight is None:
        w = np.ones(valid.sum(), dtype=float)
    else:
        w_all = np.asarray(sample_weight, dtype=float)
        w = w_all[valid]
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)

    iso = IsotonicRegression(increasing=increasing, out_of_bounds='clip')
    out[valid] = iso.fit_transform(x, y[valid], sample_weight=w)
    return _clip01(out) if clip01 else out


def _smooth_ppv_bounds_if_requested(lower, upper, nominal, prevalence=None, total_samples=None, smooth_bounds=True):
    if not smooth_bounds:
        return lower, upper

    lower = lower.copy()
    upper = upper.copy()
    nominal = nominal.copy()

    if 'ppv' not in lower.columns or 'ppv' not in upper.columns:
        return lower, upper

    if lower.index.name == 'fpr':
        fpr = lower.index.to_numpy(dtype=float)
    elif 'fpr' in lower.columns:
        fpr = pd.to_numeric(lower['fpr'], errors='coerce').to_numpy(dtype=float)
    else:
        fpr = np.arange(len(lower), dtype=float)

    if prevalence is not None and total_samples is not None and 'tpr' in nominal.columns:
        tpr = pd.to_numeric(nominal['tpr'], errors='coerce').to_numpy(dtype=float)
        ff = float(prevalence) * _clip01(tpr) + (1.0 - float(prevalence)) * _clip01(fpr)
        weights = np.maximum(float(total_samples) * ff, 1.0)
    else:
        weights = np.ones(len(lower), dtype=float)

    lower['ppv'] = _isotonic_monotone_array(lower['ppv'].to_numpy(dtype=float), increasing=False, sample_weight=weights, clip01=True)
    upper['ppv'] = _isotonic_monotone_array(upper['ppv'].to_numpy(dtype=float), increasing=False, sample_weight=weights, clip01=True)
    return lower, upper


def _lr_nondominated_mask_from_arrays(lr_minus, lr_plus, min_lrplus=1.0, max_lrminus=None, tol=1e-12):
    """Return mask for nondominated LR operating points.

    In LR space, smaller LR- and larger LR+ are preferred. A point is dominated
    if another point has LR- <= current LR- and LR+ >= current LR+.

    min_lrplus removes the no-rule-in branch near LR+ = 1. max_lrminus may be
    used to restrict to rule-out-relevant thresholds.
    """
    lr_minus = np.asarray(lr_minus, dtype=float)
    lr_plus = np.asarray(lr_plus, dtype=float)
    keep = np.zeros(lr_minus.shape, dtype=bool)

    cand = np.isfinite(lr_minus) & np.isfinite(lr_plus) & (lr_minus > 0) & (lr_plus > float(min_lrplus))
    if max_lrminus is not None:
        cand &= lr_minus < float(max_lrminus)
    if not np.any(cand):
        return keep

    idx = np.flatnonzero(cand)
    order = np.lexsort((-lr_plus[idx], lr_minus[idx]))
    idx = idx[order]

    best_lrplus = -np.inf
    for i in idx:
        if lr_plus[i] > best_lrplus + tol:
            keep[i] = True
            best_lrplus = lr_plus[i]
    return keep


def _apply_lr_frontier_mask_to_frames(nominal, lower, upper, min_lrplus=1.0, max_lrminus=None):
    nominal = nominal.copy()
    lower = lower.copy()
    upper = upper.copy()

    if 'LR-' not in nominal.columns or 'LR+' not in nominal.columns:
        return nominal, lower, upper

    mask = _lr_nondominated_mask_from_arrays(
        nominal['LR-'].to_numpy(dtype=float),
        nominal['LR+'].to_numpy(dtype=float),
        min_lrplus=min_lrplus,
        max_lrminus=max_lrminus,
    )
    bad = ~mask
    for frame in [nominal, lower, upper]:
        for col in ['LR+', 'LR-']:
            if col in frame.columns:
                frame.loc[bad, col] = np.nan
    return nominal, lower, upper


def lr_nondominated_frontier(perf_df, min_lrplus=1.0, max_lrminus=None):
    """Return the nondominated LR frontier from a performance dataframe.

    The input can be a zedstat output table with FPR as either the index or a
    column. The returned rows are thresholds for which no other threshold has
    both a lower/equal LR- and a higher/equal LR+.
    """
    pf = perf_df.reset_index().copy()
    if 'fpr' not in pf.columns:
        pf = pf.rename(columns={pf.columns[0]: 'fpr'})
    pf = pf.replace([np.inf, -np.inf], np.nan)
    if 'LR-' not in pf.columns or 'LR+' not in pf.columns:
        raise ValueError("perf_df must contain 'LR-' and 'LR+' columns")
    mask = _lr_nondominated_mask_from_arrays(
        pf['LR-'].to_numpy(dtype=float),
        pf['LR+'].to_numpy(dtype=float),
        min_lrplus=min_lrplus,
        max_lrminus=max_lrminus,
    )
    return pf.loc[mask].sort_values('LR-').reset_index(drop=True)


def _compute_pointwise_measure_frames_fixed(
    df,
    prevalence,
    alpha,
    total_samples,
    positive_samples,
    thresholdcol='threshold',
    fprcol='fpr',
    tprcol='tpr',
    lr_fpr_floor=0.001,
    lr_sp_floor=None,
    ppv_ci_method='direct',
    min_expected_flags=10.0,
    min_expected_fp=5.0,
    min_expected_tn=5.0,
):
    work, thresholdcol = _prepare_roc_dataframe(df, fprcol=fprcol, tprcol=tprcol, thresholdcol=thresholdcol)
    work = _add_endpoints(work, fprcol=fprcol, tprcol=tprcol, thresholdcol=thresholdcol)

    fpr = _clip01(work[fprcol].to_numpy(dtype=float))
    tpr = _clip01(work[tprcol].to_numpy(dtype=float))
    sp = 1.0 - fpr

    n_pos = positive_samples
    n_neg = None if total_samples is None or positive_samples is None else int(total_samples - positive_samples)

    tpr_low, tpr_high = _wilson_interval_from_proportion(tpr, n_pos, alpha=alpha)
    sp_low, sp_high = _wilson_interval_from_proportion(sp, n_neg, alpha=alpha)
    fpr_low = 1.0 - sp_high
    fpr_high = 1.0 - sp_low

    nominal = _compute_measures_from_arrays(
        fpr,
        tpr,
        prevalence=prevalence,
        lr_fpr_floor=lr_fpr_floor,
        lr_sp_floor=lr_sp_floor,
    )
    lower = pd.DataFrame({'fpr': fpr})
    upper = pd.DataFrame({'fpr': fpr})

    lower['tpr'] = tpr_low
    upper['tpr'] = tpr_high

    if ppv_ci_method == 'direct':
        ppv_low, ppv_high = _direct_ppv_bounds_from_expected_flags(
            fpr=fpr,
            tpr=tpr,
            prevalence=prevalence,
            total_samples=total_samples,
            alpha=alpha,
            min_expected_flags=min_expected_flags,
        )
        lower['ppv'] = ppv_low
        upper['ppv'] = ppv_high
    elif ppv_ci_method == 'propagated':
        lower['ppv'] = _safe_divide(tpr_low * prevalence, tpr_low * prevalence + fpr_high * (1.0 - prevalence))
        upper['ppv'] = _safe_divide(tpr_high * prevalence, tpr_high * prevalence + fpr_low * (1.0 - prevalence))
    else:
        raise ValueError("ppv_ci_method must be 'direct' or 'propagated'")

    lower['acc'] = prevalence * tpr_low + (1.0 - prevalence) * (1.0 - fpr_high)
    upper['acc'] = prevalence * tpr_high + (1.0 - prevalence) * (1.0 - fpr_low)

    lower['npv'] = _safe_divide((1.0 - fpr_low) * (1.0 - prevalence), (1.0 - fpr_low) * (1.0 - prevalence) + (1.0 - tpr_high) * prevalence)
    upper['npv'] = _safe_divide((1.0 - fpr_high) * (1.0 - prevalence), (1.0 - fpr_high) * (1.0 - prevalence) + (1.0 - tpr_low) * prevalence)

    lower['LR+'] = _safe_divide(tpr_low, fpr_high)
    upper['LR+'] = _safe_divide(tpr_high, fpr_low)

    lower['LR-'] = _safe_divide(1.0 - tpr_high, 1.0 - fpr_low)
    upper['LR-'] = _safe_divide(1.0 - tpr_low, 1.0 - fpr_high)

    # Expected-count masks for unstable LR regions.
    if total_samples is not None and n_neg is not None:
        expected_fp = float(n_neg) * fpr
        expected_tn = float(n_neg) * sp
    else:
        expected_fp = np.full_like(fpr, np.nan, dtype=float)
        expected_tn = np.full_like(fpr, np.nan, dtype=float)

    lr_plus_bad = (
        (fpr <= float(lr_fpr_floor)) |
        (fpr_low <= 0) |
        (~np.isfinite(expected_fp)) |
        (expected_fp < float(min_expected_fp))
    )
    if lr_sp_floor is None:
        lr_sp_floor = lr_fpr_floor
    lr_minus_bad = (
        (sp <= float(lr_sp_floor)) |
        (sp_low <= 0) |
        (~np.isfinite(expected_tn)) |
        (expected_tn < float(min_expected_tn))
    )

    for frame in [nominal, lower, upper]:
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        if thresholdcol is not None and thresholdcol in work.columns:
            frame[thresholdcol] = work[thresholdcol].to_numpy(dtype=float)
        frame.index = work[fprcol].to_numpy(dtype=float)
        frame.index.name = fprcol
        frame['expected_fp'] = expected_fp
        frame['expected_tn'] = expected_tn
        p = float(prevalence)
        frame['expected_flags'] = float(total_samples) * (p * tpr + (1.0 - p) * fpr) if total_samples is not None else np.nan
        if 'LR+' in frame.columns:
            frame.loc[lr_plus_bad, 'LR+'] = np.nan
        if 'LR-' in frame.columns:
            frame.loc[lr_minus_bad, 'LR-'] = np.nan

    nominal = _apply_lr_floor(nominal, lr_fpr_floor=lr_fpr_floor, lr_sp_floor=lr_sp_floor, fprcol=fprcol)
    lower = _apply_lr_floor(lower, lr_fpr_floor=lr_fpr_floor, lr_sp_floor=lr_sp_floor, fprcol=fprcol)
    upper = _apply_lr_floor(upper, lr_fpr_floor=lr_fpr_floor, lr_sp_floor=lr_sp_floor, fprcol=fprcol)

    lower, upper = _enforce_bounds_around_nominal(
        nominal,
        lower,
        upper,
        cols=['tpr', 'ppv', 'acc', 'npv', 'LR+', 'LR-'],
    )
    return nominal, lower, upper


def _processRoc_usample_fixed(self, df=None, precision=3, recompute_measures=True):
    step = 10 ** (-precision)
    grid = np.round(np.arange(0.0, 1.0 + step, step), precision)
    grid[-1] = 1.0
    source = self.df.copy() if df is None else df.copy()
    if source.index.name == self.fprcol:
        source = source.reset_index()
    source = source.sort_values(self.fprcol).reset_index(drop=True)

    out = pd.DataFrame({self.fprcol: grid})
    x = pd.to_numeric(source[self.fprcol], errors='coerce').to_numpy(dtype=float)

    # Interpolate only primary coordinates and threshold-like quantities.
    # Derived operating measures are recomputed below.
    skip_if_recompute = {'ppv', 'acc', 'npv', 'LR+', 'LR-'}
    for col in source.columns:
        if col == self.fprcol:
            continue
        if recompute_measures and col in skip_if_recompute:
            continue
        vals = pd.to_numeric(source[col], errors='coerce').to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(vals)
        if mask.sum() >= 2:
            out[col] = np.interp(grid, x[mask], vals[mask])
        elif mask.sum() == 1:
            out[col] = vals[mask][0]
        else:
            out[col] = np.nan

    if self.tprcol in out.columns:
        out[self.tprcol] = _clip01(np.maximum.accumulate(out[self.tprcol].values))

    if recompute_measures and self.prevalence is not None and self.tprcol in out.columns:
        recomputed = _compute_measures_from_arrays(
            out[self.fprcol].to_numpy(dtype=float),
            out[self.tprcol].to_numpy(dtype=float),
            prevalence=self.prevalence,
            lr_fpr_floor=self.lr_fpr_floor,
            lr_sp_floor=getattr(self, 'lr_sp_floor', self.lr_fpr_floor),
        )
        for col in ['ppv', 'acc', 'npv', 'LR+', 'LR-']:
            out[col] = recomputed[col].to_numpy(dtype=float)

    out = out.set_index(self.fprcol)
    if 'ppv' in out.columns:
        out = self._processRoc__correctPPV(out)
    out = _apply_lr_floor(
        out,
        lr_fpr_floor=self.lr_fpr_floor,
        lr_sp_floor=getattr(self, 'lr_sp_floor', self.lr_fpr_floor),
        fprcol=self.fprcol,
    )
    if df is None:
        self.df = out
    return out


def _processRoc_getBounds_fixed(
    self,
    total_samples=None,
    positive_samples=None,
    alpha=None,
    prevalence=None,
    ppv_ci_method='direct',
    min_expected_flags=10.0,
    min_expected_fp=5.0,
    min_expected_tn=5.0,
    enforce_bounds=True,
    smooth_ppv_bounds=True,
    mask_dominated_lr=True,
    lr_min_plus=1.001,
    lr_max_minus=None,
):
    if total_samples is None:
        total_samples = self.total_samples
    if positive_samples is None:
        positive_samples = self.positive_samples
    if alpha is None:
        alpha = self.alpha
    if prevalence is None:
        prevalence = self.prevalence
    if prevalence is None:
        raise ValueError('prevalence undefined')

    empirical = self.raw_df.reset_index().copy()
    nominal_emp, lower_emp, upper_emp = _compute_pointwise_measure_frames_fixed(
        empirical,
        prevalence=float(prevalence),
        alpha=alpha,
        total_samples=total_samples,
        positive_samples=positive_samples,
        thresholdcol=self.thresholdcol,
        fprcol=self.fprcol,
        tprcol=self.tprcol,
        lr_fpr_floor=self.lr_fpr_floor,
        lr_sp_floor=getattr(self, 'lr_sp_floor', self.lr_fpr_floor),
        ppv_ci_method=ppv_ci_method,
        min_expected_flags=min_expected_flags,
        min_expected_fp=min_expected_fp,
        min_expected_tn=min_expected_tn,
    )

    nominal_emp = self._processRoc__correctPPV(nominal_emp)
    lower_emp, upper_emp = _smooth_ppv_bounds_if_requested(
        lower_emp,
        upper_emp,
        nominal_emp,
        prevalence=prevalence,
        total_samples=total_samples,
        smooth_bounds=smooth_ppv_bounds,
    )
    if enforce_bounds:
        lower_emp, upper_emp = _enforce_bounds_around_nominal(
            nominal_emp,
            lower_emp,
            upper_emp,
            cols=['tpr', 'ppv', 'acc', 'npv', 'LR+', 'LR-'],
        )

    target_index = self.df.index.values.astype(float) if self.df.index.name == self.fprcol else self.df[self.fprcol].values.astype(float)
    nominal = _numeric_interp_frame_to_target_index(nominal_emp, target_index, index_name=self.fprcol)
    lower = _numeric_interp_frame_to_target_index(lower_emp, target_index, index_name=self.fprcol)
    upper = _numeric_interp_frame_to_target_index(upper_emp, target_index, index_name=self.fprcol)

    nominal = self._processRoc__correctPPV(nominal)
    lower, upper = _smooth_ppv_bounds_if_requested(
        lower,
        upper,
        nominal,
        prevalence=prevalence,
        total_samples=total_samples,
        smooth_bounds=smooth_ppv_bounds,
    )
    if enforce_bounds:
        lower, upper = _enforce_bounds_around_nominal(
            nominal,
            lower,
            upper,
            cols=['tpr', 'ppv', 'acc', 'npv', 'LR+', 'LR-'],
        )

    nominal = _apply_lr_floor(nominal, lr_fpr_floor=self.lr_fpr_floor, lr_sp_floor=getattr(self, 'lr_sp_floor', self.lr_fpr_floor), fprcol=self.fprcol)
    lower = _apply_lr_floor(lower, lr_fpr_floor=self.lr_fpr_floor, lr_sp_floor=getattr(self, 'lr_sp_floor', self.lr_fpr_floor), fprcol=self.fprcol)
    upper = _apply_lr_floor(upper, lr_fpr_floor=self.lr_fpr_floor, lr_sp_floor=getattr(self, 'lr_sp_floor', self.lr_fpr_floor), fprcol=self.fprcol)

    if mask_dominated_lr:
        nominal, lower, upper = _apply_lr_frontier_mask_to_frames(
            nominal,
            lower,
            upper,
            min_lrplus=lr_min_plus,
            max_lrminus=lr_max_minus,
        )

    # Ensure zt.get() returns the same repaired nominal calculations that are
    # used to define the displayed bounds.
    self.df = nominal.copy()
    self.df_lim['L'] = lower
    self.df_lim['U'] = upper
    self.df_measure_bounds_ = {
        'nominal_empirical': nominal_emp,
        'L_empirical': lower_emp,
        'U_empirical': upper_emp,
        'nominal_display': nominal,
        'L_display': lower,
        'U_display': upper,
        'kind': 'analytic_pointwise_fixed',
        'ppv_ci_method': ppv_ci_method,
        'min_expected_flags': min_expected_flags,
        'min_expected_fp': min_expected_fp,
        'min_expected_tn': min_expected_tn,
        'enforce_bounds': enforce_bounds,
        'smooth_ppv_bounds': smooth_ppv_bounds,
        'mask_dominated_lr': mask_dominated_lr,
        'lr_min_plus': lr_min_plus,
        'lr_max_minus': lr_max_minus,
        'lr_fpr_floor': self.lr_fpr_floor,
        'lr_sp_floor': getattr(self, 'lr_sp_floor', self.lr_fpr_floor),
    }
    return


def _processRoc_lr_frontier(self, min_lrplus=1.0, max_lrminus=None):
    return lr_nondominated_frontier(self.get(), min_lrplus=min_lrplus, max_lrminus=max_lrminus)


# Override processRoc methods with the fixed versions.
processRoc.usample = _processRoc_usample_fixed
processRoc.getBounds = _processRoc_getBounds_fixed
processRoc.lr_frontier = _processRoc_lr_frontier

try:
    __all__.append('lr_nondominated_frontier')
except Exception:
    pass


# =============================================================================
# NaN-safe final output accessor
# =============================================================================
# This accessor keeps the old behavior for zt.get(): nominal table only, no
# interpolation.  It adds two explicit final-output modes:
#   zt.get(bounds=True, interpolate=True)
#       returns nominal + upper/lower bounds and interpolates display gaps;
#   zt.get(0)
#       shorthand for the same NaN-safe final table.  The 0 is not used as a
#       numerical fill value; residual all-NaN columns remain NaN unless fillna
#       is supplied explicitly.

_PROBABILITY_BASE_COLUMNS = {'tpr', 'ppv', 'acc', 'npv'}


def _base_metric_name(col):
    col = str(col)
    for suffix in ('_upper', '_lower'):
        if col.endswith(suffix):
            return col[:-len(suffix)]
    return col


def _clip_probability_like_columns(frame):
    out = frame.copy()
    for col in out.columns:
        if _base_metric_name(col) in _PROBABILITY_BASE_COLUMNS:
            vals = pd.to_numeric(out[col], errors='coerce').to_numpy(dtype=float)
            out[col] = np.clip(vals, 0.0, 1.0)
    return out


def _interpolate_numeric_frame_for_display(frame, interpolate=True, fillna=None, limit_direction='both'):
    """Interpolate numeric display gaps in a final performance table.

    This is intentionally a final accessor/post-processing step.  It does not
    change the stored analytic bounds unless the caller assigns the returned
    dataframe.  The main use is to prevent visualization breaks from NaNs that
    arise after endpoint masking, sparse empirical ROC rows, or frontier masking.
    """
    out = frame.copy().replace([np.inf, -np.inf], np.nan)

    if interpolate:
        numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
        if numeric_cols:
            idx_numeric = np.issubdtype(out.index.dtype, np.number)
            method = 'index' if idx_numeric else 'linear'
            out[numeric_cols] = out[numeric_cols].interpolate(
                method=method,
                limit_direction=limit_direction,
                axis=0,
            )

    if fillna is not None:
        out = out.fillna(fillna)

    out = _clip_probability_like_columns(out)
    return out


def _enforce_joined_bounds_for_display(frame):
    """Ensure joined *_lower and *_upper columns bracket nominal columns."""
    out = frame.copy()
    for base in ['tpr', 'ppv', 'acc', 'npv', 'LR+', 'LR-']:
        lo_col = f'{base}_lower'
        hi_col = f'{base}_upper'
        if base not in out.columns or lo_col not in out.columns or hi_col not in out.columns:
            continue

        n = pd.to_numeric(out[base], errors='coerce').to_numpy(dtype=float)
        lo = pd.to_numeric(out[lo_col], errors='coerce').to_numpy(dtype=float)
        hi = pd.to_numeric(out[hi_col], errors='coerce').to_numpy(dtype=float)
        finite = np.isfinite(n) & np.isfinite(lo) & np.isfinite(hi)
        if not np.any(finite):
            continue

        lo2 = lo.copy()
        hi2 = hi.copy()
        lo2[finite] = np.minimum.reduce([lo[finite], hi[finite], n[finite]])
        hi2[finite] = np.maximum.reduce([lo[finite], hi[finite], n[finite]])

        if base in _PROBABILITY_BASE_COLUMNS:
            lo2 = np.clip(lo2, 0.0, 1.0)
            hi2 = np.clip(hi2, 0.0, 1.0)

        out[lo_col] = lo2
        out[hi_col] = hi2
    return out


def _processRoc_get_nan_safe(
    self,
    interpolate=False,
    bounds=False,
    fillna=None,
    limit_direction='both',
    enforce_bounds=True,
):
    """Return the processed ROC table, optionally as a NaN-safe final table.

    Parameters
    ----------
    interpolate : bool or numeric, default False
        False preserves the historical zt.get() behavior.  True interpolates
        numeric gaps in the returned display table.  Passing 0 is supported as a
        compact shorthand for zt.get(bounds=True, interpolate=True).
    bounds : bool, default False
        If True, join self.df_lim['U'] and self.df_lim['L'] using _upper and
        _lower suffixes before optional interpolation.
    fillna : scalar or None, default None
        Optional final fill for columns that remain NaN after interpolation.
        Leave as None to avoid inventing values for all-NaN columns.
    limit_direction : str, default 'both'
        Passed to pandas interpolate.  'both' fills endpoint display gaps using
        nearest finite values.
    enforce_bounds : bool, default True
        After interpolation, force lower <= nominal <= upper for joined bounds.
    """
    numeric_shorthand = isinstance(interpolate, (int, float, np.integer, np.floating)) and not isinstance(interpolate, (bool, np.bool_))
    if numeric_shorthand:
        # zt.get(0) means: return the final joined, interpolated display table.
        # The zero is not treated as a fill value because LR+ at FPR=0 should not
        # be replaced by zero.  Use fillna=0 explicitly if that is actually wanted.
        if float(interpolate) != 0.0 and fillna is None:
            fillna = float(interpolate)
        interpolate = True
        bounds = True

    out = self.df.copy()

    if bounds:
        if 'U' not in self.df_lim or 'L' not in self.df_lim:
            raise ValueError("Bounds requested, but getBounds() has not been run yet.")
        out = (
            out
            .join(self.df_lim['U'], rsuffix='_upper')
            .join(self.df_lim['L'], rsuffix='_lower')
        )

    if interpolate or fillna is not None:
        out = _interpolate_numeric_frame_for_display(
            out,
            interpolate=bool(interpolate),
            fillna=fillna,
            limit_direction=limit_direction,
        )
        if bounds and enforce_bounds:
            out = _enforce_joined_bounds_for_display(out)

    return out.copy()


def _processRoc_get_full(self, interpolate=True, fillna=None, limit_direction='both', enforce_bounds=True):
    """Convenience wrapper for the final nominal + bounds table."""
    return self.get(
        bounds=True,
        interpolate=interpolate,
        fillna=fillna,
        limit_direction=limit_direction,
        enforce_bounds=enforce_bounds,
    )


processRoc.get = _processRoc_get_nan_safe
processRoc.get_full = _processRoc_get_full

try:
    __all__.append('lr_nondominated_frontier')
except Exception:
    pass
