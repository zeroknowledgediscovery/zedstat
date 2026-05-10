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


def _nice_axis_limit(max_val):
    candidates = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00])
    target = max(0.05, min(1.0, float(max_val) * 1.08))
    idx = np.searchsorted(candidates, target, side='left')
    return 1.0 if idx >= len(candidates) else float(candidates[idx])


# -----------------------------------------------------------------------------
# calibration helpers
# -----------------------------------------------------------------------------

def crossfit_isotonic_probabilities(df, score_col, label_col, target_prevalence=None, n_splits=5, random_state=42):
    scores = pd.to_numeric(df[score_col], errors='coerce').to_numpy(dtype=float)
    y = _ensure_binary_labels(df[label_col]).to_numpy(dtype=int)
    valid = np.isfinite(scores)
    scores = scores[valid]
    y = y[valid]
    out = np.full(len(df), np.nan, dtype=float)
    actual = _choose_n_splits(y, n_splits)

    skf = StratifiedKFold(n_splits=actual, shuffle=True, random_state=random_state)
    valid_idx = np.flatnonzero(valid)
    for train_idx, val_idx in skf.split(scores, y):
        x_train = scores[train_idx]
        y_train = y[train_idx]
        x_val = scores[val_idx]
        w_train = _make_prevalence_weights(y_train, target_prevalence)
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        iso.fit(x_train, y_train, sample_weight=w_train)
        out[valid_idx[val_idx]] = iso.predict(x_val)
    return pd.Series(out, index=df.index, name='calibrated_prob_oof')


def calibration_curve_table(df, prob_col, label_col, target_prevalence=None, n_bins=10):
    d = df[[prob_col, label_col]].dropna().copy()
    d[label_col] = _ensure_binary_labels(d[label_col])
    d['weight'] = _make_prevalence_weights(d[label_col].to_numpy(), target_prevalence)
    n_bins = min(int(n_bins), len(d))
    if n_bins < 2:
        raise ValueError('Not enough rows to build a calibration curve.')

    d['bin'] = pd.qcut(d[prob_col].rank(method='first'), q=n_bins, labels=False, duplicates='drop')
    rows = []
    for b, g in d.groupby('bin', sort=True):
        rows.append({
            'bin': int(b),
            'n': int(len(g)),
            'cases_unweighted': int(g[label_col].sum()),
            'mean_pred': _weighted_mean(g[prob_col].to_numpy(), g['weight'].to_numpy()),
            'obs_rate': _weighted_mean(g[label_col].to_numpy(), g['weight'].to_numpy()),
            'prob_min': float(g[prob_col].min()),
            'prob_max': float(g[prob_col].max()),
        })
    return pd.DataFrame(rows).sort_values('mean_pred').reset_index(drop=True)


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


def _apply_lr_floor(df, lr_fpr_floor, fprcol='fpr'):
    work = df.copy()
    if work.index.name == fprcol:
        fpr_vals = work.index.to_numpy(dtype=float)
    else:
        fpr_vals = pd.to_numeric(work[fprcol], errors='coerce').to_numpy(dtype=float)
    bad = fpr_vals < float(lr_fpr_floor)
    for col in ['LR+', 'LR-', 'LR+_raw', 'LR-_raw']:
        if col in work.columns:
            work.loc[bad, col] = np.nan
    return work


def _compute_measures_from_arrays(fpr, tpr, prevalence, lr_fpr_floor=0.001):
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
    frame = _apply_lr_floor(frame, lr_fpr_floor=lr_fpr_floor, fprcol='fpr')
    return frame


def _compute_pointwise_measure_frames(df, prevalence, alpha, total_samples, positive_samples, thresholdcol='threshold', fprcol='fpr', tprcol='tpr', lr_fpr_floor=0.001):
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

    nominal = _compute_measures_from_arrays(fpr, tpr, prevalence=prevalence, lr_fpr_floor=lr_fpr_floor)
    lower = pd.DataFrame({'fpr': fpr})
    upper = pd.DataFrame({'fpr': fpr})

    lower['tpr'] = tpr_low
    upper['tpr'] = tpr_high

    lower['ppv'] = _safe_divide(tpr_low * prevalence, tpr_low * prevalence + fpr_high * (1.0 - prevalence))
    upper['ppv'] = _safe_divide(tpr_high * prevalence, tpr_high * prevalence + fpr_low * (1.0 - prevalence))

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
        frame = _apply_lr_floor(frame, lr_fpr_floor=lr_fpr_floor, fprcol=fprcol)
    nominal = _apply_lr_floor(nominal, lr_fpr_floor=lr_fpr_floor, fprcol=fprcol)
    lower = _apply_lr_floor(lower, lr_fpr_floor=lr_fpr_floor, fprcol=fprcol)
    upper = _apply_lr_floor(upper, lr_fpr_floor=lr_fpr_floor, fprcol=fprcol)
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

    def usample(self, df=None, precision=3):
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
        out = out.set_index(self.fprcol)
        if 'ppv' in out.columns:
            out = self.__correctPPV(out)
        out = _apply_lr_floor(out, lr_fpr_floor=self.lr_fpr_floor, fprcol=self.fprcol)
        if df is None:
            self.df = out
        return out

    def getBounds(self, total_samples=None, positive_samples=None, alpha=None, prevalence=None):
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
        )
        nominal_emp = self.__correctPPV(nominal_emp)
        lower_emp = self.__correctPPV(lower_emp)
        upper_emp = self.__correctPPV(upper_emp)

        target_index = self.df.index.values.astype(float) if self.df.index.name == self.fprcol else self.df[self.fprcol].values.astype(float)
        nominal = _numeric_interp_frame_to_target_index(nominal_emp, target_index, index_name=self.fprcol)
        lower = _numeric_interp_frame_to_target_index(lower_emp, target_index, index_name=self.fprcol)
        upper = _numeric_interp_frame_to_target_index(upper_emp, target_index, index_name=self.fprcol)
        nominal = self.__correctPPV(nominal)
        lower = self.__correctPPV(lower)
        upper = self.__correctPPV(upper)
        nominal = _apply_lr_floor(nominal, lr_fpr_floor=self.lr_fpr_floor, fprcol=self.fprcol)
        lower = _apply_lr_floor(lower, lr_fpr_floor=self.lr_fpr_floor, fprcol=self.fprcol)
        upper = _apply_lr_floor(upper, lr_fpr_floor=self.lr_fpr_floor, fprcol=self.fprcol)

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
