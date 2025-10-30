import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def _ppv_from_roc(roc_df: pd.DataFrame, prevalence: float) -> pd.DataFrame:
    """
    Given a ROC dataframe with columns ['fpr','tpr','thresholds'] and a prevalence p,
    compute PPV at each threshold: PPV = tpr*p / (tpr*p + fpr*(1-p)).
    Returns a dataframe with ['threshold','ppv'] (drops NaNs).
    """
    fpr = roc_df['fpr'].to_numpy(dtype=float)
    tpr = roc_df['tpr'].to_numpy(dtype=float)
    thr = roc_df['thresholds'].to_numpy(dtype=float)

    denom = tpr * prevalence + fpr * (1.0 - prevalence)
    with np.errstate(divide='ignore', invalid='ignore'):
        ppv = np.where(denom > 0, (tpr * prevalence) / denom, np.nan)

    df = pd.DataFrame({'threshold': thr, 'ppv': ppv})
    df = df.dropna().drop_duplicates(subset='threshold')
    return df

def _fit_isotonic_score_to_ppv(threshold_ppv_df: pd.DataFrame) -> IsotonicRegression:
    """
    Fit a monotone mapping score -> PPV using isotonic regression on (threshold, PPV) pairs.
    Detects monotonic direction automatically.
    """
    th = threshold_ppv_df['threshold'].to_numpy()
    ppv = threshold_ppv_df['ppv'].to_numpy()

    # Decide whether higher score should map to higher PPV.
    # Heuristic: use Spearman-like sign via correlation; fall back to increasing=True.
    corr = np.corrcoef(th, ppv)[0, 1]
    increasing = bool(corr >= 0)  # if PPV grows with threshold, keep increasing; else invert by flipping thresholds.

    # If decreasing, flip thresholds to make fit monotone increasing in the *raw score* direction.
    x = th.copy()
    y = ppv.copy()
    if not increasing:
        x = -x  # flipping makes the learned function increasing in -threshold, i.e., decreasing in threshold.

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds='clip')
    iso.fit(x, y)
    # Store whether we flipped to apply same transform at predict time.
    iso._flipped = (not increasing)
    return iso

def _predict_ppv(iso: IsotonicRegression, scores: np.ndarray) -> np.ndarray:
    x = -scores if getattr(iso, "_flipped", False) else scores
    ppv_hat = iso.predict(x)
    # Guard against exact 0/1 before logit later
    eps = 1e-6
    return np.clip(ppv_hat, eps, 1 - eps)

def calibrate_and_combine(
    scores: dict,
    rocs: dict,
    prevalence: float,
    y_union: np.ndarray,
    add_interaction: bool = True,
    C: float = 1e3,
    max_iter: int = 1000,
    random_state: int = 0
):
    """
    Parameters
    ----------
    scores : dict[str, np.ndarray]
        Raw score arrays per model, same length n. Example: {'A': sA, 'B': sB, ...}
    rocs : dict[str, pd.DataFrame]
        ROC dataframes per model with columns ['fpr','tpr','thresholds'] matching `scores` keys.
    prevalence : float
        Assumed prevalence p of the *model's target* in the ROC set (used for PPV computation).
    y_union : np.ndarray of shape (n,)
        Ground-truth label for the composite task Y = 1[A or B], used to train the GLM combiner.
    add_interaction : bool
        If True, include a multiplicative interaction term between calibrated scores.
    C, max_iter, random_state :
        LogisticRegression hyperparameters.

    Returns
    -------
    result : dict
        {
          'calibrated': pd.DataFrame of calibrated PPV features (columns match keys in `scores`),
          'glm': fitted LogisticRegression instance,
          'scaler': fitted StandardScaler (applied to logit-transformed features),
          'pp': np.ndarray of shape (n,), combined probability estimates,
          'isotonic_maps': dict[str, IsotonicRegression]
        }
    """
    # 1) Build isotonic calibrators from ROC→PPV
    isotonic_maps = {}
    for k, roc_df in rocs.items():
        thr_ppv = _ppv_from_roc(roc_df, prevalence)
        isotonic_maps[k] = _fit_isotonic_score_to_ppv(thr_ppv)

    # 2) Apply calibrators to raw scores → per-model PPV-calibrated features
    cal_feats = {}
    n = None
    for k, s in scores.items():
        s = np.asarray(s, dtype=float)
        if n is None:
            n = s.shape[0]
        elif s.shape[0] != n:
            raise ValueError("All score arrays must have the same length.")
        cal_feats[k] = _predict_ppv(isotonic_maps[k], s)

    X_ppv = pd.DataFrame(cal_feats)

    # 3) GLM combiner on logits of calibrated PPVs (well-conditioned scale)
    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p) - np.log1p(-p)

    X = np.column_stack([logit(X_ppv[col].to_numpy()) for col in X_ppv.columns])
    cols = list(X_ppv.columns)
    if add_interaction and X.shape[1] >= 2:
        # single multiplicative interaction of all features
        inter = np.prod(X, axis=1, keepdims=True)
        X = np.hstack([X, inter])
        cols = cols + ["_interaction(logit_PPVs_product)"]

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    y = np.asarray(y_union, dtype=int)
    if y.shape[0] != n:
        raise ValueError("y_union length must match scores.")

    glm = LogisticRegression(penalty='l2', C=C, solver='lbfgs', max_iter=max_iter, random_state=random_state)
    glm.fit(Xz, y)
    pp = glm.predict_proba(Xz)[:, 1]

    return {
        'calibrated': X_ppv,         # per-model calibrated PPVs
        'glm': glm,                  # trained combiner
        'scaler': scaler,            # standardizer used on logit features
        'pp': pp,                    # combined probability for Y = 1[A ∪ B]
        'isotonic_maps': isotonic_maps,
        'feature_names': cols
    }
