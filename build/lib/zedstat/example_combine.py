import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve

# --- Assume you already have these from before ---
# calibrate_and_combine(scores, rocs, prevalence, y_union, add_interaction=True, ...)
# plus its helpers, already defined in your session.

def apply_combiner_to_new(scores_new: dict, trained):
    """
    Apply frozen isotonic calibrators + scaler + GLM to new data.
    Parameters
    ----------
    scores_new : dict[str, np.ndarray]
        Raw scores per model on the NEW set, keys must match training.
    trained : dict
        Output dict from calibrate_and_combine(...) on the training set.

    Returns
    -------
    pp_new : np.ndarray
        Combined probability estimates for Y = 1[A ∪ B] on the new set.
    X_ppv_new : pd.DataFrame
        Per-model calibrated PPVs on the new set (for inspection).
    """
    # 1) per-model PPV via frozen isotonic mappers
    def _predict_ppv(iso, s):
        x = -s if getattr(iso, "_flipped", False) else s
        y = iso.predict(x)
        return np.clip(y, 1e-6, 1-1e-6)

    X_ppv_new = {}
    for k, iso in trained['isotonic_maps'].items():
        s = np.asarray(scores_new[k], dtype=float)
        X_ppv_new[k] = _predict_ppv(iso, s)
    X_ppv_new = pd.DataFrame(X_ppv_new)

    # 2) build feature matrix exactly like training: logit and optional interaction
    def logit(p):
        p = np.clip(p, 1e-6, 1-1e-6)
        return np.log(p) - np.log1p(-p)

    X_list = [logit(X_ppv_new[col].to_numpy()) for col in trained['calibrated'].columns]
    X_new = np.column_stack(X_list)
    if any("interaction" in nm for nm in trained['feature_names']):
        inter = np.prod(X_new, axis=1, keepdims=True)
        X_new = np.hstack([X_new, inter])

    # 3) standardize with the frozen scaler, then predict with frozen GLM
    Xz_new = trained['scaler'].transform(X_new)
    pp_new = trained['glm'].predict_proba(Xz_new)[:, 1]
    return pp_new, X_ppv_new

# ====== EXAMPLE WORKFLOW ======
# Inputs you have:
#   train scores for each component: sA_tr, sB_tr, ... as arrays of length n_tr
#   validation scores: sA_va, sB_va, ...
#   ROC dataframes for each component: rocA, rocB, ... (columns ['fpr','tpr','thresholds'])
#   prevalence p used for PPV calibration (per component; use a dict if they differ)
#   ground-truth A_true, B_true on train and val to form Y_or = 1[(A_true)|(B_true)]

# Make union labels:
y_or_tr = ((A_true_tr.astype(bool)) | (B_true_tr.astype(bool))).astype(int)
y_or_va = ((A_true_va.astype(bool)) | (B_true_va.astype(bool))).astype(int)

# If components have different prevalences, pass a dict e.g. {'A': pA, 'B': pB}
# For a single common prevalence p:
p = 0.20  # example

trained = calibrate_and_combine(
    scores={'A': sA_tr, 'B': sB_tr},
    rocs={'A': rocA, 'B': rocB},
    prevalence=p,
    y_union=y_or_tr,
    add_interaction=True
)

# Evaluate in-sample (for sanity, not as your headline number)
auc_tr = roc_auc_score(y_or_tr, trained['pp'])

# Apply frozen pipeline to validation
pp_va, Xppv_va = apply_combiner_to_new(
    scores_new={'A': sA_va, 'B': sB_va},
    trained=trained
)

# Report proper held-out metrics
auc_va  = roc_auc_score(y_or_va, pp_va)
aupr_va = average_precision_score(y_or_va, pp_va)
brier   = brier_score_loss(y_or_va, pp_va)

# Optional: pick an operating point from training ROC and reuse threshold on validation
fpr_tr, tpr_tr, thr_tr = roc_curve(y_or_tr, trained['pp'])
# e.g., choose training threshold at target FPR≈0.10
target_fpr = 0.10
idx = np.argmin(np.abs(fpr_tr - target_fpr))
thr = thr_tr[idx]
tpr_at_target = (pp_va[y_or_va == 1] >= thr).mean()
fpr_at_target = (pp_va[y_or_va == 0] >= thr).mean()

print({"AUC_train": auc_tr, "AUC_val": auc_va, "AUPRC_val": aupr_va,
       "Brier_val": brier, "Val_TPR@FPR≈0.10": tpr_at_target, "Val_FPR@thr": fpr_at_target})
