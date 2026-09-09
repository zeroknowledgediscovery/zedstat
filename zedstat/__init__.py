"""
# zedstat

Utilities for uncertainty quantification and deployment of ML models.

Installation::

    pip install zedstat

Useful for calculation of likelihood ratios, confidence intervals on AUC,
and simple performance interpretation.
"""

from functools import wraps
import inspect

import numpy as np
import pandas as pd

# Import the public submodule during package initialization so every supported
# import style receives the same processRoc behavior.
from . import zedstat as zedstat


def _bounds_on_current_geometry(self, nominal_df, prevalence):
    """Re-center empirical pointwise uncertainty on the current ROC geometry.

    The uncertainty widths are still estimated from the raw empirical ROC by
    processRoc.getBounds(). This helper only changes the display geometry:
    those empirical lower/upper distances are interpolated to the current FPR
    grid and re-centered on the current (possibly smoothed/convexified) ROC.

    This avoids treating a convex-hull/interpolated TPR as if it were itself an
    independent binomial observation while ensuring the displayed confidence
    band follows the ROC that the user actually requested.
    """
    info = getattr(self, "df_measure_bounds_", None) or {}
    nominal_emp = info.get("nominal_empirical")
    lower_emp = info.get("L_empirical")
    upper_emp = info.get("U_empirical")

    if nominal_emp is None or lower_emp is None or upper_emp is None:
        return

    # Recompute derived measures from the current FPR/TPR coordinates so the
    # display nominal is internally consistent with the smoothed ROC.
    nominal = self._compute_measures(
        nominal_df,
        prevalence=float(prevalence),
        apply_ppv_isotonic=True,
    )
    target_index = nominal.index.to_numpy(dtype=float)

    cols = [
        c
        for c in ["tpr", "ppv", "acc", "npv", "LR+", "LR-"]
        if c in nominal.columns
        and c in nominal_emp.columns
        and c in lower_emp.columns
        and c in upper_emp.columns
    ]

    # Empirical one-sided uncertainty widths around the raw nominal curve.
    lower_gap = pd.DataFrame(index=nominal_emp.index.copy())
    upper_gap = pd.DataFrame(index=nominal_emp.index.copy())
    lower_gap.index.name = self.fprcol
    upper_gap.index.name = self.fprcol

    for col in cols:
        n = pd.to_numeric(nominal_emp[col], errors="coerce").to_numpy(dtype=float)
        lo = pd.to_numeric(lower_emp[col], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(upper_emp[col], errors="coerce").to_numpy(dtype=float)

        lower_gap[col] = np.where(
            np.isfinite(n) & np.isfinite(lo),
            np.maximum(n - lo, 0.0),
            np.nan,
        )
        upper_gap[col] = np.where(
            np.isfinite(n) & np.isfinite(hi),
            np.maximum(hi - n, 0.0),
            np.nan,
        )

    lower_gap = zedstat._numeric_interp_frame_to_target_index(
        lower_gap,
        target_index,
        index_name=self.fprcol,
    )
    upper_gap = zedstat._numeric_interp_frame_to_target_index(
        upper_gap,
        target_index,
        index_name=self.fprcol,
    )

    lower = nominal.copy(deep=True)
    upper = nominal.copy(deep=True)

    for col in cols:
        n = pd.to_numeric(nominal[col], errors="coerce").to_numpy(dtype=float)
        lg = pd.to_numeric(lower_gap[col], errors="coerce").to_numpy(dtype=float)
        ug = pd.to_numeric(upper_gap[col], errors="coerce").to_numpy(dtype=float)
        lower[col] = n - lg
        upper[col] = n + ug

    # Probability-like quantities are bounded by [0, 1].
    for col in ["tpr", "ppv", "acc", "npv"]:
        if col in lower.columns:
            lower[col] = np.clip(
                pd.to_numeric(lower[col], errors="coerce").to_numpy(dtype=float),
                0.0,
                1.0,
            )
            upper[col] = np.clip(
                pd.to_numeric(upper[col], errors="coerce").to_numpy(dtype=float),
                0.0,
                1.0,
            )

    # ROC display limits should respect monotonic ROC geometry. We do not
    # independently convex-hull the limits because that would alter their
    # statistical interpretation more aggressively.
    if "tpr" in lower.columns:
        lower["tpr"] = np.maximum.accumulate(
            pd.to_numeric(lower["tpr"], errors="coerce").to_numpy(dtype=float)
        )
        upper["tpr"] = np.maximum.accumulate(
            pd.to_numeric(upper["tpr"], errors="coerce").to_numpy(dtype=float)
        )

    lower, upper = zedstat._enforce_bounds_around_nominal(
        nominal,
        lower,
        upper,
        cols=cols,
    )

    nominal = zedstat._apply_lr_floor(
        nominal,
        lr_fpr_floor=self.lr_fpr_floor,
        lr_sp_floor=getattr(self, "lr_sp_floor", self.lr_fpr_floor),
        fprcol=self.fprcol,
    )
    lower = zedstat._apply_lr_floor(
        lower,
        lr_fpr_floor=self.lr_fpr_floor,
        lr_sp_floor=getattr(self, "lr_sp_floor", self.lr_fpr_floor),
        fprcol=self.fprcol,
    )
    upper = zedstat._apply_lr_floor(
        upper,
        lr_fpr_floor=self.lr_fpr_floor,
        lr_sp_floor=getattr(self, "lr_sp_floor", self.lr_fpr_floor),
        fprcol=self.fprcol,
    )

    self.df_lim["L"] = lower
    self.df_lim["U"] = upper
    self.df_measure_bounds_["nominal_display"] = nominal
    self.df_measure_bounds_["L_display"] = lower
    self.df_measure_bounds_["U_display"] = upper
    self.df_measure_bounds_["geometry"] = "current"


def _install_getbounds_geometry_guard():
    """Keep the nominal ROC invariant and align display bounds when requested."""
    original = zedstat.processRoc.getBounds

    # If an older package reload left the nominal-only guard in place, unwrap it
    # before installing the newer geometry-aware wrapper.
    if getattr(original, "_bounds_geometry_v2", False):
        return
    if getattr(original, "_preserves_nominal_df", False) and hasattr(original, "__wrapped__"):
        original = original.__wrapped__

    original_sig = inspect.signature(original)

    @wraps(original)
    def guarded_getBounds(self, *args, geometry="current", **kwargs):
        geometry = str(geometry).lower()
        if geometry not in {"current", "empirical"}:
            raise ValueError("geometry must be 'current' or 'empirical'")

        nominal_df = self.df.copy(deep=True)
        bound = original_sig.bind_partial(self, *args, **kwargs)
        prevalence = bound.arguments.get("prevalence", self.prevalence)

        try:
            result = original(self, *args, **kwargs)

            if geometry == "current":
                if prevalence is None:
                    raise ValueError("prevalence undefined")
                _bounds_on_current_geometry(
                    self,
                    nominal_df=nominal_df,
                    prevalence=prevalence,
                )
            else:
                self.df_measure_bounds_["geometry"] = "empirical"

            return result
        finally:
            # Bounds are side products. The nominal/display ROC selected by
            # smooth()/usample() must never be replaced by getBounds().
            self.df = nominal_df

    # Preserve introspection while exposing the new keyword-only option.
    params = list(original_sig.parameters.values())
    params.append(
        inspect.Parameter(
            "geometry",
            kind=inspect.Parameter.KEYWORD_ONLY,
            default="current",
        )
    )
    guarded_getBounds.__signature__ = original_sig.replace(parameters=params)
    guarded_getBounds._preserves_nominal_df = True
    guarded_getBounds._bounds_geometry_v2 = True
    zedstat.processRoc.getBounds = guarded_getBounds


_install_getbounds_geometry_guard()
del _install_getbounds_geometry_guard
