"""
# zedstat

Utilities for uncertainty quantification and deployment of ML models.

Installation::

    pip install zedstat

Useful for calculation of likelihood ratios, confidence intervals on AUC,
and simple performance interpretation.
"""

from functools import wraps

# Import the public submodule during package initialization so every supported
# import style receives the same processRoc behavior.
from . import zedstat as zedstat


def _install_getbounds_nominal_guard():
    """Ensure getBounds() cannot replace the current nominal ROC dataframe.

    getBounds() is a bound-calculation operation.  It may populate df_lim and
    df_measure_bounds_, but it must not change self.df, because self.df may
    contain a smoothed/convexified ROC selected by the caller.
    """
    original = zedstat.processRoc.getBounds

    # Avoid stacking wrappers if the package is reloaded.
    if getattr(original, "_preserves_nominal_df", False):
        return

    @wraps(original)
    def guarded_getBounds(self, *args, **kwargs):
        nominal_df = self.df.copy(deep=True)
        try:
            return original(self, *args, **kwargs)
        finally:
            # Bounds are side products; the nominal/display ROC is invariant.
            self.df = nominal_df

    guarded_getBounds._preserves_nominal_df = True
    zedstat.processRoc.getBounds = guarded_getBounds


_install_getbounds_nominal_guard()
del _install_getbounds_nominal_guard
