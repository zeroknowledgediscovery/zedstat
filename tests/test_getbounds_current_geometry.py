import inspect
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from zedstat import zedstat


class GetBoundsCurrentGeometryTest(unittest.TestCase):
    @staticmethod
    def _make_zt():
        # Deliberately fluctuating ROC so convexification materially changes it.
        roc = pd.DataFrame(
            {
                "fpr": [0.0, 0.04, 0.08, 0.13, 0.20, 0.31, 0.44, 0.60, 0.78, 1.0],
                "tpr": [0.0, 0.31, 0.27, 0.53, 0.48, 0.70, 0.66, 0.84, 0.88, 1.0],
            }
        )
        zt = zedstat.processRoc(
            df=roc,
            total_samples=1200,
            positive_samples=600,
            alpha=0.01,
            prevalence=0.5,
        )
        zt.smooth(STEP=0.001, interpolate=True, convexify=True)
        zt.allmeasures(interpolate=False)
        return zt

    def test_getbounds_defaults_to_current_geometry(self):
        zt = self._make_zt()
        before = zt.get().copy(deep=True)
        auc_before = zt.auc(alpha=0.05)[0]

        zt.getBounds()

        after = zt.get().copy(deep=True)
        auc_after = zt.auc(alpha=0.05)[0]

        # getBounds is side-effect free with respect to the nominal ROC.
        assert_frame_equal(before, after, check_exact=True)
        self.assertAlmostEqual(auc_before, auc_after, places=14)

        self.assertEqual(zt.df_measure_bounds_["geometry"], "current")
        self.assertIn("L", zt.df_lim)
        self.assertIn("U", zt.df_lim)

        nominal = zt.get()["tpr"].to_numpy(dtype=float)
        lower = zt.df_lim["L"]["tpr"].to_numpy(dtype=float)
        upper = zt.df_lim["U"]["tpr"].to_numpy(dtype=float)

        finite = np.isfinite(nominal) & np.isfinite(lower) & np.isfinite(upper)
        self.assertTrue(np.all(lower[finite] <= nominal[finite] + 1e-12))
        self.assertTrue(np.all(upper[finite] >= nominal[finite] - 1e-12))

        # Display ROC limits follow monotone ROC geometry after smoothing.
        self.assertTrue(np.all(np.diff(lower[np.isfinite(lower)]) >= -1e-12))
        self.assertTrue(np.all(np.diff(upper[np.isfinite(upper)]) >= -1e-12))

    def test_empirical_geometry_remains_available(self):
        zt = self._make_zt()
        before = zt.get().copy(deep=True)

        zt.getBounds(geometry="empirical")

        assert_frame_equal(before, zt.get(), check_exact=True)
        self.assertEqual(zt.df_measure_bounds_["geometry"], "empirical")

    def test_geometry_parameter_is_visible(self):
        sig = inspect.signature(zedstat.processRoc.getBounds)
        self.assertIn("geometry", sig.parameters)
        self.assertEqual(sig.parameters["geometry"].default, "current")

    def test_invalid_geometry_raises(self):
        zt = self._make_zt()
        with self.assertRaises(ValueError):
            zt.getBounds(geometry="banana")


if __name__ == "__main__":
    unittest.main()
