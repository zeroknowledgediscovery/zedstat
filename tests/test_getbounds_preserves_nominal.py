import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from zedstat import zedstat


class GetBoundsNominalInvariantTest(unittest.TestCase):
    def test_getbounds_preserves_convexified_nominal_roc(self):
        # Deliberately fluctuating ROC so convexification materially changes it.
        roc = pd.DataFrame(
            {
                "fpr": [0.0, 0.05, 0.10, 0.18, 0.25, 0.40, 0.55, 0.75, 1.0],
                "tpr": [0.0, 0.36, 0.31, 0.60, 0.54, 0.78, 0.74, 0.91, 1.0],
            }
        )

        zt = zedstat.processRoc(
            df=roc,
            total_samples=1000,
            positive_samples=500,
            alpha=0.01,
            prevalence=0.5,
        )
        zt.smooth(STEP=0.001, interpolate=True, convexify=True)
        zt.allmeasures(interpolate=False)

        before = zt.get().copy(deep=True)
        auc_before = zt.auc(alpha=0.05)[0]

        # Confirm the fixture actually exercises the ROCCH path.
        fpr = before.index.to_numpy(dtype=float)
        tpr = before["tpr"].to_numpy(dtype=float)
        dx = np.diff(fpr)
        dy = np.diff(tpr)
        slopes = dy / dx
        self.assertTrue(np.all(dx > 0))
        self.assertTrue(np.all(dy >= -1e-12))
        self.assertTrue(np.all(np.diff(slopes) <= 1e-9))

        zt.getBounds()

        after = zt.get().copy(deep=True)
        auc_after = zt.auc(alpha=0.05)[0]

        assert_frame_equal(before, after, check_exact=True)
        self.assertAlmostEqual(auc_before, auc_after, places=14)
        self.assertIn("L", zt.df_lim)
        self.assertIn("U", zt.df_lim)


if __name__ == "__main__":
    unittest.main()
