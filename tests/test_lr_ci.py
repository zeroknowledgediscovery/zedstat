"""Regression tests for zedstat LR frontier confidence intervals."""
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from zedstat.lr_ci import lr_curve_ci_from_scores


class TestLRCurveCI(unittest.TestCase):
    @staticmethod
    def mock(n_pos=350, n_neg=1800, seed=8):
        rng = np.random.default_rng(seed)
        positive = rng.beta(4, 3, n_pos)
        negative = rng.beta(2, 5, n_neg)
        return pd.DataFrame({"target": np.r_[np.ones(n_pos), np.zeros(n_neg)],
                             "predicted_risk": np.r_[positive, negative]})

    def test_ci_encloses_nominal_and_supports_all_points(self):
        data = self.mock()
        out = lr_curve_ci_from_scores(data, n_boot=400, random_state=1)
        self.assertGreater(len(out), 10)
        self.assertTrue((out[["TP", "FP", "FN", "TN"]].min(axis=1) >= 10).all())
        self.assertTrue(out["LR-"].is_monotonic_increasing)
        self.assertTrue(out["LR+"].is_monotonic_increasing)
        for metric in ("LR+", "LR-"):
            complete = out[[f"{metric}_lower", f"{metric}_upper"]].notna().all(axis=1)
            self.assertGreater(complete.sum(), 0)
            self.assertTrue((out.loc[complete, f"{metric}_lower"] <= out.loc[complete, metric]).all())
            self.assertTrue((out.loc[complete, metric] <= out.loc[complete, f"{metric}_upper"]).all())

    def test_reproducible_and_smoothing_does_not_modify_raw_ci(self):
        data = self.mock()
        a = lr_curve_ci_from_scores(data, n_boot=400, random_state=19)
        b = lr_curve_ci_from_scores(data, n_boot=400, random_state=19, smooth_sigma=1.2)
        for col in ["LR+", "LR-", "LR+_lower", "LR+_upper", "LR-_lower", "LR-_upper"]:
            np.testing.assert_array_equal(a[col].to_numpy(), b[col].to_numpy())
        self.assertTrue(np.isfinite(b["LR+_lower_display"]).sum() > 0)

    def test_few_positives_are_handled_without_endpoint_extension(self):
        data = self.mock(n_pos=45, n_neg=1600, seed=93)
        out = lr_curve_ci_from_scores(data, n_boot=400, random_state=22)
        self.assertGreater(len(out), 0)
        self.assertTrue((out[["TP", "FP", "FN", "TN"]].min(axis=1) >= 10).all())
        for end in ("lower", "upper"):
            self.assertTrue(out[f"LR+_{end}"].isna().equals(out[f"LR+_{end}_display"].isna()))

    def test_reject_insufficient_counts(self):
        with self.assertRaisesRegex(ValueError, "Too few"):
            lr_curve_ci_from_scores(self.mock(n_pos=15, n_neg=1000), n_boot=400)


if __name__ == "__main__":
    unittest.main()
