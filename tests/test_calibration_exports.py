"""Public calibration API and migration regression checks."""
import unittest
from importlib import import_module

import numpy as np
import pandas as pd

from zedstat import calibration

core = import_module("zedstat.zedstat")


class CalibrationApiTests(unittest.TestCase):
    def test_public_exports_resolve(self):
        for module in (core, calibration):
            self.assertEqual(len(module.__all__), len(set(module.__all__)))
            for name in module.__all__:
                with self.subTest(module=module.__name__, name=name):
                    self.assertTrue(hasattr(module, name))

    def test_retired_calibration_paths_are_absent(self):
        for name in ("crossfit_isotonic_probabilities", "calibration_curve_table", "score_to_probability"):
            self.assertNotIn(name, core.__all__)
            self.assertFalse(hasattr(core, name))
        for name in ("fit_score_calibration", "calibration_curve", "plot_calibration_curve", "scoretoprobability"):
            self.assertFalse(hasattr(core.processRoc, name))

    def test_threshold_ppv_api_preserved(self):
        self.assertTrue(callable(core.score_to_threshold_ppv))
        self.assertTrue(callable(core.processRoc.score_to_threshold_ppv))

    def test_heldout_calibration_is_canonical(self):
        rng = np.random.default_rng(42)
        labels = np.tile([0, 1], 120)
        scores = labels + rng.normal(scale=0.65, size=len(labels))
        df = pd.DataFrame({"score": scores, "label": labels})
        result = calibration.heldout_isotonic_calibration_with_bootstrap(
            df, score_col="score", label_col="label",
            test_size=0.25, n_bins=6, n_boot=4, plot=False,
        )
        self.assertIn("iso_model", result)
        self.assertIn("calibration_table", result)
        self.assertIn("summary", result)
        self.assertTrue({"mean_pred", "obs_rate"}.issubset(result["calibration_table"].columns))
        prob = result["iso_model"].predict(np.array([-1.0, 0.0, 1.0, 2.0]))
        self.assertTrue(np.all(np.isfinite(prob)))
        self.assertTrue(np.all((0 <= prob) & (prob <= 1)))
        self.assertTrue(np.all(np.diff(prob) >= -1e-12))


if __name__ == "__main__":
    unittest.main()
