"""Regression checks for reusable score-level curves and plotting.

Run in the repository with: python -m unittest discover -s tests -v
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from zedstat import curveplots


class CurvePlotsAPITest(unittest.TestCase):
    @staticmethod
    def sample():
        rng = np.random.default_rng(311)
        positive = rng.beta(4, 3, 400)
        negative = rng.beta(2, 5, 2200)
        return pd.DataFrame({
            'target': np.r_[np.ones(len(positive)), np.zeros(len(negative))],
            'predicted_risk': np.r_[positive, negative],
        })

    def test_lr_frontier_columns_are_configurable(self):
        df = self.sample()
        standard = curveplots.lr_frontier_from_scores(df)
        renamed = curveplots.lr_frontier_from_scores(
            df.rename(columns={'target': 'case', 'predicted_risk': 'score'}),
            label_col='case', score_col='score',
        )
        assert_frame_equal(standard, renamed, check_exact=True)
        self.assertGreater(len(standard), 0)
        self.assertTrue(standard['LR-'].is_monotonic_increasing)
        self.assertTrue(standard['LR+'].is_monotonic_increasing)
        self.assertTrue((standard[['TP', 'FP', 'FN', 'TN']] >= 10).all().all())

    def test_fixed_specificity_bootstrap_is_seeded_and_generic(self):
        df = self.sample()
        first, info1 = curveplots.fixed_specificity_bootstrap(
            df, specificity=.95, n_boot=250, seed=120,
        )
        renamed, info2 = curveplots.fixed_specificity_bootstrap(
            df.rename(columns={'target': 'case', 'predicted_risk': 'score'}),
            label_col='case', score_col='score', specificity=.95,
            n_boot=250, seed=120,
        )
        self.assertEqual(first, renamed)
        self.assertEqual(info1, info2)
        for estimate, lower, upper in first.values():
            self.assertTrue(np.isfinite([estimate, lower, upper]).all())
            self.assertLessEqual(lower, upper)

    def test_panel_export_and_table_for_arbitrary_cohorts_and_targets(self):
        fpr = np.linspace(.01, .9, 60)
        roc = pd.DataFrame({
            'fpr': fpr, 'tpr': np.sqrt(fpr),
            'tpr_lower': np.sqrt(fpr)*.9,
            'tpr_upper': np.minimum(np.sqrt(fpr)*1.1, 1),
            'AUC': .75, 'AUC_lower': .7, 'AUC_upper': .8,
        })
        for metric, val in [('ppv', .5), ('npv', .5), ('LR+', 2), ('LR-', .5)]:
            roc[metric] = val
            roc[f'{metric}_lower'] = val*.9
            roc[f'{metric}_upper'] = val*1.1
        cohorts = ('Group1', 'Group2')
        targets = ('Disease1', 'Disease2')
        bundles = {(cohort, target): {'roc': roc}
                   for cohort in cohorts for target in targets}
        styles = {'Group1': {'linestyle':'-', 'alpha':.25},
                  'Group2': {'linestyle':'--', 'alpha':.125}}
        colors = {'Disease1': '#0000ff', 'Disease2':'#ff0000'}
        with tempfile.TemporaryDirectory() as d:
            paths = curveplots.plot_curve_panel(
                bundles, 'ROC', cohorts=cohorts, targets=targets,
                colors=colors, cohort_styles=styles,
                output_stem=Path(d)/'example', png_dpi=60,
            )
            self.assertTrue(all(p.exists() for p in paths))
            table = curveplots.performance_table_latex(
                bundles, target_labels={t:t for t in targets},
                cohorts=cohorts, specificity=.95,
                caption='Generic ROC table', label='tab:generic',
            )
            self.assertIn('Generic ROC table', table)
            self.assertIn('Group2', table)
            self.assertIn(r'\begin{table}', table)
            self.assertIn(r'\end{table}', table)


if __name__ == '__main__':
    unittest.main()
