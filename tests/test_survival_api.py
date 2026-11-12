"""Tests for general zedstat.survival APIs.

Run with: python -m unittest discover -s tests -p 'test_survival_api.py' -v
Requires lifelines, matplotlib, pandas and numpy.
"""
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zedstat import survival


class SurvivalApiTests(unittest.TestCase):
    @staticmethod
    def patient_cohorts():
        return {
            name: pd.DataFrame({
                'duration': [2, 3, 7, 15, 25, 28, 1, 8, 16, 27],
                'event_observed': [1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
                'crossed_threshold': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            })
            for name in ('cohort_a', 'cohort_b')
        }

    @staticmethod
    def metrics():
        cols = [
            'P_event_within_k_given_crossed',
            'P_no_event_within_k_given_never_crossed',
            'P_crossed_given_event_within_k',
            'P_never_crossed_given_no_event_within_k',
        ]
        return {
            name: pd.DataFrame({'k_weeks': [1, 2, 4, 8, 12, 24, 26],
                                **{c: np.linspace(.1, .9, 7) for c in cols}})
            for name in ('cohort_a', 'cohort_b')
        }

    def test_km_generic_cohort_and_group_api(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'fig.png'
            fig, ax = survival.plot_km_comparison(
                self.patient_cohorts(),
                cohort_order=('cohort_b', 'cohort_a'),
                cohort_linestyles={'cohort_a': '-', 'cohort_b': '--'},
                max_week=24, risk_set_mode='original_filter',
                outfile=file, show=False,
            )
            self.assertTrue(file.is_file())
            self.assertEqual(len(ax.lines), 4)
            self.assertTrue(all('cohort_' in line.get_label() for line in ax.lines))
            plt.close(fig)

    def test_km_administrative_is_explicit_option(self):
        fig, ax = survival.plot_km_comparison(
            self.patient_cohorts(),
            cohort_order=('cohort_a', 'cohort_b'),
            max_week=24, risk_set_mode='administrative',
        )
        self.assertEqual(len(ax.lines), 4)
        plt.close(fig)

    def test_dynamic_plot_uses_requested_horizon(self):
        fig, ax = survival.plot_finite_horizon_comparison(
            self.metrics(), max_week=24,
            cohort_order=('cohort_b', 'cohort_a'),
            landmark_weeks=(1, 2, 4, 8, 12, 24, 26),
        )
        self.assertEqual(len(ax.lines), 8)
        for line in ax.lines:
            self.assertEqual(max(line.get_xdata()), 24)
        plt.close(fig)

    def test_summary_is_general_and_does_not_load_csv_files(self):
        data = pd.DataFrame([
            {'M': target, 'source': cohort,
             'threshold': .4, 'n_crossed': 10, 'n_total': 100,
             'crossing_rate_pct': 10.0,
             'event_probability_crossed_pct': 30.0,
             'event_probability_never_crossed_pct': 4.0,
             'risk_ratio': 7.5, 'ARD_pp': 26.0, 'logrank_p': .001}
            for target in ('Event_One', 'Event_Two')
            for cohort in ('cohort_a', 'cohort_b')
        ])
        latex = survival.survival_summary_to_latex(
            data, horizon=24, specificity='0975',
            targets=('Event_One', 'Event_Two'),
            cohorts=('cohort_a', 'cohort_b'),
        )
        self.assertIn(r'97.5\%-specificity', latex)
        self.assertIn(r'\multirow{2}', latex)
        self.assertIn('cohort_a', latex)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'summary.tex'
            survival.export_survival_summary_table(
                data, out, horizon=24, specificity=.975,
                targets=('Event_One', 'Event_Two'),
                cohorts=('cohort_a', 'cohort_b'),
            )
            self.assertEqual(out.read_text(), latex)

    def test_only_general_public_api_is_exposed(self):
        expected = {
            'make_patient_level_threshold_dataset', 'compute_km_statistics',
            'dynamic_threshold_metrics', 'plot_km_comparison',
            'plot_finite_horizon_comparison', 'survival_summary_to_latex',
            'export_survival_summary_table',
        }
        self.assertEqual(set(survival.__all__), expected)
        self.assertFalse(hasattr(survival, 'run_ild_survival_publication'))
        # Keep the original plotting functions callable for backward compatibility,
        # but do not advertise them as part of the new general-purpose API.
        legacy = (
            'plot_km_paper',
            'plot_dynamic_metrics_paper',
            'plot_joined_km',
            'plot_joined_dynamic_metrics',
            'export_joined_survival_summary_table',
        )
        for name in legacy:
            self.assertNotIn(name, survival.__all__)
            self.assertTrue(callable(getattr(survival, name)))

    def test_validates_invalid_inputs(self):
        with self.assertRaises(ValueError):
            survival.plot_km_comparison({}, max_week=24)
        with self.assertRaises(ValueError):
            survival.plot_km_comparison(self.patient_cohorts(), risk_set_mode='unknown')
        with self.assertRaises(ValueError):
            survival.survival_summary_to_latex(pd.DataFrame(), horizon=24,
                                              specificity=97.5, targets=('A',), cohorts=('B',))


if __name__ == '__main__':
    unittest.main()
