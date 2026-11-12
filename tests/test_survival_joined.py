"""Regression checks for the original KM_PLOTS joined-figure behavior."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zedstat import survival


class TestJoinedSurvivalPlots(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.outdir = Path(self.temp.name)
        self.patients = {
            cohort: pd.DataFrame({
                'crossed_threshold': [1, 1, 1, 0, 0, 0, 0],
                'duration': [3, 7, 26, 4, 11, 20, 30],
                'event_observed': [1, 1, 0, 0, 1, 0, 0],
            }) for cohort in ('UKHC', 'MarketScan')
        }
        self.metrics = {}
        for cohort in ('UKHC','MarketScan'):
            k = np.array([1, 2, 4, 8, 12, 24, 26])
            self.metrics[cohort] = pd.DataFrame({
                'k_weeks': k,
                'P_event_within_k_given_crossed': np.linspace(.05, .45, len(k)),
                'P_no_event_within_k_given_never_crossed': np.linspace(.96, .73, len(k)),
                'P_crossed_given_event_within_k': np.linspace(.15, .65, len(k)),
                'P_never_crossed_given_no_event_within_k': np.linspace(.96, .81, len(k)),
            })

    def tearDown(self):
        self.temp.cleanup()
        plt.close('all')

    def test_original_24_week_selection_and_ci_curves(self):
        """Preserve the original prefilter (not administrative censoring)."""
        seen = []
        original_fit = survival.KaplanMeierFitter.fit

        def recording_fit(self, durations, *args, **kwargs):
            seen.append((len(durations), float(np.max(durations))))
            return original_fit(self, durations, *args, **kwargs)

        with patch.object(survival.KaplanMeierFitter, 'fit', new=recording_fit):
            path = survival.plot_joined_km(
                self.patients, 'ILD-AFSTDARDS', self.outdir,
                max_week=24, dpi=60, show=False
            )
        self.assertTrue(path.exists())
        self.assertEqual(path.name, 'km_ILD-AFSTDARDS.png')
        self.assertEqual([n for n, _ in seen], [2, 2, 3, 3])
        self.assertTrue(all(last <= 24 for _, last in seen))

    def test_finite_horizon_original_8_traces(self):
        saved = {}
        original_close = survival.plt.close
        def keep_figure(fig):
            saved['fig'] = fig
        with patch.object(survival.plt, 'close', side_effect=keep_figure):
            path = survival.plot_joined_dynamic_metrics(
                self.metrics, 'ILD-AFSTDARDS', self.outdir,
                max_week=24, dpi=60, show=False
            )
        self.assertTrue(path.exists())
        self.assertEqual(path.name, 'dynamic_metrics_ILD-AFSTDARDS.png')
        fig = saved['fig']
        try:
            ax = fig.axes[0]
            self.assertEqual(len(ax.lines), 8)
            self.assertTrue(all(max(line.get_xdata()) <= 24 for line in ax.lines))
            self.assertEqual([line.get_linestyle() for line in ax.lines[:4]], ['--'] * 4)
            self.assertEqual([line.get_linestyle() for line in ax.lines[4:]], ['-'] * 4)
            self.assertEqual(ax.get_xlabel(), 'Follow-up window k (weeks)')
        finally:
            original_close(fig)

    def test_summary_export_retains_original_filename(self):
        rows = []
        for outcome in ('Pulmonary_Hypertension','Oxygen_Dependence','ARDS'):
            for source in ('MarketScan','UKHC'):
                rows.append({
                    'M':outcome, 'source':source, 'threshold':.15,
                    'n_crossed':15, 'n_total':100,
                    'crossing_rate_pct':15,
                    'event_probability_crossed_pct':28.2,
                    'event_probability_never_crossed_pct':2.1,
                    'risk_ratio':13.42, 'ARD_pp':26.1,
                    'logrank_p':1e-6
                })
        path = survival.export_joined_survival_summary_table(
            pd.DataFrame(rows), self.outdir,
            max_week=24, specificity='0975', print_latex=False
        )
        self.assertEqual(path.name,'KM_SUMMARY_TABLE.tex')
        text = path.read_text()
        self.assertIn('24 weeks',text)
        self.assertIn('97.5\\%-specificity',text)
        self.assertIn('Pulmonary Hypertension',text)


if __name__ == '__main__':
    unittest.main()
