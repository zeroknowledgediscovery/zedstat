import unittest
import numpy as np
import pandas as pd
from zedstat.ci_logratio import ppv_logratio_ci, lr_logratio_ci

class TestLogRatioCI(unittest.TestCase):
    def test_ppv_formula_and_bracketing(self):
        roc = pd.DataFrame({'fpr':[0, 0.02, 0.05,0.2], 'tpr':[0,0.15,0.5,0.9]})
        out=ppv_logratio_ci(roc, positive_samples=400, negative_samples=2000,
                            prevalence=.1, min_tp=10, min_fp=10, smooth_sigma=2)
        self.assertTrue(np.isnan(out['ppv_logratio_lower'].iloc[0]))
        center=.1*.5/(.1*.5+.9*.05)
        self.assertAlmostEqual(center, float(out['ppv'].iloc[2]) if 'ppv' in out else center)
        row=out.iloc[2]
        self.assertTrue(row['ppv_logratio_lower'] < center < row['ppv_logratio_upper'])

    def test_lr_asymptotic_intervals_enclose_nominal_and_preserve_nan(self):
        counts = pd.DataFrame({'TP':[5,30,100,250],'FP':[2,20,90,160],
                               'FN':[395,370,300,150], 'TN':[1998,1980,1910,1840]})
        counts['LR+'] = (counts['TP']/400)/(counts['FP']/2000)
        counts['LR-'] = (counts['FN']/400)/(counts['TN']/2000)
        a=lr_logratio_ci(counts, smooth_sigma=0)
        b=lr_logratio_ci(counts, smooth_sigma=1.8)
        self.assertTrue(np.isnan(a['LR+_logratio_lower'].iloc[0]))
        for metric in ('LR+', 'LR-'):
            np.testing.assert_array_equal(a[f'{metric}_logratio_lower'],b[f'{metric}_logratio_lower'])
            valid=b[f'{metric}_logratio_lower_display'].notna()
            self.assertTrue((b.loc[valid, f'{metric}_logratio_lower_display'] <= b.loc[valid,metric]).all())
            self.assertTrue((b.loc[valid,metric] <= b.loc[valid,f'{metric}_logratio_upper_display']).all())

    def test_boundary_support_is_not_reconstructed(self):
        roc=pd.DataFrame({'fpr':[0,0.005,.01,.02], 'tpr':[0,.1,.2,.4]})
        out=ppv_logratio_ci(roc,positive_samples=200,negative_samples=2000,prevalence=.1,
                            min_fp=20,smooth_sigma=10)
        self.assertTrue(np.isnan(out['ppv_logratio_lower_display'].iloc[0]))
        self.assertTrue(np.isnan(out['ppv_logratio_lower_display'].iloc[1]))
        self.assertTrue(np.isfinite(out['ppv_logratio_lower_display'].iloc[2]))

if __name__ == '__main__': unittest.main()
