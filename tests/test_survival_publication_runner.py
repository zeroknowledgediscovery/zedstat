import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from zedstat import survival

class PublicationRunnerTest(unittest.TestCase):
    def test_one_call_loads_and_exports_both_families(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            models=('ILD-AFSTDPULMO','ILD-AFSTDOXYGEN','ILD-AFSTDARDS')
            model_target={'ILD-AFSTDPULMO':'Pulmonary_Hypertension',
                          'ILD-AFSTDOXYGEN':'Oxygen_Dependence','ILD-AFSTDARDS':'ARDS'}
            for cohort,path in [('MarketScan',root/'PAPER_DATA_MARKETSCAN/survival_MARKETSCAN'),
                                ('UKHC',root/'PAPER_DATA_UKHC_50TRAINING/survival_UKHC')]:
                path.mkdir(parents=True)
                pd.DataFrame([
                    dict(M=target, SPEC='0975',week=24, threshold=.5,n_crossed=2,n_total=6,
                         crossing_rate_pct=30,event_probability_crossed_pct=23,
                         event_probability_never_crossed_pct=3,risk_ratio=7,
                         ARD_pp=20,logrank_p=.02)
                    for target in model_target.values()
                ]).to_csv(path/'SUMMARY.csv',index=False)
                for model in models:
                    pd.DataFrame(dict(crossed_threshold=[0,1],duration=[12,5],
                                      event_observed=[0,1])).to_csv(
                        path/f'km_full_cohort_{model}_0975.csv',index=False)
                    pd.DataFrame(dict(k_weeks=[1,24],
                                      P_event_within_k_given_crossed=[.1,.3],
                                      P_no_event_within_k_given_never_crossed=[.9,.8],
                                      P_crossed_given_event_within_k=[.2,.7],
                                      P_never_crossed_given_no_event_within_k=[.9,.8])).to_csv(
                         path/f'dynamic_metrics_full_cohort_{model}_0975.csv',index=False)
            calls=[]
            def fake_plot(*args,**kwargs):
                model=args[1]; out=Path(args[2])
                prefix=('km_' if kwargs.get('legend_fontsize') is not None else 'dynamic_metrics_')
                result=out/f'{prefix}{model}.png'
                result.write_bytes(b'valid placeholder')
                calls.append((prefix,model,kwargs['max_week']))
                return result
            def fake_table(summary_df, output_dir, **kwargs):
                p=Path(output_dir)/'KM_SUMMARY_TABLE.tex'; p.write_text('test')
                return p
            with patch.object(survival,'plot_joined_km',side_effect=fake_plot), \
                 patch.object(survival,'plot_joined_dynamic_metrics',side_effect=fake_plot), \
                 patch.object(survival,'export_joined_survival_summary_table',side_effect=fake_table):
                result=survival.run_ild_survival_publication(root,show=False)
            self.assertEqual(len(calls),6)
            self.assertEqual(len(result['output_paths']),8)
            self.assertTrue(all(p.exists() for p in result['output_paths']))
            self.assertTrue(all(w==24 for _,_,w in calls))
            self.assertEqual(len(result['summary']),6)

if __name__=='__main__':
    unittest.main()
