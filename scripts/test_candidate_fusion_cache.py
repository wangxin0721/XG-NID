"""Offline arithmetic and cache-contract tests, no models or ML imports."""
import ast
import copy
from pathlib import Path
import sys
import unittest

from analyze_candidate_fusion_cache import (direction_values, outcome, prediction,
    sampled_runs, scan_stats, summary)


class CacheChecks(unittest.TestCase):
    def row(self):
        packet=[.1,.8,.1,0,0,0,0,0]; flow=[.8,.1,.1,0,0,0,0,0]
        preds=[1]*5+[0]*5+[1]+[0]*10
        gaps=[.7]*5+[-.7]*5+[.7]+[-.7]*10
        return {'label':1,'probabilities':{'packet':packet,'flow':flow,'equal':packet,'fusion':flow},
                'final_weights':[.3,.7],'alpha_flow_grid_predictions':preds,
                'alpha_flow_grid_true_class_margins':gaps,'correct_grid_runs_not_continuous_intervals':sampled_runs(preds,1)}

    def test_nonmonotonic_scan_and_nearest_sampled_point(self):
        result=scan_stats(self.row())
        self.assertEqual(result['correct_grid_runs'],2)
        self.assertAlmostEqual(result['nearest_correct_sampled_alpha_distance'],.1)
        self.assertAlmostEqual(result['fusion_true_class_probability_margin'],-.7)
        self.assertGreater(result['packet_weight'],.5)

    def test_bad_endpoint_and_equal_point_are_rejected(self):
        for i in (0,10,20):
            r=self.row(); r['alpha_flow_grid_predictions'][i]=(r['alpha_flow_grid_predictions'][i]+1)%8
            with self.assertRaises(ValueError): scan_stats(r)
        r=self.row(); r['alpha_flow_grid_true_class_margins'][0]=.4
        with self.assertRaises(ValueError): scan_stats(r)

    def test_false_grid_interval_and_invalid_weight_are_rejected(self):
        r=self.row(); r['correct_grid_runs_not_continuous_intervals']=[[0,.5]]
        with self.assertRaises(ValueError): scan_stats(r)
        r=self.row(); r['final_weights']=[.6,.7]
        with self.assertRaises(ValueError): scan_stats(r)

    def test_pair_outcomes_do_not_conflate_net_gain_and_repairs(self):
        states=[outcome(False,True)]*8+[outcome(True,False)]*3
        self.assertEqual(states.count('repaired'),8)
        self.assertEqual(states.count('regressed'),3)
        self.assertEqual(outcome(True,True),'both_correct')
        self.assertEqual(outcome(False,False),'both_wrong')

    def test_bidirectional_probability_gaps_are_explicit(self):
        web=[.1,.6,.1,.2,0,0,0,0]; recon=[.1,.2,.1,.6,0,0,0,0]
        values=direction_values({k:web for k in ('flow_only','packet_only','dynamic')}, {k:recon for k in ('flow','packet','fusion')})
        self.assertAlmostEqual(values['E3_fusion_pWeb_minus_pRecon'],.4)
        self.assertAlmostEqual(values['candidate_fusion_pWeb_minus_pRecon'],-.4)

    def test_empty_group_and_probability_validation(self):
        self.assertIsNone(summary([])['median'])
        self.assertEqual(summary([1,3])['median'],2)
        with self.assertRaises(ValueError): prediction([float('nan')]*8)
        with self.assertRaises(ValueError): prediction([.3]*8)
        with self.assertRaises(ValueError): summary([float('inf')])

    def test_py39_syntax_and_no_ml_imports(self):
        path=Path(__file__).with_name('analyze_candidate_fusion_cache.py')
        ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,9))
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('torch_geometric',sys.modules)


if __name__=='__main__':
    unittest.main()
