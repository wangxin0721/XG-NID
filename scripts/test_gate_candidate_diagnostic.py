"""Standard-library checks only: no torch, graph loads or model execution."""
import ast
import copy
import json
from pathlib import Path
import sys
import unittest

from diagnose_gate_candidate import (class_metrics, completed_candidate, finish_groups,
    grid_runs, group_name, groups, prediction_stats)

ROOT = Path(__file__).resolve().parents[1]


class DiagnosticChecks(unittest.TestCase):
    def test_import_has_no_ml_runtime_and_py39_syntax(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('torch_geometric', sys.modules)
        ast.parse((ROOT/'scripts/diagnose_gate_candidate.py').read_text(encoding='utf-8'), feature_version=(3, 9))

    def test_nonlinear_grid_can_have_disconnected_correct_runs(self):
        pred = [1]*21
        for i in (0, 1, 10, 12, 13, 20):
            pred[i] = 2
        self.assertEqual(grid_runs(pred, 2), [[0., .05], [.5, .5], [.6, .65], [1., 1.]])
        self.assertEqual(grid_runs([1]*21, 2), [])
        with self.assertRaises(ValueError):
            grid_runs([1]*20, 2)

    def test_signed_true_class_margin_is_not_top_two_gap(self):
        p = [.55, .01, .4, .04, 0, 0, 0, 0]
        s = prediction_stats(p, [1.,2.,3.,4.,5.,6.,7.,8.], 1)
        self.assertAlmostEqual(s['top1_top2_probability_margin'], .15)
        self.assertAlmostEqual(s['true_class_probability_margin'], -.54)
        shifted = prediction_stats(p, [11.,12.,13.,14.,15.,16.,17.,18.], 1)
        self.assertEqual(s['centered_logit_l2'], shifted['centered_logit_l2'])
        with self.assertRaises(ValueError):
            prediction_stats([float('nan')]*8, [0.]*8, 0)

    def test_groups_include_failure_denominators_and_empty_null(self):
        g = groups()
        for fc,pc in ((True,True),(True,False),(False,True),(False,False)):
            self.assertIn(group_name(fc,pc), g)
        self.assertIsNone(finish_groups(g)['both_wrong']['within_group_fusion_accuracy'])
        g['packet_only_correct'] = {'samples':196,'fusion_correct':140,'fusion_wrong':56}
        result = finish_groups(g)['packet_only_correct']
        self.assertAlmostEqual(result['within_group_fusion_accuracy'],140/196)
        self.assertEqual(result['fraction_of_population'],1.)

    def test_per_class_precision_includes_incoming_errors(self):
        m = [[0]*8 for _ in range(8)]
        m[0][0],m[0][1],m[1][0],m[1][1] = 8,2,4,6
        c = class_metrics(m, list('abcdefgh'))['a']
        self.assertEqual(c['false_positive'],4)
        self.assertEqual(c['false_negative'],2)
        self.assertAlmostEqual(c['precision'],8/12)
        self.assertAlmostEqual(c['recall'],.8)
        self.assertAlmostEqual(c['f1'],16/22)

    def test_refuses_running_wrong_variant_and_test_evaluation(self):
        metrics = {'variant':'E3a','mode':'screen','test':None,'test_evaluated':False,'initialization':{},'evaluation_batch_size':16}
        cfg = {'variant':'E3a','mode':'screen','binding':{},'initialization':{}}
        status = {'status':'completed','exit_code':0,'mode':'screen','variant':'E3a'}
        summary = {**status,'model':metrics,'binding':{},'test_evaluated':False}
        completed_candidate(summary,status,cfg,metrics,'E3a')
        for field,value in (('status','running'),('variant','E3a_equal_init'),('test_evaluated',True)):
            changed = copy.deepcopy(summary)
            changed[field] = value
            with self.assertRaises(ValueError):
                completed_candidate(changed,status,cfg,metrics,'E3a')

    def test_registered_comparison_has_explicit_loss_and_no_time_budget(self):
        d = json.loads((ROOT/'configs/fusion_acb_study_20261023.json').read_text(encoding='utf-8'))
        self.assertEqual(d['loss']['lambda'],.05)
        self.assertEqual(d['loss']['auxiliary_supervision_target'],'H(z), not z')
        self.assertFalse(d['gpu_hour_budget_enforced'])
        self.assertFalse(d['architecture']['post_gate_proxy_mix'])
        self.assertEqual(d['fusion_variants']['C_global']['s_initial'],0.)


if __name__ == '__main__':
    unittest.main()
