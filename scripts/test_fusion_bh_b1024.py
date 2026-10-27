"""Batch1024 registration checks; no tensor library or real model execution."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fusion_bh import CONTROLS
from fusion_bh_b1024_metadata import CONTROLS as LARGE
import analyze_fusion_bh_b1024_results as analysis

ROOT = Path(__file__).resolve().parent.parent

class Checks(unittest.TestCase):
    def test_only_batch_control_changes(self):
        config = json.loads((ROOT/'configs/fusion_bh_b1024_study_20261027.json').read_text(encoding='utf-8'))
        self.assertEqual(config['controls'], {**CONTROLS,'batch_size':1024})
        self.assertEqual(LARGE, config['controls'])

    def test_both_samplers_and_recorded_batch(self):
        tree = ast.parse((ROOT/'scripts/run_fusion_bh_b1024_server.py').read_text(encoding='utf-8'))
        samplers = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='TrainBatches']
        self.assertEqual(len(samplers),2)
        self.assertEqual([n.args[1].value for n in samplers],[1024,1024])
        self.assertFalse(any(isinstance(n,ast.Constant) and n.value in (512,211) for n in ast.walk(tree)))
        loaders = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='DataLoader']
        evaluation = [k.value.value for n in loaders for k in n.keywords if k.arg=='batch_size']
        self.assertEqual(evaluation,[16])

    def test_distinct_output_and_entry(self):
        script=(ROOT/'scripts/launch_fusion_bh_b1024.sh').read_text(encoding='utf-8')
        self.assertIn('outputs/fusion_bh_b1024_s42_v1',script)
        self.assertIn('run_fusion_bh_b1024_server.py',script)
        self.assertIn('gpu=${1:-1}',script)

    def test_report_does_not_claim_controlled_gate_gain(self):
        report={'metrics':{'metrics':{'fusion':{'macro_f1':.99}}},
                'decision':{'action':'freeze_candidate_for_paired_seed43_44_confirmation'}}
        with tempfile.TemporaryDirectory() as folder, patch.object(analysis,'analyze_base',return_value=report):
            result=analysis.analyze(ROOT,Path(folder),{}, {})
            self.assertEqual(result['decision']['action'],'review_exploratory_batch1024_result_with_batch512_BH')
            self.assertFalse(result['decision']['automatic_additional_training'])
            self.assertEqual(result['comparison_boundary']['candidate_updates_per_epoch'],106)

if __name__=='__main__':
    unittest.main()
