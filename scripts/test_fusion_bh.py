"""Stdlib-only structural and decision checks; real gradients checked on server."""
import ast
import copy
import json
from pathlib import Path
import unittest
from fusion_bh import prediction_statistics_ast, STATEMENTS, decide, validate_probe, CONTROLS
from e3_gate_study import pure_gate_ast

ROOT = Path(__file__).resolve().parent.parent

class Checks(unittest.TestCase):
    def source(self):
        tree = ast.parse((ROOT/'src/xgnid/model.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DualBranchLogitGatedHeteroGNN')
        return ast.unparse(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_graph_logits'))

    def test_exact_four_statistic_intervention(self):
        source = self.source()
        old, new = pure_gate_ast(source), prediction_statistics_ast(source)
        compile(new,'<static-check>','exec')
        # Reverse precisely the intended change; everything else must match.
        fn = new.body[0]
        del fn.body[1:3]
        for n in fn.body:
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in STATEMENTS:
                n.value.args[0].id=STATEMENTS[n.targets[0].id][1]
            if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='self._last_cache':
                del n.value.keys[-2:]
                del n.value.values[-2:]
        self.assertEqual(ast.dump(old),ast.dump(new))

    def test_fail_closed_on_changed_stats(self):
        with self.assertRaises(ValueError):
            prediction_statistics_ast(self.source().replace('_confidence_from_logits(flow_logits)','_margin_from_logits(flow_logits)'))

    def test_decision_boundaries(self):
        b={'B_dynamic':.9621918682153585,'C_global':.9649409450822386,'E2':.9659940017251138,'E0':.9667287438289962}
        self.assertEqual(decide(b['C_global'],b)['action'],'stop_not_above_C')
        self.assertEqual(decide(b['E2'],b)['action'],'retain_mechanism_clue_no_automatic_expansion')
        self.assertEqual(decide(b['E2']+1e-10,b)['action'],'freeze_candidate_for_paired_seed43_44_confirmation')
        self.assertFalse(decide(.99,b)['automatic_additional_training'])

    def test_invalid_probe_rejected(self):
        with self.assertRaises(ValueError):
            validate_probe({'status':'failed'}, {}, {})

    def test_config_and_legacy_loss(self):
        config=json.loads((ROOT/'configs/fusion_bh_study_20261027.json').read_text(encoding='utf-8'))
        self.assertEqual(config['controls'],CONTROLS)
        self.assertEqual(config['variant'],'B_Hstats')
        import fusion_bh, fusion_acb
        self.assertIs(fusion_bh.losses,fusion_acb.losses)

if __name__=='__main__':
    unittest.main()
