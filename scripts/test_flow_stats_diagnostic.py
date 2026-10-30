"""No-torch checks of replay interventions, source integrity and reporting."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import diagnose_flow_stats_residual as diagnostic


class Checks(unittest.TestCase):
    def states(self):
        keys = ['flow_encoder.bn.' + k for k in ('running_mean', 'running_var', 'num_batches_tracked')]
        keys += ['packet_encoder.bn.' + k for k in ('running_mean', 'running_var', 'num_batches_tracked')]
        keys += ['flow_stats_mlp.' + k for k in ('0.weight', '0.bias', '2.weight', '2.bias')]
        keys += ['flow_encoder.bn.weight', 'packet_encoder.convs.weight', 'flow_classifier.weight',
                 'packet_classifier.weight', 'shared_head.0.weight', 'joint_gate.0.weight']
        return {k: 'start:' + k for k in keys}, {k: 'last:' + k for k in keys}

    def test_only_registered_state_changes_and_inputs_are_unchanged(self):
        self.assertNotIn('torch', sys.modules)
        start, trained = self.states()
        original = copy.deepcopy((start, trained))
        for case in diagnostic.CASES:
            out = diagnostic.compose_state(start, trained, case)
            for key in out:
                expected = ('start:' if case == 'start_with_last_stats' and not key.startswith('flow_stats_mlp.')
                            or case == 'last_with_start_BN' and diagnostic.is_encoder_bn_buffer(key) else 'last:')
                self.assertEqual(out[key], expected + key)
        self.assertEqual((start, trained), original)

    def test_wrong_structure_or_case_rejected(self):
        start, trained = self.states()
        with self.assertRaises(ValueError):
            diagnostic.compose_state(start, trained, 'search_coefficients')
        with self.assertRaises(ValueError):
            diagnostic.compose_state(start, {k: v for k, v in trained.items() if 'running_var' not in k}, 'last_full')
        plain = {'head.weight': 'same'}
        for case in ('last_with_start_BN', 'start_with_last_stats'):
            with self.subTest(case=case), self.assertRaises(ValueError):
                diagnostic.compose_state(plain, plain, case)

    def fixture(self, directory):
        repo = Path(directory)
        root = repo / 'outputs/completed'
        folder = root / diagnostic.screen.VARIANT
        folder.mkdir(parents=True)
        c, binding = {'fixed_protocol': True}, {'original_entry': 'pinned'}
        m = {'checkpoint_sha256': '', 'epochs': 12}
        artifacts = {}
        for name in ('last.pt', 'best.pt'):
            # Only file-integrity fixtures; not tensor checkpoints.
            (folder / name).write_bytes(('fixture:' + name).encode())
            artifacts[name] = diagnostic.sha(folder / name)
        m['checkpoint_sha256'] = artifacts['best.pt']
        summary = {'status': 'completed', 'exit_code': 0, 'binding': binding, 'protocol': c,
                   'test_evaluated': False, 'tensor_preflight_passed': True, 'candidate': m,
                   'artifacts': artifacts}
        for path, value in [(root / 'status.json', {'status': 'completed', 'exit_code': 0}),
                            (root / 'summary.json', summary), (root / 'config.json', c),
                            (root / 'binding.json', binding), (folder / 'metrics.json', m)]:
            diagnostic.write(path, value)
        meta = (c, None, None, None, None, None, None, binding)
        return SimpleNamespace(repo=repo, training_output=Path('outputs/completed')), root, meta

    def test_source_integrity_rejects_modified_checkpoint_or_binding(self):
        for damage in (None, 'checkpoint', 'binding', 'incomplete'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                args, root, meta = self.fixture(directory)
                if damage == 'checkpoint':
                    (root / diagnostic.screen.VARIANT / 'last.pt').write_bytes(b'changed')
                if damage == 'binding':
                    diagnostic.write(root / 'binding.json', {'different': True})
                if damage == 'incomplete':
                    diagnostic.write(root / 'status.json', {'status': 'running'})
                with patch.object(diagnostic.screen, 'metadata', return_value=meta), \
                     patch.object(diagnostic.screen, 'history_evidence', return_value={}):
                    if damage is None:
                        receipt = diagnostic.source_receipt(args)
                        self.assertFalse(receipt[-1]['source_files_will_be_modified'])
                    else:
                        with self.assertRaises(ValueError):
                            diagnostic.source_receipt(args)

    def test_report_labels_interventions_without_new_training_claim(self):
        row = {'case': 'last_full', 'macro_f1': .9430319351896747, 'errors': 100,
               'mean_residual_to_graph_norm': 2., 'p95_residual_to_graph_norm': 5.}
        with tempfile.TemporaryDirectory() as directory:
            text = diagnostic.report(Path(directory), {'rows': [row], 'selected_epoch': 0, 'last_epoch': 12,
                'frozen_V2_macro_f1': .97093667251942, 'lower_lr_J2_macro_f1': .9712071203419106})
            self.assertIn('94.303194%', text)
            self.assertIn('No training', text)
            self.assertIn('not separately trained ablations', text)


if __name__ == '__main__':
    unittest.main()
