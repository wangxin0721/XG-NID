"""Check checkpoint inheritance, isolated readout seeds, and the fixed-test gate."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

import run_independent_packet_transition as run
from test_packet_transition_readout import fixture


class IndependentTransitionChecks(unittest.TestCase):
    def test_child_stages_preserve_epoch_zero_and_optimizer_groups(self):
        torch.set_num_threads(2)
        _, graphs = fixture()
        batch = run.core.Batch.from_data_list(graphs)
        device = torch.device('cpu')
        n0 = run.d32.baseline.build_experiment('E2', batch)
        run.core.initialize(n0, 'E2', batch)
        parent = run.d32.model_from_stage('D32-Gate', n0, device, 43)
        for stage, params in (('CL-Last', 15400), ('JCL-Multi', 32616), ('C', 33640)):
            metadata = {'directory': Path('parent'), 'epoch': 1,
                        'validation': {'macro_f1': 0.9}, 'checkpoint_sha256': 'ckpt',
                        'predictions_sha256': 'pred'}
            config = run.protocol(stage, 43, metadata, 'split')
            child = run.build_child(stage, parent.state_dict(), n0, device, 43)
            with torch.no_grad():
                torch.testing.assert_close(child(batch), parent(batch), atol=1e-5, rtol=1e-5)
            opt = (run.cross.optimizer(run.core, child, config) if stage == 'CL-Last' else
                   (run.legacy.optimizer if stage == 'JCL-Multi' else run.joint.optimizer)(child, config))
            self.assertEqual(sum(p.numel() for group in opt.param_groups for p in group['params']), params)
            self.assertEqual(config['seed'], 43)
            parent = child
        settings = run.raw.read(run.CONFIG)
        heads = [run.transition.make_heads({'seed': seed, 'readout_seed': 42}, device, ('B',))['B']
                 for seed in settings['seeds']]
        self.assertEqual([run.model_state(head) for head in heads], [settings['readout_initial_state_sha256']] * 2)

    def test_fixed_test_loading_waits_for_both_pairs(self):
        settings = run.raw.read(run.CONFIG)
        with TemporaryDirectory() as temp:
            output = Path(temp)
            with patch.object(run.d32.baseline, 'runtime', return_value=torch.device('cpu')), \
                 patch.object(run, 'load_data', return_value=({'split_directory': 'split', 'split_json_sha256': 'x'},
                     type('Data', (), {'metadata': {'sample_manifest': 'manifest', 'sample_manifest_sha256': 'y'}})(),
                     {'val': []})), \
                 patch.object(run, 'load_locked_pair', side_effect=[(None, None, {}, {}),
                                                                    ValueError('Seed44 incomplete')]) as loads, \
                 patch.object(run.evaluation, 'TestGraphs') as test_graphs:
                with self.assertRaisesRegex(ValueError, 'Seed44 incomplete'):
                    run.test_pairs(output, output, None, settings, 'commit')
                self.assertEqual([call.args[1] for call in loads.call_args_list], [43, 44])
                test_graphs.assert_not_called()


if __name__ == '__main__':
    unittest.main()
