"""Regression coverage for restoring a legacy interrupted Joint-Flow arm."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import resume_flow_query_joint_seeds as resume
import test_flow_query_joint_packet2 as fixtures


class RecoveryChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary = resume.raw.ROOT/'tmp'/'joint_seed_resume_checks'
        self.temporary.mkdir(parents=True, exist_ok=True)

    def test_legacy_interruption_recovers_the_original_training_trajectory(self):
        model, _, _, graphs = fixtures.joint_fixture('flow')
        data = fixtures.TinyGraphs(graphs)
        config = {**fixtures.config(seed=44, epochs=4), 'encoder_lr': 1e-6, 'workers': 0, 'prefix_epochs': 3}
        rows = {part: [{'sample_id': g.sample_id, 'label': int(g.y), 'partition': part} for g in graphs]
                for part in ('train', 'val')}
        device = torch.device('cpu')
        orders = resume.study.epoch_orders(len(rows['train']), config['batch_size'], 44, 4)
        with tempfile.TemporaryDirectory(dir=self.temporary) as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            reference, prior, target = root/'reference', root/'prior'/'seed44', root/'target'
            for folder in (reference, prior, target):
                folder.mkdir(parents=True)
            result, _ = resume.core.fit_joint('Joint-Flow', copy.deepcopy(model), data, rows, device, config, {}, reference, orders)
            evaluate = resume.core.joint_evaluate
            validations = 0

            def interrupt(*args, **kwargs):
                nonlocal validations
                validations += 1
                if validations == 4:
                    raise RuntimeError('simulated interruption in epoch three')
                return evaluate(*args, **kwargs)

            with patch.object(resume.core, 'joint_evaluate', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    resume.core.fit_joint('Joint-Flow', copy.deepcopy(model), data, rows, device, config, {}, prior, orders)
            original = prior/'Joint-Flow'
            prior_hashes = {file.name: resume.raw.sha(file) for file in original.iterdir()}
            recovered, probabilities = resume.recover_arm(prior.parent, target/'Joint-Flow', copy.deepcopy(model),
                                                          data, rows, device, config, {}, orders)
            final = target/'Joint-Flow'
            expected = reference/'Joint-Flow'
            self.assertEqual(resume.core.read_history(final/'epochs.jsonl'), resume.core.read_history(expected/'epochs.jsonl'))
            self.assertEqual(recovered['best_validation'], result['best_validation'])
            self.assertEqual(recovered['best_epoch'], result['best_epoch'])
            self.assertEqual(recovered['recovery']['last_complete_ledger_epoch'], 2)
            self.assertGreater(recovered['recovery']['resumed_from_epoch'], 0)
            self.assertEqual(len(probabilities), len(rows['val']))
            full = torch.load(expected/'last.pt', map_location='cpu', weights_only=False)
            resumed = torch.load(final/'last.pt', map_location='cpu', weights_only=False)
            for key in ('model_state', 'optimizer_state', 'sampler_rng'):
                resume.core.assert_tree_equal(resumed[key], full[key], key)
            self.assertEqual(resume.raw.rng_hash(resumed['rng']), resume.raw.rng_hash(full['rng']))
            self.assertEqual({file.name: resume.raw.sha(file) for file in original.iterdir()}, prior_hashes)
            copied, copied_probabilities = resume.copy_completed(final, root/'reused', config, rows, orders)
            self.assertEqual(copied, recovered)
            self.assertEqual(copied_probabilities, probabilities)
            self.assertEqual(resume.raw.sha(root/'reused'/'last.pt'), resume.raw.sha(final/'last.pt'))

    def test_rolling_checkpoint_and_incomplete_ledger_tail(self):
        with tempfile.TemporaryDirectory(dir=self.temporary) as tmp:
            root = Path(tmp)
            history = [{'epoch': 1, 'validation': {'macro_f1': .7}, 'train_order_sha256': 'one'},
                       {'epoch': 2, 'validation': {'macro_f1': .6}, 'train_order_sha256': 'two'}]
            (root/'epochs.jsonl').write_bytes((''.join(json.dumps(row)+'\n' for row in history)+'{"epoch":').encode())
            snapshots = {'epoch_zero.pt': {'epoch': 0, 'validation': {'macro_f1': .5}},
                         'trained_best.pt': {'epoch': 1, 'validation': history[0]['validation']},
                         'resume.pt': {'epoch': 2, 'validation': history[1]['validation']},
                         'last.pt': {'epoch': 3, 'validation': {'macro_f1': .8}}}
            config = {'epochs': 4, 'prefix_epochs': 3}
            name, prefix, kept, logged = resume.choose_checkpoint(root, config, ['one', 'two', 'three', 'four'], snapshots)
            self.assertEqual((name, prefix, logged), ('resume.pt', history, 2))
            self.assertEqual(kept['trained_best.pt'], 'trained_best.pt')


if __name__ == '__main__':
    unittest.main()
