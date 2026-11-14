"""Check seed isolation, paired budgets, and best/last prediction receipts."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

import run_packet_transition_paired_seeds as run
from test_flow_query_joint_packet2 import TinyGraphs
from test_packet_transition_readout import fixture


class PairedSeedChecks(unittest.TestCase):
    def test_frozen_backbone_with_two_readout_seeds(self):
        torch.set_num_threads(2)
        model, graphs = fixture()
        data = TinyGraphs(graphs)
        rows = {part: [{'sample_id': g.sample_id, 'label': g.y.item(), 'partition': part} for g in graphs]
                for part in ('train', 'val')}
        data.rows = rows['train']
        config = {'seed': 42, 'epochs': 2, 'batch_size': 4, 'eval_batch_size': 4, 'workers': 0,
                  'learning_rate': 0.001, 'training_samples': len(graphs), 'budget_reference_pp': 0.10}
        device = torch.device('cpu')
        frozen_sha = run.raw.state_digest(model.state_dict())
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            output = Path(tmp)
            heads = {seed: run.original.make_heads({**config, 'readout_seed': seed}, device)
                     for seed in (42, 43, 44)}
            hashes = {seed: run.initial_hashes(pair)['B'] for seed, pair in heads.items()}
            self.assertEqual(len(set(hashes.values())), 3)
            val, _, baseline = run.original.cache_split(model, data, rows['val'], 'val', config, device,
                                                        output, heads[43],
                                                        observe=lambda batch: run.check_zero(model, heads[44], batch))
            train, _, _ = run.original.cache_split(model, data, rows['train'], 'train', config, device, output)
            baseline_selection = {'epoch': 48,
                                  'validation': run.joint.study.validate_predictions(baseline, rows['val'],
                                                                                        probabilities=False),
                                  'prediction_ids_sha256': run.joint.prediction_digest(baseline)}
            receipt = run.evaluation.replay_receipt(baseline_selection['validation'], baseline, baseline_selection,
                                                    baseline)
            results = {}
            for seed in (42, 43, 44):
                folder = output / f'seed{seed}'
                folder.mkdir()
                results[seed] = run.run_seed(folder, model, data, rows, device, heads[seed],
                                             {**config, 'readout_seed': seed}, train, val, baseline,
                                             baseline_selection, receipt, val.features.lengths.tolist(), 2, frozen_sha)
                sampler = run.raw.TrainBatches(len(graphs), config['batch_size'], seed)
                orders = [row['order_sha256'] for row in
                          map(json.loads, (folder / 'epochs.jsonl').read_text().splitlines())][1:]
                for digest in orders:
                    list(sampler)
                    self.assertEqual(sampler.order_sha256, digest)
                self.assertEqual(results[seed]['updates_per_arm'], 4)
                self.assertEqual(results[seed]['initial_state_sha256'], dict.fromkeys(run.ARMS, hashes[seed]))
                self.assertTrue((folder / 'last/validation_predictions.jsonl.gz').is_file())
            run.write_summary(output, results, config)
            stats = run.raw.read(output / 'summary.json')['aggregate']
            for phase in run.PHASES:
                self.assertEqual(set(stats[phase]['all']['by_seed_pp']), {'42', '43', '44'})
                self.assertEqual(set(stats[phase]['new']['by_seed_pp']), {'43', '44'})
        self.assertEqual(run.raw.state_digest(model.state_dict()), frozen_sha)


if __name__ == '__main__':
    unittest.main()
