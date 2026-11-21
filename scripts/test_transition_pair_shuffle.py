"""CPU checks of explicit pair changes, graph isolation, fixed mappings and real resume."""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch_geometric.data import Batch, HeteroData

import run_transition_pair_shuffle as run
import launch_joint_transition_fusion as launcher
import test_joint_transition_fusion as previous_checks
from test_packet_transition_readout import fixture
from test_flow_query_joint_packet2 import TinyGraphs
from transition_pair_cache import PairCache, mapping
from xgnid.transition_pair_shuffle import derangement
from xgnid.model import FLOW_TO_PACKET, PACKET_TO_FLOW, PACKET_TO_PACKET


def graph_fixture():
    source, _ = fixture()
    generator = torch.Generator().manual_seed(107)
    graphs = []
    for label, count in enumerate((1, 2, 3, 4, 6, 5, 7, 4)):
        graph = HeteroData()
        graph['flow'].x = torch.randn(1, 82, generator=generator)
        graph['packet'].x = torch.randn(count, 1500, generator=generator)
        indices = torch.stack((torch.zeros(count, dtype=torch.long), torch.arange(count)))
        attrs = torch.randn(count, 4, generator=generator)
        attrs[:, 0], attrs[:, 3] = torch.arange(count) % 2, torch.arange(count) * 10
        graph[FLOW_TO_PACKET].edge_index, graph[FLOW_TO_PACKET].edge_attr = indices, attrs
        graph[PACKET_TO_FLOW].edge_index, graph[PACKET_TO_FLOW].edge_attr = indices.flip(0), attrs
        pairs = [(a, b) for a in range(count) for b in range(count) if a != b and
                 not (label == 7 and (a, b) == (1, 2))]
        links = torch.tensor(pairs, dtype=torch.long).t().contiguous() if pairs else torch.empty(2, 0, dtype=torch.long)
        graph[PACKET_TO_PACKET].edge_index = links
        graph[PACKET_TO_PACKET].edge_attr = torch.full((len(pairs), 1), float(count))
        graph.y, graph.sample_id = torch.tensor([label]), 'pair-example-' + str(label)
        graphs.append(graph)
    return source, graphs


class PairShuffleChecks(unittest.TestCase):
    assert_tree_equal = previous_checks.JointFusionChecks.assert_tree_equal

    def setUp(self):
        torch.set_num_threads(2)
        self.device = torch.device('cpu')
        self.settings = {**run.raw.read(run.CONFIG), 'epochs': 3, 'batch_size': 3, 'eval_batch_size': 4,
                         'training_samples': 8, 'validation_samples': 8, 'test_samples': 8}
        self.source, self.graphs = graph_fixture()
        run.frozen.independent.freeze_c(self.source)
        self.rows = [{'sample_id': g.sample_id, 'label': g.y.item(), 'partition': 'val'} for g in self.graphs]
        self.features, _ = run.frozen.encode(self.source, Batch.from_data_list(self.graphs), self.device)
        self.base = run.original.FrozenCache([self.features], [r['label'] for r in self.rows])
        self.cache = PairCache(self.base, self.rows, 'val', self.settings['pairing_seed'])
        self.reference = run.original.prediction_records(
            self.rows, run.original.baseline(self.source, self.features).argmax(-1).tolist())
        self.binding = {'source': run.raw.state_digest(self.source.state_dict()),
                        'mapping_sha256': self.cache.mapping_sha256}

    def test_derangement_is_fixed_label_free_and_does_not_consume_rng(self):
        before = run.training.rng_state()
        sampler = run.training.PairedSampler(8, 3, 42)
        generator = sampler.generator.get_state().clone()
        for count in range(21):
            for sample in range(12):
                a = derangement(str(sample), 'train', count, self.settings['pairing_seed'])
                self.assertEqual(a, derangement(str(sample), 'train', count, self.settings['pairing_seed']))
                self.assertEqual(sorted(a), list(range(count)))
                self.assertTrue(count < 2 or all(i != value for i, value in enumerate(a)))
        changed_labels = [{**r, 'label': (r['label'] + 3) % 8} for r in self.rows]
        other = PairCache(self.base, changed_labels, 'val', self.settings['pairing_seed'])
        self.assertEqual(other.mapping_sha256, self.cache.mapping_sha256)
        self.assert_tree_equal(before, run.training.rng_state())
        self.assertTrue(torch.equal(generator, sampler.generator.get_state()))
        self.assertTrue(any(derangement(str(i), 'train', 8, 20261121) !=
                            derangement(str(i), 'val', 8, 20261121) for i in range(12)))

    def test_candidate_multisets_counts_flow_isolation_and_actual_feature_ratio(self):
        self.assertEqual([r['ordered_contexts'] for r in self.cache.records], [0, 0, 1, 2, 4, 3, 5, 0])
        self.assertEqual(self.cache.records[-1]['valid_pairs'], 2)  # Two links with a missing middle do not form a context.
        self.assertEqual(self.cache.coverage['shuffleable_flows'], 4)
        self.assertEqual(self.cache.coverage['changed_pair_indices'], 14)
        for row in self.cache.records:
            self.assertEqual(sorted(row['true_right_local']), sorted(row['shuffled_right_local']))
            self.assertEqual(len(row['left_local']), len(row['shuffled_right_local']))
            self.assertTrue(all(0 <= i < row['valid_pairs'] for i in row['shuffled_right_local']))
            if row['shuffleable']:
                self.assertTrue(all(a != b for a, b in zip(row['true_right_local'], row['shuffled_right_local'])))
            else:
                self.assertEqual(row['true_right_local'], row['shuffled_right_local'])
        # Different indices may hold identical vectors: report actual feature changes separately.
        duplicate = self.features._replace(transitions=self.features.transitions.clone())
        duplicate.transitions[:, :70] = 1
        base = run.original.FrozenCache([duplicate], self.base.labels.tolist())
        cache = PairCache(base, self.rows, 'val', self.settings['pairing_seed'])
        self.assertGreater(cache.coverage['changed_pair_indices'], 0)
        self.assertEqual(cache.coverage['changed_right_features'], 0)
        self.assertEqual(cache.coverage['feature_change_fraction_of_changed_indices'], 0)

    def test_mapping_is_invariant_to_batch_order_and_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'frozen.pt'
            run.training.atomic_save(path, {'features': self.features._asdict(), 'labels': self.base.labels})
            saved = torch.load(path, weights_only=True)
            features = run.original.Features(*(saved['features'][k] for k in run.original.Features._fields))
            other = PairCache(run.original.FrozenCache([features], saved['labels'].tolist()), self.rows,
                              'val', self.settings['pairing_seed'])
            self.assertEqual(other.mapping_sha256, self.cache.mapping_sha256)
            for indices in ([6, 3, 4], [3], [6, 6, 3], [7, 2, 1, 0]):
                packed, _ = other.batch(indices, self.device)
                direct, rows = mapping(run.original.Features(*packed[:5]),
                    [self.rows[i]['sample_id'] for i in indices], 'val', self.settings['pairing_seed'])
                self.assertTrue(torch.equal(packed.shuffled_right, direct.shuffled_right))
                self.assertEqual(rows, [self.cache.records[i] for i in indices])

    def test_nonzero_readout_changes_pairs_preserves_fallback_and_matches_explicit_formula(self):
        features, _ = self.cache.batch(range(8), self.device)
        proof = run.structure_probe(self.source, features, self.settings, self.device)
        self.assertTrue(proof['passed'] and proof['true_exactly_matches_J3'])
        self.assertGreater(proof['max_abs_residual_difference'], 0)
        arms = run.make_arms(self.source, self.settings, self.device)
        with torch.no_grad():
            for arm in arms.values():
                arm.readout.output.weight.fill_(0.1)
        true = arms['pair_true'](features)[1]
        shuffled = arms['pair_shuffle'](features)[1]
        fallback = [i for i, r in enumerate(self.cache.records) if not r['shuffleable']]
        self.assertTrue(torch.equal(true[fallback], shuffled[fallback]))
        head = arms['pair_shuffle'].readout
        values = F.gelu(head.edge(features.transitions[:, :70]))
        queries = head.query(features.packet_mean)
        pooled, offset, context_offset = [], 0, 0
        for i, length in enumerate(features.lengths.tolist()):
            local = values[offset:offset + length]
            first = (local * ((local * queries[i]).sum(-1) / 4).softmax(0)[:, None]).sum(0)
            left = torch.where(features.transitions[offset:offset + length, 70].bool())[0] + offset
            if len(left):
                right = features.shuffled_right[context_offset:context_offset + len(left)]
                context = F.gelu(head.context(torch.cat((values[left], values[right]), -1)))
                second = (context * ((context * queries[i]).sum(-1) / 4).softmax(0)[:, None]).sum(0)
                first = (first + second) / 2
            pooled.append(first)
            offset += length; context_offset += len(left)
        torch.testing.assert_close(shuffled, head.output(torch.stack(pooled)), atol=2e-6, rtol=1e-5)
        # Changing a different flow's transition features cannot affect this flow's result.
        changed = features._replace(transitions=features.transitions.clone())
        start = int(features.lengths[:6].sum()); stop = start + int(features.lengths[6])
        changed.transitions[start:stop, :70] += 20
        actual = arms['pair_shuffle'](changed)[1]
        self.assertTrue(torch.equal(shuffled[:6], actual[:6]))
        self.assertTrue(torch.equal(shuffled[7:], actual[7:]))
        bad = features.shuffled_right.clone()
        owner = torch.repeat_interleave(torch.arange(8), features.lengths)
        first = torch.where(owner[bad] == 3)[0][0]; second = torch.where(owner[bad] == 4)[0][0]
        bad[first], bad[second] = bad[second].clone(), bad[first].clone()
        with self.assertRaisesRegex(ValueError, 'flow boundaries'):
            arms['pair_shuffle'](features._replace(shuffled_right=bad))

    def test_both_arms_zero_initial_gradients_and_frozen_encoders(self):
        features, labels = self.cache.batch(range(8), self.device)
        before = run.raw.state_digest(self.source.state_dict())
        arms = run.make_arms(self.source, self.settings, self.device)
        for arm in arms.values():
            scores, residual, _, _ = arm(features)
            self.assertTrue(torch.equal(scores, run.original.baseline(self.source, features)))
            self.assertFalse(bool(residual.any()))
        result = run.training.gradient_probe(self.source, features, labels, self.settings, self.device, arms=arms)
        self.assertEqual(set(result['steps']), set(run.ARMS))
        run.training.verify_source(self.source, before)

    def test_actual_paired_adam_resume_matches_continuous_training_and_rejects_new_mapping(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            def fit(folder, resume=False, stop=None, settings=None):
                return run.training.fit(run.make_arms(self.source, self.settings, self.device), self.cache, self.cache,
                    self.rows, self.reference, settings or self.settings, self.binding, self.device, folder,
                    resume=resume, stop_after_updates=stop)
            run.raw.seed_all(42)
            expected = fit(root / 'continuous')
            for stop in (1, 3, 5):
                folder = root / str(stop)
                run.raw.seed_all(42)
                fit(folder, stop=stop)
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    fit(folder, resume=True, settings={**self.settings, 'pairing_seed': 7})
                torch.rand(10)
                actual = fit(folder, resume=True)
                self.assert_tree_equal(expected, actual)
                for arm in run.ARMS:
                    self.assertTrue(all(v['step'].item() == 9 for v in actual['optimizers'][arm]['state'].values()))
                    ledger = [json.loads(line) for line in (folder / arm / 'epochs.jsonl').read_text().splitlines()]
                    self.assertEqual([r['epoch'] for r in ledger], [0, 1, 2, 3])
                    self.assertEqual(ledger[-1]['cumulative_updates_per_arm'], 9)
            self.assertEqual(expected['updates'], 9)

    def test_complete_probe_training_raw_replay_strata_and_completed_resume(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            repo = Path(temp); output = repo / run.OUTPUT; output.mkdir(parents=True)
            data = TinyGraphs(self.graphs); data.root = repo / 'graphs'; data.root.mkdir()
            rows = {'val': self.rows, 'train': [{**r, 'partition': 'train'} for r in self.rows]}
            data.rows = rows['train'] + rows['val']
            data.metadata = {'partition_sizes': {'train': 8, 'val': 8, 'test': 8}}
            config = {'seed': 42, 'workers': 0, 'batch_size': 3, 'eval_batch_size': 4}
            metric = run.frozen.joint.study.validate_predictions(self.reference, self.rows, False)
            selected = {'epoch': 48, 'validation': metric,
                        'prediction_ids_sha256': run.frozen.joint.prediction_digest(self.reference)}
            history = {'provenance': {'binding': {'source': self.binding}},
                       'summary': {'metrics': {'joint_ordered': metric}, 'selected_epochs': {'joint_ordered': 84}}}
            gold = [{**r, 'flow_weights': [0.5] * 32} for r in self.reference]
            source_result = self.source, data, rows, self.device, config, selected, gold, {}, self.binding
            with patch.object(run, 'validate_settings'), \
                 patch.object(run, 'history_inputs', return_value=(history, self.reference, {})), \
                 patch.object(run.frozen, 'load_source', return_value=source_result), \
                 patch.object(run.torch.cuda, 'get_device_name', return_value='synthetic CPU fixture'):
                probe = run.run(repo, None, output, self.settings, probe_only=True)
                self.assertFalse(probe['formal_training_started'])
                self.assertFalse((output / 'recovery.pt').exists())
                trained = run.run(repo, None, output, self.settings, resume=True)
                self.assertEqual(trained['status'], 'completed')
                state = torch.load(output / 'recovery.pt', weights_only=True)
                run.run(repo, None, output, self.settings, resume=True)
                self.assert_tree_equal(state, torch.load(output / 'recovery.pt', weights_only=True))
            summary = run.raw.read(output / 'summary.json')
            self.assertFalse(summary['test_evaluated'])
            self.assertEqual(summary['strata']['shuffleable']['samples'], 4)
            self.assertEqual(summary['strata']['unshuffleable']['samples'], 4)
            for group in summary['strata'].values():
                self.assertEqual(len(group['metrics']['pair_true']['per_class']), 8)
            self.assertTrue(run.raw.read(output / 'selected_validation_replay.json')['pair_shuffle']['fixed_mapping_replayed'])
            self.assertTrue((output / 'pairing_train.jsonl.gz').is_file())
            self.assertTrue((output / 'pairing_val.jsonl.gz').is_file())
            self.assertEqual(run.raw.read(output / 'final_verification.json')['test_graph_loads'], 0)

    def test_new_launcher_routes_failures_and_uploads_only_new_output(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            repo = Path(temp)
            process = type('Child', (), {'wait': lambda self: 7, 'poll': lambda self: 7, 'pid': 123})()
            args = argparse.Namespace(repo=repo, data_root=None, probe_only=False)
            with patch.object(launcher.subprocess, 'Popen', return_value=process) as start, \
                 patch.object(launcher, 'publish_results', return_value=[0, 0]) as publish:
                result = launcher.launch(args, output_name=run.OUTPUT_NAME, runner='run_transition_pair_shuffle.py')
            self.assertEqual(result, 7)
            self.assertIn('run_transition_pair_shuffle.py', start.call_args.args[0][3])
            self.assertEqual(publish.call_args.kwargs['output_name'], run.OUTPUT_NAME)
            output = repo / run.OUTPUT
            self.assertEqual(run.raw.read(output / 'training_exit.json')['exit_code'], 7)
            def fake_publish(command, **kwargs):
                self.assertEqual(command[-2:], ['--only', run.OUTPUT_NAME])
                kwargs['stdout'].write(b'new paired output upload\n')
                return type('Result', (), {'returncode': 0})()
            with patch.object(launcher.subprocess, 'run', side_effect=fake_publish):
                self.assertEqual(launcher.publish_results(run.raw.ROOT, repo, output, 7, {},
                                                        output_name=run.OUTPUT_NAME), [0, 0])


if __name__ == '__main__':
    unittest.main()
