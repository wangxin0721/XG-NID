"""Verify fixed learned fusion, six streams, strict native anchors and net-FP accounting."""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Batch

import run_transition_pair_cross_replay as run
import transition_pair_cross_report as report
import launch_transition_pair_cross_replay as launcher
import sync_server_outputs as sync
from test_transition_pair_shuffle import graph_fixture
from transition_pair_cache import PairCache
from xgnid.transition_pair_shuffle import PairFusion


class CrossReplayChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        source, graphs = graph_fixture()
        self.rows = [{'sample_id': g.sample_id, 'label': g.y.item(), 'partition': 'val'} for g in graphs]
        features, _ = run.frozen.encode(source, Batch.from_data_list(graphs), torch.device('cpu'))
        features = features._replace(transitions=features.transitions.clone())
        offset = int(features.lengths[:4].sum()); length = int(features.lengths[4])
        features.transitions[offset:offset + length, :70] = features.transitions[offset, :70]
        self.cache = PairCache(run.original.FrozenCache([features], [r['label'] for r in self.rows]),
                               self.rows, 'val', 20261121)
        self.settings = {**run.raw.read(run.CONFIG), 'validation_samples': 8, 'eval_batch_size': 4}
        self.checkpoints = {}
        for i, name in enumerate(('pair_true', 'pair_shuffle')):
            model = PairFusion(source.fusion, name, 42)
            with torch.no_grad():
                model.readout.output.weight.fill_(0.1 + i * 0.03)
                model.fusion.gate[-1].weight.fill_(0.02)
                model.fusion.shared_head[-1].bias[1] += 0.4 + i * 0.3
            self.checkpoints[name] = {'state': {k: v.clone() for k, v in model.state_dict().items()},
                                      'epoch': 191 if name == 'pair_true' else 183}
        self.models, self.digests = run.restore_models(self.checkpoints, self.settings, torch.device('cpu'))

    def get_records(self, output):
        with redirect_stdout(io.StringIO()):
            return run.replay(self.models, self.cache, self.rows, self.settings, torch.device('cpu'), output)

    def anchors(self, records):
        exported = run.exports(records)
        refs = {}
        for group, model in self.settings['native_archive_groups'].items():
            refs[model] = [{**r, 'flow_weights': [0.5] * 32} for r in exported[group]]
            self.checkpoints[model].update(
                validation=run.frozen.joint.study.validate_predictions(exported[group], self.rows, False),
                prediction_ids_sha256=run.frozen.joint.prediction_digest(exported[group]))
        return refs

    def test_off_uses_each_trained_checkpoint_fusion_without_residual(self):
        features, _ = self.cache.batch(range(8), torch.device('cpu'))
        result = run.batch_outputs(self.models, features)
        for off, name in (('T0', 'pair_true'), ('S0', 'pair_shuffle')):
            expected, gate = self.models[name].fusion(features.flow, features.packet)
            self.assertTrue(torch.equal(result[off]['scores'], expected))
            self.assertTrue(torch.equal(result[off]['gate'], gate))
            self.assertFalse(bool(result[off]['residual'].any()))
            self.assertFalse(bool(result[off]['injected_residual_l2'].any()))
            self.assertFalse(bool(result[off]['gate_delta'].any()))
            self.assertGreater(float(result[off]['raw_residual_l2'].sum()), 0)
        self.assertFalse(torch.equal(result['T0']['scores'], result['S0']['scores']))
        run.verify_frozen(self.models, self.digests)

    def test_six_streams_and_exact_unchanged_feature_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            records, receipt = self.get_records(Path(temp))
            self.assertEqual(len(records), 8)
            self.assertEqual(set(records[0]['predictions']), set(run.GROUPS))
            self.assertEqual(set(records[0]['diagnostics']), set(run.GROUPS))
            self.assertTrue(all(r['passed'] for r in receipt.values()))
            # Includes an index-deranged flow whose vectors are identical, not just fallbacks.
            self.assertGreater(self.cache.records[4]['indices_changed'], 0)
            self.assertEqual(self.cache.records[4]['features_changed'], 0)
            self.assertFalse(records[4]['actual_features_changed'])
            self.assertTrue(any(r['actual_features_changed'] for r in records))
            for r in records:
                for g in run.GROUPS:
                    self.assertEqual(len(r['diagnostics'][g]['flow_gate_weights']), 32)
                    self.assertEqual(len(r['diagnostics'][g]['flow_gate_delta_from_residual_off']), 32)
            run.verify_frozen(self.models, self.digests)

    def test_native_reproduction_passes_metadata_and_rejects_actual_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            records, _ = self.get_records(output)
            refs = self.anchors(records)
            receipt = run.verify_native(records, refs, self.checkpoints, self.settings, output)
            self.assertTrue(all(r['passed'] for r in receipt.values()))
            refs['pair_true'][0]['prediction'] = (refs['pair_true'][0]['prediction'] + 1) % 8
            with self.assertRaisesRegex(ValueError, 'do not reproduce archive'):
                run.verify_native(records, refs, self.checkpoints, self.settings, output)
            self.assertFalse(run.raw.read(output / 'native_archive_replay.json')['T1']['passed'])
            with self.assertRaisesRegex(ValueError, 'mandatory'):
                report.write_results(output, records, self.settings, {'T1': {'passed': False}}, {})
            self.assertFalse((output / 'summary.json').exists())

    def test_net_nine_web_false_positives_is_sixteen_added_minus_seven_removed(self):
        records = []
        for kind, count in (('shared', 31), ('added', 16), ('removed', 7)):
            for i in range(count):
                before = 1 if kind in ('shared', 'removed') else 3
                after = 1 if kind in ('shared', 'added') else 3
                records.append({'sample_id': kind + str(i), 'label': 3, 'diagnostics': {},
                    'predictions': {'T1': after, 'T2': before, 'T0': 3, 'S1': after, 'S2': before, 'S0': 3}})
        stats, changed, retained = report.fp_inventory(records, 'T1', 'S2')
        self.assertEqual((stats['added'], stats['removed'], stats['retained'], stats['net_increase']), (16, 7, 31, 9))
        self.assertEqual(len(changed), 23)
        self.assertEqual(report.web_stats(records, 'T1')['FP'], 47)
        self.assertEqual(report.web_stats(records, 'S2')['FP'], 38)
        self.assertEqual(len(retained), 31)

    def test_report_exports_fixed_comparisons_and_source_decomposition(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            records, consistency = self.get_records(output)
            refs = self.anchors(records)
            receipts = run.verify_native(records, refs, self.checkpoints, self.settings, output)
            gap = report.web_stats(records, 'T1')['FP'] - report.web_stats(records, 'S2')['FP']
            settings = {**self.settings, 'expected_native_webbased_fp_net_increase': gap}
            result = report.write_results(output, records, settings, receipts, consistency)
            self.assertEqual(result['training_epochs'], 0)
            self.assertEqual(result['optimizer_updates'], 0)
            self.assertFalse(result['previous_training_ablation']['conclusion_changed_by_replay'])
            self.assertEqual(set(result['metrics']), set(run.GROUPS))
            self.assertEqual(len(result['comparisons']), 6)
            for row in result['webbased']['net_gap_decomposition_by_source_class'].values():
                self.assertEqual(row['native_T1_minus_S2'], row['fusion_only_T0_minus_S0'] +
                                 row['True_residual_net_T1_minus_T0'] - row['Shuffle_residual_net_S2_minus_S0'])
            self.assertTrue((output / 'sample_diagnostics.jsonl.gz').is_file())
            self.assertTrue((output / 'report.md').is_file())

    def test_archive_input_sha_validation_and_tensor_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp); folder = repo / 'archive'; folder.mkdir()
            buffer = io.BytesIO(); torch.save({'weights': torch.ones(3)}, buffer)
            data = buffer.getvalue()
            (folder / 'weights.pt').write_bytes(data)
            settings = {'source_directory': 'archive', 'inputs': {
                'weights.pt': {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}}}
            archive = run.Archive(repo, settings)
            archive.verify_all()
            self.assertTrue(torch.equal(archive.tensor('weights.pt')['weights'], torch.ones(3)))
            (folder / 'weights.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA256/size'):
                archive.verify_all()

    def test_replay_failure_still_publishes_with_replay_exit_code(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            repo = Path(temp)
            args = argparse.Namespace(repo=repo, archive_root=None, snapshot=True, device='cpu',
                                      publish_remote='https://github.com/wangxin0721/XG-NID.git')
            process = type('Process', (), {'wait': lambda self: 7, 'poll': lambda self: 7, 'pid': 123})()
            def inspect_live_publish(*args, **kwargs):
                target = args[2]
                self.assertFalse((target / '.launcher.lock').exists())
                self.assertTrue((repo / 'tmp' / (launcher.OUTPUT_NAME + '.launcher.lock')).exists())
                self.assertIn('replay_exit.json', sync.inventory(target))
                return [0, 0]
            with patch.object(launcher.subprocess, 'Popen', return_value=process) as child, \
                 patch.object(launcher.publisher, 'publish_results', side_effect=inspect_live_publish) as publish:
                code = launcher.launch(args)
            self.assertEqual(code, 7)
            self.assertIn('--snapshot', child.call_args.args[0])
            self.assertEqual(publish.call_args.kwargs['exit_key'], 'replay_exit_code')
            self.assertEqual(publish.call_args.kwargs['output_name'], launcher.OUTPUT_NAME)
            self.assertEqual(publish.call_args.kwargs['remote'], args.publish_remote)
            output = repo / 'outputs' / launcher.OUTPUT_NAME
            self.assertEqual(run.raw.read(output / 'replay_exit.json')['exit_code'], 7)
            def fake_publish(command, **kwargs):
                self.assertIn(launcher.OUTPUT_NAME, command)
                self.assertEqual(command[-2:], ['--remote', args.publish_remote])
                kwargs['stdout'].write(b'fixed replay upload\n')
                return type('Result', (), {'returncode': 0})()
            with patch.object(launcher.publisher.subprocess, 'run', side_effect=fake_publish):
                launcher.publisher.publish_results(run.raw.ROOT, repo, output, 7, {}, launcher.OUTPUT_NAME,
                                                  remote=args.publish_remote, exit_key='replay_exit_code')
            status = run.raw.read(output / 'exit_codes.json')
            self.assertEqual(status['replay_exit_code'], 7)
            self.assertNotIn('training_exit_code', status)


if __name__ == '__main__':
    unittest.main()
