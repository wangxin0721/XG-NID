"""Graph-level mechanism, strict controls, recovery, launcher and archive checks."""
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
from torch_geometric.data import Batch

import run_flow_packet_node_modulation as run
import launch_flow_packet_node_modulation as launcher
import audit_flow_packet_node_archive as auditor
from test_packet_transition_readout import fixture
from test_flow_query_joint_packet2 import TinyGraphs
from xgnid.joint_transition_fusion import JointTransitionFusion
from xgnid.flow_packet_node_modulation import NodeAdjustment


class Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.device = torch.device('cpu')
        self.settings = {**run.raw.read(run.CONFIG), 'epochs': 2, 'batch_size': 3,
                         'eval_batch_size': 4, 'workers': 0, 'training_samples': 8,
                         'validation_samples': 8, 'updates_per_epoch': 3, 'updates_per_arm': 6}
        self.source, self.graphs = fixture()
        run.frozen.independent.freeze_c(self.source)
        self.ordered = JointTransitionFusion(self.source.fusion, 'joint_ordered', 42).requires_grad_(False).eval()
        with torch.no_grad():
            # Exercise learned query/readout and the first-depth skip, not only zero heads.
            self.source.cross_layer.weight[:, :64].normal_(0, .01)
            self.source.packet_pool.query.weight.normal_(0, .01)
            self.ordered.readout.output.weight.normal_(0, .01)
        self.data = TinyGraphs(self.graphs)
        self.rows = {part: [{'sample_id': g.sample_id, 'label': int(g.y), 'partition': part}
                            for g in self.graphs] for part in ('train', 'val')}

    def models(self):
        return run.make_models(self.source, self.ordered, self.device, self.settings)[0]

    def graph_batch(self):
        batch = Batch.from_data_list(self.graphs)
        return batch, run.training.topology(batch, self.device)

    def tree_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for k in a:
                self.tree_equal(a[k], b[k])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(type(a), type(b))
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.tree_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_raw_epoch0_matches_learned_ordered_and_parameter_scope(self):
        batch, links = self.graph_batch()
        features, _ = run.frozen.encode(self.source, batch, self.device)
        models, counts = run.make_models(self.source, self.ordered, self.device, self.settings)
        for n, m in models.items():
            torch.testing.assert_close(m(batch, links), self.ordered(features)[0], rtol=0, atol=0)
            m.train()
            self.assertTrue(all(not x.training for e in (m.flow_encoder, m.packet_encoder) for x in e.modules()))
            if n != 'S0':
                opt = run.training.optimizer(m, self.settings)
                self.assertEqual([g['lr'] for g in opt.param_groups][:5], [1e-6, 1e-4, 1e-4, 1e-4, 1e-3])
        self.assertEqual(counts['S2']['new'], 3008)
        self.assertEqual(counts['S2']['trainable'], counts['S3']['trainable'])
        trainable = {n for n, p in models['S1'].named_parameters() if p.requires_grad}
        self.assertFalse(any(n.startswith('flow_encoder.') or '.convs1.' in n or '.bns' in n for n in trainable))
        self.assertTrue(any('flow___contain___packet' in n for n in trainable))
        self.assertFalse(any('packet___rev_contain___flow' in n for n in trainable))

    def test_adjustment_enters_second_convolution_and_preserves_first_skip(self):
        m = self.models()['S3']
        batch, links = self.graph_batch()
        first, incoming, skip = [], [], []
        h1 = m.packet_encoder.relus1['packet'].register_forward_hook(lambda mod, args, out: first.append(out.clone()))
        h2 = m.packet_encoder.convs2.register_forward_pre_hook(lambda mod, args: incoming.append(args[0]['packet'].clone()))
        h3 = m.cross_layer.register_forward_pre_hook(lambda mod, args: skip.append(args[0][:, :64].clone()))
        with torch.no_grad():
            m.adjustment.net[-1].bias[64:].fill_(.2)
            m(batch, links)
        for h in (h1, h2, h3):
            h.remove()
        # Our hook is installed inside forward, after h1: h1 captures unmodified nodes.
        torch.testing.assert_close(incoming[0], first[0] + .2)
        from xgnid.packet_pooling import ordered_mean_pool
        torch.testing.assert_close(skip[0], ordered_mean_pool(first[0], batch['packet'].batch, batch.num_graphs))

    def test_staged_gradients_and_frozen_buffers(self):
        batch, links = self.graph_batch()
        for n, m in self.models().items():
            if n == 'S0':
                continue
            m.train()
            before = run.training.frozen_digest(m)
            opt = run.training.optimizer(m, self.settings)
            for step in range(2):
                opt.zero_grad(set_to_none=True)
                loss = F.nll_loss(m(batch, links), batch.y.view(-1))
                loss.backward()
                self.assertGreater(sum(float(p.grad.square().sum()) for mod in m.packet_relations()
                                       for p in mod.parameters() if p.grad is not None), 0)
                if m.adjustment:
                    self.assertGreater(float(m.adjustment.net[-1].weight.grad.norm()), 0)
                    norm = float(m.adjustment.net[0].weight.grad.norm())
                    self.assertEqual(norm, 0) if step == 0 else self.assertGreater(norm, 0)
                opt.step()
            self.assertEqual(run.training.frozen_digest(m), before)
            self.assertTrue(all(p.grad is None for p in m.flow_encoder.parameters()))

    def test_flow_context_is_graph_local_and_mlp_ignores_it(self):
        nodes, context, index = torch.randn(5, 64), torch.randn(2, 64), torch.tensor([0, 0, 1, 1, 1])
        for n in ('S2', 'S3'):
            m = NodeAdjustment(n)
            with torch.no_grad():
                m.net[-1].weight.normal_(0, .1)
            changed = context.clone()
            changed[0] += 5
            a, b = m(nodes, context, index), m(nodes, changed, index)
            torch.testing.assert_close(a[2:], b[2:], rtol=0, atol=0)
            if n == 'S2':
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            else:
                self.assertFalse(torch.equal(a[:2], b[:2]))

    def test_inference_does_not_use_labels(self):
        m = self.models()['S3']
        batch, links = self.graph_batch()
        with torch.no_grad():
            m.adjustment.net[-1].weight.normal_(0, .1)
            a = m(batch, links)
            del batch.y
            torch.testing.assert_close(a, m(batch, links), rtol=0, atol=0)

    def test_half_batch_fault_restores_all_models_adam_rng_sampler_and_loss(self):
        originals = {n: m for n, m in self.models().items() if n != 'S0'}
        with tempfile.TemporaryDirectory(dir=run.raw.ROOT / 'tmp') as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            run.raw.seed_all(91)
            full = run.training.fit(deepcopy(originals), self.data, self.rows, self.settings, {}, self.device, root/'full')
            run.raw.seed_all(91)
            run.training.fit(deepcopy(originals), self.data, self.rows, self.settings, {}, self.device,
                             root/'fault', stop_after_updates=1)
            def fail(n, step):
                if n == 'S2' and step == 2:
                    raise RuntimeError('half batch')
            with self.assertRaisesRegex(RuntimeError, 'half batch'):
                run.training.fit(deepcopy(originals), self.data, self.rows, self.settings, {}, self.device,
                                 root/'fault', fault=fail)
            disk = torch.load(root/'fault/recovery.pt', weights_only=True)
            self.assertEqual(disk['updates'], 1)
            torch.rand(19)
            resumed = run.training.fit(deepcopy(originals), self.data, self.rows, self.settings, {}, self.device, root/'fault')
            for key in ('models', 'optimizers', 'rng', 'sampler', 'history', 'best', 'loss_sums'):
                self.tree_equal(full[key], resumed[key])
            self.assertEqual(resumed['updates'], 6)

    def test_source_binding_mismatch_rejected_on_resume(self):
        models = {n: m for n, m in self.models().items() if n != 'S0'}
        with tempfile.TemporaryDirectory(dir=run.raw.ROOT/'tmp') as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            run.training.fit(models, self.data, self.rows, self.settings, {}, self.device, root, stop_after_updates=1)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                run.training.fit(models, self.data, self.rows, self.settings, {'changed': True}, self.device, root)

    def test_budget_and_strict_comparison_gate(self):
        s = run.raw.read(run.CONFIG)
        self.assertEqual(len(run.training.transaction.PairedSampler(107621, 512, 42)), 211)
        self.assertEqual(s['updates_per_arm'], 211 * s['epochs'])
        self.assertEqual(s['total_screen_updates'], 3 * s['updates_per_arm'])
        records = [{'sample_id': str(i), 'label': i, 'predictions': {n: i for n in run.ARMS}} for i in range(8)]
        self.assertFalse(run.summarize(records, {}, s)['decision']['S3_exceeds_both_controls'])
        records[0]['predictions'].update(S1=1, S2=1)
        result = run.summarize(records, {}, s)
        self.assertTrue(result['decision']['S3_exceeds_both_controls'])
        self.assertEqual(result['comparisons']['S3-S1']['counts']['repaired'], 1)
        self.assertFalse(result['decision']['test_authorized'])

    def test_complete_raw_pipeline_seal_and_completed_resume(self):
        features, _ = run.frozen.encode(self.source, Batch.from_data_list(self.graphs), self.device)
        gold = [{'sample_id': r['sample_id'], 'label': r['label'], 'prediction': p}
                for r, p in zip(self.rows['val'], self.ordered(features)[0].argmax(-1).tolist())]
        class Inputs:
            def rows(self, name):
                return gold
        refs = {'joint_ordered': self.ordered}
        source_before = run.sources.locked.frozen_digests(self.source, refs)
        manifest = {'seeds': {'42': {'models': {'joint_ordered': {'validation_predictions_path': 'gold'}}}}}
        with tempfile.TemporaryDirectory(dir=run.raw.ROOT/'tmp') as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            def fake_lock(*args):
                run.sources.immutable(root/'config.json', self.settings)
                run.sources.immutable(root/'protocol_lock.json', {'settings': self.settings})
                return self.settings, {}, {}, {}, Inputs(), manifest
            ctx = (root, self.source, refs, self.data, self.rows, self.device, {}, {}, {}, source_before, {'test': 0})
            with patch.object(run, 'lock', fake_lock), patch.object(run.sources, 'context', return_value=ctx):
                run.run(root, None, root)
                self.assertEqual(run.archive.verify_sealed(root)['samples'], 8)
                (root/'publication').mkdir()
                (root/'publication/training.log').write_text('launcher changes after sealing')
                run.run(root, None, root)
            self.assertFalse(run.raw.read(root/'summary.json')['test_evaluated'])

    def test_launcher_publishes_on_failure_and_preserves_child_exit(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=run.raw.ROOT/'tmp') as tmp:
            args = SimpleNamespace(repo=Path(tmp), resume='auto', data_root=None, lock_only=False,
                                   probe_only=False, publish_remote=None)
            child = SimpleNamespace(wait=lambda: 7, poll=lambda: 7)
            with patch.object(launcher, 'lfs_environment', return_value={'PATH': ''}), \
                 patch.object(launcher.subprocess, 'Popen', return_value=child), \
                 patch.object(launcher.publisher, 'publish_results', return_value=[0, 0]) as publish, \
                 patch.object(launcher, 'uploaded_commit', return_value='a'*40):
                self.assertEqual(launcher.launch(args), 7)
                publish.assert_called_once()
                self.assertEqual(publish.call_args.args[3], 7)

    def test_launcher_audits_and_republishes_receipt_on_success(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=run.raw.ROOT/'tmp') as tmp:
            args = SimpleNamespace(repo=Path(tmp), resume='auto', data_root=None, lock_only=True,
                                   probe_only=False, publish_remote=None)
            child = SimpleNamespace(wait=lambda: 0, poll=lambda: 0)
            with patch.object(launcher, 'lfs_environment', return_value={'PATH': ''}), \
                 patch.object(launcher.subprocess, 'Popen', return_value=child), \
                 patch.object(launcher.subprocess, 'run') as audit, \
                 patch.object(launcher.publisher, 'publish_results', return_value=[0, 0]) as publish, \
                 patch.object(launcher, 'uploaded_commit', return_value='a'*40):
                self.assertEqual(launcher.launch(args), 0)
                self.assertEqual(publish.call_count, 2)
                self.assertIn('audit_flow_packet_node_archive.py', audit.call_args.args[0][2])

    def test_github_snapshot_audit_and_tamper_rejection(self):
        import gzip
        from types import SimpleNamespace
        settings = self.settings
        records = [{'sample_id': str(i), 'label': i, 'predictions': {n: i for n in run.ARMS}} for i in range(8)]
        summary = run.summarize(records, {n: {'best_epoch': 0} for n in run.ARMS}, settings)
        sampler = run.training.transaction.PairedSampler(8, 3, settings['sampling_seed'])
        history = [{'epoch': 0, 'validation': summary['metrics']}]
        for epoch in (1, 2):
            sampler.start(epoch)
            history.append({'epoch': epoch, 'validation': summary['metrics'], 'samples_per_arm': 8,
                            'updates_per_arm': 3, 'cumulative_updates_per_arm': epoch*3,
                            'order_sha256': sampler.order_sha256})
            sampler.cursor = 8
        payload = {n: json.dumps(value).encode() for n, value in {
            'config.json': settings, 'protocol_lock.json': {'settings': settings,
                'validation_cohort_sha256': run.archive.canonical_sha([(r['sample_id'], r['label']) for r in records])},
            'status.json': {'status': 'completed'}, 'summary.json': summary, 'history.json': history,
            'final_verification.json': {'passed': True}}.items()}
        payload['validation_predictions.jsonl.gz'] = gzip.compress(('\n'.join(json.dumps(r) for r in records)+'\n').encode())
        prefix = 'outputs/' + run.OUTPUT + '/'
        def read_git(command, **kwargs):
            if command[1] == 'ls-tree':
                return '\n'.join(prefix+n for n in payload)
            return payload[command[2].split(':', 1)[1][len(prefix):]]
        with patch.object(run.archive, 'lfs_environment', return_value={}), \
             patch.object(auditor.subprocess, 'run', return_value=SimpleNamespace(returncode=0)), \
             patch.object(auditor.subprocess, 'check_output', side_effect=read_git):
            result = auditor.audit(run.raw.ROOT, 'a'*40)
            self.assertTrue(result['passed'] and result['completed_training_audited'])
            records[0]['predictions']['S3'] = 1
            payload['validation_predictions.jsonl.gz'] = gzip.compress(('\n'.join(json.dumps(r) for r in records)+'\n').encode())
            with self.assertRaisesRegex(ValueError, 'recomputation differs'):
                auditor.audit(run.raw.ROOT, 'a'*40)

    def test_launcher_reports_upload_failure(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=run.raw.ROOT/'tmp') as tmp, redirect_stdout(io.StringIO()):
            args = SimpleNamespace(repo=Path(tmp), resume='auto', data_root=None, lock_only=True,
                                   probe_only=False, publish_remote=None)
            with patch.object(launcher, 'lfs_environment', return_value={'PATH': ''}), \
                 patch.object(launcher.subprocess, 'Popen', return_value=SimpleNamespace(wait=lambda: 0)), \
                 patch.object(launcher.publisher, 'publish_results', return_value=[1, 1]):
                self.assertEqual(launcher.launch(args), 1)
                receipt = Path(tmp)/'outputs'/run.OUTPUT/'publication/publication_failure.json'
                self.assertEqual(run.raw.read(receipt)['training_exit_code'], 0)


if __name__ == '__main__':
    unittest.main()
