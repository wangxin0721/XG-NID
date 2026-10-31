"""Standard-library regression tests for selection, provenance and accounting."""
import copy
import json
import multiprocessing
import struct
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_packet_pool_pair as run
from summarize_packet_pool_pair import analyze, budget, summarize

ROOT = Path(__file__).resolve().parents[1]


def child_counts(counts, connection):
    counts['train'] += 1
    counts['val'] += 1
    connection.send(counts)
    connection.close()


class Contracts(unittest.TestCase):
    def config(self):
        return run.read(ROOT / 'configs/packet_pool_pair_s42_v1.json')

    def test_uniform_weighting_can_differ_from_mean_in_float32_without_tolerance_changes(self):
        f32 = lambda x: struct.unpack('f', struct.pack('f', x))[0]
        nodes, u = [1e6, 1., -1e6], f32(1 / 3)
        def add(values):
            total = f32(0.)
            for x in values:
                total = f32(total + x)
            return total
        mean = f32(add(nodes) / 3)
        legacy = add([f32(x * u) for x in nodes])
        centered = f32(mean + add([f32(f32(u - u) * f32(x - mean)) for x in nodes]))
        self.assertGreater(abs(legacy - mean), 1e-6)
        self.assertEqual(centered, mean)

    def test_exact_screen_and_mature_start(self):
        self.assertNotIn('torch', sys.modules)
        c = self.config()
        run.check_config(c)
        old = run.read(ROOT / 'configs/joint_fusion32_s42_v1.json')['variants']['J32-Gate']
        self.assertEqual((c['source_epoch'], c['source_best_sha256'], c['source_macro_f1']),
                         (old['epoch'], old['best_sha256'], old['macro_f1']))
        self.assertEqual(run.sha(ROOT / 'configs' / c['source_protocol']), c['source_config_sha256'])
        for k, v in [('include_epoch_zero', False), ('head_lr', .001), ('pool_lr', 1e-5),
                     ('encoder_mode', 'eval_with_gradients'), ('source_epoch', 0), ('source_best_sha256', 'wrong'),
                     ('score_output_bias', True), ('epochs', 30), ('patience', 12),
                     ('test_evaluated', True), ('automatic_followup', True)]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                run.check_config({**c, k: v})

    def test_architecture_budget_only_changes_packet_readout(self):
        p0, p1 = run.architecture('mean'), run.architecture('attention')
        self.assertEqual((p0['trainable_parameters'], p1['trainable_parameters']), (7144, 8200))
        self.assertEqual(p1['pool_parameters'], 64 * 16 + 16 + 16)
        for key in ('width', 'projection', 'gate', 'shared_head'):
            self.assertEqual(p0[key], p1[key])

    def history(self, scores, start=.9732070542814468):
        best, stale, selected, rows = start, 0, 0, []
        for epoch, score in enumerate(scores, 1):
            best, stale, improved = run.pair.selection_state(score, best, stale)
            if improved:
                selected = epoch
            rows.append({'epoch': epoch, 'val_macro_f1': score, 'optimizer_updates': 211,
                'train_order_sha256': 'order' + str(epoch), 'best_epoch': selected, 'stale_epochs': stale})
        return rows, {'epochs': len(scores), 'epoch_zero_eligible': True, 'epoch_zero': {'macro_f1': start},
            'last_validation': {'macro_f1': scores[-1]}, 'best_validation': {'macro_f1': best},
            'best_epoch': selected, 'selected_epoch_zero': selected == 0,
            'optimizer_updates': len(scores) * 211, 'orders': [r['train_order_sha256'] for r in rows]}

    def test_epoch_zero_retained_on_regression_or_tie_and_stops_at_patience(self):
        for score in (.96, .9732070542814468):
            h, m = self.history([score] * 20)
            run.check_history(h, m, self.config())
            self.assertEqual(m['best_epoch'], 0)
        for size in (19, 21):
            h, m = self.history([.96] * size)
            with self.assertRaises(ValueError):
                run.check_history(h, m, self.config())

    def test_strict_improvement_selects_first_tie_and_budget_is_enforced(self):
        h, m = self.history([.975] * 21)
        run.check_history(h, m, self.config())
        self.assertEqual(m['best_epoch'], 1)
        h, m = self.history([.9733 + i * .00001 for i in range(100)])
        run.check_history(h, m, self.config())
        self.assertEqual(m['best_epoch'], 100)
        for field, value in [('best_epoch', 0), ('optimizer_updates', 1), ('selected_epoch_zero', True)]:
            bad = {**m, field: value}
            with self.assertRaises(ValueError):
                run.check_history(h, bad, self.config())
        bad = copy.deepcopy(h)
        bad[3]['stale_epochs'] += 1
        with self.assertRaises(ValueError):
            run.check_history(bad, m, self.config())

    def test_budget_uses_stronger_matched_control_and_records_small_gain(self):
        b = self.config()['source_macro_f1']
        a = budget(b, b + .0005, b, .1)
        self.assertAlmostEqual(a['P1_minus_P0_pp'], .05)
        self.assertFalse(a['budget_screen_passed'])
        self.assertAlmostEqual(100 * a['required_P1_macro_f1'], 97.42070542814468)
        a = budget(b + .002, b + .0025, b, .1)
        self.assertGreater(a['P1_minus_D32_pp'], .1)
        self.assertFalse(a['budget_screen_passed'])
        self.assertAlmostEqual(a['stronger_reference_macro_f1'], b + .002)
        self.assertTrue(budget(b, b + .0011, b, .1)['budget_screen_passed'])

    def test_optimizer_has_two_rates_and_rejects_frozen_parameter_leak(self):
        class P:
            def __init__(self, n, active=True):
                self.n, self.requires_grad = n, active
            def numel(self):
                return self.n
        enc, head, pool = P(99, False), P(7144), P(1056)
        graph = SimpleNamespace(pooling='attention', fusion=SimpleNamespace(parameters=lambda: iter([head])),
            packet_pool=SimpleNamespace(parameters=lambda: iter([pool])),
            flow_encoder=SimpleNamespace(parameters=lambda: iter([enc])),
            packet_encoder=SimpleNamespace(parameters=lambda: iter([])),
            parameters=lambda: iter([enc, head, pool]))
        captured = []
        def Adam(groups, **kwargs):
            captured.append((groups, kwargs))
            return SimpleNamespace(state={})
        core = SimpleNamespace(torch=SimpleNamespace(optim=SimpleNamespace(Adam=Adam)))
        run.optimizer(core, graph, self.config())
        self.assertEqual([g['lr'] for g in captured[0][0]], [1e-5, 1e-4])
        self.assertEqual(captured[0][1], {'weight_decay': 0.})
        enc.requires_grad = True
        with self.assertRaises(ValueError):
            run.optimizer(core, graph, self.config())

    def test_parent_verified_counts_include_worker_reads_despite_private_dictionaries(self):
        counts = {'train': 0, 'val': 0, 'test': 0}
        context = multiprocessing.get_context('spawn')
        receive, send = context.Pipe(duplex=False)
        worker = context.Process(target=child_counts, args=(counts, send))
        worker.start()
        send.close()
        try:
            self.assertTrue(receive.poll(10), 'Worker failed to report')
            self.assertEqual(receive.recv(), {'train': 1, 'val': 1, 'test': 0})
            worker.join(10)
            self.assertEqual(worker.exitcode, 0)
        finally:
            receive.close()
            if worker.is_alive():
                worker.terminate()
                worker.join(10)
        self.assertEqual(counts, {'train': 0, 'val': 0, 'test': 0})
        stages = [run.graph_load_stage('extract_train', 'train', 1, 2),
                  run.graph_load_stage('initial_P0', 'val', 1, 2)]
        accounting = run.graph_load_accounting(counts, stages)
        self.assertEqual(accounting['totals'], {'train': 1, 'val': 1, 'test': 0})

    def test_counts_include_direct_warmups_probes_and_all_replays_without_double_counting(self):
        counts = {'train': 1120, 'val': 0, 'test': 0}
        stages = [run.graph_load_stage('extract_train', 'train', 107621, 2),
                  run.graph_load_stage('extract_val', 'val', 13446, 2)]
        stages += [run.graph_load_stage(phase + '_' + name, 'val', 13446, 2)
                   for phase in ('initial', 'best') for name in ('P0', 'P1')]
        accounting = run.graph_load_accounting(counts, stages)
        self.assertEqual(accounting['totals'], {'train': 108741, 'val': 67230, 'test': 0})
        direct = {'train': 19, 'val': 0, 'test': 0}
        self.assertEqual(run.graph_load_accounting(direct, [run.graph_load_stage('extract_train', 'train', 3, 0)])['totals'], direct)
        for bad in ([stages[0], stages[0]], [dict(stages[0], partition='test')],
                    [run.graph_load_stage('too_many_direct', 'val', 1, 0)]):
            with self.assertRaises(ValueError):
                run.graph_load_accounting(counts, bad)

    def test_memory_estimate_accounts_for_two_copies_and_retained_cache(self):
        m = run.cache_memory_estimate(3, 6, retained_bytes=1000, available_bytes=3000)
        self.assertEqual((m['flow64_bytes'], m['packet64_bytes'], m['labels_bytes'], m['ptr_bytes']), (768, 1536, 24, 32))
        self.assertEqual(m['final_cache_tensor_bytes'], 2360)
        self.assertEqual(m['concat_additional_allocation_bytes'], 2392)
        self.assertEqual(m['estimated_cache_tensor_peak_bytes'], 5720)
        # Available memory is sampled after the chunks/previous cache exist.
        self.assertTrue(m['additional_allocation_fits_host_available'])
        self.assertFalse(run.cache_memory_estimate(3, 6, available_bytes=2000)['additional_allocation_fits_host_available'])
        self.assertIsNone(run.cache_memory_estimate(3, 6)['additional_allocation_fits_host_available'])
        with self.assertRaises(ValueError):
            run.cache_memory_estimate(3, 2)

    def test_host_memory_reads_memavailable_and_reports_unknown_when_missing(self):
        with patch.object(Path, 'read_text', return_value='MemTotal: 99999 kB\nMemAvailable: 123 kB\n'):
            self.assertEqual(run.available_host_memory(), 123 * 1024)
        with patch.object(Path, 'read_text', side_effect=OSError):
            self.assertIsNone(run.available_host_memory())

    def fixture(self):
        # Synthetic reports isolate report arithmetic from checkpoint metadata.
        labels = ['Benign', 'WebBased', 'Spoofing', 'Recon', 'Mirai', 'Dos', 'DDos', 'BruteForce']
        cm = [[1680 if i == j else 0 for j in range(8)] for i in range(8)]
        cm[0][0] += 6
        cm[1][1], cm[1][3] = 1660, 20
        baseline = run.dim.frozen.matrix_report(cm, labels)
        c = self.config()
        c['source_macro_f1'] = baseline['macro_f1']
        history, result = self.history([baseline['macro_f1']] * 20, baseline['macro_f1'])
        models = {}
        for name, mode in run.VARIANTS.items():
            models[name] = {**copy.deepcopy(result), 'epoch_zero': baseline, 'best_validation': baseline,
                'last_validation': baseline, 'architecture': run.architecture(mode), 'training_start_rng_sha256': 'same',
                'training_wall_seconds': 1., 'training_evidence': {}, 'parameter_groups': []}
        comparisons = {n: {'counts': {'repaired': 0, 'regressed': 0}, 'net_correct': 0}
                       for n in ('P1_vs_P0', 'P0_vs_D32', 'P1_vs_D32')}
        r = {'status': 'completed', 'exit_code': 0, 'protocol': c, 'models': models,
            'source_model': {'best_validation': baseline, 'best_epoch': 62}, 'tensor_preflight_passed': True,
            'encoder_parameters_and_BN_unchanged': True, 'node_caches_unchanged': True,
            'source_artifacts_unchanged': True, 'test_evaluated': False, 'automatic_followup': False,
            'graph_load_counts': {'train': 2, 'val': 1, 'test': 0}, 'common_epochs_order_checked': 20,
            'initialization': dict.fromkeys(('P0_P1_outputs_allclose', 'source_predictions_identical',
                                            'uniform_packet_attention', 'epoch_zero_eligible'), True),
            'probes': {n: {'clean_model_and_rng_restored': True, 'optimizer_discarded': True} for n in run.VARIANTS},
            'graph_replays': {n: {'strict_state_load': True, 'predictions_identical': True,
                                  'confusion_matrix_reproduced': True} for n in run.VARIANTS},
            'comparisons': comparisons, 'prediction_transitions': {n: [] for n in comparisons}, 'packet_diagnostics': {}}
        stages = [run.graph_load_stage('extract_train', 'train', 107621, 2),
                  run.graph_load_stage('extract_val', 'val', 13446, 2)]
        stages += [run.graph_load_stage(phase + '_' + name, 'val', 13446, 2)
                   for phase in ('initial', 'best') for name in ('P0', 'P1')]
        r['graph_load_accounting'] = run.graph_load_accounting(r['graph_load_counts'], stages)
        r['graph_load_counts'] = r['graph_load_accounting']['totals']
        r['cache_memory_estimates'] = {part: run.cache_memory_estimate(n, n)
                                       for part, n in (('train', 107621), ('val', 13446))}
        return r, history

    def test_report_keeps_zero_epoch_and_class_tradeoffs_without_claiming_gain(self):
        r, history = self.fixture()
        with patch.object(run, 'check_config'):
            a = analyze(r)
            self.assertFalse(a['budget_screen_passed'])
            self.assertEqual(a['P1_minus_P0_pp'], 0.)
            self.assertTrue(a['selected_epoch_zero']['P1'])
            self.assertEqual(set(a['class_changes']), set(r['source_model']['best_validation']['per_class']))
            self.assertFalse(a['statistical_significance_claimed'])
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run.write(root / 'summary.json', r)
                for name in run.VARIANTS:
                    (root / name).mkdir()
                    (root / name / 'epochs.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in history), encoding='utf-8')
                report = summarize(root)
                self.assertIn('P1-max(P0,D32)', report)
                self.assertIn('WebBased/Recon', report)
                self.assertIn('Spoofing/BruteForce', report)
                self.assertIn('parent-verified samples', report)
                self.assertIn('concatenation tensor peak', report)
                self.assertEqual(len((root / 'per_class.csv').read_text().splitlines()), 41)
                self.assertTrue((root / 'prediction_transitions.csv').exists())

    def test_report_rejects_missing_verification_and_corrupt_selection(self):
        r, _ = self.fixture()
        with patch.object(run, 'check_config'):
            for flag in ('tensor_preflight_passed', 'node_caches_unchanged', 'encoder_parameters_and_BN_unchanged'):
                bad = {**r, flag: False}
                with self.subTest(flag=flag), self.assertRaises(ValueError):
                    analyze(bad)
            for mutate in (lambda x: x['graph_load_counts'].update(test=1),
                           lambda x: x['models']['P1'].update(epoch_zero_eligible=False),
                           lambda x: x['models']['P1']['orders'].__setitem__(0, 'wrong'),
                           lambda x: x['probes']['P1'].update(clean_model_and_rng_restored=False),
                           lambda x: x['initialization'].update(uniform_packet_attention=False),
                           lambda x: x['models']['P1']['architecture'].update(trainable_parameters=7144),
                           lambda x: x['graph_load_accounting']['loader_stages'].pop(),
                           lambda x: x['graph_load_counts'].update(val=0)):
                bad = copy.deepcopy(r)
                mutate(bad)
                with self.assertRaises(ValueError):
                    analyze(bad)


if __name__ == '__main__':
    unittest.main()
