"""Local boundary/numerical checks; no torch import, graphs or training."""
import ast
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import run_e2_concat_pair as pair

ROOT = Path(__file__).resolve().parents[1]


class PairChecks(unittest.TestCase):
    def test_py39_and_import_boundary(self):
        self.assertNotIn('torch', sys.modules)
        for name in ('run_e2_concat_pair.py', 'run_e2_n0_continue.py', 'run_e2_n1_concat.py'):
            ast.parse((ROOT/'scripts'/name).read_text(encoding='utf-8'), feature_version=(3, 9))

    def test_settings_and_reference(self):
        args = SimpleNamespace(repo=ROOT, config=ROOT/'configs/e2_concat_pair_20261028.json')
        spec = pair.metadata(args)[0]
        self.assertEqual((spec['epochs'], spec['patience'], spec['lr']), (30, 12, .0001))
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            for key, value in [('lr', .001), ('epochs', 20), ('early_stop', False), ('patience', 13),
                               ('probe_batches', 0), ('seed', 43), ('test_evaluated', True)]:
                changed = copy.deepcopy(spec)
                changed[key] = value
                path = Path(folder)/'invalid.json'
                path.write_text(json.dumps(changed), encoding='utf-8')
                with self.subTest(key=key), self.assertRaises(ValueError):
                    pair.metadata(SimpleNamespace(repo=ROOT, config=path))

    def predictions(self):
        reference = [{'sample_id': str(i), 'label': i % 8, 'prediction': i % 8} for i in range(13446)]
        candidate = copy.deepcopy(reference)
        reference[0]['prediction'] = 1  # repaired
        candidate[1]['prediction'] = 2  # regressed
        reference[2]['prediction'] = candidate[2]['prediction'] = 3  # both wrong
        return reference, candidate

    def test_epoch_zero_ties_and_patience(self):
        best, stale = .965994, 0
        for step in range(1, 13):
            best, stale, improved = pair.selection_state(.965994, best, stale)
            self.assertFalse(improved)
            self.assertEqual(stale, step)
        self.assertEqual(stale, 12)
        best, stale, improved = pair.selection_state(.966, best, stale)
        self.assertTrue(improved)
        self.assertEqual((best, stale), (.966, 0))

    def test_repairs_pair_by_id_instead_of_row_order(self):
        reference, candidate = self.predictions()
        result = pair.compare_predictions(reference, list(reversed(candidate)))
        self.assertEqual(result['counts'], dict(repaired=1, regressed=1, both_wrong=1, both_correct=13443))
        self.assertEqual(result['net_correct'], 0)
        self.assertEqual(result['by_class']['0']['repaired'], 1)
        self.assertEqual(sum(sum(c.values()) for c in result['by_class'].values()), 13446)

    def test_bad_prediction_identity_rejected(self):
        reference, candidate = self.predictions()
        candidate[1]['sample_id'] = candidate[0]['sample_id']
        with self.assertRaises(ValueError):
            pair.compare_predictions(reference, candidate)
        reference, candidate = self.predictions()
        candidate[0]['label'] = 7
        with self.assertRaises(ValueError):
            pair.compare_predictions(reference, candidate)

    def test_actual_conversion_with_numpy_tensor_double(self):
        # Execute the production conversion against a small tensor double.
        # This checks dimensions, bias preservation and numerical equivalence;
        # it does not substitute for the real checkpoint/GPU verification.
        try:
            import numpy as np
        except ImportError:
            self.skipTest('numpy unavailable')
        rng = np.random.default_rng(42)

        class Tensor:
            def __init__(self, values):
                self.values = np.array(values, dtype=np.float64)
                self.device, self.dtype = 'cpu', 'float64'
            def detach(self): return self
            def clone(self): return Tensor(self.values.copy())
            def __truediv__(self, other): return Tensor(self.values/other)
            def copy_(self, other): self.values[...] = other.values; return self
            def numel(self): return self.values.size

        class Linear:
            def __init__(self, inputs, outputs):
                self.in_features, self.out_features = inputs, outputs
                self.weight = Tensor(rng.normal(size=(outputs, inputs)))
                self.bias = Tensor(rng.normal(size=outputs))
            def to(self, **kwargs): return self
            def __call__(self, x): return x @ self.weight.values.T + self.bias.values

        class Head(list):
            def __call__(self, x):
                hidden = self[0](x)
                return self[1](np.where(hidden >= 0, hidden, .01*hidden))

        class Model:
            def __init__(self): self.shared_head = Head([Linear(8,32), Linear(32,8)])
            def state_dict(self):
                return {f'shared_head.{i}.{n}': getattr(layer,n) for i, layer in enumerate(self.shared_head)
                        for n in ('weight','bias')}
            def parameters(self):
                return list(self.state_dict().values())+[SimpleNamespace(numel=lambda: 846800-288-264)]

        model = Model()
        flow, packet = rng.normal(size=(113,8)), rng.normal(size=(113,8))
        expected = model.shared_head(.5*flow+.5*packet)
        final_weight = model.shared_head[1].weight.values.copy()
        fake_torch = SimpleNamespace(nn=SimpleNamespace(Linear=Linear), no_grad=nullcontext,
            cat=lambda xs, dim: Tensor(np.concatenate([x.values for x in xs], axis=dim)),
            equal=lambda a,b: np.array_equal(a.values,b.values))
        pair.convert(SimpleNamespace(torch=fake_torch),model)
        actual = model.shared_head(np.concatenate((flow,packet),axis=1))
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(model.shared_head[1].weight.values,final_weight)
        self.assertEqual(model.shared_head[0].weight.values.shape,(32,16))


if __name__ == '__main__':
    unittest.main()
