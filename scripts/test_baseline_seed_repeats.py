"""Structural equivalence and paired-statistics checks; standard library only."""
import ast
import copy
import inspect
import json
from pathlib import Path
import unittest
from baseline_seed_repeats import JOBS,initial_ast,probe_ast,full_ast,functions
from run_baseline_seed_repeats_server import historical_module
from analyze_baseline_seed_repeats import paired_statistics

ROOT=Path(__file__).resolve().parent.parent
STUDY=json.loads((ROOT/'configs/baseline_seed_repeats_20261028.json').read_text(encoding='utf-8'))

class Checks(unittest.TestCase):
    def test_exact_scope(self):
        self.assertEqual(JOBS,[('E0',43),('E2',43),('E0',44),('E2',44)])
        self.assertEqual([(r['model'],r['seed']) for r in STUDY['jobs']],JOBS)
        self.assertFalse(STUDY['automatic_frozen_fusion_jobs'])

    def test_initialization_only_seed_and_reference_hash_guard_change(self):
        for model in ('E0','E2'):
            module=historical_module(STUDY,model)
            old=ast.parse(inspect.getsource(module.initial_model))
            new=initial_ast(inspect.getsource(module.initial_model))
            new.body[0].args.args.pop()
            for i,n in enumerate(new.body[0].body):
                if isinstance(n,ast.If):new.body[0].body[i]=n.body[0]
            for n in ast.walk(new):
                if isinstance(n,ast.Call) and ast.unparse(n.func)=='core.seed_all':n.args[0]=ast.Constant(42)
            self.assertEqual(ast.dump(old),ast.dump(new))

    def test_epoch_math_matches_historical_training(self):
        # Remove only monitoring/checkpoint bookkeeping from the new epoch body.
        class Strip(ast.NodeTransformer):
            def visit_Expr(self,n):
                if isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='__repeat_progress':return None
                return self.generic_visit(n)
            def visit_If(self,n):
                if n.body and isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Call) and ast.unparse(n.body[0].value.func)=='__repeat_progress':return None
                return self.generic_visit(n)
            def visit_Dict(self,n):
                retained=[i for i,k in enumerate(n.keys) if not(isinstance(k,ast.Constant) and k.value in ('stale_epochs','python_random_state'))]
                n.keys=[n.keys[i] for i in retained];n.values=[n.values[i] for i in retained]
                return self.generic_visit(n)
        for model in ('E0','E2'):
            module=historical_module(STUDY,model);source=inspect.getsource(module.full)
            old=ast.parse(source);new=Strip().visit(full_ast(source))
            def loop(tree):return next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='epoch')
            self.assertEqual(ast.dump(loop(old)),ast.dump(loop(new)))

    def test_no_test_inference_or_selection(self):
        for model in ('E0','E2'):
            module=historical_module(STUDY,model);tree=full_ast(inspect.getsource(module.full))
            compile(tree,'<static-repeat>','exec')
            calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='core.evaluate']
            self.assertEqual(len(calls),1)
            self.assertEqual(ast.unparse(calls[0].args[1]),'val_loader')
            self.assertNotIn('predictions.csv',ast.unparse(tree))
            row=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='rows')
            self.assertEqual(ast.literal_eval(row.value.generators[0].iter),('train','val'))

    def test_reuses_original_update(self):
        for model in ('E0','E2'):
            module=historical_module(STUDY,model)
            _,probe,train,_=functions(module,43)
            self.assertIs(probe.__globals__['update'],module.update)
            self.assertIs(train.__globals__['update'],module.update)
            sampler=next(n for n in ast.walk(probe_ast(inspect.getsource(module.probe))) if isinstance(n,ast.Call) and ast.unparse(n.func)=='core.TrainBatches')
            self.assertEqual(ast.unparse(sampler.args[2]),'args.seed')

    def test_fail_closed_on_historical_hooks(self):
        module=historical_module(STUDY,'E0')
        with self.assertRaises(ValueError):initial_ast(inspect.getsource(module.initial_model).replace('core.seed_all(42)','core.seed_all(7)'))
        with self.assertRaises(ValueError):full_ast(inspect.getsource(module.full).replace('test_loader =','another_loader ='))

    def test_paired_sample_standard_deviation(self):
        values={42:{'E0':.90,'E2':.91},43:{'E0':.91,'E2':.90},44:{'E0':.90,'E2':.90}}
        result=paired_statistics(values)
        self.assertAlmostEqual(result['mean'],0.)
        self.assertAlmostEqual(result['sample_std'],1.)
        self.assertEqual(result['E2_higher_seeds'],1)
        self.assertEqual(result['E0_higher_seeds'],1)

    def test_reject_incomplete_or_invalid_pairs(self):
        with self.assertRaises(ValueError):paired_statistics({42:{'E0':.9,'E2':.91}})
        with self.assertRaises(ValueError):paired_statistics({s:{'E0':.9,'E2':float('nan')} for s in (42,43,44)})

if __name__=='__main__':unittest.main()
