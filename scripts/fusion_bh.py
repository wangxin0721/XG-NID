"""B-H prediction statistics adapter. Standard-library import; Linux tensor work only."""
from __future__ import annotations

import ast
import math
import inspect
from types import MethodType

from e3_gate_study import identity, pure_gate_ast, require
from fusion_acb import (build_model as build_b, losses, auxiliary_gradient_check,
                        rng_state, restore_rng, rng_hash, parameter_counts)

VARIANT = 'B_Hstats'
CONTROLS = {'seed': 42, 'batch_size': 512, 'evaluation_batch_size': 16,
            'initialization_batch_size': 16, 'optimizer': 'Adam', 'lr': .001,
            'weight_decay': 0., 'epochs': 100, 'patience': 12, 'min_delta': 0.,
            'workers': 2, 'lambda': .05, 'probe_updates': 3}
STATEMENTS = {
    'flow_conf': ('_confidence_from_logits', 'flow_logits', 'bh_flow_prediction'),
    'packet_conf': ('_confidence_from_logits', 'packet_logits', 'bh_packet_prediction'),
    'flow_margin': ('_margin_from_logits', 'flow_logits', 'bh_flow_prediction'),
    'packet_margin': ('_margin_from_logits', 'packet_logits', 'bh_packet_prediction'),
}


def prediction_statistics_ast(source):
    """Exact four-statistic replacement on the frozen no-proxy forward."""
    tree = pure_gate_ast(source)
    fn = tree.body[0]
    changed = []
    for node in fn.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in STATEMENTS:
                method, old, new = STATEMENTS[name]
                expected = ast.parse('self.%s(%s)' % (method, old), mode='eval').body
                require(ast.dump(node.value) == ast.dump(expected), 'Frozen statistic expression differs: ' + name)
                node.value.args[0] = ast.Name(id=new, ctx=ast.Load())
                changed.append(name)
    require(changed == list(STATEMENTS), 'Need exactly four original statistics in their original order')
    require(isinstance(fn.body[0], ast.Assign) and isinstance(fn.body[0].value, ast.Call)
            and ast.unparse(fn.body[0].value.func) == 'self._branch_outputs', 'Unexpected branch output position')
    fn.body[1:1] = ast.parse('bh_flow_prediction = self.shared_head(flow_logits)\n'
                            'bh_packet_prediction = self.shared_head(packet_logits)').body
    cache = next((n.value for n in fn.body if isinstance(n, ast.Assign)
                  and ast.unparse(n.targets[0]) == 'self._last_cache'), None)
    require(isinstance(cache, ast.Dict), 'Frozen forward cache missing')
    cache.keys.extend([ast.Constant(value='bh_flow_prediction'), ast.Constant(value='bh_packet_prediction')])
    cache.values.extend([ast.Name(id='bh_flow_prediction', ctx=ast.Load()), ast.Name(id='bh_packet_prediction', ctx=ast.Load())])
    ast.fix_missing_locations(tree)
    return tree


def select_source(model, source):
    require(source in ('H_prediction', 'z_representation'), 'Unknown gate statistics source')
    model._graph_logits = MethodType(model._bh_prediction_function if source == 'H_prediction'
                                    else model._bh_original_function, model)


def build_model(core, frozen, device, expected_b):
    model, b_cert = build_b(core, frozen, device, 'B_dynamic', expected_b['original_equal_init'])
    require(b_cert == expected_b, 'B-H must start from the exact recorded B state and RNG')
    before_rng = rng_state(core)
    original = type(model)._graph_logits
    source = inspect.getsource(original)
    tree = prediction_statistics_ast(source)
    namespace = dict(original.__globals__)
    exec(compile(tree, '<B-H-four-prediction-statistics>', 'exec'), namespace)
    model._bh_original_function = model._graph_logits.__func__
    model._bh_prediction_function = namespace['_graph_logits']
    select_source(model, 'H_prediction')
    require(core.state_hash(model.state_dict().items()) == b_cert['full_initial_state_sha256'], 'B-H changed parameters/buffers')
    require(rng_hash(core, rng_state(core)) == b_cert['training_start_rng_sha256'], 'B-H consumed RNG during adaptation')
    restore_rng(core, before_rng)
    return model, {'variant': VARIANT, 'base_B': b_cert,
                   'forward_ast_sha256': identity(ast.dump(tree, include_attributes=False)),
                   'changed_statistic_columns_zero_based': [16,17,18,19],
                   'parameter_counts': parameter_counts(model, 'B_dynamic'),
                   'model_forward_H_calls': 3, 'training_loss_H_calls': 5,
                   'training_H_predictions_recomputed_by_unchanged_B_loss': True}


def compare_close(core, left, right, message):
    require(bool(core.torch.allclose(left, right, rtol=0, atol=1e-6)), message)


def check_gate_input(core, model, gate_input):
    cache = model._last_cache
    zf, zp = cache['flow_logits'], cache['packet_logits']
    hf, hp = cache['bh_flow_prediction'], cache['bh_packet_prediction']
    require(gate_input.ndim == 2 and gate_input.size(1) == 21, 'Gate input must remain 21D')
    expected_stats = core.torch.stack((model._confidence_from_logits(hf), model._confidence_from_logits(hp),
                                      model._margin_from_logits(hf), model._margin_from_logits(hp)), dim=-1)
    compare_close(core, gate_input[:, :16], core.torch.cat((zf,zp), dim=-1), 'The original 16 representation dimensions changed')
    compare_close(core, gate_input[:,16:20], expected_stats, 'Four statistics are not from H(z)')
    compare_close(core, gate_input[:,20], core.F.cosine_similarity(zf,zp,dim=-1), 'Representation cosine changed')
    require(bool(core.torch.isfinite(gate_input).all()), 'Nonfinite gate inputs')


def initial_equivalence(core, model, batch, captured):
    """No optimizer updates; restore buffers/RNG after all train-mode audit forwards."""
    state = {n:t.detach().clone() for n,t in model.state_dict().items()}
    rng, mode, original_y = rng_state(core), model.training, batch.y
    outputs, inputs, values = {}, {}, {}
    try:
        model.train()
        with core.torch.no_grad():
            for name, source in [('old_B','z_representation'), ('BH','H_prediction'), ('switch_back','z_representation')]:
                model.load_state_dict(state, strict=True)
                restore_rng(core, rng)
                select_source(model, source)
                parts = losses(core, model, batch)
                outputs[name] = captured['output'].clone()
                inputs[name] = captured['input'].clone()
                values[name] = core.torch.stack(parts).clone()
                require(core.torch.equal(model._last_cache['gate_weights'], model._last_cache['gate_weights'].new_full((batch.num_graphs,2),.5)),
                        'Initial actual weights are not exactly equal')
                if name == 'BH':
                    check_gate_input(core, model, captured['input'])
            for name in ('BH','switch_back'):
                compare_close(core, outputs['old_B'],outputs[name], 'Initial fusion output differs from B')
                compare_close(core, values['old_B'],values[name], 'Initial main/branch/total loss differs from B')
            compare_close(core, inputs['old_B'][:,:16],inputs['BH'][:,:16], 'Non-statistic inputs changed')
            compare_close(core, inputs['old_B'][:,20],inputs['BH'][:,20], 'Cosine input changed')
            changed = float((inputs['old_B'][:,16:20]-inputs['BH'][:,16:20]).abs().max())
            require(changed > 1e-7, 'H and z statistics did not actually differ on the probe batch')
            # Labels are changed only for this audit; the original tensor and
            # all running buffers/RNG are restored before the first update.
            model.load_state_dict(state, strict=True)
            restore_rng(core, rng)
            select_source(model, 'H_prediction')
            batch.y = (original_y + 1) % 8
            output = model(batch)
            compare_close(core, output,outputs['BH'], 'Labels affect model forward')
            compare_close(core, captured['input'],inputs['BH'], 'Labels affect gate input')
    finally:
        batch.y = original_y
        model.load_state_dict(state, strict=True)
        restore_rng(core, rng)
        select_source(model, 'H_prediction')
        model.train(mode)
    require(core.state_hash(model.state_dict().items()) == core.state_hash(state.items()), 'Audit changed model state')
    require(rng_hash(core,rng_state(core)) == rng_hash(core,rng), 'Audit changed RNG')
    return {'initial_output_and_all_losses_match_B': True, 'switch_back_restores_B': True,
            'exact_initial_equal_weights': True, 'label_permutation_does_not_change_gate_or_output': True,
            'only_four_input_statistics_changed': True, 'four_statistics_max_change': changed,
            'state_and_rng_restored': True, 'audit_optimizer_updates': 0}


def probe_update(core, model, opt, batch, captured):
    model.train()
    opt.zero_grad(set_to_none=True)
    parts = losses(core,model,batch)
    require(all(bool(core.torch.isfinite(v)) for v in parts), 'Nonfinite probe loss')
    inp = captured['input']
    check_gate_input(core,model,inp)
    aux = auxiliary_gradient_check(core,model,parts)
    # Isolate the route loss -> four statistics -> H, excluding the main and
    # auxiliary classification routes. autograd outputs are diagnostic only.
    incoming = core.torch.autograd.grad(parts[0],inp,retain_graph=True)[0][:,16:20]
    heads = list(model.shared_head.parameters())
    routed = core.torch.autograd.grad(inp[:,16:20],heads,grad_outputs=incoming,retain_graph=True,allow_unused=True)
    require(all(bool(core.torch.isfinite(g).all()) for g in routed if g is not None), 'Nonfinite statistics-to-H gradient')
    route_max = max((float(g.abs().max()) for g in routed if g is not None),default=0.)
    before = {n:p.detach().clone() for n,p in model.gate.named_parameters()}
    parts[0].backward()
    core.torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
    gradients = {n:float(p.grad.abs().max()) if p.grad is not None else None for n,p in model.gate.named_parameters()}
    opt.step()
    deltas = {n:float((p.detach()-before[n]).abs().max()) for n,p in model.gate.named_parameters()}
    return {'loss_total':float(parts[0].detach()),'loss_fusion':float(parts[1].detach()),
            'loss_flow':float(parts[2].detach()),'loss_packet':float(parts[3].detach()),
            'gate_gradients':gradients,'gate_parameter_changes':deltas,
            'H_via_gate_statistics_gradient_max':route_max,'auxiliary_gradient_paths':aux,
            'gate_input_shape':list(inp.shape),'gate_input_H_statistics_verified':True,
            'weight_mean':model._last_cache['gate_weights'].detach().mean(0).cpu().tolist()}


def validate_probe(report,status,binding):
    for r in (report,status):
        require((r.get('status'),r.get('exit_code'),r.get('mode'),r.get('variant')) ==
                ('completed',0,'probe',VARIANT), 'Completed B-H probe required')
    require(report['binding']==binding and report['test_evaluated'] is False,'B-H probe identity differs')
    require(all(report['checks'].get(k) is True for k in ('initial_equivalence','label_independence',
            'four_statistics_only','auxiliary_gradients','gate_update','upstream_gate_update','new_H_gradient_path')),
            'B-H probe evidence incomplete')
    require(len(report['updates'])==3,'Need exactly three disposable updates')


def decide(macro,baseline):
    require(math.isfinite(macro) and 0 <= macro <= 1, 'Invalid validation score')
    if macro <= baseline['B_dynamic']:
        action='stop_no_gain_over_B'
    elif macro <= baseline['C_global']:
        action='stop_not_above_C'
    elif macro <= baseline['E2']:
        action='retain_mechanism_clue_no_automatic_expansion'
    else:
        action='freeze_candidate_for_paired_seed43_44_confirmation'
    return {'action':action,'above_E0':macro>baseline['E0'],
            'single_seed_is_not_stability_or_significance':True,'automatic_additional_training':False}
