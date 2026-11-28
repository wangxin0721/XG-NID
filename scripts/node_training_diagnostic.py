"""Controlled 20-epoch strategy diagnosis; model structure is unchanged."""
from copy import deepcopy
import hashlib
import math

import torch
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

import run_flow_packet_node_modulation as parent
import flow_packet_node_training as training
from xgnid.flow_packet_node_modulation import FlowPacketNodeClassifier

raw, require, archive = parent.raw, parent.require, parent.archive
ARMS = ('R1', 'R2', 'R3')


def make_models(source, ordered, settings, device):
    models = {}
    for name, mode in zip(ARMS, ('S1', 'S2', 'S3')):
        model = FlowPacketNodeClassifier(source, ordered, mode, settings['new_module_seed'])
        if name != 'R1':
            model.requires_grad_(False)
            model.adjustment.requires_grad_(True)
        model.diagnostic_name = name
        model.zero_grad(set_to_none=True)
        models[name] = model.to(device).eval()
    reference = FlowPacketNodeClassifier(source, ordered, 'S0').to(device).eval()
    return models, reference


def optimizer(model, settings):
    if model.diagnostic_name == 'R1':
        return training.optimizer(model, settings)
    params = list(model.adjustment.parameters())
    require({id(p) for p in params} == {id(p) for p in model.parameters() if p.requires_grad},
            'R2/R3 may update only the new module')
    return torch.optim.Adam([{'name': 'adjustment', 'params': params, 'lr': settings['new_learning_rate']}],
                            betas=tuple(settings['adam_betas']), eps=settings['adam_eps'],
                            weight_decay=settings['weight_decay'])


def diagnostic_subset(rows, settings):
    require(all(r['partition'] == 'train' for r in rows), 'Diagnostic subset must be training only')
    count, seed = settings['diagnostic_samples_per_class'], settings['diagnostic_sampling_seed']
    selected = []
    for label in range(8):
        candidates = [r for r in rows if r['label'] == label]
        require(len(candidates) >= count, 'Insufficient class coverage for fixed diagnostic subset')
        candidates.sort(key=lambda r: (hashlib.sha256(f"{seed}:{r['sample_id']}".encode()).hexdigest(), r['sample_id']))
        selected.extend(candidates[:count])
    require(len({r['sample_id'] for r in selected}) == 8*count, 'Duplicate diagnostic samples')
    return selected


def parameter_delta(model, initial):
    totals = {}
    for name, value in model.named_parameters():
        group = 'packet2' if name.startswith('packet_encoder.convs2.') else name.split('.')[0]
        original = initial[name].double()
        delta = value.detach().cpu().double() - original
        d = totals.setdefault(group, {'delta_squared': 0., 'source_squared': 0., 'changed_elements': 0})
        d['delta_squared'] += float(delta.square().sum())
        d['source_squared'] += float(original.square().sum())
        d['changed_elements'] += int(delta.ne(0).sum())
    return {k: {'absolute_l2': math.sqrt(d['delta_squared']),
                'relative_l2': math.sqrt(d['delta_squared']/d['source_squared']) if d['source_squared'] else None,
                'changed_elements': d['changed_elements']} for k, d in totals.items()}


def comparisons(rows, name):
    result = {label: {'repaired': 0, 'regressed': 0, 'wrong_changed': 0}
              for label in training.metrics.LABELS}
    for r in rows:
        a, b, y = r['predictions'][name], r['predictions']['Ordered'], r['label']
        if a != b:
            result[training.metrics.LABELS[y]]['repaired' if a == y else 'regressed' if b == y else 'wrong_changed'] += 1
    return {'counts': {k: sum(v[k] for v in result.values()) for k in ('repaired', 'regressed', 'wrong_changed')},
            'per_class': result}


def record_metrics(records, name):
    measured = training.metrics.metrics(records, name)
    measured['nll'] = -math.fsum(r['true_log_probability'][name] for r in records)/len(records)
    measured['source_comparison'] = comparisons(records, name)
    return measured


@torch.no_grad()
def evaluate(models, reference, data, rows, settings, device, output=None, phase='validation'):
    require(rows and all(r['partition'] == rows[0]['partition'] for r in rows), 'Mixed diagnostic partitions')
    require(rows[0]['partition'] in ('train', 'val'), 'Test loading prohibited')
    all_models = {'Ordered': reference, **models}
    for m in all_models.values():
        m.eval()
    loader = DataLoader(raw.GraphRows(data, rows, rows[0]['partition']), batch_size=settings['eval_batch_size'],
                        shuffle=False, num_workers=settings['workers'], generator=torch.Generator().manual_seed(242))
    records = []
    for step, batch in enumerate(loader, 1):
        expected = rows[len(records):len(records)+batch.num_graphs]
        training.identity(batch, expected)
        links = training.topology(batch, device)  # CPU parsing before in-place transfer.
        batch = batch.to(device)
        outputs = {n: m(batch, links) for n, m in all_models.items()}
        require(all(bool(torch.isfinite(x).all()) for x in outputs.values()), 'Nonfinite diagnostic output')
        predictions = {n: x.argmax(-1).tolist() for n, x in outputs.items()}
        log_y = {n: x.gather(1, batch.y.view(-1, 1)).flatten().tolist() for n, x in outputs.items()}
        for i, row in enumerate(expected):
            records.append({'sample_id': row['sample_id'], 'label': row['label'],
                            'predictions': {n: p[i] for n, p in predictions.items()},
                            'true_log_probability': {n: p[i] for n, p in log_y.items()}})
        if output and (step == 1 or step % 200 == 0 or len(records) == len(rows)):
            raw.progress(output, phase, samples=len(records), total=len(rows))
    require(len(records) == len(rows), 'Incomplete evaluation')
    return records, {n: record_metrics(records, n) for n in all_models}


def observe(models, reference, initial, data, rows, subset, settings, device, output, epoch):
    val, measured = evaluate(models, reference, data, rows['val'], settings, device, output, 'validation_'+str(epoch))
    train, fixed = evaluate(models, reference, data, subset, settings, device, output, 'fixed_train_'+str(epoch))
    folder = output/'observations'/('epoch_'+str(epoch))
    folder.mkdir(parents=True, exist_ok=True)
    raw.save_predictions(folder/'validation.jsonl.gz', val)
    raw.save_predictions(folder/'fixed_train.jsonl.gz', train)
    result = {}
    for name, model in models.items():
        result[name] = {**measured[name], 'fixed_train_nll': fixed[name]['nll'],
                        'parameter_delta': parameter_delta(model, initial[name])}
        raw.write(folder/(name+'.json'), result[name])
    raw.write(folder/'Ordered.json', {**measured['Ordered'], 'fixed_train_nll': fixed['Ordered']['nll']})
    return val, result


def gradient_probe(models, data, rows, settings, device, output):
    # This subset spans all classes, preventing a single easy-class zero loss from
    # hiding the adjustment path. Probe updates are discarded and RNG is restored.
    saved_rng = training.transaction.rng_state()
    try:
        batch = Batch.from_data_list([data.load(r) for r in rows])
        training.identity(batch, rows)
        links = training.topology(batch, device)
        batch = batch.to(device)
        receipts = {}
        for name, original in models.items():
            m = deepcopy(original).train()
            opt = optimizer(m, settings)
            frozen = training.frozen_digest(m)
            steps = []
            for step in range(2):
                opt.zero_grad(set_to_none=True)
                captured = []
                handle = m.packet_encoder.convs2.register_forward_pre_hook(
                    lambda module, args: captured.append(args[0]['packet']))
                try:
                    scores = m(batch, links)
                finally:
                    handle.remove()
                m.eval()
                with torch.no_grad():
                    eval_scores = m(batch, links)
                m.train()
                require(torch.allclose(scores.detach(), eval_scores, atol=1e-5, rtol=1e-5)
                        and torch.equal(scores.argmax(-1), eval_scores.argmax(-1)),
                        'Train/eval computation differs before an update')
                loss = F.nll_loss(scores, batch.y.view(-1))
                loss.backward()
                norms = {g['name']: sum(float(p.grad.square().sum()) for p in g['params'] if p.grad is not None)**.5
                         for g in opt.param_groups}
                require(bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all())
                                                         for p in m.parameters()), 'Invalid probe gradients')
                if name in ('R2', 'R3'):
                    require(len(captured) == 1 and captured[0].requires_grad and norms['adjustment'] > 0,
                            'Frozen GNN detached the adjustment gradient path')
                    require(all(p.grad is None for n, p in m.named_parameters() if not n.startswith('adjustment.')),
                            'An original parameter received a gradient')
                    if step == 1:
                        require(float(m.adjustment.net[0].weight.grad.norm()) > 0, 'Missing upstream adjustment gradient')
                steps.append({'gradient_norms': norms, 'train_nll': float(loss.detach()),
                              'train_eval_max_log_probability_error': float((scores.detach()-eval_scores).abs().max())})
                opt.step()
            require(training.frozen_digest(m) == frozen, 'Frozen parameters/BN changed during probe')
            changes = parameter_delta(m, training.transaction.cpu_state(original))
            require(sum(v['changed_elements'] for v in changes.values()) > 0, 'No effective parameter update')
            receipts[name] = {'passed': True, 'steps': steps, 'parameter_changes_after_two_updates': changes,
                              'two_updates_discarded': True}
        raw.write(output/'gradient_probe.json', receipts)
    finally:
        training.transaction.restore_rng(saved_rng)


def decision(best, settings):
    source = best['Ordered']['macro_f1']
    r2, r3 = best['R2']['macro_f1'], best['R3']['macro_f1']
    ds, dm = 100*(r3-source), 100*(r3-r2)
    return {'R3_minus_Ordered_pp': ds, 'R3_minus_R2_pp': dm,
            'extension_gate_passed': ds >= settings['extension_gain_pp'] and dm >= settings['extension_gain_pp'],
            'threshold_pp_each': settings['extension_gain_pp'],
            'no_arm_improves_source': all(best[n]['macro_f1'] <= source for n in ARMS),
            'R2_and_R3_both_improve_source': r2 > source and r3 > source,
            'automatic_extension': False, 'test_authorized': False,
            'scope': '20_epoch_seed42_resource_decision_not_statistical_significance'}
