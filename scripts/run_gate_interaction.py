"""Train three matched fusion heads on the pinned D32 graph-level caches."""
import argparse
import copy
import gzip
import json
import shutil
import time
from itertools import islice
from pathlib import Path

import torch
import torch_geometric
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

import run_attention_operator_pair as raw
from summarize_attention_operator_pair import LABELS, check_history, matrix_report, require
from xgnid.attention_operator_pair import stream
from xgnid.fusion_dimension import FrozenEncoderClassifier
from xgnid.gate_interaction import InteractionFusionHead


CONFIG = raw.ROOT / 'configs/gate_interaction_s42_v1.json'
MODES = {'Gate-Add': 'add', 'Gate-Interaction': 'multiply'}


def encoder_hash(model):
    return raw.state_digest({name: value for name, value in model.state_dict().items()
                             if name.startswith(('flow_encoder.', 'packet_encoder.'))})


def make_models(state, config, device):
    backbone = raw.OperatorClassifier('GAT', seed=config['seed'])
    backbone.load_state_dict(state, strict=True)
    models = {'D32-Continue': FrozenEncoderClassifier(backbone.flow_encoder, backbone.packet_encoder, backbone.fusion)}
    names = ('flow_projection', 'packet_projection', 'gate_hidden', 'gate_output', 'head_hidden', 'head_output')
    streams = {name: stream(config['seed'], 'fusion.'+name) for name in names}
    for name, mode in MODES.items():
        head = InteractionFusionHead(streams, mode, seed=config['seed'])
        head.load_state_dict({**head.state_dict(), **backbone.fusion.state_dict()}, strict=True)
        models[name] = FrozenEncoderClassifier(backbone.flow_encoder, backbone.packet_encoder, head)
    require(raw.state_digest(models['Gate-Add'].fusion.state_dict())
            == raw.state_digest(models['Gate-Interaction'].fusion.state_dict()), 'Extra projection initialization differs')
    return {name: model.to(device).eval() for name, model in models.items()}


def optimizer(head, config):
    extra = head.interaction_parameters() if isinstance(head, InteractionFusionHead) else []
    extra_ids = {id(p) for p in extra}
    old = [p for p in head.parameters() if id(p) not in extra_ids]
    require(sum(p.numel() for p in old) == 7144 and sum(p.numel() for p in extra) in (0, 768), 'Parameter count differs')
    groups = [{'params': old, 'lr': config['head_lr']}]
    if extra:
        groups.append({'params': extra, 'lr': config['interaction_lr']})
    return torch.optim.Adam(groups, weight_decay=config['weight_decay'])


def cache_batch(cache, index):
    return cache['flow64'][index], cache['packet64'][index], cache['labels'][index]


def backward(head, batch, opt):
    flow, packet, labels = batch
    opt.zero_grad(set_to_none=True)
    scores, _ = head(flow, packet)
    loss = F.nll_loss(scores, labels)
    loss.backward()
    gradients = [p.grad for p in head.parameters() if p.grad is not None]
    require(bool(torch.isfinite(loss)) and len(gradients) == len(list(head.parameters()))
            and bool(torch.isfinite(torch.stack([g.norm() for g in gradients]).norm())), 'Invalid loss or gradients')
    return float(loss.detach())


def probe(head, cache, batches, config):
    original = raw.state_digest(head.state_dict())
    rng = raw.rng_hash(raw.rng_state())
    with torch.random.fork_rng(devices=[]):
        candidate = copy.deepcopy(head).train()
    opt = optimizer(candidate, config)
    extra = candidate.interaction_parameters() if isinstance(candidate, InteractionFusionHead) else []
    norms = []
    initial = [p.detach().clone() for p in extra]
    updated = set()
    for index, numbers in enumerate(batches):
        before = [p.detach().clone() for p in extra]
        backward(candidate, cache_batch(cache, numbers), opt)
        grad = [float(p.grad.norm()) for p in extra]
        norms.append(dict(zip(('U_F', 'U_P', 'W'), grad)))
        opt.step()
        if extra and index == 0:
            require(grad[0] == grad[1] == 0 and grad[2] > 0
                    and torch.equal(extra[0], initial[0]) and torch.equal(extra[1], initial[1])
                    and not torch.equal(extra[2], initial[2]), 'Expected W-only first interaction update')
        updated.update(key for key, p, prev, norm in zip(('U_F', 'U_P', 'W'), extra, before, grad)
                       if norm > 0 and not torch.equal(p, prev))
    if extra:
        require(updated == {'U_F', 'U_P', 'W'}, 'Probe did not update both input projections and W')
    require(raw.state_digest(head.state_dict()) == original and raw.rng_hash(raw.rng_state()) == rng,
            'Probe modified training start or RNG')
    return {'steps': len(batches), 'interaction_grad_norms': norms, 'updated': sorted(updated),
            'training_start_unchanged': True}


@torch.no_grad()
def evaluate(head, cache, batch_size, export=False):
    head.eval()
    scores, weights = [], []
    for offset in range(0, len(cache['ids']), batch_size):
        flow, packet, _ = cache_batch(cache, slice(offset, offset+batch_size))
        output, alpha = head(flow, packet)
        scores.append(output)
        if export:
            weights.append(alpha)
    logits = torch.cat(scores)
    require(bool(torch.isfinite(logits).all()), 'Nonfinite validation outputs')
    predicted = logits.argmax(dim=1)
    cm = torch.bincount(cache['labels']*8+predicted, minlength=64).reshape(8, 8).tolist()
    rows = []
    if export:
        for sid, label, pred, alpha in zip(cache['ids'], cache['labels'].tolist(), predicted.tolist(), torch.cat(weights).tolist()):
            rows.append({'sample_id': sid, 'label': label, 'prediction': pred, 'flow_weights': alpha})
    return matrix_report(cm), rows


def identities(rows):
    return [(row['sample_id'], row['label'], row['prediction']) for row in rows]


@torch.no_grad()
def preflight(models, cache, gold, source_matrix, config):
    size = config['eval_batch_size']
    for offset in range(0, len(cache['ids']), size):
        flow, packet, _ = cache_batch(cache, slice(offset, offset+size))
        scores, alpha = models['D32-Continue'].fusion(flow, packet)
        for name in MODES:
            current, gate = models[name].fusion(flow, packet)
            require(torch.equal(current, scores) and torch.equal(gate, alpha), 'Zero-W outputs differ: '+name)
    measured, rows = evaluate(models['D32-Continue'].fusion, cache, size, export=True)
    require(measured['confusion_matrix'] == source_matrix and identities(rows) == identities(gold),
            'Epoch zero failed to reproduce archived D32 validation predictions')
    return {'samples': measured['samples'], 'epoch_zero': measured, 'paired_outputs_identical': True,
            'source_predictions_identical': True}


def fit(name, model, caches, config, binding, output, expected_orders=None):
    raw.seed_all(config['seed'])
    head = model.fusion
    folder = output/name
    folder.mkdir()
    opt = optimizer(head, config)
    sampler = raw.TrainBatches(len(caches['train']['ids']), config['batch_size'], config['seed'])
    size = config['eval_batch_size']
    before = encoder_hash(model)
    rng = raw.rng_hash(raw.rng_state())
    cfg = {'variant': name, 'protocol': config, 'binding': binding, 'labels': LABELS,
           'initial_state_sha256': raw.state_digest(model.state_dict())}
    zero, _ = evaluate(head, caches['val'], size)
    raw.save_checkpoint(folder/'epoch_zero.pt', model, opt, 0, zero, cfg, sampler)
    selected, selected_epoch = zero, 0
    trained, trained_epoch = None, None
    history = []
    started = time.perf_counter()
    with (folder/'epochs.jsonl').open('x', encoding='utf-8') as ledger:
        for epoch in range(1, config['epochs']+1):
            model.train()
            steps = samples = 0
            total = 0.0
            for numbers in sampler:
                if steps == 0 and expected_orders is not None:
                    require(sampler.order_sha256 == expected_orders[epoch-1], 'Paired sample order differs')
                total += backward(head, cache_batch(caches['train'], numbers), opt)*len(numbers)
                opt.step()
                steps += 1
                samples += len(numbers)
            require(samples == len(caches['train']['ids']) and steps == len(sampler)
                    and encoder_hash(model) == before, 'Incomplete epoch or frozen encoder changed')
            metrics, _ = evaluate(head, caches['val'], size)
            if trained is None or metrics['macro_f1'] > trained['macro_f1']:
                trained, trained_epoch = metrics, epoch
                raw.save_checkpoint(folder/'trained_best.pt', model, opt, epoch, metrics, cfg, sampler)
            if metrics['macro_f1'] > selected['macro_f1']:
                selected, selected_epoch = metrics, epoch
            row = {'epoch': epoch, 'validation': metrics, 'best_epoch': selected_epoch,
                   'best_trained_epoch': trained_epoch, 'samples': samples, 'optimizer_updates': steps,
                   'train_nll': total/samples, 'train_order_sha256': sampler.order_sha256}
            history.append(row)
            ledger.write(json.dumps(row)+'\n')
            ledger.flush()
            print(json.dumps({'model': name, 'epoch': epoch, 'val_macro_f1': metrics['macro_f1'],
                              'best_epoch': selected_epoch}), flush=True)
    raw.save_checkpoint(folder/'last.pt', model, opt, config['epochs'], metrics, cfg, sampler)
    shutil.copyfile(folder/('epoch_zero.pt' if selected_epoch == 0 else 'trained_best.pt'), folder/'best.pt')
    raw.strict_reload(model, folder/'best.pt', caches['val']['flow64'].device, cfg, selected_epoch, selected)
    replay, rows = evaluate(head, caches['val'], size, export=True)
    require(replay == selected and encoder_hash(model) == before, 'Cached checkpoint replay differs')
    raw.save_predictions(folder/'validation_predictions.jsonl.gz', rows)
    result = {'epoch_zero': zero, 'best_validation': selected, 'best_epoch': selected_epoch,
              'trained_best_validation': trained, 'best_trained_epoch': trained_epoch, 'last_validation': metrics,
              'epochs': config['epochs'], 'optimizer_updates': sum(r['optimizer_updates'] for r in history),
              'training_samples_per_epoch': samples, 'orders': [r['train_order_sha256'] for r in history],
              'training_start_rng_sha256': rng, 'frozen_state_sha256': before, 'wall_seconds': time.perf_counter()-started,
              'trainable_parameters': sum(p.numel() for p in head.parameters()),
              'checkpoint_sha256': {n: raw.sha(folder/n) for n in ('epoch_zero.pt','trained_best.pt','best.pt','last.pt')},
              'predictions_sha256': raw.sha(folder/'validation_predictions.jsonl.gz')}
    check_history(history, result, config['epochs'], len(sampler), len(caches['val']['ids']))
    return result, rows


def load_inputs(repo, config, rows, device):
    source = repo/config['source_directory']
    expected = {source/'best.pt': config['source_sha256'],
                source/'validation_predictions.jsonl.gz': config['source_predictions_sha256']}
    caches = {}
    for part in ('train', 'val'):
        path = repo/config['cache_directory']/(part+'_encoder64.pt')
        expected[path] = config['cache_sha256'][part]
    for path, digest in expected.items():
        require(raw.sha(path) == digest, 'Pinned input hash differs: '+str(path))
    for part, count in [('train', 107621), ('val', 13446)]:
        path = repo/config['cache_directory']/(part+'_encoder64.pt')
        cache = torch.load(path, map_location='cpu', weights_only=False)
        require(cache['metadata']['partition'] == part and cache['metadata']['source']['split_id'] == config['split_id']
                and cache['metadata']['source']['encoder_sha256'] == config['encoder_sha256'], 'Wrong cache provenance')
        require(len(rows[part]) == count and cache['ids'] == [r['sample_id'] for r in rows[part]]
                and cache['labels'].dtype == torch.int64 and cache['labels'].tolist() == [r['label'] for r in rows[part]],
                'Cache sample IDs or labels differ from manifest')
        require(all(cache[key].shape == (count, 64) and cache[key].dtype == torch.float32
                    and bool(torch.isfinite(cache[key]).all()) for key in ('flow64', 'packet64')), 'Invalid cache tensor')
        caches[part] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cache.items()}
    with gzip.open(source/'validation_predictions.jsonl.gz', 'rt', encoding='utf-8') as stream:
        gold = [json.loads(line) for line in stream]
    checkpoint = torch.load(source/'best.pt', map_location='cpu', weights_only=False)
    require(checkpoint['epoch'] == config['source_epoch'] and checkpoint['validation']['macro_f1'] == config['source_macro_f1'],
            'Wrong source checkpoint epoch or score')
    return checkpoint, gold, caches, expected


def run(repo, output, train):
    config = raw.read(CONFIG)
    require(torch.cuda.is_available() and torch.__version__.split('+')[0] == '2.3.1'
            and torch_geometric.__version__ == '2.5.3', 'Requires CUDA, Torch 2.3.1 and PyG 2.5.3')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    data = raw.FrozenGraphs(repo/config['split_directory'], config['split_id'])
    rows = {part: raw.choose_rows(data, part) for part in ('train', 'val')}
    checkpoint, gold, caches, expected = load_inputs(repo, config, rows, device)
    models = make_models(checkpoint['model_state'], config, device)
    require(all(encoder_hash(model) == config['encoder_sha256'] for model in models.values()), 'Encoder differs from cache')
    binding = {'config_sha256': raw.sha(CONFIG), 'input_sha256': {str(p.relative_to(repo)): h for p, h in expected.items()},
               'sources': {path: raw.sha(raw.ROOT/path) for path in (
                   'scripts/run_gate_interaction.py', 'src/xgnid/gate_interaction.py', 'src/xgnid/fusion_dimension.py',
                   'scripts/run_attention_operator_pair.py', 'scripts/summarize_attention_operator_pair.py',
                   'scripts/grouped_graphs.py', 'src/xgnid/attention_operator_pair.py', 'src/xgnid/model.py')}}
    raw.write(output/'config.json', config)
    raw.write(output/'binding.json', binding)
    initial = preflight(models, caches['val'], gold, checkpoint['validation']['confusion_matrix'], config)
    raw.write(output/'initialization.json', initial)
    sampler = raw.TrainBatches(len(rows['train']), config['batch_size'], config['seed'])
    batches = list(islice(sampler, config['probe_batches']))
    raw.write(output/'probes.json', {name: probe(model.fusion, caches['train'], batches, config)
                                     for name, model in models.items()})
    if not train:
        return
    val_batches = DataLoader(raw.GraphRows(data, rows['val'], 'val'), batch_size=config['eval_batch_size'],
                             shuffle=False, num_workers=config['workers'])
    results, exports = {}, {}
    for name, model in models.items():
        orders = results['D32-Continue']['orders'] if results else None
        result, cached = fit(name, model, caches, config, binding, output, orders)
        require(result['epoch_zero'] == initial['epoch_zero'], 'Training epoch zero differs from preflight')
        measured, replay = raw.evaluate(model, val_batches, rows['val'], device, output, name+'_raw_replay', True)
        require(measured == result['best_validation'] and identities(replay) == identities(cached)
                and encoder_hash(model) == config['encoder_sha256'], 'Best raw-graph replay differs: '+name)
        result['raw_replay_passed'] = True
        results[name], exports[name] = result, replay
        raw.write(output/name/'metrics.json', result)
    require(len({r['training_start_rng_sha256'] for r in results.values()}) == 1, 'Paired starting RNG differs')
    scores = {name: result['best_validation']['macro_f1'] for name, result in results.items()}
    candidate = scores['Gate-Interaction']
    delta = 100*(candidate-max(scores['D32-Continue'], scores['Gate-Add']))
    analysis = {'Interaction_minus_Add_pp': 100*(candidate-scores['Gate-Add']),
                'Interaction_minus_Continue_pp': 100*(candidate-scores['D32-Continue']),
                'Add_minus_Continue_pp': 100*(scores['Gate-Add']-scores['D32-Continue']),
                'budget_delta_pp': delta, 'budget_passed': delta >= config['budget_threshold_pp'],
                'Interaction_minus_CL_Multi_pp': 100*(candidate-config['CL_Multi_reference_macro_f1']),
                'Interaction_minus_JCL_Multi_pp': 100*(candidate-config['JCL_Multi_reference_macro_f1']),
                'per_class': {label: {name: result['best_validation']['per_class'][label] for name, result in results.items()}
                              for label in LABELS}, 'comparisons': {}}
    for new, old in [('Gate-Interaction', 'Gate-Add'), ('Gate-Interaction', 'D32-Continue'), ('Gate-Add', 'D32-Continue')]:
        key = new+'_vs_'+old
        comparison, changed = raw.compare(exports[old], exports[new])
        analysis['comparisons'][key] = comparison
        raw.save_predictions(output/(key+'_changed_samples.jsonl.gz'), changed)
    for path, digest in expected.items():
        require(raw.sha(path) == digest, 'Pinned input changed during experiment: '+str(path))
    raw.write(output/'summary.json', {'status': 'completed', 'protocol': config, 'binding': binding,
                                     'models': results, 'analysis': analysis, 'test_evaluated': False})
    raw.write(output/'analysis.json', analysis)
    lines = ['| Model | Epoch zero F1 | Selected F1 | Selected epoch | Last F1 |', '|---|---:|---:|---:|---:|']
    for name, result in results.items():
        lines.append(f"| {name} | {100*result['epoch_zero']['macro_f1']:.6f}% | {100*scores[name]:.6f}% | "
                     f"{result['best_epoch']} | {100*result['last_validation']['macro_f1']:.6f}% |")
    lines.extend(['', f"Interaction−Add {analysis['Interaction_minus_Add_pp']:+.6f} pp.",
                  f"Interaction−Continue {analysis['Interaction_minus_Continue_pp']:+.6f} pp.",
                  f"Budget delta {delta:+.6f} pp. Budget passed {analysis['budget_passed']}."])
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    repo = args.repo.resolve()
    name = 'gate_interaction_s42_v1' if args.train else 'gate_interaction_preflight_s42_v1'
    output = repo/'outputs'/name
    output.mkdir()
    raw.write(output/'status.json', {'status': 'running'})
    try:
        run(repo, output, args.train)
    except BaseException as error:
        raw.write(output/'status.json', {'status': 'failed', 'error': str(error)})
        raise
    raw.write(output/'status.json', {'status': 'completed', 'exit_code': 0, 'training_started': args.train})
    print(output, flush=True)


if __name__ == '__main__':
    main()
