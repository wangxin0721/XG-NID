"""Fixed-budget matched query sources on the archived D32 node caches."""
import argparse
import copy
import gzip
import json
import shutil
import time
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_geometric
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

import run_attention_operator_pair as raw
import run_packet_pool_pair as packet
from run_gate_interaction import encoder_hash, identities
from summarize_attention_operator_pair import LABELS, check_history, matrix_report, require
from xgnid.flow_query_pooling import QueryPacketClassifier
from xgnid.packet_pooling import ordered_mean_pool, uniform_packet_weights


CONFIG = raw.ROOT/'configs/flow_query_pooling_s42_v1.json'
CORE = SimpleNamespace(torch=torch)


def build_models(state, config, device):
    models = {}
    for name, mode in config['variants'].items():
        base = raw.OperatorClassifier('GAT', seed=config['seed'])
        base.load_state_dict(state, strict=True)
        models[name] = QueryPacketClassifier(base.flow_encoder, base.packet_encoder, base.fusion,
                                             mode, config['seed']).to(device).eval()
    require(len({raw.state_digest(m.state_dict()) for m in models.values()}) == 1, 'Initial states differ')
    return models


def optimizer(model, config):
    pool, head = list(model.packet_pool.parameters()), list(model.fusion.parameters())
    require(sum(p.numel() for p in pool) == 1024 and sum(p.numel() for p in head) == 7144,
            'Expected 1024 readout and 7144 head parameters')
    require({id(p) for p in pool+head} == {id(p) for p in model.parameters() if p.requires_grad},
            'Unexpected trainable parameters')
    return torch.optim.Adam([{'params': head, 'lr': config['head_lr']},
                             {'params': pool, 'lr': config['pool_lr']}], weight_decay=config['weight_decay'])


def step(model, cache, numbers, device, opt):
    features, labels = packet.minibatch(CORE, cache, numbers, device)
    opt.zero_grad(set_to_none=True)
    scores, _ = model.forward_features(*features)
    loss = F.nll_loss(scores, labels)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    require(bool(torch.isfinite(loss)) and all(g is not None for g in grads)
            and bool(torch.isfinite(torch.stack([g.norm() for g in grads]).norm())), 'Invalid loss or gradients')
    opt.step()
    return float(loss.detach())


def probe(model, cache, batches, config, device):
    initial = raw.state_digest(model.state_dict())
    rng = raw.rng_hash(raw.rng_state())
    with torch.random.fork_rng(devices=[]):
        candidate = copy.deepcopy(model).train()
    opt = optimizer(candidate, config)
    weights = {'query': candidate.packet_pool.query.weight, 'key': candidate.packet_pool.key.weight}
    history, updated = [], set()
    for index, batch in enumerate(batches):
        before = {name: p.detach().clone() for name, p in weights.items()}
        step(candidate, cache, batch, device, opt)
        norms = {name: float(p.grad.norm()) for name, p in weights.items()}
        changed = {name for name, p in weights.items() if not torch.equal(p, before[name])}
        if index == 0:
            require(changed == {'query'} and norms['query'] > 0 and norms['key'] == 0,
                    'Expected query update and zero key gradient at step one')
        history.append({'gradient_norms': norms, 'updated': sorted(changed)})
        updated.update(changed)
    require(updated == {'query', 'key'}, 'Both readout projections must update during the probe')
    require(raw.state_digest(model.state_dict()) == initial and raw.rng_hash(raw.rng_state()) == rng
            and encoder_hash(candidate) == encoder_hash(model), 'Probe changed training start, RNG or frozen encoders')
    return {'steps': history, 'training_start_unchanged': True}


def validation_indices(cache, size):
    for start in range(0, len(cache['ids']), size):
        yield list(range(start, min(start+size, len(cache['ids']))))


@torch.no_grad()
def evaluate(model, cache, device, size, export=False):
    model.eval()
    outputs, labels, rows = [], [], []
    for numbers in validation_indices(cache, size):
        features, batch_labels = packet.minibatch(CORE, cache, numbers, device)
        scores, alpha, beta = model.forward_with_attention(*features)
        outputs.append(scores)
        labels.append(batch_labels)
        if export:
            lengths = (cache['ptr'][[i+1 for i in numbers]]-cache['ptr'][numbers]).tolist()
            attention = beta.cpu().split(lengths)
            for i, y, p, a, b in zip(numbers, batch_labels.tolist(), scores.argmax(-1).tolist(), alpha.tolist(), attention):
                require(bool(torch.isfinite(b).all()) and abs(float(b.sum())-1) < 1e-5, 'Invalid per-graph attention')
                rows.append({'sample_id': cache['ids'][i], 'label': y, 'prediction': p,
                             'flow_weights': a, 'packet_attention': b.tolist()})
    scores, truth = torch.cat(outputs), torch.cat(labels)
    require(bool(torch.isfinite(scores).all()), 'Invalid validation scores')
    cm = torch.bincount(truth*8+scores.argmax(-1), minlength=64).reshape(8, 8).tolist()
    return matrix_report(cm), rows


@torch.no_grad()
def initial_check(models, cache, device, config, gold, source):
    baseline = models['Mean-Query']
    for numbers in validation_indices(cache, config['eval_batch_size']):
        (flow, nodes, index, size), _ = packet.minibatch(CORE, cache, numbers, device)
        mean = ordered_mean_pool(nodes, index, size)
        uniform = uniform_packet_weights(nodes, index, size)
        for model in models.values():
            pooled, beta = model.packet_pool(flow, nodes, index, size)
            require(torch.equal(pooled, mean) and torch.equal(beta, uniform), 'Initial readout differs from ordered mean')
    measured, rows = evaluate(baseline, cache, device, config['eval_batch_size'], True)
    require(measured['confusion_matrix'] == source['confusion_matrix'] and identities(rows) == identities(gold),
            'Initial predictions differ from archived D32')
    return {'samples': len(rows), 'uniform_attention_and_ordered_mean_exact': True,
            'd32_predictions_identical': True, 'epoch_zero': measured}


def fit(name, model, caches, device, config, binding, output, expected_orders=None):
    raw.seed_all(config['seed'])
    folder = output/name
    folder.mkdir()
    opt = optimizer(model, config)
    sampler = raw.TrainBatches(len(caches['train']['ids']), config['batch_size'], config['seed'])
    size = config['eval_batch_size']
    frozen = encoder_hash(model)
    rng = raw.rng_hash(raw.rng_state())
    cfg = {'variant': name, 'protocol': config, 'binding': binding, 'labels': LABELS,
           'initial_state_sha256': raw.state_digest(model.state_dict())}
    zero, _ = evaluate(model, caches['val'], device, size)
    raw.save_checkpoint(folder/'epoch_zero.pt', model, opt, 0, zero, cfg, sampler)
    selected, selected_epoch = zero, 0
    trained, trained_epoch = None, None
    history = []
    started = time.perf_counter()
    with (folder/'epochs.jsonl').open('x', encoding='utf-8') as ledger:
        for epoch in range(1, config['epochs']+1):
            model.train()
            count = steps = 0
            total = 0.0
            for numbers in sampler:
                if steps == 0 and expected_orders is not None:
                    require(sampler.order_sha256 == expected_orders[epoch-1], 'Paired order differs')
                total += step(model, caches['train'], numbers, device, opt)*len(numbers)
                count += len(numbers)
                steps += 1
            require(count == len(caches['train']['ids']) and steps == len(sampler)
                    and encoder_hash(model) == frozen, 'Incomplete epoch or frozen state changed')
            metrics, _ = evaluate(model, caches['val'], device, size)
            if trained is None or metrics['macro_f1'] > trained['macro_f1']:
                trained, trained_epoch = metrics, epoch
                raw.save_checkpoint(folder/'trained_best.pt', model, opt, epoch, metrics, cfg, sampler)
            if metrics['macro_f1'] > selected['macro_f1']:
                selected, selected_epoch = metrics, epoch
            row = {'epoch': epoch, 'validation': metrics, 'best_epoch': selected_epoch, 'best_trained_epoch': trained_epoch,
                   'samples': count, 'optimizer_updates': steps, 'train_nll': total/count,
                   'train_order_sha256': sampler.order_sha256}
            history.append(row)
            ledger.write(json.dumps(row)+'\n')
            ledger.flush()
            print(json.dumps({'model': name, 'epoch': epoch, 'val_macro_f1': metrics['macro_f1'],
                              'best_epoch': selected_epoch}), flush=True)
    raw.save_checkpoint(folder/'last.pt', model, opt, config['epochs'], metrics, cfg, sampler)
    shutil.copyfile(folder/('epoch_zero.pt' if selected_epoch == 0 else 'trained_best.pt'), folder/'best.pt')
    raw.strict_reload(model, folder/'best.pt', device, cfg, selected_epoch, selected)
    replay, rows = evaluate(model, caches['val'], device, size, True)
    require(replay == selected and encoder_hash(model) == frozen, 'Cached replay or frozen state differs')
    raw.save_predictions(folder/'validation_predictions.jsonl.gz', rows)
    result = {'epoch_zero': zero, 'best_validation': selected, 'best_epoch': selected_epoch,
              'trained_best_validation': trained, 'best_trained_epoch': trained_epoch, 'last_validation': metrics,
              'epochs': config['epochs'], 'optimizer_updates': sum(r['optimizer_updates'] for r in history),
              'training_samples_per_epoch': count, 'orders': [r['train_order_sha256'] for r in history],
              'training_start_rng_sha256': rng, 'frozen_state_sha256': frozen, 'wall_seconds': time.perf_counter()-started,
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'checkpoint_sha256': {n: raw.sha(folder/n) for n in ('epoch_zero.pt', 'trained_best.pt', 'best.pt', 'last.pt')},
              'predictions_sha256': raw.sha(folder/'validation_predictions.jsonl.gz')}
    check_history(history, result, config['epochs'], len(sampler), len(caches['val']['ids']))
    return result, rows


def run(repo, output, train):
    config = raw.read(CONFIG)
    require(torch.cuda.is_available() and torch.__version__.split('+')[0] == '2.3.1'
            and torch_geometric.__version__ == '2.5.3', 'Requires CUDA, Torch 2.3.1 and PyG 2.5.3')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    data = raw.FrozenGraphs(repo/config['split_directory'], config['split_id'])
    rows = {part: raw.choose_rows(data, part) for part in ('train', 'val')}
    require((len(rows['train']), len(rows['val'])) == (107621, 13446), 'Wrong split sizes')
    source = repo/config['source_directory']
    cache_paths = {part: repo/config['cache_directory']/(part+'_packet_nodes.pt') for part in rows}
    inputs = {source/'best.pt': config['source_sha256'],
              source/'validation_predictions.jsonl.gz': config['source_predictions_sha256'],
              **{path: config['cache_sha256'][part] for part, path in cache_paths.items()}}
    for path, digest in inputs.items():
        require(raw.sha(path) == digest, 'Pinned input hash differs: '+str(path))
    payload = torch.load(source/'best.pt', map_location='cpu', weights_only=False)
    require(payload['epoch'] == config['source_epoch'] and payload['validation']['macro_f1'] == config['source_macro_f1'],
            'Wrong mature D32 checkpoint')
    with gzip.open(source/'validation_predictions.jsonl.gz', 'rt', encoding='utf-8') as stream:
        gold = [json.loads(line) for line in stream]
    caches = {}
    for part, path in cache_paths.items():
        cache = torch.load(path, map_location='cpu', weights_only=False)
        split = cache['metadata']['split_binding']
        require(split['split_id'] == config['split_id'] and split['manifest_sha256'] == data.metadata['sample_manifest_sha256'],
                'Node cache split or manifest differs')
        packet.validate_cache(CORE, cache, rows[part], part, config['encoder_sha256'], split)
        caches[part] = cache
    models = build_models(payload['model_state'], config, device)
    require(all(encoder_hash(m) == config['encoder_sha256'] for m in models.values()), 'D32 encoder hash differs')
    binding = {'config_sha256': raw.sha(CONFIG), 'input_sha256': {str(path.relative_to(repo)): sha for path, sha in inputs.items()},
               'sources': {path: raw.sha(raw.ROOT/path) for path in (
                   'scripts/run_flow_query_pooling.py', 'src/xgnid/flow_query_pooling.py', 'src/xgnid/packet_pooling.py',
                   'scripts/run_packet_pool_pair.py', 'scripts/run_gate_interaction.py', 'scripts/run_attention_operator_pair.py',
                   'scripts/summarize_attention_operator_pair.py', 'scripts/grouped_graphs.py', 'src/xgnid/fusion_dimension.py',
                   'src/xgnid/attention_operator_pair.py', 'src/xgnid/model.py')}}
    raw.write(output/'config.json', config)
    raw.write(output/'binding.json', binding)
    initial = initial_check(models, caches['val'], device, config, gold, payload['validation'])
    raw.write(output/'initialization.json', initial)
    sampler = raw.TrainBatches(len(rows['train']), config['batch_size'], config['seed'])
    batches = list(islice(sampler, config['probe_batches']))
    raw.write(output/'probes.json', {name: probe(model, caches['train'], batches, config, device)
                                     for name, model in models.items()})
    if not train:
        return
    val_loader = DataLoader(raw.GraphRows(data, rows['val'], 'val'), batch_size=config['eval_batch_size'],
                            shuffle=False, num_workers=config['workers'])
    results, exports = {}, {}
    for name, model in models.items():
        orders = results['Mean-Query']['orders'] if results else None
        result, cached = fit(name, model, caches, device, config, binding, output, orders)
        require(result['epoch_zero'] == initial['epoch_zero'], 'Training start differs from preflight')
        measured, replay = raw.evaluate(model, val_loader, rows['val'], device, output, name+'_raw_replay', True)
        require(measured == result['best_validation'] and identities(replay) == identities(cached)
                and encoder_hash(model) == config['encoder_sha256'], 'Full original-graph replay differs: '+name)
        result['raw_replay_passed'] = True
        results[name], exports[name] = result, cached
        raw.write(output/name/'metrics.json', result)
    mean, flow = [results[name]['best_validation']['macro_f1'] for name in ('Mean-Query', 'Flow-Query')]
    require(len({r['training_start_rng_sha256'] for r in results.values()}) == 1, 'Paired RNG differs')
    a, b = [exports[name] for name in ('Mean-Query', 'Flow-Query')]
    comparison, changed = raw.compare(a, b)
    raw.save_predictions(output/'Flow-Query_vs_Mean-Query_changed_samples.jsonl.gz', changed)
    focus = {}
    for label in config['focus_classes']:
        y = LABELS.index(label)
        counts, _ = raw.compare([r for r in a if r['label'] == y], [r for r in b if r['label'] == y])
        focus[label] = counts
    analysis = {'Flow_Query_minus_Mean_Query_pp': 100*(flow-mean),
                'Flow_Query_minus_JCL_Multi_pp': 100*(flow-config['JCL_Multi_reference_macro_f1']),
                'comparison': comparison, 'focus_comparisons': focus,
                'per_class': {label: {name: r['best_validation']['per_class'][label] for name, r in results.items()}
                              for label in LABELS}}
    for path, digest in inputs.items():
        require(raw.sha(path) == digest, 'Pinned input changed during run: '+str(path))
    raw.write(output/'summary.json', {'status': 'completed', 'protocol': config, 'binding': binding,
                                     'models': results, 'analysis': analysis, 'test_evaluated': False})
    raw.write(output/'analysis.json', analysis)
    lines = ['| Model | Epoch zero F1 | Selected F1 | Selected epoch | Last F1 |', '|---|---:|---:|---:|---:|']
    for name, r in results.items():
        lines.append(f"| {name} | {100*r['epoch_zero']['macro_f1']:.6f}% | {100*r['best_validation']['macro_f1']:.6f}% | "
                     f"{r['best_epoch']} | {100*r['last_validation']['macro_f1']:.6f}% |")
    lines.extend(['', f'Flow−Mean {100*(flow-mean):+.6f} pp.',
                  f"Flow−JCL-Multi {analysis['Flow_Query_minus_JCL_Multi_pp']:+.6f} pp.", '',
                  '| True class | Repaired | Regressed |', '|---|---:|---:|'])
    for label, counts in focus.items():
        lines.append(f"| {label} | {counts['counts']['repaired']} | {counts['counts']['regressed']} |")
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    repo = args.repo.resolve()
    name = 'flow_query_pooling_s42_v1' if args.train else 'flow_query_pooling_preflight_s42_v1'
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
