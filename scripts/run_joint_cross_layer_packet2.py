"""Server acceptance and fixed-budget Packet layer-two fine-tuning pair."""
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
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

import run_attention_operator_pair as raw
from summarize_attention_operator_pair import LABELS, check_history, matrix_report, require
from xgnid.cross_layer import CrossLayerClassifier
from xgnid.joint_cross_layer import JointCrossLayerClassifier, reparameterize_projection


VARIANTS = {'JCL-Last': 'last', 'JCL-Multi': 'multi'}
CONFIG = raw.ROOT / 'configs/joint_cross_layer_packet2_s42_v1.json'


def make_models(payload, device):
    models = {}
    for name, mode in {'source': 'last', **VARIANTS}.items():
        backbone = raw.OperatorClassifier('GAT')
        cls = CrossLayerClassifier if name == 'source' else JointCrossLayerClassifier
        model = cls(backbone.flow_encoder, backbone.packet_encoder, backbone.fusion, mode)
        model.load_state_dict(payload['model_state'], strict=True)
        if name != 'source':
            reparameterize_projection(model)
        models[name] = model.to(device).eval()
    require(raw.state_digest(models['JCL-Last'].state_dict()) == raw.state_digest(models['JCL-Multi'].state_dict()),
            'Paired initial states differ')
    return models


def optimizer(model, config):
    groups = [
        {'name': 'flow_to_packet', 'params': list(model.packet_relations()[0].parameters()), 'lr': config['encoder_lr']},
        {'name': 'packet_to_packet', 'params': list(model.packet_relations()[1].parameters()), 'lr': config['encoder_lr']},
        {'name': 'projection', 'params': list(model.cross_layer.parameters()), 'lr': config['projection_lr']},
        {'name': 'fusion', 'params': list(model.fusion.parameters()), 'lr': config['head_lr']},
    ]
    ids = [id(parameter) for group in groups for parameter in group['params']]
    require(len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters() if p.requires_grad},
            'Optimizer parameter scope differs')
    require([sum(p.numel() for p in group['params']) for group in groups] == [8704, 8512, 8256, 7144],
            'Unexpected optimizer group sizes')
    return torch.optim.Adam(groups, weight_decay=config['weight_decay'])


def frozen_hash(model):
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    return raw.state_digest({name: tensor for name, tensor in model.state_dict().items() if name not in trainable})


def identities(rows):
    return [(row['sample_id'], row['label'], row['prediction']) for row in rows]


def predictions(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def loader(data, rows, part, config, sampler=None):
    dataset = raw.GraphRows(data, rows, part)
    generator = torch.Generator().manual_seed(config['seed'] + (100 if part == 'train' else 200))
    if sampler is not None:
        return DataLoader(dataset, batch_sampler=sampler, num_workers=config['workers'], generator=generator)
    return DataLoader(dataset, batch_size=config['eval_batch_size'], shuffle=False,
                      num_workers=config['workers'], generator=generator)


@torch.no_grad()
def check_initial(models, batches, rows, gold, metrics, device, output):
    errors = {key: {'max_abs_error': 0.0, 'sample_id': None} for key in ('paired_scores', 'paired_gate', 'source_scores', 'source_gate')}
    failures = set()
    offset = 0
    for batch in batches:
        batch = batch.to(device)
        features, _ = models['source'].encode(batch)
        source = models['source'].forward_features(*features)
        paired = [models[name].forward_features(*features) for name in VARIANTS]
        comparisons = [(*pair, tol) for pair, tol in [
            ((paired[0][0], paired[1][0]), 1e-6), ((paired[0][1], paired[1][1]), 1e-6),
            ((source[0], paired[0][0]), 1e-5), ((source[1], paired[0][1]), 1e-5)]]
        for key, (a, b, tolerance) in zip(errors, comparisons):
            diff = (a-b).abs().amax(dim=1)
            index = int(diff.argmax())
            if float(diff[index]) > errors[key]['max_abs_error']:
                errors[key] = {'max_abs_error': float(diff[index]), 'sample_id': batch.sample_id[index]}
            if not torch.allclose(a, b, atol=tolerance, rtol=tolerance):
                failures.add(key)
        expected = identities(gold[offset:offset+batch.num_graphs])
        # Full raw forwards also verify the trainable encoder path, independently of the shared-feature test.
        for name, scores in [('source', source[0]), *((name, models[name](batch)) for name in VARIANTS)]:
            actual = list(zip(batch.sample_id, batch.y.view(-1).tolist(), scores.argmax(-1).tolist()))
            if actual != expected:
                failures.add(name + '_raw_predictions')
        offset += batch.num_graphs
        if offset == len(rows) or offset // 3200 != (offset-batch.num_graphs) // 3200:
            raw.progress(output, 'initial_equivalence', samples=offset, total=len(rows))
    cm = [[0]*8 for _ in range(8)]
    for row in gold:
        cm[row['label']][row['prediction']] += 1
    require(offset == len(rows) == len(gold) and matrix_report(cm) == metrics, 'Reference validation coverage or matrix differs')
    receipt = {'samples': offset, 'errors': errors, 'failures': sorted(failures), 'passed': not failures,
               'initial_state_sha256': {name: raw.state_digest(models[name].state_dict()) for name in VARIANTS}}
    raw.write(output/'initialization.json', receipt)
    require(not failures, 'Initial equivalence failed; see initialization.json')


def step(model, batch, opt):
    opt.zero_grad(set_to_none=True)
    loss = F.nll_loss(model(batch), batch.y.view(-1))
    require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    require(bool(gradients) and bool(torch.isfinite(torch.stack([grad.norm() for grad in gradients]).norm())),
            'Nonfinite or missing gradients')
    opt.step()
    return float(loss.detach())


def probe(model, batches, device, config):
    # Probe a separate model instance to preserve all training weights and Adam state.
    with torch.random.fork_rng(devices=[]):
        candidate = copy.deepcopy(model).train()
    before = frozen_hash(candidate)
    opt = optimizer(candidate, config)
    changed = set()
    rng = raw.rng_hash(raw.rng_state())
    for batch in batches:
        snapshots = {group['name']: [p.detach().clone() for p in group['params']] for group in opt.param_groups}
        step(candidate, batch.to(device), opt)
        for group in opt.param_groups:
            if any(p.grad is not None and bool(p.grad.ne(0).any()) and not torch.equal(p, old)
                   for p, old in zip(group['params'], snapshots[group['name']])):
                changed.add(group['name'])
    require(changed == {group['name'] for group in opt.param_groups}, 'Probe did not update every registered group')
    require(frozen_hash(candidate) == before and raw.rng_hash(raw.rng_state()) == rng, 'Probe changed frozen state or RNG')
    return {'groups_with_gradient_and_update': sorted(changed), 'batches': len(batches), 'frozen_state_sha256': before}


def fit(name, model, data, train_rows, val_rows, config, binding, device, output, expected_orders=None):
    raw.seed_all(config['seed'])
    folder = output/name
    folder.mkdir()
    sampler = raw.TrainBatches(len(train_rows), config['batch_size'], config['seed'])
    train_batches = loader(data, train_rows, 'train', config, sampler)
    val_batches = loader(data, val_rows, 'val', config)
    opt = optimizer(model, config)
    cfg = {'variant': name, 'protocol': config, 'binding': binding, 'labels': LABELS,
           'initial_state_sha256': raw.state_digest(model.state_dict())}
    before = frozen_hash(model)
    initial_rng = raw.rng_hash(raw.rng_state())
    started = time.perf_counter()
    zero, zero_predictions = raw.evaluate(model, val_batches, val_rows, device, output, name+'_epoch_zero', True)
    raw.save_checkpoint(folder/'epoch_zero.pt', model, opt, 0, zero, cfg, sampler)
    require(raw.rng_hash(raw.rng_state()) == initial_rng, 'Initial evaluation changed RNG')
    selected, selected_epoch, selected_predictions = zero, 0, zero_predictions
    trained, trained_epoch = None, None
    history = []
    peak = 0
    for epoch in range(1, config['epochs']+1):
        model.train()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        count = steps = 0
        total_loss = 0.0
        for batch in train_batches:
            if steps == 0 and expected_orders is not None:
                require(sampler.order_sha256 == expected_orders[epoch-1], 'Paired epoch orders differ')
            expected = [train_rows[i] for i in sampler.order[count:count+batch.num_graphs]]
            require([(r['sample_id'], r['label']) for r in expected] == list(zip(batch.sample_id, batch.y.view(-1).tolist())),
                    'Training IDs or labels differ from sampler')
            total_loss += step(model, batch.to(device), opt)*batch.num_graphs
            count += batch.num_graphs
            steps += 1
            if steps == 1 or steps % 50 == 0:
                raw.progress(output, name+'_train', epoch=epoch, samples=count, total=len(train_rows))
        require(count == len(train_rows) and steps == len(sampler), 'Incomplete training epoch')
        require(frozen_hash(model) == before, 'Frozen parameters or BN changed')
        if device.type == 'cuda':
            peak = max(peak, torch.cuda.max_memory_allocated(device))
        metrics, rows = raw.evaluate(model, val_batches, val_rows, device, output, name+'_val', True)
        if trained is None or metrics['macro_f1'] > trained['macro_f1']:
            trained, trained_epoch = metrics, epoch
            raw.save_checkpoint(folder/'trained_best.pt', model, opt, epoch, metrics, cfg, sampler)
        if metrics['macro_f1'] > selected['macro_f1']:
            selected, selected_epoch, selected_predictions = metrics, epoch, rows
        record = {'epoch': epoch, 'validation': metrics, 'best_epoch': selected_epoch,
                  'best_trained_epoch': trained_epoch, 'samples': count, 'optimizer_updates': steps,
                  'train_nll': total_loss/count, 'train_order_sha256': sampler.order_sha256}
        history.append(record)
        with (folder/'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps({'model': name, 'epoch': epoch, 'val_macro_f1': metrics['macro_f1'],
                          'best_epoch': selected_epoch}), flush=True)
    raw.save_checkpoint(folder/'last.pt', model, opt, config['epochs'], metrics, cfg, sampler)
    shutil.copyfile(folder/('epoch_zero.pt' if selected_epoch == 0 else 'trained_best.pt'), folder/'best.pt')
    raw.strict_reload(model, folder/'best.pt', device, cfg, selected_epoch, selected)
    replay, replay_rows = raw.evaluate(model, val_batches, val_rows, device, output, name+'_raw_replay', True)
    require(replay == selected and identities(replay_rows) == identities(selected_predictions)
            and frozen_hash(model) == before, 'Selected checkpoint raw replay failed')
    raw.save_predictions(folder/'validation_predictions.jsonl.gz', replay_rows)
    result = {'epoch_zero': zero, 'best_validation': selected, 'best_epoch': selected_epoch,
              'trained_best_validation': trained, 'best_trained_epoch': trained_epoch,
              'last_validation': metrics, 'epochs': config['epochs'], 'optimizer_updates': sum(r['optimizer_updates'] for r in history),
              'training_samples_per_epoch': len(train_rows), 'orders': [r['train_order_sha256'] for r in history],
              'training_start_rng_sha256': initial_rng, 'frozen_state_sha256': before, 'raw_replay_passed': True,
              'wall_seconds': time.perf_counter()-started, 'peak_cuda_allocated_bytes': peak,
              'checkpoint_sha256': {n: raw.sha(folder/n) for n in ('epoch_zero.pt','trained_best.pt','best.pt','last.pt')},
              'predictions_sha256': raw.sha(folder/'validation_predictions.jsonl.gz')}
    check_history(history, result, config['epochs'], len(sampler), len(val_rows))
    raw.write(folder/'metrics.json', result)
    return result, replay_rows


def run(repo, output, train):
    config = raw.read(CONFIG)
    require(torch.cuda.is_available() and torch.__version__.split('+')[0] == '2.3.1'
            and torch_geometric.__version__ == '2.5.3', 'Requires CUDA, Torch 2.3.1 and PyG 2.5.3')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    references = {name: repo/ref['directory']/'best.pt' for name, ref in config['references'].items()}
    for name, path in references.items():
        require(raw.sha(path) == config['references'][name]['sha256'], 'Reference SHA256 differs: '+name)
    source = references['CL-Last']
    payload = torch.load(source, map_location='cpu', weights_only=False)
    archived = raw.read(raw.ROOT/'docs/cross_layer_pair_server_summary_20261102.json')['models']['CL-Last']
    gold_path = source.parent/'validation_predictions.jsonl.gz'
    require(payload['epoch'] == 113 and payload['validation'] == archived['best_validation']
            and raw.sha(gold_path) == archived['predictions_sha256'], 'CL-Last validation identity differs')
    gold = predictions(gold_path)
    data = raw.FrozenGraphs(repo/config['split_directory'], config['split_id'])
    train_rows, val_rows = raw.choose_rows(data, 'train'), raw.choose_rows(data, 'val')
    require((len(train_rows), len(val_rows)) == (107621, 13446), 'Wrong split counts')
    models = make_models(payload, device)
    binding = {'config_sha256': raw.sha(CONFIG), 'references': config['references'],
               'split_id': config['split_id'], 'source_predictions_sha256': raw.sha(gold_path),
               'sources': {path: raw.sha(raw.ROOT/path) for path in (
                   'scripts/run_joint_cross_layer_packet2.py', 'src/xgnid/joint_cross_layer.py',
                   'src/xgnid/cross_layer.py', 'src/xgnid/model.py', 'src/xgnid/packet_pooling.py',
                   'src/xgnid/fusion_dimension.py', 'scripts/grouped_graphs.py',
                   'scripts/summarize_attention_operator_pair.py', 'scripts/run_attention_operator_pair.py',
                   'src/xgnid/attention_operator_pair.py')}}
    raw.write(output/'config.json', config)
    raw.write(output/'binding.json', binding)
    check_initial(models, loader(data, val_rows, 'val', config), val_rows, gold, payload['validation'], device, output)
    del models['source']
    sampler = raw.TrainBatches(len(train_rows), config['batch_size'], config['seed'])
    dataset = raw.GraphRows(data, train_rows, 'train')
    batches = [Batch.from_data_list([dataset[i] for i in indices]) for indices in islice(sampler, config['probe_batches'])]
    raw.write(output/'probes.json', {name: probe(model, batches, device, config) for name, model in models.items()})
    del batches
    if not train:
        return
    results, exports = {}, {}
    for name, model in models.items():
        expected = results['JCL-Last']['orders'] if name == 'JCL-Multi' else None
        results[name], exports[name] = fit(name, model, data, train_rows, val_rows, config, binding, device, output, expected)
        require(results[name]['epoch_zero'] == payload['validation'], 'Training epoch zero differs from CL-Last')
    last, multi = (results[name] for name in VARIANTS)
    require(last['training_start_rng_sha256'] == multi['training_start_rng_sha256'] and last['orders'] == multi['orders'],
            'Paired training order or RNG differs')
    counts, changed = raw.compare(exports['JCL-Last'], exports['JCL-Multi'])
    raw.save_predictions(output/'JCL-Multi_vs_JCL-Last_changed_samples.jsonl.gz', changed)
    a, b = last['best_validation'], multi['best_validation']
    delta = 100*(b['macro_f1']-max(a['macro_f1'], config['references']['D32-Gate']['macro_f1']))
    historical = 100*(b['macro_f1']-config['references']['CL-Multi']['macro_f1'])
    analysis = {'JCL_Multi_minus_JCL_Last_pp': 100*(b['macro_f1']-a['macro_f1']), 'budget_delta_pp': delta,
                'JCL_Multi_minus_CL_Multi_pp': historical,
                'budget_passed': delta >= config['budget_threshold_pp'] and historical > 0,
                'comparison': counts, 'per_class': {label: {name: results[name]['best_validation']['per_class'][label]
                                                           for name in VARIANTS} for label in LABELS}}
    for name, path in references.items():
        require(raw.sha(path) == config['references'][name]['sha256'], 'Reference changed during run: '+name)
    raw.write(output/'summary.json', {'status': 'completed', 'protocol': config, 'binding': binding,
                                    'models': results, 'analysis': analysis, 'test_evaluated': False})
    raw.write(output/'analysis.json', analysis)
    lines = ['| Model | Epoch zero F1 | Trained best F1 | Trained best epoch | Selected F1 | Selected epoch | Last F1 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, result in results.items():
        lines.append(f"| {name} | {100*result['epoch_zero']['macro_f1']:.6f}% | "
                     f"{100*result['trained_best_validation']['macro_f1']:.6f}% | {result['best_trained_epoch']} | "
                     f"{100*result['best_validation']['macro_f1']:.6f}% | {result['best_epoch']} | "
                     f"{100*result['last_validation']['macro_f1']:.6f}% |")
    lines.extend(['', f"Paired delta {analysis['JCL_Multi_minus_JCL_Last_pp']:+.6f} pp.",
                  f'Budget delta {delta:+.6f} pp. Historical CL-Multi delta {historical:+.6f} pp.',
                  f"Budget passed {analysis['budget_passed']}.", '',
                  'See analysis.json for class metrics and prediction changes.'])
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    repo = args.repo.resolve()
    name = 'joint_cross_layer_packet2_s42_v1' if args.train else 'joint_cross_layer_packet2_preflight_s42_v1'
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
