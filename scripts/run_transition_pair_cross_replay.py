"""Six fixed-checkpoint validation replays: no optimizer and zero training epochs."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import traceback

from filelock import FileLock
import torch
from torch.nn import functional as F
import torch_geometric

import joint_transition_training as training
from transition_pair_cache import PairCache
from xgnid.fusion_dimension import FusionHead
from xgnid.transition_pair_shuffle import PairFusion

raw, original, frozen, require = training.raw, training.original, training.frozen, training.require
OUTPUT_NAME = 'transition_pair_cross_replay_s42_v1'
CONFIG = raw.ROOT / ('configs/' + OUTPUT_NAME + '.json')
GROUPS = {'T1': ('pair_true', 'true'), 'T2': ('pair_true', 'shuffle'), 'T0': ('pair_true', 'off'),
          'S1': ('pair_shuffle', 'true'), 'S2': ('pair_shuffle', 'shuffle'), 'S0': ('pair_shuffle', 'off')}
COMPUTE_FILES = ('src/xgnid/fusion_dimension.py', 'src/xgnid/ordered_transition_readout.py',
                 'src/xgnid/packet_transition_readout.py', 'src/xgnid/packet_pooling.py',
                 'src/xgnid/joint_transition_fusion.py', 'src/xgnid/transition_pair_shuffle.py',
                 'scripts/transition_pair_cache.py')


class Archive:
    """Read hash-bound filesystem inputs or immutable Git blobs and existing LFS objects."""
    def __init__(self, repo, settings, archive_root=None, snapshot=False):
        self.repo, self.settings, self.snapshot = repo, settings, snapshot
        self.root = archive_root or repo / settings['source_directory']
        self.values, self.verified = {}, {}

    def locate(self, name):
        if name not in self.values:
            require(name in self.settings['inputs'], 'Unregistered archive input: ' + name)
            if self.snapshot:
                ref = self.settings['source_result_commit'] + ':' + self.settings['source_directory'] + '/' + name
                value = subprocess.check_output(['git', 'show', ref], cwd=self.repo)
                if value.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
                    lines = value.decode().splitlines()
                    digest = lines[1].removeprefix('oid sha256:')
                    require(digest == self.settings['inputs'][name]['sha256'], 'LFS pointer differs: ' + name)
                    environment = subprocess.check_output(['git', 'lfs', 'env'], cwd=self.repo, text=True)
                    folder = next(Path(line.split('=', 1)[1]) for line in environment.splitlines()
                                  if line.startswith('LocalMediaDir='))
                    value = folder / digest[:2] / digest[2:4] / digest
            else:
                value = self.root / name
            self.values[name] = value
        return self.values[name]

    def verify(self, name):
        value, expected = self.locate(name), self.settings['inputs'][name]
        if isinstance(value, Path):
            require(value.is_file(), 'Missing archive/LFS input: ' + str(value))
            size, digest = value.stat().st_size, raw.sha(value)
        else:
            size, digest = len(value), hashlib.sha256(value).hexdigest()
        require(size == expected['bytes'] and digest == expected['sha256'], 'Archive SHA256/size differs: ' + name)
        self.verified[name] = {'bytes': size, 'sha256': digest}

    def verify_all(self):
        for name in self.settings['inputs']:
            self.verify(name)

    def data(self, name):
        value = self.locate(name)
        return value.read_bytes() if isinstance(value, Path) else value

    def json(self, name):
        return json.loads(self.data(name))

    def rows(self, name):
        return [json.loads(line) for line in gzip.decompress(self.data(name)).splitlines()]

    def tensor(self, name):
        value = self.locate(name)
        return torch.load(value if isinstance(value, Path) else io.BytesIO(value),
                          map_location='cpu', weights_only=True)


def restore_models(checkpoints, settings, device):
    models, digests = {}, {}
    streams = {name: 42 for name in ('flow_projection', 'packet_projection', 'gate_hidden',
                                    'gate_output', 'head_hidden', 'head_output')}
    for name, saved in checkpoints.items():
        require(saved['epoch'] == settings['checkpoint_epochs'][name], 'Wrong selected checkpoint epoch')
        model = PairFusion(FusionHead(32, streams), name, 42)
        model.load_state_dict(saved['state'], strict=True)
        model.zero_grad(set_to_none=True)
        model.requires_grad_(False).to(device).eval()
        models[name], digests[name] = model, raw.state_digest(model.state_dict())
    return models, digests


def verify_frozen(models, digests):
    for name, model in models.items():
        require(raw.state_digest(model.state_dict()) == digests[name]
                and all(not p.requires_grad and p.grad is None for p in model.parameters())
                and all(not m.training for m in model.modules()), 'Fixed replay checkpoint changed: ' + name)


@torch.no_grad()
def batch_outputs(models, features):
    outputs = {}
    for checkpoint, model in models.items():
        tag = 'T' if checkpoint == 'pair_true' else 'S'
        residuals = {
            'true': model.readout(features.packet_mean, features.transitions, features.lengths)[0],
            'shuffle': model.readout(features.packet_mean, features.transitions, features.lengths,
                                     context_right=features.shuffled_right)[0]}
        off_scores, off_gate = model.fusion(features.flow, features.packet)
        native = residuals['true' if tag == 'T' else 'shuffle']
        packet_norm = features.packet.norm(dim=-1).clamp_min(1e-12)
        for suffix, pairing in (('1', 'true'), ('2', 'shuffle'), ('0', 'off')):
            residual = native if pairing == 'off' else residuals[pairing]
            injected = torch.zeros_like(residual) if pairing == 'off' else residual
            scores, gate = (off_scores, off_gate) if pairing == 'off' else model.fusion(
                features.flow, features.packet + residual)
            require(bool(torch.isfinite(scores).all() and torch.isfinite(gate).all()
                         and torch.isfinite(residual).all()), 'Nonfinite fixed-checkpoint replay output')
            outputs[tag + suffix] = {'scores': scores, 'gate': gate, 'residual': injected,
                'raw_residual_l2': residual.norm(dim=-1), 'injected_residual_l2': injected.norm(dim=-1),
                'residual_packet_ratio': injected.norm(dim=-1) / packet_norm,
                'projected_injected_residual_l2': F.linear(injected, model.fusion.packet_projection.weight).norm(dim=-1),
                'gate_delta': gate - off_gate}
    return outputs


@torch.no_grad()
def replay(models, cache, rows, settings, device, output):
    all_rows = []
    consistency = {tag: {'samples': 0, 'prediction_mismatches': 0,
                        'max_abs_log_probability_difference': 0.0,
                        'max_abs_residual_difference': 0.0, 'max_abs_gate_difference': 0.0}
                   for tag in ('T', 'S')}
    for start in range(0, len(rows), settings['eval_batch_size']):
        end = min(start + settings['eval_batch_size'], len(rows))
        features, _ = cache.batch(range(start, end), device)
        outputs = batch_outputs(models, features)
        unchanged = torch.tensor([r['features_changed'] == 0 for r in cache.records[start:end]], device=device)
        for tag in ('T', 'S'):
            if bool(unchanged.any()):
                target, a, b = consistency[tag], outputs[tag + '1'], outputs[tag + '2']
                target['samples'] += int(unchanged.sum())
                target['prediction_mismatches'] += int((a['scores'].argmax(-1)[unchanged] != b['scores'].argmax(-1)[unchanged]).sum())
                for field, key in (('scores', 'max_abs_log_probability_difference'),
                                   ('residual', 'max_abs_residual_difference'), ('gate', 'max_abs_gate_difference')):
                    target[key] = max(target[key], float((a[field][unchanged] - b[field][unchanged]).abs().max()))
        current = [{**row, 'pairing': pair, 'actual_features_changed': pair['features_changed'] > 0,
                    'predictions': {}, 'diagnostics': {}}
                   for row, pair in zip(rows[start:end], cache.records[start:end])]
        for group in GROUPS:
            values = outputs[group]
            scores, gates, delta = (values[key].cpu().tolist() for key in ('scores', 'gate', 'gate_delta'))
            scalar_fields = ('raw_residual_l2', 'injected_residual_l2', 'residual_packet_ratio', 'projected_injected_residual_l2')
            scalars = {key: values[key].cpu().tolist() for key in scalar_fields}
            predictions = values['scores'].argmax(-1).cpu().tolist()
            for i, row in enumerate(current):
                row['predictions'][group] = predictions[i]
                row['diagnostics'][group] = {key: scalars[key][i] for key in scalar_fields}
                row['diagnostics'][group].update(log_probabilities=scores[i], flow_gate_weights=gates[i],
                    flow_gate_delta_from_residual_off=delta[i], flow_gate_mean=sum(gates[i]) / len(gates[i]),
                    gate_mean_abs_change_from_off=sum(abs(x) for x in delta[i]) / len(delta[i]))
        all_rows.extend(current)
        if start == 0 or end % (200 * settings['eval_batch_size']) == 0 or end == len(rows):
            raw.progress(output, 'six_group_validation_replay', samples=end, total=len(rows),
                         validation_groups=6, training_epochs=0, optimizer_updates=0)
    for values in consistency.values():
        values['passed'] = all(values[key] == 0 for key in ('prediction_mismatches',
            'max_abs_log_probability_difference', 'max_abs_residual_difference', 'max_abs_gate_difference'))
    raw.write(output / 'unchanged_feature_consistency.json', consistency)
    require(all(values['passed'] for values in consistency.values()), 'Unchanged-feature pairing outputs differ')
    return all_rows, consistency


def exports(records):
    return {group: original.prediction_records(records, [r['predictions'][group] for r in records]) for group in GROUPS}


def verify_native(records, references, checkpoints, settings, output):
    exported, receipts = exports(records), {}
    for group, checkpoint in settings['native_archive_groups'].items():
        items = exported[group]
        metric = frozen.joint.study.validate_predictions(items, references[checkpoint], probabilities=False)
        receipts[group] = frozen.evaluation.replay_receipt(metric, items, checkpoints[checkpoint], references[checkpoint])
        receipts[group].update(checkpoint_epoch=checkpoints[checkpoint]['epoch'], checkpoint=checkpoint)
    raw.write(output / 'native_archive_replay.json', receipts)
    require(all(r['passed'] for r in receipts.values()), 'T1/S2 do not reproduce archive; interpret no cross-replay results')
    return receipts


def load_inputs(archive, settings, device):
    archive.verify_all()
    source = archive.json('provenance.json')
    require(archive.json('status.json')['status'] == 'completed'
            and archive.json('final_verification.json')['passed']
            and source['binding']['code_commit'] == settings['source_code_commit'], 'Wrong source run')
    for name in COMPUTE_FILES:
        canonical = (raw.ROOT / name).read_bytes().replace(b'\r\n', b'\n')
        require(hashlib.sha256(canonical).hexdigest() == source['binding']['source_files_sha256'][name],
                'Inference computation changed from archived code: ' + name)
    checkpoints = {name: archive.tensor(name + '/best.pt') for name in settings['checkpoint_epochs']}
    saved = archive.tensor('frozen_val_features.pt')
    binding = checkpoints['pair_true']['binding']
    require(checkpoints['pair_shuffle']['binding'] == binding
            and {k: v for k, v in binding.items() if k not in ('mapping_sha256', 'frozen_cache_sha256')} == saved['binding']
            == source['binding'], 'Checkpoint/cache source bindings differ')
    require(binding['frozen_cache_sha256']['val'] == settings['inputs']['frozen_val_features.pt']['sha256'], 'Wrong frozen validation cache')
    features = original.Features(*(saved['features'][key] for key in original.Features._fields))
    rows = [{'sample_id': sid, 'label': label, 'partition': 'val'} for sid, label in zip(saved['ids'], saved['labels'])]
    require(len(rows) == settings['validation_samples'] and len({r['sample_id'] for r in rows}) == len(rows), 'Wrong validation identities')
    cache = PairCache(original.FrozenCache([features], saved['labels']), rows, 'val', settings['pairing_seed'])
    require(cache.records == archive.rows('pairing_val.jsonl.gz')
            and cache.mapping_sha256 == binding['mapping_sha256']['val']
            == archive.json('pairing_val_receipt.json')['mapping_sha256'], 'Fixed validation shuffle mapping differs')
    references = {name: archive.rows(name + '/best_validation_predictions.jsonl.gz') for name in checkpoints}
    models, digests = restore_models(checkpoints, settings, device)
    for name in checkpoints:
        metric = frozen.joint.study.validate_predictions(references[name], rows, probabilities=False)
        require(metric == checkpoints[name]['validation']
                and frozen.joint.prediction_digest(references[name]) == checkpoints[name]['prediction_ids_sha256'], 'Archived native selection differs')
    return models, digests, cache, rows, references, checkpoints


def run(repo, output, settings, device_name='auto', archive_root=None, snapshot=False, resume=False):
    started = time.monotonic()
    require(settings['training_epochs'] == settings['optimizer_updates'] == 0 and settings['groups'] == list(GROUPS), 'Changed zero-training protocol')
    require(torch.__version__.split('+')[0] == '2.3.1' and torch_geometric.__version__ == '2.5.3', 'Expected Torch 2.3.1 / PyG 2.5.3')
    device = torch.device('cuda:0' if device_name == 'auto' and torch.cuda.is_available() else
                          'cpu' if device_name == 'auto' else device_name)
    require(device.type != 'cuda' or torch.cuda.is_available(), 'Requested CUDA is unavailable')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    archive = Archive(repo, settings, archive_root, snapshot)
    models, digests, cache, rows, references, checkpoints = load_inputs(archive, settings, device)
    environment = {'device': str(device), 'torch': str(torch.__version__), 'pyg': torch_geometric.__version__,
                   'python': platform.python_version(), 'platform': platform.platform()}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=raw.ROOT, text=True).strip()
    provenance = {'code_commit': commit, 'settings_sha256': raw.sha(CONFIG), 'environment': environment,
                  'inputs': archive.verified, 'source_result_commit': settings['source_result_commit'],
                  'source_code_commit': settings['source_code_commit'], 'checkpoint_state_sha256': digests,
                  'mapping_sha256': cache.mapping_sha256, 'validation_samples': len(rows),
                  'training_epochs': 0, 'optimizer_updates': 0, 'scope': settings['scope'],
                  'encoders_executed': False, 'frozen_encoder_features_reused': True,
                  'source_sha256': {name: raw.sha(raw.ROOT / name) for name in
                      (*COMPUTE_FILES, 'scripts/run_transition_pair_cross_replay.py', 'scripts/transition_pair_cross_report.py')}}
    if (output / 'provenance.json').exists():
        require(resume and raw.read(output / 'provenance.json') == provenance, 'Existing replay has different inputs/code/device; preserve it')
    raw.write(output / 'config.json', settings)
    raw.write(output / 'provenance.json', provenance)
    folder = output / 'checkpoints'; folder.mkdir(exist_ok=True)
    for name in checkpoints:
        (folder / (name + '.pt')).write_bytes(archive.data(name + '/best.pt'))
    (output / 'pairing_val.jsonl.gz').write_bytes(archive.data('pairing_val.jsonl.gz'))
    records, consistency = replay(models, cache, rows, settings, device, output)
    for group, values in exports(records).items():
        folder = output / group; folder.mkdir(exist_ok=True)
        raw.save_predictions(folder / 'validation_predictions.jsonl.gz', values)
    receipts = verify_native(records, references, checkpoints, settings, output)
    from transition_pair_cross_report import write_results
    result = write_results(output, records, settings, receipts, consistency)
    verify_frozen(models, digests)
    archive.verify_all()
    raw.write(output / 'final_verification.json', {'passed': True, 'native_archive_predictions_identical': True,
        'unchanged_feature_outputs_identical': True, 'all_checkpoints_frozen_eval_unchanged': True,
        'inputs_unchanged': True, 'validation_samples_per_group': len(rows), 'groups': len(GROUPS),
        'training_epochs': 0, 'optimizer_updates': 0, 'test_evaluated': False, 'seconds': time.monotonic() - started})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--archive-root', type=Path)
    parser.add_argument('--snapshot', action='store_true', help='Read the pinned Git snapshot and pre-fetched LFS objects')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    repo = args.repo.resolve(); output = repo / 'outputs' / OUTPUT_NAME
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / '.run.lock'), timeout=0):
        try:
            raw.write(output / 'status.json', {'status': 'running', 'training_epochs': 0, 'optimizer_updates': 0, 'test_evaluated': False})
            run(repo, output, raw.read(CONFIG), args.device, args.archive_root, args.snapshot, args.resume)
        except BaseException as exc:
            raw.write(output / 'status.json', {'status': 'failed', 'exception': repr(exc), 'traceback': traceback.format_exc(),
                      'training_epochs': 0, 'optimizer_updates': 0, 'test_evaluated': False})
            raise
        else:
            status = {'status': 'completed', 'training_epochs': 0, 'optimizer_updates': 0, 'validation_groups': 6,
                      'validation_samples_per_group': 13446, 'test_evaluated': False}
            raw.write(output / 'status.json', status)
            print(json.dumps({**status, 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
