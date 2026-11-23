"""Optional CPU preflight on hash-bound archived validation features; never formal training."""
import argparse
import json
from pathlib import Path
import platform
import subprocess

import torch
import torch_geometric

import run_residual_injection as run
from run_transition_pair_cross_replay import Archive


def probe(repo, output):
    settings = run.raw.read(run.CONFIG)
    run.validate_settings(settings)
    torch.set_num_threads(2)
    cross = run.raw.read(run.raw.ROOT / 'configs/transition_pair_cross_replay_s42_v1.json')
    spec = {**cross, 'inputs': {name: cross['inputs'][name] for name in ('provenance.json', 'frozen_val_features.pt')}}
    archive = Archive(repo, spec, snapshot=True); archive.verify_all()
    saved = archive.tensor('frozen_val_features.pt')
    source_binding = archive.json('provenance.json')['binding']['source']
    run.require(saved['binding']['source'] == source_binding, 'CPU probe cache source mismatch')
    source_config = run.original.base_config()
    checkpoint = source_config['checkpoints']['C']
    source_spec = {'source_result_commit': cross['source_result_commit'],
        'source_directory': source_config['source_directory'],
        'inputs': {checkpoint['path']: {'sha256': checkpoint['sha256'], 'bytes': 3823196}}}
    source_archive = Archive(repo, source_spec, snapshot=True); source_archive.verify_all()
    # Same trusted, hash-bound original-C archive and constructor used by load_source;
    # its saved metadata contains NumPy objects, as in the existing checkpoint loader.
    payload = torch.load(source_archive.locate(checkpoint['path']), map_location='cpu', weights_only=False)
    run.require(payload['epoch'] == checkpoint['epoch'] == 48, 'Wrong original C epoch')
    base = run.raw.OperatorClassifier('GAT', seed=42)
    source = run.frozen.joint.JointCrossLayerQueryClassifier(base.flow_encoder, base.packet_encoder, base.fusion, 'flow', 42)
    source.load_state_dict(payload['model_state'], strict=True)
    digest = run.frozen.independent.freeze_c(source)
    run.require(digest == source_binding['frozen_c_sha256'], 'Full original C state differs from source binding')
    features = run.original.Features(*(saved['features'][key] for key in run.original.Features._fields))
    cache = run.original.FrozenCache([features], saved['labels'])
    rows = [{'sample_id': sid, 'label': label, 'partition': 'val'} for sid, label in zip(saved['ids'], saved['labels'])]
    run.require(len(rows) == 13446 and len({r['sample_id'] for r in rows}) == 13446, 'Wrong validation membership')
    reference = run.original.prediction_records(rows, [r['prediction'] for r in saved['reference']])
    run.require(run.frozen.joint.prediction_digest(reference) == source_binding['validation_digest'],
                'Original C validation identity digest differs')
    device, arms = torch.device('cpu'), run.training.make_arms(source, settings, torch.device('cpu'))
    predictions, max_errors = {name: [] for name in arms}, {name: 0.0 for name in arms}
    with torch.no_grad():
        for start in range(0, len(rows), 16):
            batch, _ = cache.batch(range(start, min(start + 16, len(rows))), device)
            baseline = run.original.baseline(source, batch)
            for name, arm in arms.items():
                arm.eval(); out = arm.detailed_forward(batch)
                run.require(torch.equal(out.scores, baseline) and not bool(out.residual.any())
                            and bool((out.beta == 1).all()), name + ' epoch-zero mismatch')
                predictions[name].extend(out.scores.argmax(-1).tolist())
                max_errors[name] = max(max_errors[name], float((out.scores - baseline).abs().max()))
    for name, values in predictions.items():
        run.require(run.original.prediction_records(rows, values) == reference, name + ' archived C predictions differ')
    counts = torch.bincount(torch.repeat_interleave(torch.arange(len(rows)), features.lengths)[
        features.transitions[:, 70].bool()], minlength=len(rows))
    indices = [next(i for i, row in enumerate(rows) if row['label'] == label and counts[i] > 0)
               for label in sorted(set(saved['labels'])) if any(row['label'] == label and counts[i] > 0
               for i, row in enumerate(rows))]
    batch, labels = cache.batch(indices, device)
    gradients = run.training.gradient_probe(source, batch, labels, settings, device)
    run.training.parent.verify_source(source, digest)
    archive.verify_all(); source_archive.verify_all()
    output.mkdir(parents=True, exist_ok=True)
    run.raw.write(output / 'real_cached_cpu_preflight.json', {'passed': True, 'samples_per_arm': len(rows),
        'arms': list(arms), 'formal_training_started': False, 'formal_optimizer_updates': 0,
        'encoders_executed': False, 'raw_graph_replay': False, 'beta_all_one': True,
        'zero_residual_all_samples': True, 'max_abs_log_probability_difference': max_errors,
        'original_C_source_binding': source_binding, 'source_checkpoint_sha256': checkpoint['sha256'],
        'input_sha256': {**archive.verified, **source_archive.verified},
        'configuration_sha256': run.raw.sha(run.CONFIG),
        'source_files_sha256': {name: run.raw.sha(run.raw.ROOT / name) for name in run.SOURCE_FILES},
        'prediction_ids_sha256': source_binding['validation_digest'], 'source_state_unchanged': True,
        'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                        'pyg': torch_geometric.__version__, 'device': 'cpu'}})
    run.raw.write(output / 'real_cached_cpu_gradients.json', gradients)
    print(json.dumps({'passed': True, 'samples_per_arm': len(rows), 'formal_training_started': False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=run.raw.ROOT / 'docs/residual_injection_s42_v1')
    args = parser.parse_args()
    probe(args.repo.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
