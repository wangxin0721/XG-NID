"""Byte-level evidence checks; never imports torch or deserializes weights."""
from pathlib import Path
import json
from e3_gate_study import read, sha, require, inside, reference_metadata
from fusion_bh import CONTROLS as B512_CONTROLS, VARIANT
CONTROLS = {**B512_CONTROLS, 'batch_size': 1024}

HERE = Path(__file__).resolve().parent

def metadata(args):
    config = read(args.config)
    require(config['schema'] == 'xgnid.fusion-bh-b1024-study.v1' and config['variant'] == VARIANT
            and config['controls'] == CONTROLS and config['test_evaluated'] is False
            and config['automatic_additional_training'] is False, 'B-H configuration changed')
    root = inside(args.repo, config['acb_root'])
    summary = read(root / 'summary.json')
    require(sha(root / 'summary.json') == config['acb_summary_sha256'], 'Bound ACB summary changed')
    require(summary['status'] == 'completed' and summary['test_evaluated'] is False, 'ACB incomplete')
    old_binding = summary['binding']
    for name, digest in old_binding['sources'].items():
        require(sha(HERE / name) == digest, 'Historical source changed: ' + name)
    require(sha(args.reference_config) == old_binding['reference_config_sha256'], 'Reference config changed')
    spec = read(args.reference_config)
    reference = reference_metadata(args.repo, spec)
    require(reference == old_binding['reference'], 'Frozen graph/source/reference mismatch')
    baselines = {}
    for name, ref in config['baselines'].items():
        path = inside(args.repo, ref['path'])
        require(sha(path) == ref['sha256'], 'Baseline metrics changed: ' + name)
        require(read(path)['best_validation']['macro_f1'] == ref['macro_f1'], 'Baseline score differs')
        baselines[name] = ref['macro_f1']
    receipts = {}
    for name in ('A_equal', 'C_global', 'B_dynamic'):
        folder = root / name
        status, metrics, cfg = [read(folder / f) for f in ('status.json','metrics.json','config.json')]
        require((status['status'], status['exit_code']) == ('completed',0), 'ACB run incomplete')
        require(metrics == summary['models'][name] and cfg['binding'] == old_binding, 'ACB records disagree')
        require(sha(folder/'config.json') == metrics['config_sha256'], 'ACB config changed')
        require(sha(folder/'diagnostics.json') == metrics['diagnostics_sha256'], 'ACB diagnostics changed')
        diag = read(folder/'diagnostics.json')
        require(sha(folder/'validation_predictions.jsonl.gz') == diag['predictions_sha256'], 'ACB predictions changed')
        for f, digest in metrics['checkpoint_sha256'].items():
            require(sha(folder/f) == digest, 'ACB checkpoint bytes changed')
        epochs = [json.loads(line) for line in (folder/'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
        require([r['epoch'] for r in epochs] == list(range(1,metrics['stopped_epoch']+1)), 'Incomplete ACB epochs')
        require(sha(folder/'epochs.jsonl') == config['acb_epoch_sha256'][name], 'Registered ACB epoch orders changed')
        receipts[name] = {'epochs_sha256':sha(folder/'epochs.jsonl'), 'predictions_sha256':diag['predictions_sha256']}
    cert = read(root/'B_dynamic/metrics.json')['initialization']
    require(cert == read(root/'probe/summary.json')['variants']['B_dynamic']['initialization'], 'B probe initialization differs')
    zpath = inside(args.repo,config['z_diagnostic'])
    z = read(zpath)
    require(sha(zpath) == old_binding['candidate_evidence']['E3a_equal_init']['diagnostic_sha256'], 'Z diagnostic changed')
    require(sha(zpath.parent/'validation_cache.jsonl.gz') == z['cache_sha256'], 'Z prediction cache changed')
    # The completed prediction evidence suffices; do not depend on stale local Z weights/logs.
    sources = ('fusion_bh.py','fusion_bh_metadata.py','run_fusion_bh_server.py','analyze_fusion_bh_results.py',
               'analyze_fusion_acb_results.py','fusion_bh_b1024_metadata.py',
               'run_fusion_bh_b1024_server.py','analyze_fusion_bh_b1024_results.py')
    binding = {'config_sha256':sha(args.config), 'reference':reference,
               'acb_summary_sha256':config['acb_summary_sha256'], 'acb_receipts':receipts,
               'z_diagnostic_sha256':sha(zpath),'z_cache_sha256':z['cache_sha256'],
               'sources':{f:sha(HERE/f) for f in sources}, 'baseline_macro_f1':baselines,
               'B_initialization':cert, 'training_batch_size':1024,
               'comparison_note':'Historical B/C/E2/E0 use batch512; batch and update count differ'}
    return config, spec, binding, cert
