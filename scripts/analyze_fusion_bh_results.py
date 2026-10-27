"""Validation-only B-H decision report, using identity-aligned private caches."""
from e3_gate_study import read, sha, require, inside
from analyze_fusion_acb_results import predictions, paired, compare_metrics
from fusion_bh import VARIANT, decide

def analyze(repo, output, config, binding):
    folder = output / VARIANT
    status, metrics, cfg, diag = [read(folder/f) for f in ('status.json','metrics.json','config.json','diagnostics.json')]
    require((status['status'],status['exit_code']) == ('completed',0), 'B-H training incomplete')
    require(cfg['binding']==binding and sha(folder/'config.json')==metrics['config_sha256'], 'B-H config identity differs')
    require(sha(folder/'diagnostics.json')==metrics['diagnostics_sha256'], 'B-H diagnostic changed')
    require(sha(folder/'validation_predictions.jsonl.gz')==diag['predictions_sha256'], 'B-H predictions changed')
    for name, digest in metrics['checkpoint_sha256'].items():
        require(sha(folder/name)==digest, 'B-H checkpoint bytes changed')
    data, aggregate = predictions(folder/'validation_predictions.jsonl.gz')
    compare_metrics(aggregate['metrics']['fusion'],metrics['best_validation'])
    require(aggregate['groups']==diag['groups'] and aggregate['per_class']==diag['per_class'], 'Diagnostic counts differ')
    root = inside(repo,config['acb_root'])
    paired_results, old_data = {}, {}
    for name in ('B_dynamic','C_global'):
        path = root/name/'validation_predictions.jsonl.gz'
        require(sha(path)==binding['acb_receipts'][name]['predictions_sha256'], 'Comparison cache changed')
        old_data[name], old_aggregate = predictions(path)
        compare_metrics(old_aggregate['metrics']['fusion'],read(root/name/'metrics.json')['best_validation'])
        paired_results[name+'_to_BH']=paired(old_data[name],data)
    zpath = inside(repo,config['z_diagnostic']).parent/'validation_cache.jsonl.gz'
    require(sha(zpath)==binding['z_cache_sha256'], 'Z cache changed')
    z, _ = predictions(zpath)
    paired_results['Z_to_BH']=paired(z,data)
    b = old_data['B_dynamic']
    subset = [s for s,r in z.items() if r['label']==1 and r['group']=='both_wrong'
              and r['predictions']['fusion']==1 and b[s]['predictions']['fusion']!=1]
    require(len(subset)==64, 'Registered WebBased subset changed')
    recovered = sum(data[s]['predictions']['fusion']==1 for s in subset)
    score = aggregate['metrics']['fusion']['macro_f1']
    baseline = binding['baseline_macro_f1']
    result = {'schema':'xgnid.fusion-bh-analysis.v1','status':'completed','exit_code':0,
              'variant':VARIANT,'partition':'val','test_evaluated':False,
              'metrics':aggregate,'paired':paired_results,
              'registered_64_WebBased':{'samples':64,'recovered':recovered,'still_wrong':64-recovered},
              'B_to_BH_all_new_regressions':paired_results['B_dynamic_to_BH']['total']['regressed'],
              'baseline_macro_f1':baseline,'delta_percentage_points':{k:100*(score-v) for k,v in baseline.items()},
              'decision':decide(score,baseline),
              'interpretation':'H-derived statistics are not necessarily calibrated; new H-to-gate gradient is part of the intervention.'}
    decision = result['decision']['action']
    lines = ['# B-H seed42 validation decision','',f'Macro-F1: {100*score:.6f}%',f'Decision: {decision}',
             '', '| Control | Difference (percentage points) |','|---|---:|']
    lines += [f'| {k} | {v:+.6f} |' for k,v in result['delta_percentage_points'].items()]
    lines += ['',f'Recovered {recovered}/64 registered WebBased samples; all B-to-BH new regressions: {result["B_to_BH_all_new_regressions"]}.',
              'This exploratory seed does not establish stability or statistical significance. No further training is automatically scheduled.',
              'Full per-class precision/recall/F1, confusion matrices and paired rescue/regression counts are in analysis.json.']
    (output/'decision.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
