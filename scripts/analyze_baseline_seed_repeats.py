"""Validate four baseline repeats and summarize paired seeds42/43/44; stdlib."""
import json
import math
from pathlib import Path
import statistics
from e3_gate_study import inside, read, require, sha, matrix_metrics
from baseline_seed_repeats import JOBS, study_controls

LABELS=('Benign','WebBased','Spoofing','Recon','Mirai','Dos','DDos','BruteForce')

def paired_statistics(values):
    require(set(values)=={42,43,44} and all(set(v)=={'E0','E2'} for v in values.values()),'Need all paired seeds42/43/44')
    require(all(math.isfinite(x) and 0<=x<=1 for v in values.values() for x in v.values()),'Invalid Macro-F1')
    differences={str(s):100*(values[s]['E2']-values[s]['E0']) for s in (42,43,44)}
    model_stats={m:{'mean_percent':100*statistics.mean(values[s][m] for s in (42,43,44)),
                    'sample_std_percentage_points':100*statistics.stdev(values[s][m] for s in (42,43,44))}
                 for m in ('E0','E2')}
    return {'per_seed_E2_minus_E0':differences,'mean':statistics.mean(differences.values()),
            'sample_std':statistics.stdev(differences.values()),'units':'percentage_points',
            'E2_higher_seeds':sum(x>0 for x in differences.values()),
            'E0_higher_seeds':sum(x<0 for x in differences.values()),'models':model_stats,
            'three_seeds_are_initial_variability_evidence_only':True}

def analyze(repo,output,study,receipt):
    per_seed={s:{} for s in (42,43,44)};orders={}
    hashes={}
    for seed in (42,43,44):
        for model in ('E0','E2'):
            reference=study['references'][model]
            folder=(inside(repo,reference['directory'])/model if seed==42 else output/f'{model}_s{seed}'/model)
            metrics,cfg=read(folder/'metrics.json'),read(folder/'config.json')
            require(sha(folder/'config.json')==metrics['config_sha256'],'Run config changed')
            require(cfg['controls']==study_controls(study,seed),'Model/seed training controls differ')
            require((cfg['epochs'],cfg['workers'],cfg['initialization_batch_size'],cfg['evaluation_batch_size'])==(100,2,16,16),'Budget/batching differs')
            if seed==42:
                require(sha(folder/'metrics.json')==reference['metrics_sha256'],'Seed42 metrics changed')
            else:
                status=read(folder.parent/'status.json');summary=read(folder.parent/'summary.json')
                require((status['status'],status['exit_code'],status['mode'])==('completed',0,'full'),'Repeat incomplete')
                require(read(folder/'status.json')['status']=='completed','Model incomplete')
                require(summary['models'][model]==metrics and summary['binding']==cfg['binding'],'Run records disagree')
                require(metrics['test'] is None and metrics['test_evaluated'] is False and summary['test_evaluated'] is False,'New repeat evaluated test')
                require(cfg['binding']['repeat_study']==receipt and cfg['binding']['repeat_seed']==seed
                        and cfg['binding']['repeat_model']==model and cfg['binding']['seed42_initialization_reproduced'] is True,
                        'Run reference/seed binding differs')
                require(cfg['binding']['entry_sha256']==reference['entry_sha256'],'Wrong historical entry')
                require(cfg['binding']['initial_state_sha256']==metrics['initial_state_sha256']
                        and metrics['initial_state_sha256']!=reference['initial_state_sha256'],'New seed did not change initialization')
            for f,digest in metrics['checkpoint_sha256'].items():
                require(sha(folder/f)==digest,'Checkpoint bytes changed')
            history=[json.loads(line) for line in (folder/'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
            stop=metrics['stopped_epoch'];best_epoch=metrics['best_epoch']
            require([r['epoch'] for r in history]==list(range(1,stop+1)) and 1<=stop<=100,'Incomplete epoch history')
            require(all(r['train_samples']==107621 and r['optimizer_updates']==211
                        and r['cumulative_optimizer_updates']==211*r['epoch'] for r in history),'Update/sample accounting differs')
            scores=[r['val_macro_f1'] for r in history]
            require(all(math.isfinite(v) and 0<=v<=1 for v in scores),'Invalid validation history')
            require(best_epoch==scores.index(max(scores))+1 and (stop==100 or stop-best_epoch==12),'Strict best/early stopping differs')
            require(metrics['total_optimizer_updates']==stop*211 and metrics['training_samples_seen']==stop*107621,'Actual training amount differs')
            validation=metrics['best_validation'];rebuilt=matrix_metrics(validation['confusion_matrix'])
            require(abs(validation['macro_f1']-max(scores))<1e-12,'Selected score is not best validation')
            for field in ('accuracy','macro_f1','weighted_f1'):
                require(abs(rebuilt[field]-validation[field])<1e-12,'Matrix/metrics mismatch')
            matrix=validation['confusion_matrix']
            require(sum(map(sum,matrix))==13446,'Validation sample count differs')
            for i,label in enumerate(LABELS):
                tp=matrix[i][i];support=sum(matrix[i]);predicted=sum(row[i] for row in matrix)
                precision=tp/predicted if predicted else 0.;recall=tp/support if support else 0.
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.
                item=validation['per_class'][label]
                require(item['support']==support and all(abs(item[k]-v)<1e-12 for k,v in
                        [('precision',precision),('recall',recall),('f1-score',f1)]),'Per-class P/R/F1 differs')
            require(metrics['parameter_count']==(431704 if model=='E0' else 846800),'Architecture parameter count differs')
            per_seed[seed][model]={'macro_f1':validation['macro_f1'],'accuracy':validation['accuracy'],
                'weighted_f1':validation['weighted_f1'],'per_class':{c:validation['per_class'][c] for c in LABELS},
                'confusion_matrix':validation['confusion_matrix'],'best_epoch':best_epoch,'stopped_epoch':stop,
                'best_epoch_optimizer_updates':best_epoch*211,'total_optimizer_updates':stop*211,
                'training_samples_seen':stop*107621,'parameter_count':metrics['parameter_count'],
                'selected_validation_forward_seconds':validation.get('forward_seconds'),
                'training_validation_checkpoint_wall_seconds':metrics.get('training_validation_checkpoint_wall_seconds')}
            orders[(seed,model)]=[r['train_order_sha256'] for r in history]
            hashes[f'{model}_s{seed}']={'metrics_sha256':sha(folder/'metrics.json'),'epochs_sha256':sha(folder/'epochs.jsonl')}
    for seed in (42,43,44):
        first,second=orders[(seed,'E0')],orders[(seed,'E2')]
        overlap=min(len(first),len(second))
        require(first[:overlap]==second[:overlap],'Paired data permutations differ')
    values={s:{m:per_seed[s][m]['macro_f1'] for m in ('E0','E2')} for s in per_seed}
    stats=paired_statistics(values)
    classes={}
    for label in LABELS:
        classes[label]={'per_seed_E2_minus_E0_f1_percentage_points':{str(s):100*(per_seed[s]['E2']['per_class'][label]['f1-score']-per_seed[s]['E0']['per_class'][label]['f1-score']) for s in per_seed},
            'models':{m:{field:{'mean_percent':100*statistics.mean(per_seed[s][m]['per_class'][label][field] for s in per_seed),
                               'sample_std_percentage_points':100*statistics.stdev(per_seed[s][m]['per_class'][label][field] for s in per_seed)}
                         for field in ('precision','recall','f1-score')} for m in ('E0','E2')}}
    result={'schema':'xgnid.baseline-seed-repeat-analysis.v1','status':'completed','exit_code':0,
        'partition':'val','new_training_runs':4,'new_runs_test_evaluated':False,'prior_seed42_test_results_available':True,
        'seed42_was_available_during_prior_exploration':True,'per_seed':per_seed,
        'paired_macro_f1_percentage_points':stats,'per_class_summary':classes,'artifact_hashes':hashes,
        'paired_epoch_data_orders_match':True,'frozen_fusion_jobs_started':False,
        'interpretation':'Review paired direction, magnitude, class losses and computation costs before choosing the main baseline. Three seeds do not establish statistical significance; no automatic F0/F1/F2 scheduling.',
        'timing_scope':'GPU sharing permitted; reported timings are descriptive and not isolated cost benchmarks',
        'structural_comparison':'Original E0/E2 architectures differ in encoders, aggregation, heads and total/effective parameter counts; this is not a parameter-matched isolation of encoder count.'}
    lines=['# E0/E2 paired baseline repeats','', '| Seed | E0 Macro-F1 | E2 Macro-F1 | E2−E0 (pp) |','|---|---:|---:|---:|']
    for seed in (42,43,44):
        lines.append(f"| {seed} | {100*values[seed]['E0']:.6f}% | {100*values[seed]['E2']:.6f}% | {stats['per_seed_E2_minus_E0'][str(seed)]:+.6f} |")
    lines += ['',f"Paired mean difference: {stats['mean']:+.6f} pp; sample standard deviation: {stats['sample_std']:.6f} pp.",
              f"E0 mean: {stats['models']['E0']['mean_percent']:.6f}%, sample std: {stats['models']['E0']['sample_std_percentage_points']:.6f} pp.",
              f"E2 mean: {stats['models']['E2']['mean_percent']:.6f}%, sample std: {stats['models']['E2']['sample_std_percentage_points']:.6f} pp.",
              'Three seeds provide initial evidence of direction and variability; a small mean lead is not stable/significant superiority.',
              'Full class P/R/F1, confusions, epochs, optimizer updates, exposure and model means/std: analysis.json.',
              'Seed42 was already known during previous exploration. New runs use validation only. Shared-GPU timings are descriptive.',
              'No frozen-fusion experiments are automatically scheduled.']
    lines += ['', '| Seed | Model | Best/stop epoch | Updates | Training samples seen |',
              '|---|---|---:|---:|---:|']
    for seed in (42,43,44):
        for model in ('E0','E2'):
            r=per_seed[seed][model]
            lines.append(f"| {seed} | {model} | {r['best_epoch']}/{r['stopped_epoch']} | {r['total_optimizer_updates']} | {r['training_samples_seen']} |")
    lines += ['', '| Seed | Model | Class | Precision | Recall | F1 | Support |',
              '|---|---|---|---:|---:|---:|---:|']
    for seed in (42,43,44):
        for model in ('E0','E2'):
            for label in ('WebBased','Recon','BruteForce'):
                c=per_seed[seed][model]['per_class'][label]
                lines.append(f"| {seed} | {model} | {label} | {100*c['precision']:.4f}% | {100*c['recall']:.4f}% | {100*c['f1-score']:.4f}% | {int(c['support'])} |")
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
