"""Read a precise published Git commit, resolving LFS, and recompute predictions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
import subprocess

import prepare_conditional_complement as prepare
import summarize_conditional_complement as report

raw,require,archive=prepare.raw,prepare.require,prepare.archive


def audit(repo,commit,remote='origin'):
    require(len(commit)==40 and all(c in '0123456789abcdef' for c in commit),'Audit requires an exact commit SHA')
    env=archive.lfs_environment(repo)
    subprocess.run(['git','fetch','--no-tags',remote,commit],cwd=repo,env=env,check=True)
    prefix='outputs/'+prepare.OUTPUT_NAME+'/'
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',commit,'--',prefix],cwd=repo,text=True,env=env).splitlines()
    selected=[n for n in names if n.endswith(('/summary.json','/predictions.jsonl.gz','/diagnosis.json',
        '/train_diagnosis.jsonl.gz','/val_diagnosis.jsonl.gz','/config.json'))]
    spec={'source_result_commit':commit,'source_directory':prefix.rstrip('/'),'inputs':{}}
    for path in selected:
        b=subprocess.check_output(['git','show',commit+':'+path],cwd=repo,env=env)
        if b.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
            lines=b.decode().splitlines();meta={'sha256':lines[1].split(':')[1],'bytes':int(lines[2].split()[1])}
        else:meta={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
        spec['inputs'][path[len(prefix):]]=meta
    require('config.json' in spec['inputs'],'Published configuration missing')
    # Fetch LFS only for the report inputs selected above; caches/weights aren't re-downloaded.
    inputs=archive.Archive(repo,spec,snapshot=True,env=env)
    missing=[prefix+n for n in spec['inputs'] if isinstance(inputs.locate(n),Path) and not inputs.locate(n).is_file()]
    if missing:subprocess.run(['git','lfs','fetch','--include='+','.join(missing),'--exclude=',remote,commit],cwd=repo,env=env,check=True)
    inputs.verify_all();settings=inputs.json('config.json');audited={};preparations={};seed_summaries={}
    for n in spec['inputs']:
        if n.endswith('/preparation/diagnosis.json'):
            summary=inputs.json(n);base=n.rsplit('/',1)[0]
            for part in ('train','val'):
                rows=inputs.rows(base+'/'+part+'_diagnosis.jsonl.gz')
                require(all(r['teacher_group']==report.group(r['label'],r['teacher_predictions']['flow'],r['teacher_predictions']['packet']) for r in rows),
                        'Teacher grouping differs from exported teacher predictions')
                for teacher in ('teacher_flow','teacher_packet'):report.check_metric(report.metrics(rows,teacher),summary[part]['teacher_metrics'][teacher])
                require(report.subgroup(rows,'source_ordered')==summary[part]['groups'],'Published preparation groups differ')
                opportunities=sum(r['teacher_group'] in report.GROUPS[:2] and r['predictions']['source_ordered']!=r['label'] for r in rows)
                require(opportunities==summary[part]['repairable_disagreement_samples'],'Published disagreement count differs')
            preparations[base]={'passed':True,'train_samples':settings['training_samples'],'val_samples':settings['validation_samples']}
        if not n.endswith('/summary.json'):continue
        summary=inputs.json(n)
        if summary.get('schema')!='xgnid.conditional-complement-results.v1':continue
        base=n.rsplit('/',1)[0];part='test' if summary['test_evaluated'] else 'validation'
        seed_summaries[(summary['partition'],summary['backbone_seed'])]=summary
        rows=inputs.rows(base+'/'+part+'/predictions.jsonl.gz')
        require(all(r['teacher_group']==report.group(r['label'],r['teacher_predictions']['flow'],r['teacher_predictions']['packet']) for r in rows),
                'Published teacher grouping differs')
        require(len(rows)==summary['samples'] and len({r['sample_id'] for r in rows})==len(rows),'Published cohort differs')
        for model,m in summary['metrics'].items():
            report.check_metric(report.metrics(rows,model),m)
            require(report.subgroup(rows,model)==summary['groups'][model],'Published groups differ')
        for key,c in summary['comparisons'].items():
            a,b=key.split('-');measured,_=report.compare(rows,b,a)
            require(all(measured[k]==c[k] for k in ('counts','teacher_groups','per_class')),'Published pair counts differ')
            delta=100*(summary['metrics'][a]['macro_f1']-summary['metrics'][b]['macro_f1'])
            require(delta==c['delta_macro_f1_pp'],'Published paired difference differs')
        if 'decision' in summary:require(report.decision(summary,settings)==summary['decision'],'Published decision differs')
        audited[base]={'passed':True,'samples':len(rows),'models':len(summary['metrics']),'pairs':len(summary['comparisons'])}
    aggregates={}
    def verify_stats(saved,values):
        require(saved['values']==values and math.isclose(saved['mean'],mean(values),rel_tol=0,abs_tol=2e-15)
            and math.isclose(saved['sample_std'],stdev(values),rel_tol=0,abs_tol=2e-15),'Published mean/sample standard deviation differs')
    for n in spec['inputs']:
        if not n.endswith('/aggregate/summary.json') and n!='aggregate/summary.json':continue
        summary=inputs.json(n);part=summary['partition']
        for scope in ('all_three','new_replications_only'):
            group=summary[scope];seeds=group['backbone_seeds']
            for model,items in group['models'].items():
                for metric,saved in items.items():verify_stats(saved,[seed_summaries[(part,s)]['metrics'][model][metric] for s in seeds])
            for key,saved in group['paired_macro_f1_pp'].items():
                a,b=key.split('-')
                verify_stats(saved,[100*(seed_summaries[(part,s)]['metrics'][a]['macro_f1']-
                    seed_summaries[(part,s)]['metrics'][b]['macro_f1']) for s in seeds])
            for model,classes in group['per_class'].items():
                for label,items in classes.items():
                    for metric,saved in items.items():verify_stats(saved,[seed_summaries[(part,s)]['metrics'][model]['per_class'][label][metric] for s in seeds])
        if part=='val':
            c=summary['candidate'];m=summary['matched_control'];group=summary['all_three']['paired_macro_f1_pp']
            per_seed={s:report.qualify(seed_summaries[(part,s)],c,settings) for s in (42,43,44)}
            checks={'candidate_T0_all_positive':all(v>0 for v in group[c+'-cc_t0']['values']),
                'candidate_T0_mean':group[c+'-cc_t0']['mean']>=.10,'candidate_MLP_mean':group[c+'-'+m]['mean']>=.05,
                'candidate_MLP_new_both_positive':all(v>0 for v in summary['new_replications_only']['paired_macro_f1_pp'][c+'-'+m]['values']),
                'focus_classes':all(q['checks']['focus_classes'] for q in per_seed.values()),
                'destruction':all(q['checks']['destruction'] for q in per_seed.values())}
            require(checks==summary['replication_checks'] and all(checks.values())==summary['passed'],'Published replication gate differs')
        aggregates[n]={'passed':True,'scope':'per-backbone paired means and n-1 sample standard deviations'}
    return {'passed':True,'result_commit':commit,'published_prediction_audits':audited,
        'published_preparation_audits':preparations,'published_aggregate_audits':aggregates,'archive_payload_sha256':inputs.verified,
        'method':'reread immutable Git/LFS bytes; integer confusion counts and paired sample transitions',
        'not_a_new_original_graph_replay':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--remote',default='origin');p.add_argument('--output',type=Path)
    a=p.parse_args();result=audit(a.repo.resolve(),a.commit,a.remote)
    if a.output:raw.write(a.output,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
