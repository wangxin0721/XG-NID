"""Offline candidate scans and identity-paired E3 comparisons. Standard library only.

Reads existing validation caches; no torch, checkpoint, inference, optimizer or
test data. Label-informed scans describe grid points, never oracle predictions.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import traceback

LABELS = ('Benign','WebBased','Spoofing','Recon','Mirai','Dos','DDos','BruteForce')
OUTCOMES = ('repaired','regressed','both_correct','both_wrong')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def records(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def prediction(p):
    require(len(p)==8 and all(math.isfinite(x) and 0<=x<=1 for x in p) and abs(sum(p)-1)<1e-5, 'Invalid probabilities')
    return max(range(8), key=lambda k:p[k])


def margin(p, y):
    return p[y]-max(v for k,v in enumerate(p) if k!=y)


def outcome(old_correct, new_correct):
    return 'both_correct' if old_correct and new_correct else 'regressed' if old_correct else 'repaired' if new_correct else 'both_wrong'


def correctness(flow, packet, label):
    return 'both_correct' if flow==packet==label else 'flow_only_correct' if flow==label else 'packet_only_correct' if packet==label else 'both_wrong'


def summary(values):
    if not values:
        return {'n':0,'mean':None,'min':None,'p10':None,'median':None,'p90':None,'max':None}
    require(all(math.isfinite(x) for x in values), 'Nonfinite statistic')
    ordered=sorted(values)
    def q(f):
        pos=(len(ordered)-1)*f
        low=int(pos); high=min(low+1,len(ordered)-1)
        return ordered[low]+(ordered[high]-ordered[low])*(pos-low)
    return {'n':len(values),'mean':sum(values)/len(values),'min':ordered[0], 'p10':q(.1),'median':q(.5),'p90':q(.9),'max':ordered[-1]}


def sampled_runs(preds, label):
    result=[]
    for i,p in enumerate(preds):
        if p==label:
            if i and preds[i-1]==label:
                result[-1][1]=i/20
            else:
                result.append([i/20,i/20])
    return result


def scan_stats(row):
    y=row['label']; p=row['probabilities']; grid=row['alpha_flow_grid_predictions']; gaps=row['alpha_flow_grid_true_class_margins']
    require(len(grid)==len(gaps)==21 and all(isinstance(v,int) and 0<=v<8 for v in grid), 'Invalid scan grid')
    require(all(math.isfinite(v) and -1<=v<=1 for v in gaps), 'Invalid scan margins')
    require(grid[0]==prediction(p['packet']) and grid[20]==prediction(p['flow']), 'Scan endpoints disagree with H(z) branches')
    require(abs(gaps[0]-margin(p['packet'],y))<1e-6 and abs(gaps[20]-margin(p['flow'],y))<1e-6, 'Endpoint margins disagree')
    require(grid[10]==prediction(p['equal']) and abs(gaps[10]-margin(p['equal'],y))<1e-6, 'Equal grid point differs')
    for predicted,gap in zip(grid,gaps):
        require((gap<=0 or predicted==y) and (gap>=0 or predicted!=y), 'Grid prediction/margin inconsistent')
    require(sampled_runs(grid,y)==row['correct_grid_runs_not_continuous_intervals'], 'Cached grid runs differ')
    w=row['final_weights']
    require(len(w)==2 and all(math.isfinite(v) and 0<=v<=1 for v in w) and abs(sum(w)-1)<1e-5,'Invalid final weight')
    indices=[i for i,pred in enumerate(grid) if pred==y]
    return {'flow_weight':w[0], 'packet_weight':w[1],
            'fusion_true_class_probability_margin':margin(p['fusion'],y),
            'flow_true_class_probability_margin':margin(p['flow'],y),
            'packet_true_class_probability_margin':margin(p['packet'],y),
            'nearest_correct_sampled_alpha_distance':min(abs(w[0]-i/20) for i in indices) if indices else None,
            'correct_grid_points':len(indices),'correct_grid_runs':len(sampled_runs(grid,y))}


def direction_values(old, new):
    values={}
    for prefix,probs,mapping in [('E3',old,{'flow':'flow_only','packet':'packet_only','fusion':'dynamic'}),
                                 ('candidate',new,{'flow':'flow','packet':'packet','fusion':'fusion'})]:
        for branch,key in mapping.items():
            values[prefix+'_'+branch+'_pWeb_minus_pRecon']=probs[key][1]-probs[key][3]
    return values


def analyze(original_path, candidate_path, output):
    old_report,new_report=read(original_path),read(candidate_path)
    require((old_report['schema'],old_report['status'],old_report['exit_code'])==('xgnid.e3-gate-diagnostic.v1','completed',0),'Original diagnosis incomplete')
    require((new_report['schema'],new_report['status'],new_report['exit_code'],new_report['partition'])==('xgnid.gate-candidate-diagnostic.v1','completed',0,'val'),'Candidate diagnosis incomplete')
    require(new_report['variant'] in ('E3a','E3a_equal_init') and new_report['saved_validation_reproduced'] is True and new_report['test_evaluated'] is False,'Wrong candidate state')
    for path in (original_path,candidate_path):
        status=read(path.parent/'status.json')
        require((status['status'],status['exit_code'])==('completed',0),'Source process incomplete')
    require(new_report['receipt']['reference']==old_report['reference'] and
            new_report['receipt']['diagnostic']['report_sha256']==sha(original_path),'Reference or original diagnostic identity differs')
    paths={'original':original_path.parent/'E3'/'validation_cache.jsonl.gz',
           'candidate':candidate_path.parent/'validation_cache.jsonl.gz',
           'scan':candidate_path.parent/'webbased_packet_only_scan.jsonl.gz'}
    digests={'original':old_report['models']['E3']['cache_sha256'],'candidate':new_report['cache_sha256'],'scan':new_report['scan_sha256']}
    for k,path in paths.items():
        require(sha(path)==digests[k], 'Cache SHA differs: '+k)
    require(not output.exists(),'Use a fresh output directory')
    output.mkdir(parents=True)
    write(output/'status.json',{'status':'running','mode':'offline_cache_analysis'})
    try:
        old_rows={}; old_ids=[]
        matrices={key:[[0]*8 for _ in range(8)] for key in ('E3','candidate','flow','packet','equal')}
        for row in records(paths['original']):
            sid,y=row['sample_id'],row['label']
            require(sid not in old_rows and isinstance(y,int) and 0<=y<8,'Original duplicate/invalid identity')
            pred=prediction(row['probabilities']['dynamic'])
            for key in ('flow_only','packet_only'): prediction(row['probabilities'][key])
            matrices['E3'][y][pred]+=1
            old_rows[sid]={'label':y,'node_hash':row['node_hash'],'probabilities':row['probabilities']}
            old_ids.append((sid,y))
        require(identity(old_ids)==old_report['models']['E3']['ordered_validation_identity_sha256'],'Original order hash differs')
        paired={label:Counter({k:0 for k in OUTCOMES}) for label in LABELS}
        transitions={label:Counter() for label in LABELS}
        group_counts={label:{k:Counter({'samples':0,'fusion_correct':0,'fusion_wrong':0}) for k in ('both_correct','flow_only_correct','packet_only_correct','both_wrong')} for label in LABELS}
        packet_values={'success':defaultdict(list),'failure':defaultdict(list)}
        packet_grids={'success':Counter(),'failure':Counter()}
        direction_groups=defaultdict(lambda:defaultdict(list)); direction_votes=defaultdict(Counter)
        direction_changes={name:Counter({'persistent':0,'introduced':0,'resolved_to_correct':0,'changed_to_another_error':0}) for name in ('WebBased_to_Recon','Recon_to_WebBased')}
        candidate_ids=[]; seen=set(); scan_rows={}
        details=output/'paired_diagnostics.jsonl.gz'
        with gzip.open(details,'xt',encoding='utf-8') as stream:
            for row in records(paths['candidate']):
                sid,y=row['sample_id'],row['label']
                require(sid in old_rows and sid not in seen,'Candidate unknown/duplicate sample')
                before=old_rows[sid]
                require((y,row['node_hash'])==(before['label'],before['node_hash']),'Paired label/node hash differs')
                seen.add(sid); candidate_ids.append((sid,y)); probs=row['probabilities']
                preds={key:prediction(probs[key]) for key in ('fusion','flow','packet','equal')}
                old_pred=prediction(before['probabilities']['dynamic'])
                for key in preds: matrices['candidate' if key=='fusion' else key][y][preds[key]]+=1
                name=LABELS[y]; group=correctness(preds['flow'],preds['packet'],y)
                require(group==row['group'],'Candidate correctness group differs')
                state=outcome(old_pred==y,preds['fusion']==y); paired[name][state]+=1
                old_group=correctness(prediction(before['probabilities']['flow_only']),prediction(before['probabilities']['packet_only']),y)
                transitions[name][old_group+' -> '+group]+=1
                target=group_counts[name][group]; target['samples']+=1
                target['fusion_correct']+=int(preds['fusion']==y); target['fusion_wrong']+=int(preds['fusion']!=y)
                scans=scan_stats(row)
                for key in preds:
                    stats=row['classification_statistics'][key]
                    require(stats['prediction']==preds[key] and abs(stats['true_class_probability_margin']-margin(probs[key],y))<1e-6,'Cached classification statistics differ')
                if y==1 and group=='packet_only_correct':
                    require(row['alpha_flow_grid_predictions'][0]==y,'Packet-only endpoint is not correct')
                    scan_rows[sid]=identity(row)
                    bucket='success' if preds['fusion']==y else 'failure'
                    for key,value in scans.items():
                        require(value is not None,'Packet-only must have a correct endpoint')
                        packet_values[bucket][key].append(value)
                    for i,pred in enumerate(row['alpha_flow_grid_predictions']):
                        if pred==y: packet_grids[bucket][i]+=1
                if y in (1,3):
                    other=3 if y==1 else 1; direction=name+'_to_'+LABELS[other]
                    a,b=old_pred==other,preds['fusion']==other
                    if a or b:
                        category='persistent' if a and b else 'introduced' if b else 'resolved_to_correct' if preds['fusion']==y else 'changed_to_another_error'
                        direction_changes[direction][category]+=1
                        values=direction_values(before['probabilities'],probs)
                        subsets=[direction+'/'+category]+([direction+'/original_errors'] if a else [])+([direction+'/candidate_errors'] if b else [])
                        for subset in subsets:
                            for key,value in values.items(): direction_groups[subset][key].append(value)
                            for prefix,pp,mapping in [('E3',before['probabilities'],{'flow':'flow_only','packet':'packet_only','fusion':'dynamic'}),('candidate',probs,{'flow':'flow','packet':'packet','fusion':'fusion'})]:
                                for branch,key in mapping.items(): direction_votes[subset][prefix+'_'+branch+'_'+LABELS[prediction(pp[key])]]+=1
                stream.write(json.dumps({'sample_id':sid,'label':y,'E3_prediction':old_pred,'candidate_prediction':preds['fusion'],
                    'outcome':state,'E3_group':old_group,'candidate_group':group,'scan_statistics':scans,
                    'WebBased_minus_Recon_probability_differences':direction_values(before['probabilities'],probs) if y in (1,3) else None},allow_nan=False)+'\n')
        require(len(seen)==len(old_rows)==13446 and seen==set(old_rows),'Incomplete paired validation')
        require(identity(candidate_ids)==new_report['ordered_validation_identity_sha256']==identity(old_ids),'Candidate ordered identities differ')
        require(matrices['E3']==old_report['models']['E3']['metrics']['dynamic']['confusion_matrix'],'Original matrix differs')
        for key in ('candidate','flow','packet','equal'):
            require(matrices[key]==new_report['metrics']['fusion' if key=='candidate' else key]['confusion_matrix'],'Candidate matrix differs: '+key)
        for label in LABELS:
            for group in group_counts[label]:
                for key,v in group_counts[label][group].items():
                    require(v==new_report['groups_by_class'][label][group][key],'Group count differs')
        scan_seen=set()
        for row in records(paths['scan']):
            sid=row['sample_id']
            require(sid not in scan_seen and sid in scan_rows and identity(row)==scan_rows[sid],'Scan subset differs from full cache')
            scan_seen.add(sid)
        require(scan_seen==set(scan_rows) and len(scan_seen)==new_report['webbased_packet_only']['samples'],'Incomplete Packet-only scan subset')
        require(len(packet_values['failure']['packet_weight'])==new_report['webbased_packet_only']['fusion_wrong'],'Packet-only failure count differs')
        totals=Counter({key:sum(p[key] for p in paired.values()) for key in OUTCOMES})
        for label in LABELS:
            i=LABELS.index(label)
            require(paired[label]['repaired']-paired[label]['regressed']==matrices['candidate'][i][i]-matrices['E3'][i][i],'Paired net change differs')
        for key,path in paths.items(): require(sha(path)==digests[key],'Source cache changed during analysis')
        result={'schema':'xgnid.candidate-fusion-cache-analysis.v1','status':'completed','exit_code':0,'partition':'val','variant':new_report['variant'],
            'samples':len(seen),'report_sha256':{'original':sha(original_path),'candidate':sha(candidate_path)},'cache_sha256':digests,
            'paired_outcomes':totals,'paired_outcomes_by_class':paired,'branch_group_transitions_by_class':transitions,
            'WebBased_packet_only':{bucket:{'samples':len(packet_values[bucket]['packet_weight']),
                'statistics':{k:summary(v) for k,v in packet_values[bucket].items()},
                'correct_count_at_each_alpha_flow_grid_point':[packet_grids[bucket][i] for i in range(21)]} for bucket in packet_values},
            'WebBased_Recon_direction_changes':direction_changes,
            'WebBased_Recon_probability_gap_summaries':{subset:{k:summary(v) for k,v in values.items()} for subset,values in direction_groups.items()},
            'WebBased_Recon_branch_votes':direction_votes,'checks':{'cache_hashes_match':True,'identities_labels_nodes_match':True,'prediction_matrices_match':True,
                'all_scan_branch_endpoints_and_equal_points_match':True,'scan_subset_matches_full_cache':True,'paired_net_changes_match':True},
            'details_sha256':sha(details),'entry_sha256':sha(Path(__file__)),
            'interpretation':'Probability gaps (not raw logits); nearest successful sampled-alpha distance is descriptive, not an exact boundary or causal estimate. No per-sample best-weight prediction or oracle score. H(z) are joint-model interventions.',
            'gpu_touched':False,'model_loaded':False,'training_started':False,'test_evaluated':False}
        write(output/'report.json',result)
        write(output/'status.json',{'status':'completed','exit_code':0,'mode':'offline_cache_analysis'})
        return result
    except BaseException:
        write(output/'status.json',{'status':'failed','exit_code':1,'mode':'offline_cache_analysis'})
        (output/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-report',type=Path,required=True)
    p.add_argument('--candidate-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    r=analyze(args.original_report.resolve(),args.candidate_report.resolve(),args.output.resolve())
    print(json.dumps({'status':r['status'],'report':str(args.output/'report.json'),'paired_outcomes':r['paired_outcomes'],
        'WebBased_paired_outcomes':r['paired_outcomes_by_class']['WebBased'],'WebBased_packet_only':r['WebBased_packet_only'],
        'gpu_touched':False},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
