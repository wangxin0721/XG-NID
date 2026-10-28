"""Stdlib evidence checks and narrow adapters over the exact seed42 trainers."""
import ast
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import random
import textwrap
from e3_gate_study import inside, read, require, sha, identity

JOBS = [('E0',43),('E2',43),('E0',44),('E2',44)]
HERE = Path(__file__).resolve().parent

def progress(args, phase, epoch, batches):
    path=args.output/'progress.json'
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'updated_utc':datetime.now(timezone.utc).isoformat(),
        'model':args.model,'seed':args.seed,'phase':phase,'epoch':epoch,
        'batches_done':batches,'expected_batches':211},indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)

def metadata(args):
    study=read(args.config)
    require(study['schema']=='xgnid.baseline-seed-repeats.v1' and set(study['references'])=={'E0','E2'}
            and [(r['model'],r['seed']) for r in study['jobs']]==JOBS
            and study['test_evaluated'] is False and study['automatic_frozen_fusion_jobs'] is False,
            'Only the registered four validation runs are allowed')
    expected={'train_batch_size':512,'evaluation_batch_size':16,'initialization_batch_size':16,
              'lr':.001,'optimizer':'Adam','weight_decay':0.,'epochs':100,'patience':12,'workers':2}
    require(all(study.get(k)==v for k,v in expected.items()),'Repeated baseline controls changed')
    receipts={}
    for model,ref in study['references'].items():
        folder=inside(args.repo,ref['directory'])
        cfg=read(folder/model/'config.json');metrics=read(folder/model/'metrics.json')
        for relative,digest in [(ref['entry'],ref['entry_sha256']),(ref['study_config'],ref['study_config_sha256'])]:
            require(sha(HERE.parent/relative)==digest,'Frozen historical snapshot differs: '+relative)
        require(ref['entry_sha256']==cfg['binding']['entry_sha256'] and
                ref['study_config_sha256']==cfg['binding']['config_sha256'],'Snapshot is not the seed42 version')
        for name,key in [('summary.json','summary_sha256'),(model+'/config.json','config_sha256'),
                         (model+'/metrics.json','metrics_sha256'),(model+'/epochs.jsonl','epochs_sha256')]:
            require(sha(folder/name)==ref[key],'Seed42 evidence changed: '+str(folder/name))
        require(read(folder/'summary.json')['models'][model]==metrics,'Seed42 summary/metrics differ')
        status=read(folder/'status.json')
        require((status['status'],status['exit_code'])==('completed',0),'Seed42 baseline incomplete')
        for name,digest in ref['sources'].items():
            require(sha(inside(args.repo,name))==digest,'Frozen model/data source changed: '+name)
        for name,digest in ref['checkpoint_sha256'].items():
            require(sha(folder/model/name)==digest,'Seed42 checkpoint bytes changed')
        require(cfg['controls']=={**study_controls(study,42)} and cfg['epochs']==100
                and cfg['workers']==2 and cfg['evaluation_batch_size']==16 and cfg['initialization_batch_size']==16,
                'Seed42 baseline training settings differ')
        require(cfg['binding']['split_id']==study['split_id'],'Seed42 split differs')
        receipts[model]={'metrics_sha256':ref['metrics_sha256'],'config_sha256':ref['config_sha256'],
                         'entry_sha256':ref['entry_sha256'],'study_config_sha256':ref['study_config_sha256']}
    split_dir=inside(args.repo,study['split_directory']);split=read(split_dir/'split.json')
    require(split['split_id']==study['split_id'] and split['partition_sizes']=={'train':107621,'val':13446,'test':13448},'Wrong split')
    require(sha(inside(split_dir,split['sample_manifest']))==split['sample_manifest_sha256'],'Sample manifest changed')
    for model,ref in study['references'].items():
        cfg=read(inside(args.repo,ref['directory'])/model/'config.json')
        require(cfg['binding']['samples_sha256']==split['sample_manifest_sha256'],'Seed42 manifest differs')
    files=('baseline_seed_repeats.py','run_baseline_seed_repeats_server.py','analyze_baseline_seed_repeats.py')
    binding={'study_sha256':sha(args.config),'seed42':receipts,'split_id':study['split_id'],
             'manifest_sha256':split['sample_manifest_sha256'],
             'adapter_sources':{f:sha(HERE/f) for f in files},'jobs':study['jobs']}
    return study,binding

def study_controls(study,seed):
    return {'seed':seed,'batch_size':512,'lr':.001,'hidden_dim':64,'optimizer':'Adam',
            'early_stop_patience':12,'early_stop_min_delta':0.,'auxiliary_losses':False,
            'calibration':False,'class_weighting':False,'hard_negatives':False,'loss':'plain NLL','device_family':'cuda'}

def initial_ast(source):
    tree=ast.parse(textwrap.dedent(source));fn=tree.body[0]
    require(fn.name=='initial_model','Unexpected historical initialization')
    fn.args.args.append(ast.arg(arg='seed'))
    changed={'seed':0,'hash':0}
    for index,node in enumerate(fn.body):
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
            call=node.value
            if ast.unparse(call.func)=='core.seed_all':
                require(len(call.args)==1 and ast.literal_eval(call.args[0])==42,'Unexpected old seed call')
                call.args[0]=ast.Name(id='seed',ctx=ast.Load());changed['seed']+=1
            elif ast.unparse(call.func)=='require' and 'binding[\'initial_state_sha256\']' in ast.unparse(call.args[0]):
                fn.body[index]=ast.If(test=ast.Compare(left=ast.Name(id='seed',ctx=ast.Load()),ops=[ast.Eq()],comparators=[ast.Constant(42)]),body=[node],orelse=[])
                changed['hash']+=1
    require(changed=={'seed':1,'hash':1},'Initialization adapter hooks differ')
    ast.fix_missing_locations(tree)
    return tree

def probe_ast(source):
    tree=ast.parse(textwrap.dedent(source));changed=0
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and ast.unparse(n.func)=='core.TrainBatches':
            require(len(n.args)==3 and ast.literal_eval(n.args[2])==42,'Old probe sampler differs')
            n.args[2]=ast.Attribute(value=ast.Name(id='args',ctx=ast.Load()),attr='seed',ctx=ast.Load());changed+=1
    require(changed==1,'Need one probe sampler seed change')
    ast.fix_missing_locations(tree)
    return tree

def full_ast(source):
    tree=ast.parse(textwrap.dedent(source));counts={k:0 for k in ('rows','sampler','controls','test_loader','test_evaluation','test_export','metrics','payload','epoch','batch','validation')}
    class Adapt(ast.NodeTransformer):
        def visit_Assign(self,node):
            target=ast.unparse(node.targets[0]) if len(node.targets)==1 else ''
            if target=='rows':
                require(isinstance(node.value,ast.DictComp) and ast.literal_eval(node.value.generators[0].iter)==('train','val','test'),'Old partitions differ')
                node.value.generators[0].iter=ast.parse("('train','val')",mode='eval').body;counts['rows']+=1
            elif target=='sampler':
                require(ast.unparse(node.value.func)=='core.TrainBatches' and ast.literal_eval(node.value.args[2])==42,'Old sampler differs')
                node.value.args[2]=ast.parse('args.seed',mode='eval').body;counts['sampler']+=1
            elif target=='controls':
                require(ast.unparse(node.value.func)=='dict','Old controls differ')
                node.value.keywords.append(ast.keyword(arg='seed',value=ast.parse('args.seed',mode='eval').body));counts['controls']+=1
            elif target=='test_loader':
                counts['test_loader']+=1;return None
            elif target=='(test, predictions)':
                require(ast.unparse(node.value.func)=='core.evaluate','Unexpected test inference')
                counts['test_evaluation']+=1;return ast.parse('test = None').body[0]
            elif target=='metrics':
                node.value.keys.append(ast.Constant('test_evaluated'));node.value.values.append(ast.Constant(False));counts['metrics']+=1
            elif target=='payload':
                node.value.keys.extend([ast.Constant('stale_epochs'),ast.Constant('python_random_state')])
                node.value.values.extend([ast.Name(id='stale',ctx=ast.Load()),ast.parse('__repeat_random_state()',mode='eval').body]);counts['payload']+=1
            return self.generic_visit(node)
        def visit_With(self,node):
            if 'predictions.csv' in ast.unparse(node.items[0].context_expr):
                counts['test_export']+=1;return None
            return self.generic_visit(node)
        def visit_For(self,node):
            node=self.generic_visit(node)
            if ast.unparse(node.target)=='epoch':
                node.body.insert(0,ast.parse("__repeat_progress(args,'training',epoch,0)").body[0]);counts['epoch']+=1
            return node
        def visit_AugAssign(self,node):
            if ast.unparse(node.target)=='epoch_updates':
                counts['batch']+=1
                return [node,*ast.parse("if epoch_updates == 1 or epoch_updates % 10 == 0:\n    __repeat_progress(args,'training',epoch,epoch_updates)").body]
            return node
        def visit_Expr(self,node):
            return self.generic_visit(node)
    tree=Adapt().visit(tree)
    fn=tree.body[0]
    for n in ast.walk(fn):
        if isinstance(n,ast.For) and ast.unparse(n.target)=='epoch':
            for index,node in enumerate(n.body):
                if isinstance(node,ast.Assign) and ast.unparse(node.targets[0])=='(validation, _)':
                    n.body.insert(index,ast.parse("__repeat_progress(args,'validation',epoch,epoch_updates)").body[0]);counts['validation']+=1;break
    require(all(v==1 for v in counts.values()),'Historical full-training adapter hooks differ: '+str(counts))
    calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and ast.unparse(n.func)=='core.evaluate']
    require(len(calls)==1 and ast.unparse(calls[0].args[1])=='val_loader','Unexpected evaluation partition')
    require(not any(isinstance(n,ast.Name) and n.id=='test_loader' for n in ast.walk(fn)),'Test loader remains')
    ast.fix_missing_locations(tree)
    return tree

def functions(module,seed):
    initial_tree=initial_ast(inspect.getsource(module.initial_model))
    namespace=dict(module.__dict__)
    exec(compile(initial_tree,'<baseline-seeded-initialization>','exec'),namespace)
    seeded_initial=namespace['initial_model']
    def initial(core,frozen,device,binding):
        model=seeded_initial(core,frozen,device,binding,seed)
        require(core.state_hash(model.state_dict().items())==binding['initial_state_sha256'],'Seeded initialization changed between audit/probe/full')
        actual_rng=identity({'python':random.getstate(),'torch':core.state_hash(
            [('cpu',core.torch.get_rng_state())]+[('cuda_%d'%i,s) for i,s in enumerate(core.torch.cuda.get_rng_state_all())])})
        require(actual_rng==binding['seeded_initial_rng_sha256'],'Seeded initialization RNG changed')
        return model
    probe_tree=probe_ast(inspect.getsource(module.probe));full_tree=full_ast(inspect.getsource(module.full))
    namespace.update(initial_model=initial,__repeat_progress=progress,__repeat_random_state=random.getstate)
    exec(compile(probe_tree,'<baseline-seeded-probe>','exec'),namespace)
    exec(compile(full_tree,'<baseline-validation-only-training>','exec'),namespace)
    return seeded_initial,namespace['probe'],namespace['full'],{
        'initial_ast_sha256':identity(ast.dump(initial_tree)),'probe_ast_sha256':identity(ast.dump(probe_tree)),
        'training_ast_sha256':identity(ast.dump(full_tree))}
