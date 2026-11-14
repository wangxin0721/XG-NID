"""Real PyG frozen-depth CPU integration; missing dependencies fail."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch
import run_cross_layer_pair as run
import test_relation_pair_tensor as fixtures
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from xgnid.cross_layer import CrossLayerClassifier
from xgnid.packet_pooling import ordered_mean_pool


class Tensors(unittest.TestCase):
    def setUp(self):torch.manual_seed(7);torch.set_num_threads(2)
    def models(self,empty=False):
        fixture=fixtures.TensorChecks();graphs=fixture.graphs(empty)
        original,_,batch=fixture.models(graphs)
        # PyG deep-copy reconstruction can consume RNG. Prepare independent
        # inputs first, then measure only the constructor under test, exactly
        # as build() does after creating the pretrained backbone and head.
        prepared=[(copy.deepcopy(original.flow_encoder),copy.deepcopy(original.packet_encoder),copy.deepcopy(original.fusion)) for _ in ('last','multi')]
        models=[]
        for mode,inputs in zip(('last','multi'),prepared):
            rng=torch.get_rng_state().clone()
            model=CrossLayerClassifier(*inputs,mode).eval()
            self.assertTrue(torch.equal(rng,torch.get_rng_state()),'Cross-layer construction changed RNG: '+mode)
            models.append(model)
        return models,graphs,batch,original

    def test_initial_identity_post_activation_and_capacity_including_empty_edges(self):
        for empty in (False,True):
            models,graphs,batch,original=self.models(empty)
            first=[];handle=models[1].packet_encoder.convs2.register_forward_pre_hook(lambda m,a:first.append(a[0]['packet'].detach()))
            f,_=models[1].encode(batch);handle.remove()
            torch.testing.assert_close(f[1],ordered_mean_pool(first[0],batch['packet'].batch,3),atol=0,rtol=0)
            for m in models:
                torch.testing.assert_close(m(batch),original(batch),atol=1e-6,rtol=1e-6)
                torch.testing.assert_close(m.cross_layer(torch.cat((f[1],f[2]),-1)),f[2],atol=0,rtol=0)
                self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad),15400)
                self.assertEqual(sum(p.numel() for p in m.cross_layer.parameters()),8256)

    def test_control_ignores_shallow_features_candidate_can_use_them_and_strict_reload(self):
        models,graphs,batch,_=self.models();f,_=models[0].encode(batch)
        for m in models:
            with torch.no_grad():m.cross_layer.weight[:,:64].copy_(torch.eye(64))
        a=models[0].forward_features(*f)[0];b=models[0].forward_features(f[0],f[1]+100,f[2])[0]
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertFalse(torch.equal(models[1].forward_features(*f)[0],models[1].forward_features(f[0],f[1]+1,f[2])[0]))
        cache={'ids':[str(i) for i in range(3)],'flow64':f[0],'first64':f[1],'second64':f[2],'labels':batch.y}
        core=SimpleNamespace(torch=torch);reordered,_=run.minibatch(core,cache,[2,0,1],torch.device('cpu'))
        torch.testing.assert_close(models[1].forward_features(*reordered)[0],models[1](Batch.from_data_list([graphs[i] for i in (2,0,1)])),atol=1e-5,rtol=1e-5)
        fresh,_,_,_=self.models();fresh[1].load_state_dict(models[1].state_dict(),strict=True);fresh[1].eval()
        torch.testing.assert_close(fresh[1](batch),models[1](batch))

    def test_raw_probe_restores_weights_gradients_rng_and_both_projection_blocks_learn(self):
        models,graphs,batch,_=self.models()
        core=SimpleNamespace(torch=torch,F=F,Batch=Batch,state_hash=fixtures.state_hash,
            select_rows=lambda *args:[{'sample_id':str(i),'label':i} for i in range(3)])
        data=SimpleNamespace(load=lambda row:graphs[int(row['sample_id'])]);c=run.read(run.HERE.parent/'configs/cross_layer_pair_s42_v1.json')
        for m in models:
            f,_=m.encode(batch);cache={'ids':[str(i) for i in range(3)],'flow64':f[0],'first64':f[1],'second64':f[2],'labels':batch.y}
            enc=fixtures.state_hash(run.dim.encoder_values(m));clean=fixtures.state_hash(m.state_dict().items());rng=run.pair.rng_state(core)
            r=run.probe(core,m,data,cache,torch.device('cpu'),c,enc)
            self.assertTrue(r['clean_model_and_rng_restored']);self.assertEqual(clean,fixtures.state_hash(m.state_dict().items()))
            self.assertEqual(run.pair.rng_hash(core,rng),run.pair.rng_hash(core,run.pair.rng_state(core)))
            opt=run.optimizer(core,m,c);step=opt.step
            def fail():step();raise RuntimeError('injected')
            opt.step=fail
            with patch.object(run,'optimizer',return_value=opt),self.assertRaisesRegex(RuntimeError,'injected'):
                run.probe(core,m,data,cache,torch.device('cpu'),c,enc)
            self.assertEqual(clean,fixtures.state_hash(m.state_dict().items()));self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_raw_probe_replays_cache_batch_boundaries_and_rejects_changed_features(self):
        models, graphs, batch, _ = self.models()
        model = models[0]
        device = torch.device('cpu')
        rows = [{'sample_id': sid, 'label': label} for sid, label in zip(batch.sample_id, batch.y.tolist())]
        core = SimpleNamespace(torch=torch, F=F, Batch=Batch, state_hash=fixtures.state_hash,
                               select_rows=lambda *args: rows)
        data = SimpleNamespace(load=lambda row: graphs[int(row['sample_id'])])
        config = run.read(run.HERE.parent / 'configs/cross_layer_pair_s42_v1.json')
        config['eval_batch_size'] = 2
        chunks = [Batch.from_data_list(graphs[:2]), Batch.from_data_list(graphs[2:])]
        encoded = [model.encode(chunk)[0] for chunk in chunks]
        cache = {'ids': list(batch.sample_id), 'labels': batch.y}
        cache.update({name: torch.cat(parts) for name, parts in zip(('flow64', 'first64', 'second64'), zip(*encoded))})
        frozen = fixtures.state_hash(run.dim.encoder_values(model))
        encode = model.encode

        def require_cache_batch_size(graph_batch):
            self.assertLessEqual(graph_batch.num_graphs, config['eval_batch_size'])
            return encode(graph_batch)

        with patch.object(model, 'encode', side_effect=require_cache_batch_size) as called:
            receipt = run.probe(core, model, data, cache, device, config, frozen)
            self.assertEqual([call.args[0].num_graphs for call in called.call_args_list], [2, 1])
        self.assertEqual(receipt['train_graphs'], 3)
        self.assertEqual(receipt['encoding_batch_size'], 2)
        self.assertTrue(all(update['max_abs_raw_cached_log_score_error'] == 0 for update in receipt['updates']))
        cache['second64'] = cache['second64'] + 100
        with self.assertRaisesRegex(ValueError, 'max abs log-score error'):
            run.probe(core, model, data, cache, device, config, frozen)

    def test_hooks_cleanup_on_error_and_device_preflight(self):
        models,_,batch,_=self.models();m=models[0];before=len(m.packet_encoder.relus1['packet']._forward_hooks)
        with patch.object(m.packet_encoder,'forward',side_effect=RuntimeError('injected')),self.assertRaises(RuntimeError):m.encode(batch)
        self.assertEqual(before,len(m.packet_encoder.relus1['packet']._forward_hooks))
        core=SimpleNamespace(torch=torch,F=F,state_hash=fixtures.state_hash)
        with tempfile.TemporaryDirectory() as folder:
            result=run.device_preflight(core,dict(zip(run.VARIANTS,models)),torch.device('cpu'),SimpleNamespace(output=Path(folder)))
            self.assertEqual(result['status'],'passed');self.assertTrue(result['model_and_rng_unchanged'])

    def test_cache_identity_schema_and_finiteness(self):
        models,_,batch,_=self.models();f,legacy=models[0].encode(batch)
        rows=[{'sample_id':str(i),'label':i} for i in range(3)];ids=[r['sample_id'] for r in rows]
        cache={'ids':ids,'flow64':f[0],'first64':f[1],'second64':f[2],'legacy_packet_mean64':legacy,'labels':batch.y,
            'metadata':{'schema':'xgnid.cross-layer-cache.v1','partition':'train','encoder_sha256':'enc','split_binding':{},'samples':3,
                'sample_order_sha256':run.identity(ids)}}
        core=SimpleNamespace(torch=torch);run.validate_cache(core,cache,rows,'train','enc',{})
        for corrupt in (lambda x:x['metadata'].update(schema='xgnid.packet-node64-cache.v1'),
                        lambda x:x['ids'].reverse(),lambda x:x['first64'].__setitem__((0,0),float('nan'))):
            bad=copy.deepcopy(cache);corrupt(bad)
            with self.assertRaises(ValueError):run.validate_cache(core,bad,rows,'train','enc',{})


if __name__=='__main__':unittest.main()
