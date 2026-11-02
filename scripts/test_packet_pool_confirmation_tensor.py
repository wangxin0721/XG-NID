"""CPU integration of the confirmation's packet operators and cache reuse."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

import run_packet_pool_confirmation as run
import test_packet_pooling_tensor as fixtures


class ConfirmationTensors(unittest.TestCase):
    def setUp(self):torch.manual_seed(17);torch.set_num_threads(2)

    def test_adapted_training_operators_preserve_packet_updates_and_encoder_state(self):
        fixture=fixtures.Tensors();graphs=fixture.data()
        original=fixture.graph().eval();candidate=fixture.graph().eval()
        candidate.load_state_dict(original.state_dict(),strict=True);candidate.eval()
        cache=fixture.cache(original,graphs)
        core=SimpleNamespace(torch=torch,state_hash=fixtures.state_hash,
            LABELS=run.dim.verify_metrics.__globals__['LABELS'])
        c=run.read(run.HERE.parent/'configs/packet_pool_confirmation_s42_v1.json')
        bound=run.train.__globals__
        opts=[run.packet.optimizer(core,original,c),bound['optimizer'](core,candidate,c)]
        enc=fixtures.state_hash(run.dim.encoder_values(candidate))
        before=candidate.packet_pool.aggregation.gate_nn[0].weight.detach().clone()
        for step in range(3):
            for model,opt,operator in ((original,opts[0],run.packet.minibatch),(candidate,opts[1],bound['minibatch'])):
                model.train();features,labels=operator(core,cache,[2,0,1],torch.device('cpu'))
                opt.zero_grad(set_to_none=True)
                loss=F.nll_loss(model.forward_features(*features)[0],labels);loss.backward();opt.step()
            for name,value in original.state_dict().items():
                torch.testing.assert_close(value,candidate.state_dict()[name],atol=0,rtol=0)
        self.assertFalse(torch.equal(before,candidate.packet_pool.aggregation.gate_nn[0].weight))
        run.dim.assert_frozen(core,candidate,enc)
        # The unchanged evaluator requires the registered13446 validation
        # samples. Repeat the tiny fixture rather than bypassing that guard.
        ix=torch.arange(13446)%3
        lengths=(cache['ptr'][1:]-cache['ptr'][:-1])[ix]
        validation={'ids':[str(i) for i in range(13446)],'flow64':cache['flow64'][ix],
            'packet64':torch.cat([cache['packet64'][int(cache['ptr'][i]):int(cache['ptr'][i+1])] for i in ix.tolist()]),
            'ptr':torch.cat((torch.zeros(1,dtype=torch.long),lengths.cumsum(0))), 'labels':cache['labels'][ix]}
        old,_=run.packet.evaluate(core,original,validation,torch.device('cpu'))
        new,_=bound['evaluate'](core,candidate,validation,torch.device('cpu'))
        self.assertEqual(old,new)

    def test_copy_revalidates_ragged_cache_and_rejects_changed_source(self):
        fixture=fixtures.Tensors();graphs=fixture.data();model=fixture.graph().eval()
        cache=fixture.cache(model,graphs);encoder=fixtures.state_hash(run.dim.encoder_values(model))
        rows=[{'sample_id':str(i),'label':i} for i in range(3)];split={'split_id':'cpu-fixture'}
        cache['metadata']={'schema':'xgnid.packet-node64-cache.v1','partition':'train','samples':3,
            'encoder_sha256':encoder,'split_binding':split,'sample_order_sha256':run.packet.identity(cache['ids'])}
        core=SimpleNamespace(torch=torch,state_hash=fixtures.state_hash,select_rows=lambda data,part:rows)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'previous';source.mkdir();output=root/'confirmation';output.mkdir()
            path=source/'train_packet_nodes.pt';torch.save(cache,path)
            receipt={'directory':str(source),'cache_receipts':{'train':{'sha256':run.sha(path),
                'tensor_sha256':run.packet.tensor_hash(core,cache),'metadata':copy.deepcopy(cache['metadata']),
                'memory_estimate':{},'wall_seconds':1.}}}
            args=SimpleNamespace(output=output);before=run.sha(path)
            copied,record=run.cached_nodes(receipt)(core,args,model,None,torch.device('cpu'),'train',{},encoder,split)
            self.assertEqual(run.sha(path),before);self.assertTrue(record['reused'])
            self.assertEqual(copied['ids'],cache['ids'])
            for key in run.packet.TENSORS:torch.testing.assert_close(copied[key],cache[key])
            with self.assertRaises(ValueError):
                run.cached_nodes(receipt)(core,args,model,None,torch.device('cpu'),'train',{},encoder,split)
            output2=root/'changed';output2.mkdir();path.write_bytes(b'changed cache')
            with self.assertRaises(ValueError):
                run.cached_nodes(receipt)(core,SimpleNamespace(output=output2),model,None,torch.device('cpu'),'train',{},encoder,split)


if __name__ == '__main__':unittest.main()
