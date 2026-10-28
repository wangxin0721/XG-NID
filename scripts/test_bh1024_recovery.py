"""Recovery invariants with scalar metadata and fake /proc; no ML execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from bh1024_recovery import history, validate_checkpoint, active_runs

def fixture(epoch=21, best_epoch=21):
    cfg={'variant':'B_Hstats','batch_size':1024,'epochs':100,'patience':12}
    rows=[{'epoch':i,'best_epoch':min(i,best_epoch),'stale_epochs':max(0,i-best_epoch),
           'val_macro_f1':.9+min(i,best_epoch)/1000,'optimizer_updates':106,
           'cumulative_optimizer_updates':106*i,'train_samples':107621} for i in range(1,epoch+1)]
    saved={'model_state':{},'optimizer_state':{'param_groups':[{'lr':.001,'weight_decay':0}]},
           'config':cfg,'epoch':epoch,'best_epoch':best_epoch,'stale_epochs':epoch-best_epoch,
           'best_val_macro_f1':rows[-1]['val_macro_f1'],'validation':{'macro_f1':rows[-1]['val_macro_f1']},
           'sampler_rng_state':'test-placeholder','rng_state':{'python':None,'torch':None,'cuda':None},
           'optimizer_updates':106*epoch}
    return cfg,rows,saved

class Checks(unittest.TestCase):
    def test_resume_22_preserves_budget(self):
        cfg,rows,saved=fixture()
        result=validate_checkpoint(saved,cfg,rows)
        self.assertEqual((result['next_epoch'],result['optimizer_updates'],result['stale_epochs']),(22,2226,0))
        self.assertEqual(result['epoch_limit'],100)
        self.assertEqual(result['new_optimizer_updates_for_restoration'],0)

    def test_nonzero_stale_counter_is_retained(self):
        cfg,rows,saved=fixture(21,18)
        self.assertEqual(validate_checkpoint(saved,cfg,rows)['stale_epochs'],3)

    def test_reject_missing_rng_or_optimizer(self):
        cfg,rows,saved=fixture()
        for name in ('rng_state','sampler_rng_state','optimizer_state'):
            broken=copy.deepcopy(saved);del broken[name]
            with self.assertRaises(ValueError):
                validate_checkpoint(broken,cfg,rows)

    def test_reject_reset_budget_or_inconsistent_logs(self):
        cfg,rows,saved=fixture()
        for change in ({'optimizer_updates':0},{'stale_epochs':1},{'config':{**cfg,'epochs':121}}):
            with self.assertRaises(ValueError):
                validate_checkpoint({**saved,**change},cfg,rows)
        with self.assertRaises(ValueError):
            validate_checkpoint(saved,cfg,rows[:-1])

    def test_partial_final_log_is_preserved_as_source_and_ignored(self):
        _,rows,_=fixture(2,2)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'epochs.jsonl'
            content=''.join(json.dumps(r)+'\n' for r in rows)+'{"epoch":3'
            p.write_text(content,encoding='utf-8')
            result,raw=history(p)
            self.assertEqual(result,rows)
            self.assertEqual(len(raw),2)
            self.assertEqual(p.read_text(encoding='utf-8'),content)

    def test_corrupt_middle_log_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'epochs.jsonl';p.write_text('broken\n{}\n',encoding='utf-8')
            with self.assertRaises(ValueError):
                history(p)

    def test_only_matching_output_process_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);target=root/'outputs/run'
            for pid,output in ((12,target),(13,root/'outputs/other'),(14,target)):
                path=root/str(pid);path.mkdir()
                (path/'cmdline').write_bytes(('python\0train.py\0--output\0'+str(output)+'\0').encode())
            result=active_runs([target],proc=root,own_pid=14)
            self.assertEqual([r['pid'] for r in result],[12])

    def test_historical_modules_are_reused(self):
        import fusion_acb
        import run_fusion_bh_b1024_server as original
        import resume_fusion_bh_b1024_server as recovery
        self.assertIs(recovery.update,fusion_acb.update)
        self.assertIs(recovery.optimizer,original.optimizer)
        self.assertIs(recovery.best_diagnostic,original.best_diagnostic)

if __name__=='__main__':
    unittest.main()
