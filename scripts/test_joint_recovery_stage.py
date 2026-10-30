"""Real file/hash receipt checks without torch, graphs or optimizer steps."""
import copy,json,tempfile,unittest
from pathlib import Path
import run_joint_gate as joint

ROOT=Path(__file__).resolve().parents[1]


class StageChecks(unittest.TestCase):
    def fixture(self,directory):
        repo=Path(directory);c=json.loads((ROOT/'configs/joint_gate_recovery_v2_20261030.json').read_text())
        joint.check_config(c)
        stage=repo/c['stage_A'];root=stage.parent;(stage/'V2').mkdir(parents=True)
        parent={'vector_cache':{'train':'same','val':'same'},'n0':'same'}
        fusion={'lr':.001,'seed':42,'epochs':30,'patience':12}
        val=json.loads((ROOT/'configs/n0_seed42_reference_20261029.json').read_text())['model']['best_validation']
        for folder in (root,stage,stage/'V2'):joint.write(folder/'status.json',{'status':'completed','exit_code':0})
        joint.write(root/'recovery_plan.json',{'binding':parent,'source':'original_directory'})
        cfg={'binding':parent,'variant':'V2','protocol':fusion,'trainable_parameters':488}
        joint.write(stage/'V2/config.json',cfg)
        m={'variant':'V2','test_evaluated':False,'best_epoch':3,'best_validation':val,'checkpoint_sha256':{}}
        for name in ('best.pt','last.pt'):
            (stage/'V2'/name).write_bytes(('fake '+name).encode());m['checkpoint_sha256'][name]=joint.sha(stage/'V2'/name)
        (stage/'V2/validation_predictions.jsonl.gz').write_bytes(b'fake prediction file')
        m['predictions_sha256']=joint.sha(stage/'V2/validation_predictions.jsonl.gz')
        joint.write(stage/'V2/metrics.json',m)
        report={'job':'V2','binding':parent,'metrics':m,'test_evaluated':False,'frozen_model_unchanged':True}
        joint.write(stage/'result.json',report)
        joint.write(root/'summary.json',{'binding':parent,'results':{'V2':report},'test_evaluated':False,
                                       'vector_screening':{'budget_threshold_passed':False}})
        return repo,c,parent,fusion,stage

    def test_recovery_reused_even_when_old_budget_screen_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            repo,c,parent,fusion,stage=self.fixture(directory)
            actual,report,receipt=joint.stage_receipt(repo,c,parent,fusion)
            self.assertEqual(actual,stage)
            self.assertTrue(receipt['reused']);self.assertFalse(receipt['prewarm_training_started'])
            self.assertFalse(receipt['existing_vector_screening']['budget_threshold_passed'])

    def test_bad_weights_or_protocol_rejected(self):
        for failure in ('weights','protocol','incomplete','epoch0','summary'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as directory:
                repo,c,parent,fusion,stage=self.fixture(directory)
                if failure=='weights':(stage/'V2/best.pt').write_bytes(b'other model')
                if failure=='protocol':
                    cfg=joint.read(stage/'V2/config.json');cfg['protocol']['lr']=.1;joint.write(stage/'V2/config.json',cfg)
                if failure=='incomplete':joint.write(stage/'status.json',{'status':'running'})
                if failure=='epoch0':
                    # Keep all records mutually consistent while changing only
                    # the selected epoch, to exercise the trained-gate guard.
                    m=joint.read(stage/'V2/metrics.json');m['best_epoch']=0;joint.write(stage/'V2/metrics.json',m)
                    r=joint.read(stage/'result.json');r['metrics']=m;joint.write(stage/'result.json',r)
                    s=joint.read(stage.parent/'summary.json');s['results']['V2']=r;joint.write(stage.parent/'summary.json',s)
                if failure=='summary':
                    s=joint.read(stage.parent/'summary.json');s['binding']={'other':True};joint.write(stage.parent/'summary.json',s)
                with self.assertRaises(ValueError):joint.stage_receipt(repo,c,parent,fusion)

    def test_missing_recovery_does_not_fall_back_to_old_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            repo,c,parent,fusion,stage=self.fixture(directory)
            # The valid tree exists, but the explicit selection is absent.
            c['stage_A']='outputs/missing_recovery/V2'
            before=joint.sha(stage/'V2/best.pt')
            with self.assertRaises(FileNotFoundError):joint.stage_receipt(repo,c,parent,fusion)
            self.assertEqual(before,joint.sha(stage/'V2/best.pt'))


if __name__=='__main__':unittest.main()
