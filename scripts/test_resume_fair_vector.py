"""Recovery ledger/queue boundary checks without tensor deserialization."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import resume_fair_vector as recovery


class Checks(unittest.TestCase):
    def saved(self):
        return {'epoch':7,'best_epoch':4,'stale_epochs':3,'optimizer_updates':1477,
                'best_val_macro_f1':.9657,'validation':{'macro_f1':.9655}}

    def rows(self):
        values=[.962,.9621,.9649,.9657,.9626,.9647,.9655]
        return [{'epoch':i,'val_macro_f1':v,'train_samples':107621,'optimizer_updates':211,
                 'cumulative_optimizer_updates':i*211,'train_order_sha256':str(i)} for i,v in enumerate(values,1)]

    def test_py39_stdlib(self):
        self.assertNotIn('torch',sys.modules)
        ast.parse(Path(recovery.__file__).read_text(encoding='utf-8'),feature_version=(3,9))

    def test_seventh_epoch_retains_best_stale_and_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'epochs.jsonl';rows=self.rows()
            p.write_text('\n'.join(json.dumps(r) for r in rows))
            self.assertEqual(recovery.history(p,self.saved(),.963),rows)
            for key,value in [('stale_epochs',0),('best_epoch',7),('optimizer_updates',0)]:
                saved=self.saved();saved[key]=value
                with self.assertRaises(ValueError):recovery.history(p,saved,.963)

    def test_one_missing_or_torn_row_allowed_larger_gap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'epochs.jsonl';rows=self.rows()
            p.write_text('\n'.join(json.dumps(r) for r in rows[:-1])+'\n{"epoch":')
            self.assertEqual(len(recovery.history(p,self.saved(),.963)),6)
            p.write_text('\n'.join(json.dumps(r) for r in rows[:-2]))
            with self.assertRaises(ValueError):recovery.history(p,self.saved(),.963)

    def test_plan_skips_done_resumes_43_and_never_resets_vector_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'E0FT43/E0FT';folder.mkdir(parents=True)
            (folder/'last.pt').write_bytes(b'fixture checkpoint')
            with patch.object(recovery,'complete',side_effect=lambda folder,job,binding:job=='E0FT42'):
                plan=recovery.source_plan(root,{})
                self.assertEqual([r['action'] for r in plan],['skip_completed','resume','new','new','new'])
                v=root/'V1/V1';v.mkdir(parents=True);(v/'last.pt').write_bytes(b'partial vector')
                with self.assertRaises(ValueError):recovery.source_plan(root,{})


if __name__=='__main__':unittest.main()
