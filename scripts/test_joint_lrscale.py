import copy,json,tempfile,unittest
from pathlib import Path
import run_joint_gate as joint

ROOT=Path(__file__).resolve().parents[1]


class Checks(unittest.TestCase):
    def test_only_learning_rates_change(self):
        path=ROOT/'configs/joint_gate_lrscale01_20261030.json'
        c=json.loads(path.read_text());joint.check_config(c);joint.validate_scaled_config(path,c)
        self.assertEqual((c['encoder_lr'],c['head_lr'],c['gate_lr']),(1e-6,1e-5,1e-5))
        for key,value in [('head_lr',1e-4),('weight_decay',.01),('epochs',10),('patience',5),('stage_A','outputs/other/V2')]:
            altered=copy.deepcopy(c);altered[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):joint.validate_scaled_config(path,altered)

    def test_epoch_zero_floor_does_not_hide_trained_decline(self):
        evidence=joint.trained_evidence(.97093667251942,[
            {'epoch':1,'val_macro_f1':.9697401376346687},
            {'epoch':2,'val_macro_f1':.9697943966593489},
            {'epoch':3,'val_macro_f1':.9698297541433616}])
        self.assertEqual(evidence['best_trained_epoch'],3)
        self.assertAlmostEqual(evidence['best_trained_minus_start_pp'],-.11069183760584)
        with self.assertRaises(ValueError):joint.trained_evidence(.97,[])
        with self.assertRaises(ValueError):joint.trained_evidence(.97,[{'epoch':2,'val_macro_f1':.97}])

    def test_tie_keeps_first_trained_epoch_and_last_is_recorded(self):
        e=joint.trained_evidence(.97,[{'epoch':1,'val_macro_f1':.971},{'epoch':2,'val_macro_f1':.971},
                                     {'epoch':3,'val_macro_f1':.969}])
        self.assertEqual(e['best_trained_epoch'],1)
        self.assertAlmostEqual(e['last_trained_minus_start_pp'],-.1)


if __name__=='__main__':unittest.main()
