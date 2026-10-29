import ast,json,sys,unittest
from pathlib import Path
from types import SimpleNamespace
import run_joint_gate as j

class Checks(unittest.TestCase):
 def test_syntax_config(self):
  self.assertNotIn('torch',sys.modules)
  root=Path(__file__).resolve().parents[1]
  ast.parse((root/'scripts/run_joint_gate.py').read_text(),feature_version=(3,9))
  j.check_config(json.loads((root/'configs/joint_gate_20261029.json').read_text()))
 def test_groups_and_dedup(self):
  class P:
   def requires_grad_(self,v):self.requires_grad=v
  a,b,c=P(),P(),P()
  m=SimpleNamespace(named_parameters=lambda:iter([('flow_encoder.a',a),('packet_encoder.shared',a),('shared_head.x',b),('joint_gate.x',c)]))
  config={'encoder_lr':1e-5,'head_lr':1e-4,'gate_lr':1e-4}
  result=j.groups(m,'J2',config)
  self.assertEqual([len(g['params']) for g in result],[1,1,1])
  self.assertEqual(j.group_name('gate.net.0.weight','J2'),None)
  self.assertEqual(j.group_name('joint_gate.0.weight','J0'),None)
 def test_named_paths(self):
  for n in ('flow_classifier.weight','packet_classifier.bias','shared_head.0.weight'):self.assertEqual(j.group_name(n,'J2'),'head')

if __name__=='__main__':unittest.main()
