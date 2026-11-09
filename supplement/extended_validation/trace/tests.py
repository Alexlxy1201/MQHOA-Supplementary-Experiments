import unittest,math
from engine import *

class Tests(unittest.TestCase):
 def setUp(self):
  self.x=Instance('fixture-trace',[9.,6.,3.,12.],[3.,2.,1.,4.],[(0,1,5.),(0,2,5.),(1,3,5.),(2,3,5.)],[0,3],{'kind':'fixture'})
 def test_matrix_and_params(self):
  tasks=make_tasks({self.x.identifier:self.x});self.assertEqual(len(tasks),62);cfg,frozen=original_parameters()
  self.assertEqual(len(set(t['run_id'] for t in tasks)),62)
  for t in tasks:
   if t['method']=='PSO':self.assertEqual(t['params'],frozen['PSO'])
   elif t['method'] in ['MQHOA','DirectGeometryMQHOA']:self.assertEqual(t['params'],cfg['mqhoa'])
 def test_all_methods_schedule_and_budget(self):
  tasks=make_tasks({self.x.identifier:self.x})
  for method in CONFIG['stochastic_methods']+CONFIG['deterministic_methods']:
   t=next(t for t in tasks if t['method']==method);t['pool_size']=64
   if method in CONFIG['stochastic_methods']:t['budget']=100
   row=run_one((t,self.x.data(),{'input_file_sha256':'fixture'}));self.assertTrue(row['validator_ok'])
   if method in CONFIG['stochastic_methods']:self.assertEqual(row['evals_used'],100)
   self.assertLessEqual(row['evals_used'],t['budget']);self.assertTrue(math.isfinite(row['best_makespan']))
 def test_pool_shared_across_methods(self):
  tasks=make_tasks({self.x.identifier:self.x});a=next(x for x in tasks if x['method']=='MQHOA');b=next(x for x in tasks if x['method']=='PSO');self.assertEqual((a['pool_size'],a['pool_seed']),(b['pool_size'],b['pool_seed']))
 def test_schedule_checker_rejects_pinned(self):
  with self.assertRaises(ValueError):schedule(self.x,[1,0,0,0],[0,1,2,3])
 def test_multiterminal_pins_all_roots_sinks(self):
  x=Instance('fixture-multi',[3.,4.,5.,4.,2.,3.],[1.,4/3,5/3,4/3,2/3,1.],[(0,2,2.),(1,2,2.),(2,3,2.),(3,4,2.),(3,5,2.)],[0,1,4,5],{'kind':'fixture'})
  tasks=make_tasks({x.identifier:x})
  for method in CONFIG['stochastic_methods']+CONFIG['deterministic_methods']:
   t=next(t for t in tasks if t['method']==method);t['pool_size']=64
   if method in CONFIG['stochastic_methods']:t['budget']=100
   r=run_one((t,x.data(),{}));self.assertTrue(all(r['best_l'][i]==0 for i in x.pinned));self.assertTrue(r['validator_ok'])
 def test_missing_gate_blocks_input(self):
  from unittest.mock import patch
  with patch('engine.read',side_effect=FileNotFoundError('missing parent gate')),patch('engine.sha') as checksum:
   with self.assertRaises(FileNotFoundError):load_inputs()
   checksum.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)
