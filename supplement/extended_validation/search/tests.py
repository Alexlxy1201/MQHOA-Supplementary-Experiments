import unittest,itertools
import numpy as np
from engine import *

class Tests(unittest.TestCase):
 def setUp(self):
  self.i=Instance('toy',[2.,3.,1.,2.],[1.,2.,.5,1.],[(0,1,.3),(0,2,.2),(1,3,.4),(2,3,.1)],[0,3])
  self.task={'seed':0,'pool_size':64,'pool_seed':99,'params':{'k':2,'m':3,'lambda':1.3},'method':'MQHOA'}
 def test_bijection(self):
  p=PermutedPool(Pool(self.i,64,99),123)
  for a,b in itertools.product(range(64),repeat=2):self.assertEqual(p.latent(p.encode([a,b])),[a,b])
 def test_conjugacy(self):
  a=run_task((self.task,self.i.data(),'MQHOA',120));b=run_task((self.task,self.i.data(),'Conjugate',120));self.assertEqual(a['stream_sha256'],b['stream_sha256'])
 def test_relabel_initial(self):
  #20 budget includes20 initializer only, matching production k*m.
  self.task['params']={'k':4,'m':5,'lambda':1.3}
  a=run_task((self.task,self.i.data(),'MQHOA',20));b=run_task((self.task,self.i.data(),'SamePoolRelabeledMQHOA',20));self.assertEqual(a['stream_sha256'],b['stream_sha256'])
 def test_prefix(self):
  for method,params in [('MQHOA',{'k':2,'m':3,'lambda':1.3}),('PSO',{'population':12,'inertia':.7,'c1':2,'c2':2,'velocity_fraction':.2}),('GA',{'population':12,'crossover':.8,'mutation':.06})]:
   self.task.update(method=method,params=params)
   a=run_task((self.task,self.i.data(),method,100));b=run_task((self.task,self.i.data(),method,200));self.assertEqual(a['best_makespan'],next(x[2] for x in b['trace'] if x[0]==100))
 def test_geometry_validity(self):
  a=run_task((self.task,self.i.data(),'DirectGeometryMQHOA',120));self.assertEqual(a['evals_used'],120);self.assertTrue(a['validator_ok'])
 def test_solver_against_exhaustive(self):
  exactrow=exact(self.i,10);self.assertTrue(exactrow['proven'])
  best=float('inf')
  for x in itertools.product([0,1],repeat=2):
   l=[0,*x,0]
   for pi in itertools.permutations(range(4)):
    try:v=schedule(self.i,l,pi)[0]
    except ValueError:continue
    best=min(best,v)
  self.assertAlmostEqual(best,exactrow['ub'])
 def test_integerization_rejects(self):
  self.i.ts[0]=.333
  with self.assertRaises(ValueError):exact(self.i,1)

if __name__=='__main__':unittest.main(verbosity=2)
