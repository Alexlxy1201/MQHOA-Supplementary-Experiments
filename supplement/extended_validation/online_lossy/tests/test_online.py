import copy
import json
from pathlib import Path
import unittest

from extended_validation.online_lossy.model import build_problem, loss_draw, simulate, validate, visible_problem, State
from extended_validation.online_lossy.methods import make_planner, topological

CONFIG = json.loads((Path(__file__).parents[1] / 'config.json').read_text())


def fixed(problem, state, seed, epoch, budget, weight, kappa, network_power):
    loc = {t['id']: state.placed.get(t['id'], t['owner'] if t['pinned'] else problem['users']) for t in problem['tasks']}
    return loc, topological(problem), 1, {}


class OnlineTests(unittest.TestCase):
    def test_actual_retry_payload_airtime_and_energy(self):
        p = {'id':'hand', 'users':1, 'servers':1,
             'tasks':[{'id':0,'owner':0,'work':1.,'pinned':True}, {'id':1,'owner':0,'work':1.,'pinned':False}],
             'edges':[(0,1,4.)], 'jobs':[{'user':0,'arrival':0.,'tasks':[0,1]}],
             'task_release':{0:0.,1:.5}, 'frequency_ghz':[1.,2.], 'loss_probability':.15,
             'events':[{'time':2.,'kind':'link','value':.5}], 'loss_seed':0}
        for key in range(1000):
            p['loss_seed']=key
            if loss_draw(p,0,1,0)<.15 and loss_draw(p,0,1,1)>=.15:
                break
        r=simulate(p,fixed,0,1000,0.)
        self.assertEqual(r['attempts'],2)
        self.assertEqual(r['failed_attempts'],1)
        self.assertAlmostEqual(r['useful_payload_work'],4.)
        self.assertAlmostEqual(r['attempt_work'],8.)
        self.assertAlmostEqual(r['failed_attempt_work'],4.)
        self.assertAlmostEqual(r['network_airtime'],15.)
        self.assertAlmostEqual(r['network_energy_j'],30.)
        self.assertAlmostEqual(r['energy_j'],32.5)
        self.assertAlmostEqual(r['makespan'],16.5)
        self.assertEqual(r['transfer_schedule'][0]['attempt_index'],1)
        bad=copy.deepcopy(r)
        bad['energy_j']-=r['attempt_schedule'][0]['energy_j']
        self.assertIn('failed energy missing from total',validate(p,bad))

    def test_incremental_nodes_and_edges_only_after_release(self):
        p=build_problem(74001,2,2,6,True)
        r=simulate(p,make_planner('MQHOA',CONFIG),0,1000,0.)
        first=r['observations'][0]
        self.assertEqual(first['visible_tasks'],[0,1,2,3])
        before=[o for o in r['observations'] if o['time']<5.]
        self.assertTrue(all(4 not in o['visible_tasks'] and 5 not in o['visible_tasks'] for o in before))
        at5=next(o for o in r['observations'] if o['time']==5.)
        self.assertIn(4,at5['visible_tasks'])
        self.assertIn(5,at5['visible_tasks'])
        self.assertTrue(at5['started_placements'])
        state=State(visible={0,1,2,3})
        observed=visible_problem(p,state)
        self.assertEqual(observed['jobs'][0]['tasks'],[0,1,2,3])
        self.assertTrue(all(a<4 and b<4 for a,b,_ in observed['edges']))
        self.assertNotIn('task_release',observed)
        self.assertNotIn('loss_seed',observed)

    def test_future_incremental_tasks_events_and_loss_draws_do_not_change_initial_plan(self):
        p=build_problem(74003,3,2,6,True)
        q=copy.deepcopy(p)
        for task in q['tasks']:
            if q['task_release'][str(task['id'])]>0:
                task['work']*=2
        q['loss_seed']+=100
        q['events'].pop()
        a=simulate(p,make_planner('MQHOA',CONFIG),3,1000,0.)
        b=simulate(q,make_planner('MQHOA',CONFIG),3,1000,0.)
        self.assertEqual(a['plans'][0],b['plans'][0])
        self.assertEqual(a['observations'][0],b['observations'][0])

    def test_common_draws_immutable_started_tasks_and_actual_call_counts(self):
        p=build_problem(74002,2,3,6,True)
        results={}
        for m in ['MQHOA','PSO']:
            r=simulate(p,make_planner(m,CONFIG),2,1000,0.)
            self.assertEqual(r['evals_used'],800)
            self.assertTrue(r['validator_ok'])
            for o,plan in zip(r['observations'],r['plans']):
                for j,res in o['started_placements'].items():
                    self.assertEqual(plan['placement'][j],res)
            results[m]=r
        aa={(x['a'],x['b'],x['attempt_index']):x['uniform_draw'] for x in results['MQHOA']['attempt_schedule']}
        bb={(x['a'],x['b'],x['attempt_index']):x['uniform_draw'] for x in results['PSO']['attempt_schedule']}
        common=set(aa)&set(bb)
        self.assertTrue(common)
        self.assertTrue(all(aa[k]==bb[k] for k in common))


if __name__=='__main__':
    unittest.main()
