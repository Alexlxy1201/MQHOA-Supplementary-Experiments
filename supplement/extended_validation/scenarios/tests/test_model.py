import copy
import json
from pathlib import Path
import unittest

from extended_validation.scenarios.model import State, build_problem, forecast, simulate, validate, visible_problem
from extended_validation.scenarios.methods import make_planner, topological

CONFIG = json.loads((Path(__file__).parents[1] / 'config.json').read_text())


def fixed_placement(problem, state, seed, epoch, budget, weight, kappa, network_power):
    loc = {t['id']: state.placed.get(t['id'], t['owner'] if t['pinned'] else problem['users']) for t in problem['tasks']}
    return loc, topological(problem), 1, {}


class ModelTests(unittest.TestCase):
    def test_shared_link_and_independent_compute(self):
        p = build_problem(1, 2, 2, 6)
        r = simulate(p, fixed_placement, 0, 1000, 0.)
        self.assertTrue(r['validator_ok'])
        self.assertGreater(len(r['transfer_schedule']), 1)
        transfers = sorted(r['transfer_schedule'], key=lambda x: x['start'])
        self.assertTrue(all(a['finish'] <= b['start'] + 1e-8 for a, b in zip(transfers, transfers[1:])))
        local = [x for x in r['task_schedule'] if x['task'] in (0, 6)]
        self.assertEqual([x['start'] for x in local], [0., 0.])

    def test_midtask_frequency_and_energy(self):
        p = {'id': 'one', 'users': 1, 'servers': 1,
             'tasks': [{'id': 0, 'owner': 0, 'work': 4., 'pinned': True}], 'edges': [],
             'jobs': [{'user': 0, 'tasks': [0], 'arrival': 0.}], 'frequency_ghz': [1., 2.],
             'events': [{'time': 2., 'kind': 'compute', 'resource': 0, 'value': .5}]}
        r = simulate(p, make_planner('FIFO-Local', CONFIG), 0, 200, 0.)
        self.assertAlmostEqual(r['makespan'], 6.)
        self.assertAlmostEqual(r['energy_j'], 1.25)
        self.assertAlmostEqual(r['device_energy_j'], 1.25)
        self.assertEqual(len(r['segments']), 2)

    def test_future_information_isolation(self):
        p = build_problem(44, 2, 2, 6, dynamic=True)
        q = copy.deepcopy(p)
        q['events'][0]['value'] = .2
        q['events'].pop()
        for task in q['tasks'][6:]:
            task['work'] *= 10
        a = simulate(p, make_planner('MQHOA', CONFIG), 5, 200, 0.)
        b = simulate(q, make_planner('MQHOA', CONFIG), 5, 200, 0.)
        self.assertEqual(a['plans'][0], b['plans'][0])
        self.assertEqual(a['observations'][0], b['observations'][0])
        self.assertNotIn('events', visible_problem(p, State(visible=set(range(6)))))

    def test_forecast_waits_for_all_parents_before_transfers(self):
        p = {'id': 'fork', 'users': 1, 'servers': 1,
             'tasks': [{'id': i, 'owner': 0, 'work': 1., 'pinned': False} for i in range(3)],
             'edges': [(0, 2, 2.), (1, 2, 2.)],
             'jobs': [{'user': 0, 'tasks': [0, 1, 2], 'arrival': 0.}], 'frequency_ghz': [1., 2.]}
        p['tasks'][1]['work'] = 5.
        state = State(visible={0, 1, 2}, multipliers=[1., 1.])
        delay, _ = forecast(p, state, {0: 0, 1: 0, 2: 1}, [0, 1, 2])
        self.assertAlmostEqual(delay, 10.5)

    def test_started_placement_freeze_and_budget(self):
        p = build_problem(3, 3, 2, 6, dynamic=True)
        for method in ('MQHOA', 'PSO'):
            r = simulate(p, make_planner(method, CONFIG), 2, 1000, 0.)
            self.assertEqual(r['evals_used'], 1000)
            self.assertTrue(any(x['started_placements'] for x in r['observations'][1:]))
            for observation, plan in zip(r['observations'], r['plans']):
                for task, resource in observation['started_placements'].items():
                    self.assertEqual(plan['placement'][task], resource)

    def test_validator_rejects_wrong_release_ownership_and_energy(self):
        p = build_problem(4, 2, 2, 6, dynamic=True)
        r = simulate(p, make_planner('HEFT-List', CONFIG), 0, 1000, 0.)
        for field, value in [('start', -1.), ('resource', 1)]:
            bad = copy.deepcopy(r)
            bad['task_schedule'][0][field] = value
            self.assertTrue(validate(p, bad))
        bad = copy.deepcopy(r)
        bad['energy_j'] += 1.
        self.assertIn('total energy', validate(p, bad))

    def test_link_mid_transfer_changes_actual_completion(self):
        p = {'id': 'link', 'users': 1, 'servers': 1,
             'tasks': [{'id': 0, 'owner': 0, 'work': 1., 'pinned': True},
                       {'id': 1, 'owner': 0, 'work': 1., 'pinned': False}],
             'edges': [(0, 1, 4.)], 'jobs': [{'user': 0, 'tasks': [0, 1], 'arrival': 0.}],
             'frequency_ghz': [1., 2.],
             'events': [{'time': 2., 'kind': 'link', 'value': .5}]}
        r = simulate(p, fixed_placement, 0, 200, 0.)
        self.assertAlmostEqual(r['transfer_schedule'][0]['finish'], 8.)
        self.assertAlmostEqual(r['makespan'], 8.5)
        self.assertAlmostEqual(r['network_energy_j'], 14.)
        self.assertAlmostEqual(r['device_network_energy_j'], 7.)


if __name__ == '__main__':
    unittest.main()
