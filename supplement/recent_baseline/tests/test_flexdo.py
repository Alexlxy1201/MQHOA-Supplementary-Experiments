import itertools
import json
from pathlib import Path
import unittest
from mqhoa_supp.model import Instance, schedule, validate
from mqhoa_supp.protocol import tasks
from recent_baseline.flexdo import candidates, critical_order, solve


def star():
    return Instance('star', [1, 6, 4, 1], [1, 1, 3, 1],
                    [(0, 1, .2), (0, 2, .2), (1, 3, .2), (2, 3, .2)], [0, 3])


class FlexDOTests(unittest.TestCase):
    def test_anchor_gain_and_greedy_retention(self):
        rows, details = candidates(star(), '0')
        self.assertEqual([x[0] for x in rows], [(0, 0, 0, 0), (0, 1, 0, 0), (0, 1, 1, 0)])
        self.assertEqual(details['greedy_distinct'], 3)
        anchored = Instance('negative-anchor-scores', [2] * 6, [1] * 6,
            [(0, 1, 20), (1, 2, 10), (2, 5, 20), (0, 3, 4), (3, 4, 5), (4, 5, 4)], [0, 5])
        ordered, _ = candidates(anchored, 'N')
        self.assertEqual(ordered[1][0], (0, 0, 0, 1, 1, 0))

    def test_intermediate_schedule_is_considered(self):
        inst = star()
        rows, _ = candidates(inst, '0')
        expected = min(schedule(inst, p, critical_order(inst, p))[0] for p, _ in rows)
        self.assertEqual(solve(inst, '0')['best_makespan'], expected)

    def test_pair_overshoot_and_distinct_enumeration(self):
        inst = Instance('pairs', [2] * 7, [1] * 7,
                        [(0, 1, 0), (0, 3, 0), (0, 5, 0), (1, 2, 9),
                         (3, 4, 8), (2, 6, 0), (4, 6, 0), (5, 6, 0)], [0, 6])
        rows, d = candidates(inst, 'N')
        self.assertEqual(d['target_free_bits'], 2)
        self.assertEqual(d['actual_free_bits'], 1)
        self.assertEqual(len(rows), len({p for p, _ in rows}))
        self.assertEqual(d['duplicates_removed'], 1)

    def test_placement_specific_ready_order_and_zero_ties(self):
        inst = Instance('fork', [0, 9, 2, 0], [0, 1, 8, 0],
                        [(0, 1, 0), (0, 2, 0), (1, 3, 0), (2, 3, 0)], [0, 3])
        self.assertEqual(critical_order(inst, [0, 0, 0, 0]), [0, 1, 2, 3])
        self.assertEqual(critical_order(inst, [0, 1, 1, 0]), [0, 2, 1, 3])
        zero = Instance('reverse-label', [0, 0, 0], [0, 0, 0], [(2, 1, 0), (1, 0, 0)], [0, 2])
        self.assertEqual(critical_order(zero, [0, 0, 0]), [2, 1, 0])

    def test_deterministic_budget_and_full_enumeration(self):
        inst = star()
        a = solve(inst, 'N2'); b = solve(inst, 'N2')
        for key in ('best_makespan', 'best_l', 'best_pi', 'details', 'evals_used'):
            self.assertEqual(a[key], b[key])
        expected = min(schedule(inst, [0, *bits, 0], critical_order(inst, [0, *bits, 0]))[0]
                       for bits in itertools.product((0, 1), repeat=2))
        self.assertEqual(a['best_makespan'], expected)
        self.assertEqual(a['evals_used'], 4)
        with self.assertRaises(ValueError): solve(inst, 'N2', limit=3)
        self.assertFalse(validate(inst, a['best_l'], a['best_pi'], a['starts'], a['finishes']))

    def test_archived_engine_and_task_matrix_unchanged(self):
        from recent_baseline.campaign import load, check_source, base_tasks
        repo = Path(__file__).resolve().parents[3]
        instances, rows, config, frozen = load(repo)
        check_source(frozen)
        matrix = tasks(list(instances.values()), config, 'evaluation', frozen)
        self.assertEqual(len(matrix), 3374)
        self.assertEqual({r['run_id'] for r in matrix}, {r['run_id'] for r in rows})
        self.assertEqual(len(base_tasks(rows, list(range(5)))), 70)


if __name__ == '__main__': unittest.main()
