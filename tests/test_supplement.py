"""Tests for new code only; no archived scripts are imported or executed."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from mqhoa_supp.instances import Pool, random_solution, synthetic
from mqhoa_supp.methods import Budget, gaussian_index, optimize
from mqhoa_supp.model import Instance, check_solution, schedule, validate
from mqhoa_supp.protocol import sensitivity, tasks


class SchedulingTests(unittest.TestCase):
    def test_all_predecessors_and_resource_exclusion(self):
        instance = Instance("diamond", [1,10,1,1], [1,10,1,1], [(0,1,0),(0,2,0),(1,3,0),(2,3,0)], [])
        value, starts, ends = schedule(instance, [0,0,1,0], [0,1,2,3])
        self.assertEqual(starts[3], 11)
        self.assertEqual(value, 12)
        self.assertEqual(validate(instance, [0,0,1,0], [0,1,2,3], starts, ends), [])
        starts[3] = 2
        self.assertTrue(validate(instance, [0,0,1,0], [0,1,2,3], starts, ends))

    def test_fractional_first_remote_multiple_sinks(self):
        instance = Instance("multiple", [1.25,9.5,2], [0.25,3.75,1], [(0,1,0.4),(0,2,0)], [])
        value, starts, finishes = schedule(instance, [0,1,0], [0,1,2])
        self.assertAlmostEqual(starts[1], 1.65)
        self.assertAlmostEqual(value, 5.4)
        self.assertAlmostEqual(finishes[2], 3.25)

    def test_topology_and_pinning(self):
        instance = Instance("topo", [1]*4, [1]*4, [(0,1,0),(1,2,0),(0,3,0)], [0,3])
        self.assertEqual(instance.depth[3], 1)
        with self.assertRaises(ValueError):
            schedule(instance, [0,0,0,0], [0,2,1,3])
        with self.assertRaises(ValueError):
            schedule(instance, [1,0,0,0], [0,1,2,3])
        with self.assertRaises(ValueError):
            schedule(instance, [0,0,0,0], [0,1,1,3])

    def test_non_numeric_topological_order_and_invalid_instances(self):
        instance = Instance("ids", [1,2,3], [1,1,1], [(2,0,0),(0,1,0)], [])
        self.assertEqual(instance.topo, [2,0,1])
        with self.assertRaises(ValueError):
            Instance("cycle", [1,1], [1,1], [(0,1,0),(1,0,0)], [])
        with self.assertRaises(ValueError):
            Instance("nan", [float("nan")], [1], [], [])

    def test_reference_agreement(self):
        reference = Path(__file__).resolve().parents[2] / "revision" / "audit" / "refeval.py"
        if not reference.exists():
            self.skipTest("reference evaluator is not part of standalone public source export")
        spec = importlib.util.spec_from_file_location("reference_evaluator", reference)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rng = np.random.default_rng(8989)
        for size in (5,10,20):
            instance = synthetic(size, "wide", size, f"check{size}", "test")
            for _ in range(100):
                loc, order = random_solution(instance, rng)
                ours = schedule(instance, loc, order)
                theirs = module.evaluate(loc, order, instance.edges, instance.tl, instance.ts)
                np.testing.assert_allclose(ours[0], theirs[0])
                np.testing.assert_allclose(ours[1:], theirs[1:])
                self.assertEqual(validate(instance, loc, order, ours[1], ours[2]), [])


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.instance = synthetic(10, "wide", 1337, "unit", "test")
        self.pool = Pool(self.instance, 64, 2718)

    def test_pool_and_rng_reproducibility(self):
        other = Pool(self.instance, 64, 2718)
        self.assertEqual(self.pool.sha256, other.sha256)
        for i in range(64):
            check_solution(self.instance, *self.pool.decode([i,i]))

    def test_gaussian_bounds(self):
        rng = np.random.default_rng(1)
        for sigma in (0,0.01,32,100000):
            for center in (0,31,63):
                values = [gaussian_index(center, sigma, 64, rng) for _ in range(20)]
                self.assertTrue(all(0 <= x < 64 for x in values))

    def test_budgets_and_shared_decoding(self):
        params = {"MQHOA": {"k":4,"m":5,"lambda":1.3},
                  "NoHamming": {"k":4,"m":5,"lambda":1.3},
                  "PSO": {"population":20,"inertia":0.7,"c1":1.5,"c2":1.5,"velocity_fraction":0.2},
                  "GA": {"population":12,"crossover":0.8,"mutation":0.06},
                  "RandomPool": {}, "DirectRandom": {}, "HEFT": {}}
        for method, settings in params.items():
            with self.subTest(method=method):
                budget = Budget(self.instance, 1 if method == "HEFT" else 123, 20)
                details = optimize(method, self.instance, self.pool, budget, np.random.default_rng(7), settings)
                result = budget.result()
                self.assertEqual(result["evals_used"], budget.limit)
                self.assertTrue(result["validator_ok"])
                self.assertAlmostEqual(schedule(self.instance, result["best_l"], result["best_pi"])[0], result["best_makespan"])
                self.assertTrue(all(a[2] >= b[2] for a,b in zip(result["trace"],result["trace"][1:])))
                if method == "NoHamming":
                    self.assertEqual(details["scale_reductions"], 0)
                    self.assertEqual(details["sigma_final"], 32)
                with self.assertRaises(RuntimeError):
                    budget.evaluate(result["best_l"], result["best_pi"])

    def test_persistent_scale_and_determinism(self):
        results = []
        for _ in range(2):
            budget = Budget(self.instance, 300, 50)
            details = optimize("MQHOA", self.instance, self.pool, budget, np.random.default_rng(1), {"k":2,"m":3,"lambda":1.3})
            results.append((budget.best, budget.solution[:2], details))
        self.assertEqual(results[0], results[1])
        self.assertGreater(results[0][2]["scale_reductions"], 0)
        self.assertLess(results[0][2]["sigma_final"], 32)

    def test_protocol_seven_settings_no_duplicate_full_runs(self):
        config = {"mqhoa":{"k":4,"m":5,"lambda":1.3},
                  "sensitivity":{"k":[2,4,6],"m":[3,5,8],"lambda":[1.2,1.3,1.5]},
                  "pool_large":64,"pool_small":64,"large_pool_threshold":50,"pool_seed_base":1,
                  "budget_evals":100,"seeds":list(range(20)),"trace_every":10}
        self.assertEqual(len(sensitivity(config)), 7)
        jobs = tasks([self.instance], config, "evaluation", {"PSO":{},"GA":{}})
        self.assertEqual(len({t["run_id"] for t in jobs}), len(jobs))
        self.assertEqual(sum(t["method"] == "MQHOA" and t["variant"] == "full" for t in jobs), 20)


if __name__ == "__main__":
    unittest.main()
