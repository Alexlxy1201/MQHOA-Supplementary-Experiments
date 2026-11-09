"""Budgeted supplementary methods, implemented separately from archived scripts.

MQHOA follows the manuscript's Algorithm 2 (including its stated index/Hamming
scale comparison). No-Hamming holds sigma fixed at M/2. Direct random search is
a whole-search-strategy control, not an isolated causal encoding ablation.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .instances import random_solution
from .model import schedule, validate


class Budget:
    def __init__(self, instance, limit, trace_every=100):
        self.instance, self.limit = instance, int(limit)
        self.used = 0
        self.best = math.inf
        self.solution = None
        self.trace = []
        self.t_eval = 0.0
        self.started = time.perf_counter()
        self.trace_every = int(trace_every)

    @property
    def remaining(self):
        return self.limit - self.used

    def evaluate(self, loc, order):
        if not self.remaining:
            raise RuntimeError("evaluation budget exhausted")
        tic = time.perf_counter()
        # Candidate validity is guaranteed by the generator; all reported best
        # schedules are independently validated before being written to a log.
        value, starts, finishes = schedule(self.instance, loc, order, check=False)
        self.t_eval += time.perf_counter() - tic
        self.used += 1
        if value < self.best:
            self.best = value
            self.solution = (list(loc), list(order), starts, finishes)
        if self.used == 1 or self.used % self.trace_every == 0 or not self.remaining:
            self.trace.append([self.used, time.perf_counter() - self.started, self.best])
        return value

    def result(self):
        if self.solution is None:
            raise RuntimeError("no evaluated solution")
        if not self.trace or self.trace[-1][0] != self.used:
            self.trace.append([self.used, time.perf_counter() - self.started, self.best])
        loc, order, starts, finishes = self.solution
        violations = validate(self.instance, loc, order, starts, finishes)
        if violations:
            raise RuntimeError(f"invalid reported schedule: {violations}")
        return {"best_makespan": self.best, "best_l": loc, "best_pi": order,
                "starts": starts, "finishes": finishes, "validator_ok": True,
                "evals_used": self.used, "trace": self.trace, "t_eval": self.t_eval}


def gaussian_index(center, sigma, size, rng):
    # Explicit Box-Muller, floor, then clamp, matching the manuscript encoding.
    u1 = max(float(rng.random()), np.finfo(float).tiny)
    z = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * float(rng.random()))
    return min(size - 1, max(0, math.floor(float(center) + sigma * z)))


def mqhoa(pool, evaluator, rng, params, fixed_scale=False):
    k, m = int(params["k"]), int(params["m"])
    reduction = float(params["lambda"])
    if k < 1 or m < 1 or reduction <= 1 or evaluator.limit < k * m:
        raise ValueError("invalid MQHOA parameters/budget")
    initial = rng.integers(0, pool.size, (k * m, 2))
    fitness = [evaluator.evaluate(*pool.decode(pair)) for pair in initial]
    centers = initial[np.argsort(fitness, kind="stable")[:k]].copy()
    sigma = pool.size / 2
    iterations = reductions = 0
    sigma_trace = []
    while evaluator.remaining:
        candidates = []
        for c in range(k):
            group = []
            for _ in range(min(m, evaluator.remaining)):
                pair = [gaussian_index(centers[c, d], sigma, pool.size, rng) for d in range(2)]
                loc, order = pool.decode(pair)
                value = evaluator.evaluate(loc, order)
                item = (value, pair, loc, order)
                group.append(item)
                candidates.append(item)
            if group:
                centers[c] = min(group, key=lambda x: x[0])[1]
            if not evaluator.remaining:
                break
        best = min(candidates, key=lambda x: x[0])
        maximum_hamming = max(sum(a != b for a, b in zip(item[2], best[2]))
                              + sum(a != b for a, b in zip(item[3], best[3]))
                              for item in candidates)
        if not fixed_scale and maximum_hamming < sigma:
            sigma /= reduction
            reductions += 1
        iterations += 1
        if iterations == 1 or iterations % 10 == 0 or not evaluator.remaining:
            sigma_trace.append([evaluator.used, sigma, maximum_hamming])
    return {"iterations": iterations, "scale_reductions": reductions,
            "sigma_final": sigma, "sigma_trace": sigma_trace}


def pso(pool, evaluator, rng, params):
    """Global-best PSO, continuous 2-index coordinates, floor-and-clamp decoder."""
    pop = min(int(params["population"]), evaluator.limit)
    if pop < 1:
        raise ValueError("PSO needs a population")
    x = rng.integers(0, pool.size, (pop, 2)).astype(float)
    vmax = pool.size * float(params["velocity_fraction"])
    v = rng.uniform(-vmax, vmax, (pop, 2))
    personal_x = x.copy()
    personal_f = np.asarray([evaluator.evaluate(*pool.decode(pair.astype(int))) for pair in x])
    global_x = personal_x[int(np.argmin(personal_f))].copy()
    global_f = float(personal_f.min())
    iterations = 0
    while evaluator.remaining:
        # Synchronous generation: all particles use the previous global best.
        v = (float(params["inertia"]) * v
             + float(params["c1"]) * rng.random((pop, 2)) * (personal_x - x)
             + float(params["c2"]) * rng.random((pop, 2)) * (global_x - x))
        v = np.clip(v, -vmax, vmax)
        proposed = x + v
        hit = (proposed < 0) | (proposed > pool.size - 1)
        x = np.clip(proposed, 0, pool.size - 1)
        v[hit] = 0.0
        for i in range(min(pop, evaluator.remaining)):
            value = evaluator.evaluate(*pool.decode(np.floor(x[i]).astype(int)))
            if value < personal_f[i]:
                personal_f[i], personal_x[i] = value, x[i].copy()
            if value < global_f:
                global_f, global_x = value, x[i].copy()
        iterations += 1
    return {"iterations": iterations, "encoding": "two-index-floor-clamp"}


def ga(pool, evaluator, rng, params):
    """Supplementary index-coded GA; not a rerun of archived GA outputs."""
    pop = min(int(params["population"]), evaluator.limit)
    population = rng.integers(0, pool.size, (pop, 2))
    fitness = np.asarray([evaluator.evaluate(*pool.decode(pair)) for pair in population])
    generations = 0
    while evaluator.remaining:
        # Inverse-makespan roulette selection with copied parents and elitism.
        weights = 1 / np.maximum(fitness, 1e-12)
        weights /= weights.sum()
        count = min(pop, evaluator.remaining)
        children, values = [], []
        for _ in range(count):
            parents = rng.choice(pop, size=2, p=weights)
            child = population[parents[0]].copy()
            if rng.random() < float(params["crossover"]):
                child[1] = population[parents[1], 1]
            for d in range(2):
                if rng.random() < float(params["mutation"]):
                    child[d] = rng.integers(0, pool.size)
            children.append(child)
            values.append(evaluator.evaluate(*pool.decode(child)))
        # Elitism changes no evaluation count; every evaluated child is counted.
        elite = int(np.argmin(fitness))
        merged = np.vstack([population[elite], np.asarray(children)])
        merged_f = np.asarray([fitness[elite]] + values)
        take = np.argsort(merged_f, kind="stable")[:min(pop, len(merged))]
        population, fitness = merged[take], merged_f[take]
        pop = len(population)
        generations += 1
    return {"iterations": generations, "encoding": "two-index", "elitism": 1}


def random_search(instance, pool, evaluator, rng, direct=False):
    while evaluator.remaining:
        if direct:
            solution = random_solution(instance, rng)
        else:
            solution = pool.decode(rng.integers(0, pool.size, 2))
        evaluator.evaluate(*solution)
    return {"encoding": "direct-level-random" if direct else "two-index-uniform",
            "interpretation": "whole-search-strategy control" if direct else "same-pool diagnostic"}


def heft(instance, evaluator):
    """Two-resource HEFT insertion schedule, converted to an equivalent pi.

    There is one symmetric inter-resource link, so its mean transfer cost is c.
    Pinned tasks have only resource 0 as an admissible assignment.
    """
    rank = [0.0] * instance.n
    for i in reversed(instance.topo):
        computation = instance.tl[i] if i in instance.pinned else (instance.tl[i] + instance.ts[i]) / 2
        rank[i] = computation + max((c + rank[j] for j, c in instance.succ[i]), default=0)
    # Kahn priority scheduling avoids invalid order in zero-cost rank ties.
    indegree = [len(p) for p in instance.pre]
    ready = [i for i, d in enumerate(indegree) if not d]
    calendars = [[], []]
    loc, starts, finish = [0] * instance.n, [0.0] * instance.n, [0.0] * instance.n
    while ready:
        j = min(ready, key=lambda i: (-rank[i], i))
        ready.remove(j)
        options = []
        for r in ([0] if j in instance.pinned else [0, 1]):
            release = max((finish[i] + (c if loc[i] != r else 0) for i, c in instance.pre[j]), default=0)
            duration = instance.ts[j] if r else instance.tl[j]
            start = release
            for s, f, _ in sorted(calendars[r]):
                if start + duration <= s + 1e-12:
                    break
                start = max(start, f)
            options.append((start + duration, start, r))
        finish[j], starts[j], loc[j] = min(options)
        calendars[loc[j]].append((starts[j], finish[j], j))
        for child, _ in instance.succ[j]:
            indegree[child] -= 1
            if not indegree[child]:
                ready.append(child)
    # Build a topological order with explicit HEFT resource-order arcs. The
    # common non-insertion decoder can then reproduce (or left-shift) HEFT.
    arcs = {(a, b) for a, b, _ in instance.edges}
    topo_position = {j: i for i, j in enumerate(instance.topo)}
    for calendar in calendars:
        tasks = [j for _, _, j in sorted(calendar, key=lambda item: (item[0], item[1], topo_position[item[2]]))]
        arcs.update(zip(tasks, tasks[1:]))
    incoming = [0] * instance.n
    children = [[] for _ in range(instance.n)]
    for a, b in sorted(arcs):
        incoming[b] += 1
        children[a].append(b)
    ready = [i for i, d in enumerate(incoming) if not d]
    order = []
    while ready:
        j = min(ready, key=lambda i: (starts[i], -rank[i], i))
        ready.remove(j)
        order.append(j)
        for child in children[j]:
            incoming[child] -= 1
            if not incoming[child]:
                ready.append(child)
    if len(order) != instance.n:
        raise RuntimeError("HEFT resource-order conversion produced a cycle")
    evaluator.evaluate(loc, order)
    return {"one_shot": True, "heft_insertion_makespan": max(finish),
            "rank_communication": "mean over the symmetric inter-resource link = c"}


def optimize(method, instance, pool, evaluator, rng, params):
    if method == "MQHOA":
        return mqhoa(pool, evaluator, rng, params)
    if method == "NoHamming":
        return mqhoa(pool, evaluator, rng, params, fixed_scale=True)
    if method == "PSO":
        return pso(pool, evaluator, rng, params)
    if method == "GA":
        return ga(pool, evaluator, rng, params)
    if method in {"RandomPool", "DirectRandom"}:
        return random_search(instance, pool, evaluator, rng, direct=method == "DirectRandom")
    if method == "HEFT":
        return heft(instance, evaluator)
    raise ValueError(f"unknown method {method}")
