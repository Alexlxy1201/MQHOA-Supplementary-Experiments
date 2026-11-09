"""Generalized two-index methods sharing the same observed forecast model."""
from __future__ import annotations

import math
import numpy as np

from mqhoa_supp.methods import mqhoa, pso
from .model import allowed, digest, forecast


def topological(problem, priority=None):
    tasks = {t['id']: t for t in problem['tasks']}
    pre = {i: [] for i in tasks}
    succ = {i: [] for i in tasks}
    for a, b, c in problem['edges']:
        pre[b].append(a)
        succ[a].append(b)
    degree = {j: len(p) for j, p in pre.items()}
    ready = [j for j, d in degree.items() if not d]
    order = []
    while ready:
        j = min(ready, key=lambda i: (priority.get(i, i) if priority else i, i))
        ready.remove(j)
        order.append(j)
        for child in succ[j]:
            degree[child] -= 1
            if degree[child] == 0:
                ready.append(child)
    if len(order) != len(tasks):
        raise ValueError('cycle')
    return order


class Pool:
    def __init__(self, problem, state, size, seed):
        self.ids = sorted(t['id'] for t in problem['tasks'])
        self.size = size
        rng = np.random.default_rng(seed)
        self.loc = np.empty((size, len(self.ids)), dtype=np.int16)
        for col, task in enumerate(sorted(problem['tasks'], key=lambda t: t['id'])):
            options = allowed(problem, task)
            self.loc[:, col] = rng.choice(options, size=size)
            if task['id'] in state.placed:
                self.loc[:, col] = state.placed[task['id']]
        # Level-consistent permutations, matching the original representation.
        depth = {}
        for j in topological(problem):
            depth[j] = 1 + max((depth[a] for a, b, _ in problem['edges'] if b == j), default=-1)
        levels = [[j for j in self.ids if depth[j] == d] for d in range(max(depth.values()) + 1)]
        rows = []
        for _ in range(size):
            rows.append([int(j) for level in levels for j in rng.permutation(level)])
        self.order = np.asarray(rows)
        self.sha256 = digest({'ids': self.ids, 'loc': self.loc.tolist(), 'order': self.order.tolist()})

    def decode(self, pair):
        a, b = map(int, pair)
        return self.loc[a].tolist(), self.order[b].tolist()


class ForecastBudget:
    def __init__(self, problem, state, ids, limit, weight, kappa, network_power):
        self.problem, self.state, self.ids = problem, state, ids
        self.limit, self.used = limit, 0
        self.weight, self.kappa, self.network_power = weight, kappa, network_power
        self.best, self.solution = math.inf, None
        self.trace = []
        pending = [t for t in problem['tasks'] if t['id'] not in state.completed]
        self.time_scale = max(1e-9, sum(t['work'] / problem['frequency_ghz'][t['owner']] for t in pending))
        self.energy_scale = max(1e-9, sum(kappa * (problem['frequency_ghz'][t['owner']] * 1e9) ** 3
                                        * t['work'] / problem['frequency_ghz'][t['owner']] for t in pending))

    @property
    def remaining(self):
        return self.limit - self.used

    def evaluate(self, loc, order):
        if self.used >= self.limit:
            raise RuntimeError('forecast evaluation budget exhausted')
        placement = dict(zip(self.ids, map(int, loc)))
        delay, energy = forecast(self.problem, self.state, placement, order, self.kappa, self.network_power)
        score = delay / self.time_scale + self.weight * energy / self.energy_scale
        self.used += 1
        if score < self.best:
            self.best, self.solution = score, (placement, list(map(int, order)))
        if self.used == 1 or self.used % 100 == 0 or not self.remaining:
            self.trace.append([self.used, self.best])
        return score


def heuristic(problem, state, weight, kappa, network_power):
    tasks = {t['id']: t for t in problem['tasks']}
    rank = {}
    freqs = [f * state.multipliers[i] for i, f in enumerate(problem['frequency_ghz'])]
    for j in reversed(topological(problem)):
        average = np.mean([tasks[j]['work'] / freqs[r] for r in allowed(problem, tasks[j])])
        rank[j] = float(average) + max((c / ((1-problem['loss_probability'])*state.link_rate) + rank[b]
                                      for a, b, c in problem['edges'] if a == j), default=0)
    order = topological(problem, {j: -x for j, x in rank.items()})
    loc = dict(state.placed)
    ends = dict(state.completed)
    available = [state.t] * len(freqs)
    for r, op in state.running.items():
        ends[op['task']] = state.t + op['remaining'] / freqs[r]
        available[r] = ends[op['task']]
    link_free = state.t + ((state.transfer['remaining'] + state.transfer['amount']*problem['loss_probability']
                           /(1-problem['loss_probability']))/state.link_rate if state.transfer else 0.)
    pre = {j: [] for j in tasks}
    for a, b, c in problem['edges']:
        pre[b].append((a, c/(1-problem['loss_probability'])))
    time_scale = max(1e-9, sum(t['work'] / freqs[t['owner']] for t in tasks.values()))
    energy_scale = max(1e-9, sum(kappa * (freqs[t['owner']] * 1e9) ** 3 * t['work'] / freqs[t['owner']]
                                for t in tasks.values()))
    for j in order:
        if j in ends:
            continue
        options = []
        for r in ([loc[j]] if j in loc else allowed(problem, tasks[j])):
            ready = max([state.t] + [ends[a] for a, _ in pre[j]])
            link, comm_energy = link_free, 0.
            for a, c in pre[j]:
                ready = max(ready, ends[a])
                if loc[a] != r and (a, j) not in state.delivered:
                    if state.transfer and (a, j) == (state.transfer['a'], state.transfer['b']):
                        ready = max(ready, link_free)
                    else:
                        link = max(ready, link) + c / state.link_rate
                        comm_energy += network_power * c / state.link_rate
                        ready = max(ready, link)
            finish = max(available[r], ready) + tasks[j]['work'] / freqs[r]
            energy = kappa * (freqs[r] * 1e9) ** 3 * tasks[j]['work'] / freqs[r] + comm_energy
            score = finish / time_scale + weight * energy / energy_scale
            options.append((score, finish, r, link))
        _, ends[j], loc[j], link_free = min(options)
        available[loc[j]] = ends[j]
    return loc, order


def make_planner(method, config):
    def planner(problem, state, seed, epoch, budget, weight, kappa, network_power):
        ids = sorted(t['id'] for t in problem['tasks'])
        evaluator = ForecastBudget(problem, state, ids, budget, weight, kappa, network_power)
        if method in ('HEFT-List', 'FIFO-Local'):
            if method == 'HEFT-List':
                loc, order = heuristic(problem, state, weight, kappa, network_power)
            else:
                loc = {t['id']: state.placed.get(t['id'], t['owner']) for t in problem['tasks']}
                order = topological(problem)
            evaluator.evaluate([loc[j] for j in ids], order)
            return loc, order, evaluator.used, {'forecast_score': evaluator.best, 'adaptation': method}
        pool_seed = int(digest({'visible_ids': ids, 'scenario': problem['id'], 'epoch': epoch})[:8], 16)
        pool = Pool(problem, state, config['pool_size'], pool_seed)
        rng = np.random.default_rng(seed + 1009 * epoch)
        if method == 'MQHOA':
            auxiliary = mqhoa(pool, evaluator, rng, config['mqhoa'])
        elif method == 'PSO':
            auxiliary = pso(pool, evaluator, rng, config['pso'])
        else:
            raise ValueError(method)
        loc, order = evaluator.solution
        return loc, order, evaluator.used, {**auxiliary, 'forecast_score': evaluator.best,
                                           'trace': evaluator.trace, 'pool_sha256': pool.sha256}
    return planner
