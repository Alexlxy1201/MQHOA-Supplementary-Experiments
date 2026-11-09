"""Work-conserving event simulator; forecast never receives future events."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
import time

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def build_problem(seed, users, servers, nodes, dynamic=False):
    rng = np.random.default_rng(seed)
    tasks, edges, jobs = [], [], []
    for u in range(users):
        ids = list(range(u * nodes, (u + 1) * nodes))
        for j in range(nodes):
            tasks.append({'id': ids[j], 'owner': u, 'work': float(rng.uniform(3, 9)),
                          'pinned': j in (0, nodes - 1)})
        # One real fork/join per job, with optional internal dependency chains.
        middle = ids[1:-1]
        pairs = {(ids[0], j) for j in middle} | {(j, ids[-1]) for j in middle}
        for a, b in zip(middle, middle[1:]):
            if rng.random() < .4:
                pairs.add((a, b))
        for a, b in sorted(pairs):
            edges.append((a, b, float(rng.uniform(.2, .8))))
        jobs.append({'user': u, 'tasks': ids, 'arrival': float(3 * u if dynamic else 0)})
    frequency = [1. + .1 * u for u in range(users)] + [2. + .5 * s for s in range(servers)]
    events = []
    if dynamic:
        events = [{'time': 4., 'kind': 'compute', 'resource': users, 'value': .5},
                  {'time': 7., 'kind': 'link', 'value': .5},
                  {'time': 12., 'kind': 'compute', 'resource': users, 'value': 1.25},
                  {'time': 15., 'kind': 'link', 'value': 1.4}]
    releases = {str(t['id']): float(3*t['owner'] + (5 if t['id'] % nodes >= nodes-2 else 0)) for t in tasks}
    return {'id': f'lossy-s{seed}', 'seed': seed, 'users': users,
            'servers': servers, 'tasks': tasks, 'edges': edges, 'jobs': jobs,
            'frequency_ghz': frequency, 'events': events, 'dynamic': dynamic,
            'task_release': releases, 'loss_probability': .15, 'loss_seed': seed+3900001}


def loss_draw(problem, a, b, attempt):
    """Common semantic-edge keyed Bernoulli randomness; never passed to planner."""
    seed = np.random.SeedSequence([problem['loss_seed'], int(a), int(b), int(attempt)])
    return float(np.random.default_rng(seed).random())


@dataclass
class State:
    t: float = 0.
    visible: set = field(default_factory=set)
    completed: dict = field(default_factory=dict)
    placed: dict = field(default_factory=dict)
    running: dict = field(default_factory=dict)
    transfer: dict | None = None
    delivered: set = field(default_factory=set)
    task_log: dict = field(default_factory=dict)
    transfer_log: list = field(default_factory=list)
    attempt_log: list = field(default_factory=list)
    segments: list = field(default_factory=list)
    multipliers: list = field(default_factory=list)
    link_rate: float = 1.
    energy: float = 0.


def incoming(problem):
    pre = [[] for _ in problem['tasks']]
    for a, b, amount in problem['edges']:
        pre[b].append((a, amount))
    return pre


def visible_problem(problem, state):
    """Explicit capability boundary: excludes unreleased jobs and all future events."""
    ids = set(state.visible)
    return {'id': problem['id'], 'users': problem['users'], 'servers': problem['servers'],
            'tasks': [copy.deepcopy(t) for t in problem['tasks'] if t['id'] in ids],
            'edges': [(a, b, c) for a, b, c in problem['edges'] if a in ids and b in ids],
            'jobs': [{'user': j['user'], 'arrival': j['arrival'],
                      'tasks': [i for i in j['tasks'] if i in ids]}
                     for j in problem['jobs'] if any(i in ids for i in j['tasks'])],
            'frequency_ghz': list(problem['frequency_ghz']),
            'loss_probability': problem['loss_probability']}


def allowed(problem, task):
    return [task['owner']] if task['pinned'] else [task['owner']] + list(range(problem['users'], problem['users'] + problem['servers']))


def forecast(problem, state, loc, order, kappa=5e-28, network_power=2.):
    """Non-insertion list forecast for currently visible tasks at current rates.

    Running compute and active link operations are frozen. Future rate changes,
    future arrivals and their task definitions are absent from problem.
    """
    tasks = {x['id']: x for x in problem['tasks']}
    if set(order) != set(tasks) or set(loc) != set(tasks):
        raise ValueError('forecast must cover exactly the currently visible tasks')
    available = [state.t] * len(problem['frequency_ghz'])
    end = dict(state.completed)
    positions = dict(state.placed)
    link_free = state.t
    energy = 0.
    for r, operation in state.running.items():
        f = problem['frequency_ghz'][r] * state.multipliers[r]
        finish = state.t + operation['remaining'] / f
        available[r] = finish
        end[operation['task']] = finish
        energy += kappa * (f * 1e9) ** 3 * (finish - state.t)
    current_edge = None
    if state.transfer:
        extra = state.transfer['amount'] * problem['loss_probability'] / (1-problem['loss_probability'])
        link_free = state.t + (state.transfer['remaining'] + extra) / state.link_rate
        current_edge = (state.transfer['a'], state.transfer['b'])
        energy += network_power * (link_free - state.t)
    pre = {j: [] for j in tasks}
    for a, b, c in problem['edges']:
        pre[b].append((a, c))
    for j in order:
        if j in end:
            continue
        task = tasks[j]
        r = positions.get(j, loc[j])
        if r not in allowed(problem, task):
            raise ValueError('invalid placement')
        ready = max([state.t] + [end[a] for a, _ in pre[j]])
        for a, amount in pre[j]:
            if a not in end:
                raise ValueError('non-topological forecast order')
            ready = max(ready, end[a])
            if positions[a] != r and (a, j) not in state.delivered:
                if (a, j) == current_edge:
                    ready = max(ready, link_free)
                else:
                    begin = max(ready, link_free, state.t)
                    expected_amount = amount / (1-problem['loss_probability'])
                    link_free = begin + expected_amount / state.link_rate
                    ready = max(ready, link_free)
                    energy += network_power * expected_amount / state.link_rate
        f = problem['frequency_ghz'][r] * state.multipliers[r]
        start = max(ready, available[r])
        end[j] = start + task['work'] / f
        available[r] = end[j]
        positions[j] = r
        energy += kappa * (f * 1e9) ** 3 * task['work'] / f
    return max(end.values(), default=state.t) - state.t, energy


def dispatch(problem, state, loc, order):
    """Start ready operations only; current placement is frozen on first activity."""
    pre = incoming(problem)
    priority = {j: k for k, j in enumerate(order)}
    tasks = problem['tasks']
    candidates = [j for j in state.visible if j not in state.completed
                  and all(a in state.completed for a, _ in pre[j])]
    candidates.sort(key=lambda j: (priority.get(j, len(tasks)), j))
    active = {x['task'] for x in state.running.values()}
    for j in candidates:
        if j in active:
            continue
        r = state.placed.get(j, loc[j])
        transfers = [(a, amount) for a, amount in pre[j]
                     if state.placed[a] != r and (a, j) not in state.delivered]
        if not transfers and r not in state.running:
            state.placed[j] = r
            state.running[r] = {'task': j, 'remaining': tasks[j]['work']}
            state.task_log[j] = {'task': j, 'resource': r, 'start': state.t, 'finish': None,
                                 'work': tasks[j]['work'], 'energy_j': 0.}
    if state.transfer is None:
        for j in candidates:
            if j in {x['task'] for x in state.running.values()}:
                continue
            r = state.placed.get(j, loc[j])
            missing = [(a, amount) for a, amount in pre[j]
                       if state.placed[a] != r and (a, j) not in state.delivered]
            if missing:
                a, amount = sorted(missing)[0]
                state.placed[j] = r
                state.transfer = {'a': a, 'b': j, 'source_resource': state.placed[a],
                                  'target_resource': r, 'remaining': amount, 'amount': amount,
                                  'start': state.t, 'energy_j': 0., 'attempt_index': 0}
                break


def advance(problem, state, new_time, kappa=5e-28, network_power=2.):
    dt = new_time - state.t
    if dt < -1e-9:
        raise ValueError('backward time')
    for r, op in list(state.running.items()):
        f = problem['frequency_ghz'][r] * state.multipliers[r]
        energy = kappa * (f * 1e9) ** 3 * dt
        work = f * dt
        op['remaining'] -= work
        state.energy += energy
        state.task_log[op['task']]['energy_j'] += energy
        state.segments.append({'kind': 'compute', 'task': op['task'], 'resource': r,
                               'start': state.t, 'finish': new_time, 'rate': f,
                               'work': work, 'energy_j': energy})
        if op['remaining'] <= 1e-8:
            state.completed[op['task']] = new_time
            state.task_log[op['task']]['finish'] = new_time
            del state.running[r]
    if state.transfer:
        op = state.transfer
        energy = network_power * dt
        work = state.link_rate * dt
        op['remaining'] -= work
        op['energy_j'] += energy
        state.energy += energy
        state.segments.append({'kind': 'transfer', 'a': op['a'], 'b': op['b'],
                               'start': state.t, 'finish': new_time, 'rate': state.link_rate,
                               'work': work, 'energy_j': energy, 'attempt_index': op['attempt_index']})
        if op['remaining'] <= 1e-8:
            draw = loss_draw(problem, op['a'], op['b'], op['attempt_index'])
            failed = draw < problem['loss_probability']
            record = {k: v for k, v in op.items() if k != 'remaining'} | {
                'finish': new_time, 'uniform_draw': draw, 'failed': failed}
            state.attempt_log.append(record)
            if failed:
                state.transfer = {**op, 'remaining': op['amount'], 'start': new_time,
                                  'energy_j': 0., 'attempt_index': op['attempt_index']+1}
            else:
                state.delivered.add((op['a'], op['b']))
                state.transfer_log.append(record)
                state.transfer = None
    state.t = new_time


def simulate(problem, planner, seed, budget, weight, kappa=5e-28, network_power=2., decision_slots=10):
    state = State(multipliers=[1.] * len(problem['frequency_ghz']))
    arrivals = sorted((at, int(i)) for i, at in problem['task_release'].items())
    events = sorted(problem.get('events', []), key=lambda x: (x['time'], x['kind']))
    # A fixed public slot count, never the number of future arrivals/events.
    # Unused slots remain unused; no final-epoch look-ahead redistribution.
    allocation = [budget // decision_slots + (i < budget % decision_slots) for i in range(decision_slots)]
    arrived, applied, observations, plans = set(), set(), [], []
    loc, order = {}, []
    calls, planning_seconds = 0, 0.
    while len(state.completed) < len(problem['tasks']):
        changed = False
        for at, task in arrivals:
            if at <= state.t + 1e-9 and task not in arrived:
                state.visible.add(task)
                arrived.add(task)
                changed = True
        for i, event in enumerate(events):
            if event['time'] <= state.t + 1e-9 and i not in applied:
                if event['kind'] == 'compute':
                    state.multipliers[event['resource']] = event['value']
                else:
                    state.link_rate = event['value']
                applied.add(i)
                changed = True
        if changed and state.visible - state.completed.keys():
            observed = visible_problem(problem, state)
            epoch = len(observations)
            tic = time.perf_counter()
            if epoch >= decision_slots:
                raise RuntimeError('predeclared decision-slot cap exceeded')
            loc, order, used, auxiliary = planner(observed, copy.deepcopy(state), seed, epoch,
                                                 allocation[epoch], weight, kappa, network_power)
            planning_seconds += time.perf_counter() - tic
            calls += used
            observations.append({'time': state.t, 'visible_tasks': sorted(state.visible),
                                 'completed_tasks': sorted(state.completed), 'started_placements': dict(state.placed),
                                 'compute_multipliers': list(state.multipliers), 'link_rate': state.link_rate,
                                 'visible_problem_sha256': digest(observed), 'evaluations': used})
            plans.append({'time': state.t, 'placement': loc, 'order': order, 'auxiliary': auxiliary})
        dispatch(problem, state, loc, order)
        future = [at for at, u in arrivals if u not in arrived]
        future += [e['time'] for i, e in enumerate(events) if i not in applied]
        for r, op in state.running.items():
            f = problem['frequency_ghz'][r] * state.multipliers[r]
            future.append(state.t + op['remaining'] / f)
        if state.transfer:
            future.append(state.t + state.transfer['remaining'] / state.link_rate)
        if not future:
            raise RuntimeError('deadlock')
        next_time = min(x for x in future if x > state.t + 1e-10)
        advance(problem, state, next_time, kappa, network_power)
    response = [max(state.completed[i] for i in j['tasks']) - j['arrival'] for j in problem['jobs']]
    local_compute = sum(x['energy_j'] for x in state.task_log.values() if x['resource'] < problem['users'])
    server_compute = sum(x['energy_j'] for x in state.task_log.values() if x['resource'] >= problem['users'])
    network_energy = sum(x['energy_j'] for x in state.attempt_log)
    device_network = sum(x['energy_j'] / 2 * ((x['source_resource'] < problem['users'])
                                               + (x['target_resource'] < problem['users'])) for x in state.attempt_log)
    result = {'makespan': state.t, 'mean_response_time': float(np.mean(response)),
              'max_response_time': max(response), 'min_response_time': min(response),
              'energy_j': state.energy, 'per_user_response': response, 'evals_used': calls,
              'local_compute_energy_j': local_compute, 'server_compute_energy_j': server_compute,
              'network_energy_j': network_energy, 'device_network_energy_j': device_network,
              'device_energy_j': local_compute + device_network,
              'planning_wall_seconds': planning_seconds, 'virtual_time_excludes_planning': True,
              'task_schedule': [state.task_log[i] for i in sorted(state.task_log)],
              'transfer_schedule': state.transfer_log, 'segments': state.segments,
              'attempt_schedule': state.attempt_log,
              'attempts': len(state.attempt_log),
              'failed_attempts': sum(x['failed'] for x in state.attempt_log),
              'useful_payload_work': sum(x['amount'] for x in state.transfer_log),
              'attempt_work': sum(x['amount'] for x in state.attempt_log),
              'failed_attempt_work': sum(x['amount'] for x in state.attempt_log if x['failed']),
              'network_airtime': sum(x['finish']-x['start'] for x in state.attempt_log),
              'observations': observations, 'plans': plans, 'placement': state.placed,
              'budget_cap': budget, 'energy_weight': weight}
    all_local_time = sum(t['work'] / problem['frequency_ghz'][t['owner']] for t in problem['tasks'])
    all_local_energy = sum(kappa * (problem['frequency_ghz'][t['owner']] * 1e9) ** 3
                           * t['work'] / problem['frequency_ghz'][t['owner']] for t in problem['tasks'])
    result['actual_weighted_cost'] = state.t / all_local_time + weight * state.energy / all_local_energy
    result['postrun_reporting_normalizers_all_jobs'] = {'all_local_serial_time': all_local_time,
                                                       'all_local_compute_energy': all_local_energy}
    errors = validate(problem, result, kappa, network_power)
    if errors:
        raise RuntimeError('independent validator failed: ' + str(errors))
    result['validator_ok'] = True
    return result


def _validate_successful_only(problem, result, kappa=5e-28, network_power=2., eps=1e-6):
    """Independent release, ownership, resource, work, rate, energy and leakage check."""
    errors = []
    logs = {x['task']: x for x in result['task_schedule']}
    placement = {int(k): v for k, v in result['placement'].items()}
    if set(logs) != {t['id'] for t in problem['tasks']}:
        errors.append('missing/extra tasks')
        return errors
    arrivals = {int(i): at for i, at in problem['task_release'].items()}
    for task in problem['tasks']:
        log = logs[task['id']]
        if log['resource'] not in allowed(problem, task) or placement[task['id']] != log['resource']:
            errors.append('ownership/pinning')
        if log['start'] < arrivals[task['id']] - eps or log['finish'] < log['start'] - eps:
            errors.append('release/negative duration')
    for r in range(len(problem['frequency_ghz'])):
        calendar = sorted((x['start'], x['finish']) for x in logs.values() if x['resource'] == r)
        if any(b > c + eps for (_, b), (c, _) in zip(calendar, calendar[1:])):
            errors.append('compute overlap')
    transfers = {(x['a'], x['b']): x for x in result['transfer_schedule']}
    if len(transfers) != len(result['transfer_schedule']):
        errors.append('duplicate transfer')
    link_calendar = sorted((x['start'], x['finish']) for x in transfers.values())
    if any(b > c + eps for (_, b), (c, _) in zip(link_calendar, link_calendar[1:])):
        errors.append('link overlap')
    expected_transfers = set()
    for a, b, amount in problem['edges']:
        if logs[b]['start'] < logs[a]['finish'] - eps:
            errors.append('precedence')
        if placement[a] != placement[b]:
            expected_transfers.add((a, b))
            tr = transfers.get((a, b))
            if not tr or tr['start'] < logs[a]['finish'] - eps or tr['finish'] > logs[b]['start'] + eps:
                errors.append('cross-edge communication')
            elif tr['source_resource'] != placement[a] or tr['target_resource'] != placement[b]:
                errors.append('transfer endpoints')
    if set(transfers) != expected_transfers:
        errors.append('extra/missing transfer')
    total_energy, work, transferred = 0., {}, {}
    task_energy = {}
    edge_energy = {}
    for segment in result['segments']:
        duration = segment['finish'] - segment['start']
        midpoint = (segment['finish'] + segment['start']) / 2
        if duration < -eps:
            errors.append('negative segment')
        if segment['kind'] == 'compute':
            r, j = segment['resource'], segment['task']
            multiplier = 1.
            for event in problem.get('events', []):
                if event['kind'] == 'compute' and event['resource'] == r and event['time'] <= midpoint:
                    multiplier = event['value']
            expected_rate = problem['frequency_ghz'][r] * multiplier
            expected_energy = kappa * (expected_rate * 1e9) ** 3 * duration
            work[j] = work.get(j, 0.) + duration * expected_rate
            task_energy[j] = task_energy.get(j, 0.) + expected_energy
            if segment['start'] < logs[j]['start'] - eps or segment['finish'] > logs[j]['finish'] + eps:
                errors.append('compute segment bounds')
        else:
            expected_rate = 1.
            for event in problem.get('events', []):
                if event['kind'] == 'link' and event['time'] <= midpoint:
                    expected_rate = event['value']
            expected_energy = network_power * duration
            edge = (segment['a'], segment['b'])
            transferred[edge] = transferred.get(edge, 0.) + duration * expected_rate
            edge_energy[edge] = edge_energy.get(edge, 0.) + expected_energy
        if abs(segment['rate'] - expected_rate) > eps or abs(segment['work'] - duration * expected_rate) > eps:
            errors.append('rate/work integration')
        if abs(segment['energy_j'] - expected_energy) > eps:
            errors.append('segment energy')
        total_energy += expected_energy
    for task in problem['tasks']:
        if abs(work.get(task['id'], 0) - task['work']) > eps:
            errors.append('compute work')
        if abs(logs[task['id']]['energy_j'] - task_energy.get(task['id'], 0)) > eps:
            errors.append('task energy')
    for a, b, amount in problem['edges']:
        if (a, b) in expected_transfers and abs(transferred.get((a, b), 0) - amount) > eps:
            errors.append('transfer work')
    if abs(total_energy - result['energy_j']) > eps:
        errors.append('total energy')
    for observation, plan in zip(result['observations'], result['plans']):
        visible = {int(i) for i, at in problem['task_release'].items() if at <= observation['time'] + eps}
        if set(observation['visible_tasks']) != visible or set(plan['order']) != visible:
            errors.append('future job visibility')
        plan_placement = {int(k): v for k, v in plan['placement'].items()}
        for j, r in observation['started_placements'].items():
            if plan_placement[int(j)] != r:
                errors.append('started placement changed')
    if result['evals_used'] > result['budget_cap']:
        errors.append('budget overflow')
    return errors


def validate(problem, result, kappa=5e-28, network_power=2., eps=1e-6):
    """Validate actual failed/successful attempts, then an independent success-only view."""
    errors = []
    attempts = result['attempt_schedule']
    by_edge = {}
    energy = 0.
    for attempt in attempts:
        a, b, k = attempt['a'], attempt['b'], attempt['attempt_index']
        by_edge.setdefault((a,b), []).append(attempt)
        expected = loss_draw(problem, a, b, k)
        if abs(attempt['uniform_draw']-expected) > 1e-14 or attempt['failed'] != (expected < problem['loss_probability']):
            errors.append('common Bernoulli draw/outcome')
        segments = [s for s in result['segments'] if s['kind']=='transfer'
                    and (s['a'],s['b'],s['attempt_index'])==(a,b,k)]
        amount = sum(s['work'] for s in segments)
        segment_energy = sum(s['energy_j'] for s in segments)
        duration = attempt['finish']-attempt['start']
        if abs(amount-attempt['amount'])>eps or abs(segment_energy-attempt['energy_j'])>eps:
            errors.append('attempt work/energy')
        if abs(attempt['energy_j']-network_power*duration)>eps:
            errors.append('attempt airtime energy')
        if abs(sum(s['finish']-s['start'] for s in segments)-duration)>eps:
            errors.append('attempt segment coverage')
        for s in segments:
            midpoint=(s['start']+s['finish'])/2
            expected_rate=1.
            for e in problem.get('events',[]):
                if e['kind']=='link' and e['time']<=midpoint:
                    expected_rate=e['value']
            if abs(s['rate']-expected_rate)>eps or abs(s['work']-expected_rate*(s['finish']-s['start']))>eps:
                errors.append('attempt changing link-rate integration')
        energy += attempt['energy_j']
    calendar = sorted((a['start'],a['finish']) for a in attempts)
    if any(b>c+eps for (_,b),(c,_) in zip(calendar,calendar[1:])):
        errors.append('retry link overlap')
    for edge, aa in by_edge.items():
        aa.sort(key=lambda x:x['attempt_index'])
        if [a['attempt_index'] for a in aa] != list(range(len(aa))) or aa[-1]['failed'] or any(not a['failed'] for a in aa[:-1]):
            errors.append('stop-and-wait retry chain')
        if any(abs(a['finish']-b['start'])>eps for a,b in zip(aa,aa[1:])):
            errors.append('retry continuity')
    failed_keys={(a['a'],a['b'],a['attempt_index']) for a in attempts if a['failed']}
    clean = copy.deepcopy(result)
    clean['segments']=[s for s in result['segments'] if s['kind']=='compute'
                       or (s['a'],s['b'],s['attempt_index']) not in failed_keys]
    failed_energy=sum(a['energy_j'] for a in attempts if a['failed'])
    clean['energy_j']-=failed_energy
    errors += _validate_successful_only(problem, clean, kappa, network_power, eps)
    expected_total=sum(s['energy_j'] for s in result['segments'])
    if abs(expected_total-result['energy_j'])>eps or abs(energy-result['network_energy_j'])>eps:
        errors.append('failed energy missing from total')
    if result['attempts']!=len(attempts) or result['failed_attempts']!=len(failed_keys):
        errors.append('attempt denominator')
    if abs(result['attempt_work']-result['useful_payload_work']-result['failed_attempt_work'])>eps:
        errors.append('useful versus transmitted work')
    if len(result['transfer_schedule']) != len(by_edge):
        errors.append('successful payload denominator')
    return errors
