from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import unittest

import numpy as np

from .methods import make_planner
from .model import build_problem, digest, simulate, validate

HERE = Path(__file__).resolve().parent


def config():
    return json.loads((HERE / 'config.json').read_text())


def source():
    files = sorted([p for p in HERE.rglob('*') if p.suffix in ('.py', '.json', '.md')])
    return digest({str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def problem(config, index, group):
    p = build_problem(config['instance_seeds'][index], *config['layouts'][index], dynamic=True)
    p['loss_probability'] = config['loss_probability']
    nodes = config['layouts'][index][2]
    p['task_release'] = {str(t['id']): float(3*t['owner'] + (config['incremental_delay']
                         if t['id'] % nodes >= nodes-2 else 0)) for t in p['tasks']}
    return p


def row(config, index, group, method, seed, weight, run_id):
    p = problem(config, index, group)
    tic = time.perf_counter()
    r = simulate(p, make_planner(method, config), seed, config['budget_per_run'], weight,
                 config['kappa_si'], config['network_dynamic_power_w'],
                 decision_slots=config['decision_slots'])
    return {**r, 'run_id': run_id, 'group': group, 'instance_id': p['id'],
            'problem_sha256': digest(p), 'method': method, 'seed': seed,
            'wall_seconds': time.perf_counter() - tic, 'scenario_index': index,
            'source_sha256': source(), 'config_sha256': digest(config),
            'host': platform.node(), 'pid': os.getpid()}


def matrix(config):
    rows = []
    for group in ('online_lossy_incremental',):
        weights = [0.]
        for i in range(len(config['instance_seeds'])):
            for weight in weights:
                for method in ('MQHOA', 'PSO', 'HEFT-List', 'FIFO-Local'):
                    seeds = config['run_seeds'] if method in ('MQHOA', 'PSO') else [0]
                    for seed in seeds:
                        identifier = f'{group}-i{i}-w{weight:g}-{method}-seed{seed}'
                        rows.append((i, group, method, seed, weight, identifier))
    # Seeded interleaving protects wall-time descriptive comparisons from order drift.
    rng = np.random.default_rng(20261109)
    return [rows[int(k)] for k in rng.permutation(len(rows))]


def pilot(out):
    cfg = config()
    if (out / 'pilot_receipt.json').exists():
        raise RuntimeError('pilot exists; use separate attempt output')
    out.mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.discover(str(HERE / 'tests'))
    with (out / 'tests_formal_gate.txt').open('w') as log:
        passed = unittest.TextTestRunner(stream=log, verbosity=2).run(suite).wasSuccessful()
    if not passed:
        raise RuntimeError('tests failed')
    records = []
    with gzip.open(out / 'pilot_runs.jsonl.gz', 'wt', encoding='utf-8') as handle:
        for i in (0, 5):
            for group in ('online_lossy_incremental',):
                for method in ('MQHOA', 'PSO', 'HEFT-List', 'FIFO-Local'):
                    r = row(cfg, i, group, method, 1000, 0.,
                            f'pilot-{i}-{group}-{method}')
                    handle.write(json.dumps(r, separators=(',', ':'), allow_nan=False) + '\n')
                    records.append({k: r[k] for k in ('scenario_index', 'group', 'method', 'wall_seconds',
                                                      'planning_wall_seconds', 'evals_used', 'validator_ok')})
    # Conservative projection uses larger measured small/large-case time per group/method.
    estimate = sum(max(r['wall_seconds'] for r in records if r['group'] == t[1] and r['method'] == t[2])
                   for t in matrix(cfg))
    receipt = {'source_sha256': source(), 'config_sha256': digest(cfg), 'tests_passed': True,
               'pilot_rows': records, 'formal_runs': len(matrix(cfg)), 'estimated_seconds': estimate,
               'estimate_not_guarantee': True, 'environment': {'python': sys.version, 'numpy': np.__version__,
                     'platform': platform.platform(), 'host': platform.node()},
               'time': time.time(), 'pid': os.getpid()}
    dump(out / 'pilot_receipt.json', receipt)
    dump(out / 'frozen_config.json', cfg)
    dump(out / 'frozen_matrix.json', matrix(cfg))
    dump(out / 'problems.json', {f'{g}-i{i}': problem(cfg, i, g)
                                for g in ('online_lossy_incremental',) for i in range(6)})
    print(json.dumps({'formal_runs': len(matrix(cfg)), 'estimated_seconds': estimate,
                      'source_sha256': source(), 'pilot_rows': len(records)}), flush=True)


def run(out):
    cfg = config()
    receipt = json.loads((out / 'pilot_receipt.json').read_text())
    if source() != receipt['source_sha256'] or digest(cfg) != receipt['config_sha256']:
        raise RuntimeError('code/protocol changed after pilot; new attempt needed')
    if (out / 'raw_runs.jsonl.gz').exists():
        raise RuntimeError('formal run exists; preserve it; no silent retry')
    tic = time.perf_counter()
    with gzip.open(out / 'raw_runs.jsonl.gz', 'wt', encoding='utf-8') as handle:
        for k, task in enumerate(matrix(cfg)):
            if time.perf_counter() - tic > cfg['hard_timeout_seconds']:
                raise TimeoutError('predeclared 4-hour hard timeout')
            r = row(cfg, *task)
            handle.write(json.dumps(r, separators=(',', ':'), allow_nan=False) + '\n')
            handle.flush()
            if k % 25 == 0 or k + 1 == len(matrix(cfg)):
                print(json.dumps({'completed': k + 1, 'total': len(matrix(cfg)), 'last_run': task[-1],
                                  'elapsed_seconds': time.perf_counter() - tic, 'pid': os.getpid()}), flush=True)
    dump(out / 'execution_receipt.json', {'exit_code': 0, 'runs': len(matrix(cfg)),
          'seconds': time.perf_counter() - tic, 'source_sha256': source(), 'config_sha256': digest(cfg),
          'raw_sha256': hashlib.sha256((out / 'raw_runs.jsonl.gz').read_bytes()).hexdigest()})


def verify(out):
    cfg = json.loads((out / 'frozen_config.json').read_text())
    expected = {x[-1] for x in matrix(cfg)}
    seen, calls, dynamic_changes, adverse = set(), 0, 0, []
    task_count, transfer_count, segments, observations, attempts, failed = 0,0,0,0,0,0
    successful_work, attempted_work, failed_work, airtime, network_energy = 0.,0.,0.,0.,0.
    with gzip.open(out / 'raw_runs.jsonl.gz', 'rt', encoding='utf-8') as handle:
        for line in handle:
            r = json.loads(line)
            if r['run_id'] in seen:
                raise RuntimeError('duplicate run')
            seen.add(r['run_id'])
            p = problem(cfg, r['scenario_index'], r['group'])
            if r['problem_sha256'] != digest(p):
                raise RuntimeError('input hash mismatch')
            errors = validate(p, r, cfg['kappa_si'], cfg['network_dynamic_power_w'])
            if errors:
                raise RuntimeError((r['run_id'], errors))
            if r['evals_used'] != sum(x['evaluations'] for x in r['observations']):
                raise RuntimeError('evaluation sum mismatch')
            if r['method'] in ('MQHOA', 'PSO') and r['evals_used'] != cfg['budget_per_run']:
                adverse.append({'run_id': r['run_id'], 'unused_calls': cfg['budget_per_run'] - r['evals_used']})
            if r['group'] == 'online_lossy_incremental':
                for j in range(len(p['tasks'])):
                    rates = {x['rate'] for x in r['segments'] if x['kind'] == 'compute' and x['task'] == j}
                    dynamic_changes += len(rates) > 1
            calls += r['evals_used']
            task_count += len(r['task_schedule'])
            transfer_count += len(r['transfer_schedule'])
            segments += len(r['segments'])
            observations += len(r['observations'])
            attempts += r['attempts']
            failed += r['failed_attempts']
            successful_work += r['useful_payload_work']
            attempted_work += r['attempt_work']
            failed_work += r['failed_attempt_work']
            airtime += r['network_airtime']
            network_energy += r['network_energy_j']
    if seen != expected:
        raise RuntimeError('incomplete/extra campaign matrix')
    result = {'verified': True, 'runs': len(seen), 'independent_scenarios': 6,
              'total_evaluations': calls, 'mid_running_task_rate_changes': dynamic_changes,
              'unused_budget_rows': adverse, 'all_constraints_energy_work_checked': True,
              'task_schedules': task_count, 'successful_payload_transfers': transfer_count,
              'compute_and_transfer_segments': segments, 'online_observations':observations,
              'attempts': attempts, 'failed_attempts': failed,
              'useful_payload_work':successful_work, 'attempt_work':attempted_work,
              'failed_attempt_work':failed_work, 'network_airtime_s':airtime,
              'network_dynamic_energy_j':network_energy}
    dump(out / 'verification.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['pilot', 'run', 'verify'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    {'pilot': pilot, 'run': run, 'verify': verify}[args.stage](args.out)


if __name__ == '__main__':
    main()
