from __future__ import annotations
import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from mqhoa_supp.model import Instance, canonical, digest, schedule
from mqhoa_supp.runner import run_one, code_hash, provenance, append_json, read_rows
from .flexdo import solve


def load(repo):
    archive = Path(repo) / 'supplement/results/20261108_v1/raw/quality_dataset.zip'
    with ZipFile(archive) as z:
        config = json.loads(z.read('protocol.json'))
        frozen = json.loads(z.read('frozen_defaults.json'))
        instances = {i.identifier: i for n in sorted(z.namelist()) if n.startswith('instances/')
                     and n.endswith('.json') for i in [Instance(**json.loads(z.read(n)))]
                     if i.metadata.get('split') != 'tuning'}
        rows = [json.loads(line) for n in sorted(z.namelist())
                if n.startswith('results/evaluation-shard') and n.endswith('.jsonl')
                for line in z.read(n).decode().splitlines() if line]
    return instances, rows, config, frozen


def source(repo):
    settings = json.loads(Path(__file__).with_name('config.json').read_text())
    files = sorted(Path(__file__).parent.rglob('*.py')) + [Path(__file__).with_name('config.json'),
             Path(repo) / 'supplement/RECENT_BASELINE_PLAN.md']
    return {**provenance(), 'recent_code_sha256': digest([(p.name, hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]),
            'recent_protocol_sha256': digest(settings)}, settings


def base_tasks(rows, seeds):
    return sorted([r for r in rows if r['method'] == 'MQHOA' and r['variant'] == 'full'
                   and r['seed'] in seeds], key=lambda r: (r['instance_id'], r['seed']))


def mq_run(old, instance, src):
    return run_one((dict(old), instance.data(), src))


def flex_run(instance, variant, src, config):
    start = time.perf_counter()
    row = solve(instance, variant, config['budget_evals_cap'])
    return {**row, **src, 't_total': time.perf_counter() - start, 'seed': None}


def check_source(frozen):
    # The archived Windows implementation hashes native path separators too.
    # Reconstruct that byte-for-byte rule on Mac; never change the old engine.
    from mqhoa_supp import runner
    root = Path(runner.__file__).resolve().parent.parent
    files = sorted(list((root / 'mqhoa_supp').rglob('*.py')) + list((root / 'tests').rglob('*.py')))
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(root).as_posix().replace('/', '\\').encode())
        h.update(p.read_bytes())
    if h.hexdigest() != frozen['code_sha256']:
        raise RuntimeError('archived Windows-path MQHOA package hash differs; no gate bypass allowed')
    if platform.system() == 'Windows' and code_hash() != frozen['code_sha256']:
        raise RuntimeError('native Windows source mismatch')


def receipt(out, stage, src, **extra):
    p = out / (stage + '_receipt.json')
    if p.exists():
        raise ValueError('refusing to overwrite stage receipt')
    p.write_text(canonical({**src, **extra}) + '\n')


def require_receipt(out, stage, src):
    old = json.loads((out / (stage + '_receipt.json')).read_text())
    if old['recent_code_sha256'] != src['recent_code_sha256'] or old['recent_protocol_sha256'] != src['recent_protocol_sha256']:
        raise RuntimeError('code/protocol changed after stage; use a new output directory')
    return old


def pilot(repo, out):
    import unittest
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / 'tests'))
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise RuntimeError('tests failed')
    src, config = source(repo)
    instances, old, _, frozen = load(repo)
    check_source(frozen)
    if (out / 'pilot_receipt.json').exists():
        raise ValueError('pilot already exists')
    if any(out.glob('*.jsonl')):
        raise ValueError('new output root required')
    out.mkdir(parents=True, exist_ok=True)
    timings = []
    for n in (10, 50, 100):
        inst = next(i for k, i in sorted(instances.items()) if i.n == n)
        for variant in config['variants']:
            row = flex_run(inst, variant, src, config)
            timings.append({'n': n, 'variant': variant, 'seconds': row['t_total'], 'evals': row['evals_used']})
    mq = base_tasks(old, [0])
    mq_times = []
    for n in (10, 50, 100):
        task = next(t for t in mq if instances[t['instance_id']].n == n)
        row = mq_run(task, instances[task['instance_id']], src)
        if row['best_makespan'] != task['best_makespan'] or row['pool_sha256'] != task['pool_sha256']:
            raise RuntimeError('pilot MQHOA reproduction mismatch')
        mq_times.append({'n': n, 'seconds': row['t_total']})
    estimate = sum(next(x['seconds'] for x in timings if x['n'] == i.n and x['variant'] == v)
                   for i in instances.values() for v in config['variants'])
    mq_estimate = sum(next(x['seconds'] for x in mq_times if x['n'] == i.n) * 5 for i in instances.values())
    receipt(out, 'pilot', src, tests_passed=True, flexdo_pilots=timings, mqhoa_pilots=mq_times,
            estimated_quality_seconds=estimate + mq_estimate,
            estimated_timing_seconds=10 * estimate + mq_estimate + config['warmup_seconds'],
            estimate_scope='pilot projection, excludes sentinels and data I/O; not a guarantee')
    print((out / 'pilot_receipt.json').read_text(), flush=True)


def quality(repo, out):
    src, config = source(repo)
    require_receipt(out, 'pilot', src)
    instances, old, _, frozen = load(repo)
    check_source(frozen)
    if list(out.glob('*quality*.jsonl')) or (out / 'quality_receipt.json').exists():
        raise ValueError('quality attempt exists; preserve it, no automatic retry')
    tasks = base_tasks(old, config['reproduction_seeds'])
    if len(tasks) != 70 or len(instances) != 14:
        raise ValueError('incomplete archived matrix')
    parity = []
    for k, task in enumerate(tasks):
        row = mq_run(task, instances[task['instance_id']], src)
        for key in ('best_makespan', 'instance_sha256', 'pool_sha256', 'params', 'seed', 'evals_used', 'pool_seed'):
            if row[key] != task[key]:
                raise RuntimeError('reproduction mismatch: ' + str((task['instance_id'], task['seed'], key)))
        append_json(out / 'mqhoa_reproduction_quality.jsonl', row)
        parity.append({'instance_id': row['instance_id'], 'seed': row['seed'], 'exact_match': True})
        if (k + 1) % 10 == 0:
            print('reproduction', k + 1, '/70', flush=True)
    receipt(out, 'reproduction', src, matches=parity, exact_matches=70)
    for k, (identifier, variant) in enumerate((i, v) for i in sorted(instances) for v in config['variants']):
        row = flex_run(instances[identifier], variant, src, config)
        append_json(out / 'flexdo_quality.jsonl', row)
        print('FlexDO', k + 1, '/42', identifier, variant, row['best_makespan'], row['evals_used'], flush=True)
    receipt(out, 'quality', src, reproduction_runs=70, flexdo_runs=42)


def timing(repo, out):
    # Use the original Windows affinity helper without editing the old tool.
    import sys
    import statistics
    sys.path.insert(0, str(Path(repo) / 'supplement/analysis_tools'))
    from pinned_timing import pin_current_process
    src, config = source(repo)
    require_receipt(out, 'quality', src)
    if platform.node() != config['timing_host']:
        raise RuntimeError('dedicated timing requires the original recorded host')
    if (out / 'timing.jsonl').exists() or (out / 'timing_receipt.json').exists():
        raise ValueError('timing attempt exists; preserve it, no automatic retry')
    affinity = pin_current_process(config['timing_cpu_index'])
    instances, old, _, frozen = load(repo)
    check_source(frozen)
    jobs = [('FlexDO', i, v, repeat) for i in sorted(instances)
            for v in config['variants'] for repeat in range(config['timing_repeats_per_flexdo_variant'])]
    mq = {(r['instance_id'], r['seed']): r for r in base_tasks(old, config['timing_mqhoa_seeds'])}
    jobs += [('MQHOA', i, 'full', seed) for i, seed in sorted(mq)]
    if len(jobs) != 490:
        raise ValueError('incorrect timing matrix')
    random.Random(config['shuffle_seed']).shuffle(jobs)
    anchor = instances['archive10']
    started = time.perf_counter()
    while time.perf_counter() - started < config['warmup_seconds']:
        schedule(anchor, [0] * anchor.n, anchor.topo, check=False)
    controls = []

    def sentinel(position):
        samples = []
        for _ in range(3):
            tic = time.perf_counter()
            for _ in range(10000):
                schedule(anchor, [0] * anchor.n, anchor.topo, check=False)
            samples.append(time.perf_counter() - tic)
        row = {'after_timing_runs': position, 'samples_seconds': samples,
               'median_seconds': statistics.median(samples)}
        controls.append(row)
        append_json(out / 'timing_controls.jsonl', row)

    sentinel(0)
    for index, (method, identifier, variant, repeat) in enumerate(jobs, 1):
        inst = instances[identifier]
        row = flex_run(inst, variant, src, config) if method == 'FlexDO' else mq_run(mq[identifier, repeat], inst, src)
        row.update({'timing_repeat': repeat, 'parallel_workers': 1,
                    'cpu_index': config['timing_cpu_index'], 'phase': 'recent_timing'})
        append_json(out / 'timing.jsonl', row)
        if index % 10 == 0 or index == len(jobs):
            print('timing', index, '/490', flush=True)
        if index % 50 == 0 or index == len(jobs):
            sentinel(index)
    medians = [c['median_seconds'] for c in controls]
    ratio = max(medians) / min(medians)
    receipt(out, 'timing', src, runs=490, affinity=affinity, controls=controls,
            sentinel_ratio=ratio, stable=ratio <= config['stability_ratio_limit'],
            elapsed_seconds=time.perf_counter() - started)
    if ratio > config['stability_ratio_limit']:
        raise RuntimeError('timing stability gate failed; preserve attempt for diagnosis')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['pilot', 'quality', 'timing', 'analyze'])
    parser.add_argument('--repo', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.stage == 'analyze':
        from .analysis import analyze
        analyze(Path(args.repo), Path(args.out))
    else:
        globals()[args.stage](Path(args.repo), Path(args.out))
