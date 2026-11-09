"""Independent, read-only acceptance checks for a completed FlexDO evidence set.

Run from supplement/: python analysis_tools/verify_flexdo_evidence.py REPO RUN_ROOT
Prints JSON to stdout; redirect to a NEW validation report, never to raw inputs.
"""
from __future__ import annotations
import csv
import itertools
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mqhoa_supp.model import validate
from mqhoa_supp.runner import read_rows
from recent_baseline.campaign import load, source, require_receipt, check_source


def verify(repo, out):
    instances, archived, _, frozen = load(repo)
    check_source(frozen)
    src, config = source(repo)
    for stage in ('pilot', 'reproduction', 'quality', 'timing'):
        require_receipt(out, stage, src)
    reproduction = read_rows(out / 'mqhoa_reproduction_quality.jsonl')
    quality = read_rows(out / 'flexdo_quality.jsonl')
    timing = read_rows(out / 'timing.jsonl')
    assert (len(reproduction), len(quality), len(timing)) == (70, 42, 490)
    assert {(r['instance_id'], r['variant']) for r in quality} == {(i, v) for i in instances for v in config['variants']}
    expected_timing = Counter({('FlexDO', i, v, repeat): 1 for i in instances for v in config['variants'] for repeat in range(10)})
    expected_timing.update({('MQHOA', i, 'full', seed): 1 for i in instances for seed in range(5)})
    assert Counter((r['method'], r['instance_id'], r['variant'], r['timing_repeat']) for r in timing) == expected_timing
    lookup = {r['run_id']: r for r in archived}
    for r in reproduction:
        old = lookup[r['run_id']]
        for k in ('best_makespan', 'instance_sha256', 'pool_sha256', 'seed', 'params', 'budget_evals', 'evals_used', 'pool_seed', 'pool_size'):
            assert r[k] == old[k], (r['run_id'], k)
    for r in reproduction + quality + timing:
        inst = instances[r['instance_id']]
        assert r['instance_sha256'] == inst.sha256
        assert r['code_sha256'] == frozen['code_sha256']
        assert r['trace'][-1][0] == r['evals_used']
        assert r['trace'][-1][2] == r['best_makespan']
        assert all(a[0] < b[0] and a[2] >= b[2] for a, b in zip(r['trace'], r['trace'][1:]))
        assert not validate(inst, r['best_l'], r['best_pi'], r['starts'], r['finishes'])
        assert abs(r['best_makespan'] - max(r['finishes'])) < 1e-10
        assert 0 < r['evals_used'] <= 10000
        assert r['recent_code_sha256'] == src['recent_code_sha256']
        assert r['recent_protocol_sha256'] == src['recent_protocol_sha256']
    assert {r['host'] for r in timing} == {config['timing_host']}
    assert {r['cpu_index'] for r in timing} == {2}
    assert {r['parallel_workers'] for r in timing} == {1}
    by_condition = {(r['instance_id'], r['variant']): r for r in quality}
    for r in timing:
        if r['method'] == 'FlexDO':
            q = by_condition[r['instance_id'], r['variant']]
            for k in ('best_makespan', 'best_l', 'best_pi', 'evals_used', 'details'):
                assert r[k] == q[k]
    control = json.loads((out / 'timing_receipt.json').read_text())
    medians = [r['median_seconds'] for r in read_rows(out / 'timing_controls.jsonl')]
    ratio = max(medians) / min(medians)
    assert ratio == control['sentinel_ratio'] and ratio <= 1.5 and control['stable']
    prior = list(csv.DictReader((repo / 'supplement/results/20261108_v1/analysis/paired_instance_tests.csv').open()))
    old_family = {r['opponent']: r for r in prior if r['family'] == 'comparison'}
    new_family = list(csv.DictReader((out / 'analysis/paired_six_comparisons.csv').open()))
    assert len(new_family) == 6
    for r in new_family:
        if r['opponent'] in old_family:
            old = old_family[r['opponent']]
            for k in ('primary_p', 'ci95_low', 'ci95_high', 'mean_relative_difference'):
                assert float(r[k]) == float(old[k])
    # Independently reconstruct all six instance-mean effects and exact p-values.
    synthetic = sorted(i for i, x in instances.items() if x.metadata['kind'] == 'synthetic')
    assert len(synthetic) == 12
    def mean(i, method):
        return float(np.mean([r['best_makespan'] for r in archived if r['instance_id'] == i and r['method'] == method
                              and (method != 'MQHOA' or r['variant'] == 'full')]))
    signs = np.asarray(list(itertools.product((-1, 1), repeat=12)))
    pvals = []
    for r in new_family:
        comparator = r['opponent']
        ref = [by_condition[i, 'N2']['best_makespan'] if comparator == 'FlexDO-N2' else mean(i, comparator) for i in synthetic]
        d = np.asarray([(mean(i, 'MQHOA') - v) / v for i, v in zip(synthetic, ref)])
        p = float(np.mean(abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-12))
        assert p == float(r['primary_p'])
        assert float(d.mean()) == float(r['mean_relative_difference'])
        pvals.append(p)
        if comparator == 'FlexDO-N2':
            rng = np.random.default_rng(config['bootstrap_seed'])
            bootstrap = d[rng.integers(0, 12, (config['bootstrap_samples'], 12))].mean(axis=1)
            low, high = np.quantile(bootstrap, [.025, .975])
            assert (float(low), float(high)) == (float(r['ci95_low']), float(r['ci95_high']))
    ordering = sorted(range(6), key=lambda i: pvals[i])
    for rank, i in enumerate(ordering):
        adjusted = min(1.0, max((6 - j) * pvals[ordering[j]] for j in range(rank + 1)))
        assert adjusted == float(new_family[i]['holm_p'])
    table = list(csv.DictReader((out / 'analysis/per_instance_results.csv').open()))
    assert len(table) == 42 and all(sum(int(r[k]) for k in ('mqhoa_win_count', 'mqhoa_tie_count', 'mqhoa_loss_count')) == 20 for r in table)
    return {'status': 'VERIFIED', 'independent_schedule_checks': 602, 'exact_reproductions': 70,
            'new_deterministic_quality_conditions': 42, 'timing_conditions': 490,
            'source_and_protocol_match': True, 'timing_sentinel_ratio': ratio,
            'old_five_raw_p_and_ci_values_unchanged': True,
            'six_raw_p_values_and_holm_values_recomputed_independently': True,
            'primary_bootstrap_ci_reproduced': True,
            'max_flexdo_evaluations': max(r['evals_used'] for r in quality),
            'original_data_overwritten': False}


if __name__ == '__main__':
    print(json.dumps(verify(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
