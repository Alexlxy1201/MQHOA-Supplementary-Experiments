"""Reproduce summaries from the complete immutable scenario raw matrix."""
from __future__ import annotations
from collections import defaultdict
import csv
import gzip
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def dump_csv(name, rows):
    with (ROOT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def ci(values, rng, repetitions=10000):
    x = np.asarray(values, dtype=float)
    if len(x) == 1:
        return [float(x[0]), float(x[0])]
    simulated = np.mean(rng.choice(x, (repetitions, len(x))), axis=1)
    return [float(v) for v in np.percentile(simulated, [2.5, 97.5])]


def main():
    if not json.loads((ROOT / 'verification.json').read_text())['verified']:
        raise RuntimeError('complete independently verified raw data required')
    rows = [json.loads(line) for line in gzip.open(ROOT / 'raw_runs.jsonl.gz', 'rt')]
    rng = np.random.default_rng(20261109)
    groups = defaultdict(list)
    for r in rows:
        groups[r['group'], r['scenario_index'], r['energy_weight'], r['method']].append(r)
    metrics = ['makespan', 'mean_response_time', 'max_response_time', 'energy_j', 'device_energy_j',
               'local_compute_energy_j', 'server_compute_energy_j', 'network_energy_j',
               'actual_weighted_cost', 'planning_wall_seconds', 'wall_seconds', 'evals_used']
    summaries = []
    lookup = {}
    for key, values in sorted(groups.items()):
        group, instance, weight, method = key
        summary = {'group': group, 'scenario_index': instance, 'instance_id': values[0]['instance_id'],
                   'energy_weight': weight, 'method': method, 'algorithm_runs': len(values)}
        for metric in metrics:
            x = [r[metric] for r in values]
            bounds = ci(x, rng)
            summary[metric + '_mean'] = float(np.mean(x))
            summary[metric + '_ci95_low'], summary[metric + '_ci95_high'] = bounds
        summaries.append(summary)
        lookup[key] = summary
    dump_csv('per_scenario_summary.csv', summaries)
    tests = []
    for group, metric, weight in [('static', 'makespan', 0.), ('dynamic', 'mean_response_time', 0.),
                                   ('energy', 'actual_weighted_cost', 1.)]:
        for opponent in ('PSO', 'HEFT-List'):
            differences = [(lookup[group, i, weight, 'MQHOA'][metric + '_mean']
                            / lookup[group, i, weight, opponent][metric + '_mean'] - 1) for i in range(6)]
            x = np.asarray(differences)
            signs = np.asarray(list(itertools.product([-1., 1.], repeat=len(x))))
            p = float(np.mean(np.abs(np.mean(signs * x, axis=1)) >= abs(np.mean(x)) - 1e-14))
            bounds = ci(x, rng)
            tests.append({'group': group, 'metric': metric, 'weight': weight, 'reference': 'MQHOA',
                          'opponent': opponent, 'independent_scenarios': 6,
                          'relative_difference_mean': float(np.mean(x)),
                          'ci95_low': bounds[0], 'ci95_high': bounds[1], 'p_raw': p,
                          'scenario_differences': json.dumps(differences),
                          'direction': 'negative favors MQHOA'})
    running = 0.
    for rank, r in enumerate(sorted(tests, key=lambda r: r['p_raw'])):
        running = max(running, min(1., (len(tests) - rank) * r['p_raw']))
        r['p_holm_six'] = running
    dump_csv('primary_paired_tests.csv', tests)
    fronts = []
    for i in range(6):
        for method in ('MQHOA', 'PSO', 'HEFT-List', 'FIFO-Local'):
            candidates = [lookup['energy', i, w, method] for w in (0., .1, .3, 1., 3.)]
            for r in candidates:
                t, e = r['makespan_mean'], r['energy_j_mean']
                nd = not any(q['makespan_mean'] <= t + 1e-10 and q['energy_j_mean'] <= e + 1e-10
                             and (q['makespan_mean'] < t - 1e-10 or q['energy_j_mean'] < e - 1e-10)
                             for q in candidates)
                fronts.append({'scenario_index': i, 'method': method, 'weight': r['energy_weight'],
                               'makespan_mean': t, 'system_energy_j_mean': e,
                               'device_energy_j_mean': r['device_energy_j_mean'],
                               'nondominated_among_same_method_weights': nd})
    dump_csv('observed_energy_tradeoffs.csv', fronts)
    flat = [{k: r[k] for k in ['run_id', 'group', 'scenario_index', 'method', 'seed', 'energy_weight',
                                'makespan', 'mean_response_time', 'energy_j', 'device_energy_j',
                                'actual_weighted_cost', 'planning_wall_seconds', 'wall_seconds', 'evals_used']}
            for r in rows]
    dump_csv('run_metrics.csv', flat)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    colors = {'MQHOA': '#d45500', 'PSO': '#0072b2', 'HEFT-List': '#009e73', 'FIFO-Local': '#777777'}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (group, metric, weight, title) in zip(axes, [('static', 'makespan', 0., 'Static: makespan'),
                ('dynamic', 'mean_response_time', 0., 'Online: mean response'),
                ('energy', 'actual_weighted_cost', 1., 'Energy: weighted cost (w=1)')]):
        for method in colors:
            values = [lookup[group, i, weight, method][metric + '_mean'] /
                      lookup[group, i, weight, 'FIFO-Local'][metric + '_mean'] for i in range(6)]
            ax.plot(range(1, 7), values, marker='o', label=method, color=colors[method])
        ax.set_title(title)
        ax.set_xlabel('Independent scenario')
        ax.set_xticks(range(1, 7))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean / FIFO-Local')
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / 'scenario_comparison.pdf', bbox_inches='tight')
    fig.savefig(ROOT / 'scenario_comparison.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8))
    for i, ax in enumerate(axes.flat):
        for method in colors:
            vals = [r for r in fronts if r['scenario_index'] == i and r['method'] == method]
            ax.plot([r['makespan_mean'] for r in vals], [r['system_energy_j_mean'] for r in vals],
                    'o-', color=colors[method], label=method, markersize=4)
        ax.set_title(f'Scenario {i+1}')
        ax.set_xlabel('Mean makespan (simulated s)')
        ax.set_ylabel('Mean system dynamic energy (J)')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / 'energy_tradeoff.pdf', bbox_inches='tight')
    fig.savefig(ROOT / 'energy_tradeoff.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    exemplar = next(r for r in rows if r['group'] == 'dynamic' and r['scenario_index'] == 5
                    and r['method'] == 'MQHOA' and r['seed'] == 0)
    problems = json.loads((ROOT / 'problems.json').read_text())
    p = problems['dynamic-i5']
    fig, ax = plt.subplots(figsize=(11, 4.5))
    palette = ['#0072b2', '#d45500', '#009e73']
    for task in exemplar['task_schedule']:
        owner = p['tasks'][task['task']]['owner']
        y = task['resource']
        ax.broken_barh([(task['start'], task['finish'] - task['start'])], (y - .35, .7), facecolors=palette[owner], edgecolors='white', linewidth=.6)
        ax.text((task['start'] + task['finish'])/2, y, str(task['task']), va='center', ha='center', fontsize=8, color='white')
    network_y = len(p['frequency_ghz'])
    for tr in exemplar['transfer_schedule']:
        ax.broken_barh([(tr['start'], tr['finish'] - tr['start'])], (network_y - .35, .7), facecolors='#555555')
    for event in p['events']:
        ax.axvline(event['time'], linestyle='--', linewidth=.8, color='black', alpha=.45)
    ax.set_yticks(range(network_y+1), [f'Local {j}' if j < p['users'] else f'Server {j-p["users"]}'
                                    for j in range(network_y)] + ['Shared link'])
    ax.set_xlabel('Simulated time (s); dashed lines: resource/link rate events')
    ax.set_title('Online execution: scenario 6, MQHOA seed 0; colors identify users')
    fig.tight_layout()
    fig.savefig(ROOT / 'online_execution.pdf', bbox_inches='tight')
    fig.savefig(ROOT / 'online_execution.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    output = {'runs': len(rows), 'independent_scenarios': 6, 'primary_tests': tests,
              'all_comparison_conditions': len(summaries), 'front_points': len(fronts),
              'maximum_planning_wall_seconds': max(r['planning_wall_seconds'] for r in rows),
              'raw_rows_by_group': {g: sum(r['group'] == g for r in rows) for g in ('static', 'dynamic', 'energy')}}
    (ROOT / 'analysis_summary.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
