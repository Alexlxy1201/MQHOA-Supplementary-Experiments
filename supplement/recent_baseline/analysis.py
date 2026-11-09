from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from mqhoa_supp.statistics import paired_analysis, holm, write_csv
from mqhoa_supp.runner import read_rows
from .campaign import load, source, require_receipt


def analyze(repo, out):
    src, config = source(repo)
    require_receipt(out, 'quality', src)
    timing_receipt = require_receipt(out, 'timing', src)
    if not timing_receipt['stable']:
        raise RuntimeError('timing failed stability gate')
    instances, old, _, frozen = load(repo)
    quality = read_rows(out / 'flexdo_quality.jsonl')
    reproduction = read_rows(out / 'mqhoa_reproduction_quality.jsonl')
    timing = read_rows(out / 'timing.jsonl')
    if len(quality) != 42 or len(reproduction) != 70 or len(timing) != 490:
        raise ValueError('incomplete new matrix')
    if len({(r['instance_id'], r['variant']) for r in quality}) != 42:
        raise ValueError('duplicate quality condition')
    if any(not r['validator_ok'] or r['evals_used'] > 10000 for r in quality + reproduction + timing):
        raise ValueError('validation or budget failure')
    if any(r['recent_code_sha256'] != src['recent_code_sha256'] for r in quality + timing):
        raise ValueError('mixed recent source')
    destination = out / 'analysis'
    if destination.exists():
        raise ValueError('preserve existing analysis directory')
    destination.mkdir()
    archive_analysis = Path(repo) / 'supplement/results/20261108_v1/analysis'
    prior = list(csv.DictReader((archive_analysis / 'paired_instance_tests.csv').open()))
    prior_means = {(r['instance_id'], r['method']): r for r in csv.DictReader((archive_analysis / 'per_instance_summary.csv').open())}
    groups = defaultdict(list)
    for r in old:
        if r['method'] == 'MQHOA' and r['variant'] == 'full':
            groups[r['instance_id'], 'MQHOA'].append(r)
        elif r['method'] in ('PSO', 'GA', 'HEFT', 'RandomPool', 'DirectRandom'):
            groups[r['instance_id'], r['method']].append(r)
    means = {k: float(np.mean([r['best_makespan'] for r in rows])) for k, rows in groups.items()}
    results = {(r['instance_id'], r['variant']): r for r in quality}
    synthetic = sorted(k for k, inst in instances.items() if inst.metadata['kind'] == 'synthetic')
    rows = []
    for identifier in sorted(instances):
        scores = np.asarray([r['best_makespan'] for r in groups[identifier, 'MQHOA']])
        if len(scores) != 20:
            raise ValueError('MQHOA requires 20 archived seeds per instance')
        for variant in config['variants']:
            r = results[identifier, variant]
            matching = [x for x in timing if x['method'] == 'FlexDO' and x['instance_id'] == identifier and x['variant'] == variant]
            mq_times = [x['t_total'] for x in timing if x['method'] == 'MQHOA' and x['instance_id'] == identifier]
            if len(matching) != 10 or len(mq_times) != 5:
                raise ValueError('incorrect per-condition timing replication')
            value = r['best_makespan']
            rows.append({'instance_id': identifier, 'kind': instances[identifier].metadata['kind'], 'n': instances[identifier].n,
                         'variant': variant, 'flexdo_makespan': value, 'mqhoa_mean': float(scores.mean()),
                         'mqhoa_ci95_low': float(prior_means[identifier, 'MQHOA']['ci95_low']),
                         'mqhoa_ci95_high': float(prior_means[identifier, 'MQHOA']['ci95_high']),
                         'mqhoa_win_count': int(np.sum(scores < value - 1e-9)), 'mqhoa_tie_count': int(np.sum(abs(scores - value) <= 1e-9)),
                         'mqhoa_loss_count': int(np.sum(scores > value + 1e-9)),
                         'mqhoa_relative_difference': (float(scores.mean()) - value) / value,
                         'flexdo_evals': r['evals_used'], 'flexdo_greedy_candidates': r['details']['greedy_distinct'],
                         'flexdo_enumeration_attempted': r['details']['enumeration_attempted'],
                         'flexdo_time_mean_s': float(np.mean([x['t_total'] for x in matching])),
                         'mqhoa_time_mean_s': float(np.mean(mq_times)),
                         'mqhoa_over_flexdo_time_ratio': float(np.mean(mq_times) / np.mean([x['t_total'] for x in matching]))})
    write_csv(destination / 'per_instance_results.csv', rows)
    rng = np.random.default_rng(config['bootstrap_seed'])
    family = []
    for row in prior:
        if row['family'] != 'comparison':
            continue
        opponent = row['opponent']
        differences = [(means[i, 'MQHOA'] - means[i, opponent]) / means[i, opponent] for i in synthetic]
        p = paired_analysis(differences, np.random.default_rng(0), 10)['primary_p']
        if p != float(row['primary_p']):
            raise RuntimeError('archived raw p-value does not reproduce: ' + opponent)
        family.append({**row, 'primary_p': p, 'holm_p_original_five': float(row['holm_p']), 'analysis_scope': 'expanded-six-comparison-family'})
    differences = [(means[i, 'MQHOA'] - results[i, 'N2']['best_makespan']) / results[i, 'N2']['best_makespan'] for i in synthetic]
    new = {'family': 'comparison', 'reference': 'MQHOA', 'opponent': 'FlexDO-N2', 'direction': 'negative favours MQHOA',
           **paired_analysis(differences, rng, config['bootstrap_samples']),
           'holm_p_original_five': '', 'analysis_scope': 'expanded-six-comparison-family'}
    family.append(new)
    holm(family)
    # Keep one consistent schema and original CIs unchanged for the older rows.
    keys = list(dict.fromkeys(k for row in family for k in row))
    write_csv(destination / 'paired_six_comparisons.csv', [{k: r.get(k, '') for k in keys} for r in family])
    write_csv(destination / 'reproduction_check.csv', [{'instance_id': r['instance_id'], 'seed': r['seed'],
               'best_makespan': r['best_makespan'], 'exact_match': True} for r in reproduction])
    _plots(destination, synthetic, rows, groups, results)
    primary_rows = [r for r in rows if r['variant'] == 'N2' and r['kind'] == 'synthetic']
    mean_relative = float(np.mean([r['mqhoa_relative_difference'] for r in primary_rows]))
    receipt = {'reproduction_exact_matches': 70, 'quality_conditions': 42, 'timing_runs': 490,
               'timing_stable': True, 'sentinel_ratio': timing_receipt['sentinel_ratio'],
               'old_raw_p_values_reproduced': 5, 'expanded_holm_family_size': 6,
               'primary_mean_relative_difference': mean_relative, 'primary_ci95': [new['ci95_low'], new['ci95_high']],
               'primary_raw_p': new['primary_p'], 'primary_holm_p_six': new['holm_p'],
               'old_data_overwritten': False, 'model': 'shared serial two-resource decoder; placement-specific critical-path order',
               'recent_code_sha256': src['recent_code_sha256'], 'recent_protocol_sha256': src['recent_protocol_sha256']}
    (destination / 'completion.json').write_text(json.dumps(receipt, indent=2) + '\n')
    _reports(destination, new, mean_relative, timing_receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def _plots(out, synthetic, rows, groups, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for variant in ('0', 'N', 'N2'):
        subset = [next(r for r in rows if r['instance_id'] == i and r['variant'] == variant) for i in synthetic]
        axes[0].plot(range(12), [r['flexdo_makespan'] / r['mqhoa_mean'] for r in subset], 'o-', label='FlexDO-' + variant)
    axes[0].axhline(1, color='black', linestyle='--', label='MQHOA mean')
    axes[0].set_xticks(range(12), [i.replace('eval-', '') for i in synthetic], rotation=60, ha='right')
    axes[0].set_ylabel('Makespan / archived MQHOA mean')
    axes[0].legend(fontsize=8)
    main = [r for r in rows if r['variant'] == 'N2' and r['kind'] == 'synthetic']
    for n in (10, 50, 100):
        subset = [r for r in main if r['n'] == n]
        axes[1].scatter([n] * len(subset), [r['mqhoa_over_flexdo_time_ratio'] for r in subset])
    axes[1].axhline(1, color='black', linestyle='--')
    axes[1].set(xlabel='Tasks', ylabel='New timing: MQHOA / FlexDO-N2', yscale='log')
    fig.tight_layout()
    for ext in ('png', 'pdf'): fig.savefig(out / ('comparison_runtime.' + ext), dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    grid = np.r_[np.arange(1, 10000, 100), 10000]
    for name in ('MQHOA', '0', 'N', 'N2'):
        curves = []
        for i in synthetic:
            items = groups[i, 'MQHOA'] if name == 'MQHOA' else [results[i, name]]
            within = []
            for item in items:
                trace = np.asarray(item['trace'])
                indices = np.maximum(0, np.searchsorted(trace[:, 0], grid, side='right') - 1)
                within.append(trace[indices, 2] / np.mean([r['best_makespan'] for r in groups[i, 'MQHOA']]))
            curves.append(np.mean(within, axis=0))
        ax.step(grid, np.mean(curves, axis=0), where='post', label=name if name == 'MQHOA' else 'FlexDO-' + name)
    ax.set(xlabel='Objective evaluations (including initialization)', ylabel='Best so far / archived MQHOA mean')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    for ext in ('png', 'pdf'): fig.savefig(out / ('convergence.' + ext), dpi=180)
    plt.close(fig)


def _reports(out, row, effect, timing):
    en = f'''# FlexDO supplementary baseline results

Seventy MQHOA reruns exactly reproduced archived makespans and input/pool hashes. The 42 deterministic FlexDO quality conditions and 490 single-core timing conditions completed with independent schedule validation. Timing stability ratio: {timing['sentinel_ratio']:.4f} (limit 1.5).

The pre-fixed primary comparator is FlexDO-N2. Across the 12 synthetic DAGs, mean relative makespan (MQHOA minus FlexDO, divided by FlexDO) was {effect:+.2%}, with 95% instance-bootstrap CI [{row['ci95_low']:+.2%}, {row['ci95_high']:+.2%}]. The exact paired sign-flip p was {row['primary_p']:.8f}; Holm p across the expanded six-comparison family was {row['holm_p']:.8f}. Negative differences favor MQHOA. The archived graphs are descriptive. Deterministic timing repeats and the 70 reproduction runs are not additional independent quality samples.

FlexDO-0 and FlexDO-N are descriptive variants. Per-instance results include wins/ties/losses over the existing 20 MQHOA seeds and actual objective-call counts. All five older raw p-values reproduced exactly. Their older five-comparison analysis is preserved; `paired_six_comparisons.csv` is a separately labeled expanded analysis. Old CIs are copied unchanged; the new contrast uses a separate fixed bootstrap stream.

## Methods and reviewer R2.9 addition

We added the 2024 journal method FlexDO (de Queiroz et al.; DOI 10.1016/j.jnca.2023.103791; preprint 2023). The candidate-generation mechanism follows Algorithm 1: retain greedy placements and enumerate the remaining assignments. The two serial resources, communication model, evaluator and entry/exit restrictions are those of this study. Each placement receives its own critical-path-ready task order; the paper's CPU/channel-sharing simulator is not reproduced. The frozen plan states rounding, pair overshoot, stable ties, adjusted anchor scores and duplicate-set handling. Thus the experiment evaluates an explicit adaptation, not a reproduction of the paper's reported performance.

The fixed primary setting had no tuning on evaluation data. All actual makespan calls, including greedy candidates, count toward a 10,000-call cap. Deterministic runs stop after their complete candidate set. Dedicated paired timing uses the original Windows host and logical CPU 2, a 60-second warmup and seeded interleaving. Curves carry the final deterministic best value forward after termination for display only, without claiming further calls. Existing data, manuscript files and earlier analyses are preserved; these additions have not been submitted.
'''
    zh = f'''# FlexDO补充实验结果

70次MQHOA复现的makespan及输入/候选池哈希与归档精确一致。42个确定性FlexDO质量条件和490次单核计时完成，报告方案均通过独立可行性检查。计时稳定性比值为{timing['sentinel_ratio']:.4f}，阈值1.5。

预先固定的主要对照为FlexDO-N2。12个合成DAG上，相对差异（MQHOA减FlexDO，再除FlexDO）的均值为{effect:+.2%}，95%实例bootstrap区间为[{row['ci95_low']:+.2%}, {row['ci95_high']:+.2%}]。精确配对符号翻转检验原始p={row['primary_p']:.8f}；扩展六项比较家族的Holm校正p={row['holm_p']:.8f}。负值有利于MQHOA。两个归档图单独描述；确定性计时重复和70次复现不计为新增独立质量样本。

FlexDO-0和FlexDO-N作为描述性变体；逐实例表列出相对MQHOA已有20次运行的胜/平/负及实际评价次数。旧五项原始p值均精确复现；旧五项校正表保留，新增六项家族另存。旧置信区间原样保留，新对照使用独立固定bootstrap随机流。

## 方法与审稿人R2.9补充回复

新增2024年期刊方法FlexDO（DOI 10.1016/j.jnca.2023.103791，预印本2023）。候选生成遵循算法1，保留贪心过程中的放置并枚举未标记任务。评价使用本研究的双串行资源、通信模型、评价器和入口/出口固定规则；每个放置独立生成关键路径就绪顺序，没有复现原文的CPU/信道共享模拟器。取整、成对标记越界、稳定平局、锚边调整及候选集合去重均在冻结计划中说明。因此结果是明确模型适配下的比较，不是原论文性能的复现。

主要变体事先确定，没有根据评价集调参。所有实际makespan调用（含贪心候选）均计入10,000次上限；确定性方法完成候选集后即停止。专门计时采用原Windows主机逻辑CPU2、60秒预热及交错顺序。曲线仅为展示在确定性方法结束后延续最终最好值，不计为额外评价。原论文、数据和旧分析保留；本补充尚未投稿。
'''
    (out / 'RESULTS_AND_RESPONSE_EN.md').write_text(en)
    (out / 'RESULTS_AND_RESPONSE_ZH.md').write_text(zh)
