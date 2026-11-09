import argparse
import csv
import json
from pathlib import Path
import numpy as np

def main(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[json.loads(s) for s in (out/'e2e_repeated.jsonl').read_text().splitlines()]
    assert len(rows)==75
    cases=['asm256-p3','asm512-p5','asm1024-p8']
    methods=['MQHOA','PSO','HEFT','FlexDO-N2','AllLocal']
    rng=np.random.default_rng(20261110)
    result=[]
    for case in cases:
        for method in methods:
            group=[r for r in rows if r['case_id']==case and r['method']==method]
            assert sorted(r['repetition'] for r in group)==list(range(5))
            values=np.array([r['wall_seconds'] for r in group])
            ci=np.percentile(rng.choice(values,(10000,5),replace=True).mean(axis=1),[2.5,97.5])
            result.append(dict(case_id=case,method=method,n_repeats=5,mean_execution_seconds=float(values.mean()),
                               median_execution_seconds=float(np.median(values)),minimum_execution_seconds=float(values.min()),
                               maximum_execution_seconds=float(values.max()),ci_low=float(ci[0]),ci_high=float(ci[1]),
                               planning_seconds=group[0]['planning_seconds'],
                               mean_planning_plus_execution_seconds=float(np.mean([r['planning_plus_execution_seconds'] for r in group])),
                               predicted_seconds=group[0]['predicted_seconds'],gpu_tasks=sum(group[0]['best_l'])))
    with (out/'e2e_repeated_summary.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    figure,axes=plt.subplots(1,2,figsize=(12,4.4))
    colors=['#2769a8','#d2872b','#409775','#9369a6','#777777']
    x=np.arange(3)
    for i,method in enumerate(methods):
        group=[r for r in result if r['method']==method]
        y=np.asarray([r['mean_execution_seconds'] for r in group])
        lower=np.asarray([r['ci_low'] for r in group])
        upper=np.asarray([r['ci_high'] for r in group])
        axes[0].bar(x+(i-2)*.155,y,.15,label=method,color=colors[i],yerr=np.vstack([y-lower,upper-y]),capsize=2)
        axes[1].bar(x+(i-2)*.155,[r['mean_planning_plus_execution_seconds'] for r in group],.15,color=colors[i])
    axes[0].set_title('Actual execution: matched persistent worker')
    axes[0].set_ylabel('Execution wall time (s)')
    axes[0].legend(fontsize=8)
    axes[1].set_title('Planning cost + actual execution')
    axes[1].set_ylabel('Composite cost (s)')
    for ax in axes:
        ax.set_xticks(x,cases)
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    figure.tight_layout()
    for ext in ('pdf','png'):
        figure.savefig(out/f'optical_matched_execution.{ext}',dpi=180)
    plt.close(figure)
    (out/'e2e_repeated_analysis_receipt.json').write_text(json.dumps(dict(rows=75,conditions=15,bootstrap_seed=20261110,
                                                                      bootstrap_samples=10000,ci='95% percentile bootstrap over five timing rounds within one fixed schedule',
                                                                      primary_scientific_units='three designed optical workflows, not 75 independent applications',
                                                                      no_significance_claim=True),indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
