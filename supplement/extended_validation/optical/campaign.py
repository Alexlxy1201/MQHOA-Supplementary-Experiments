"""Measured-profile DAG comparisons and optional real execution, additive only."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'supplement'))
from mqhoa_supp.instances import Pool
from mqhoa_supp.methods import Budget,optimize
from mqhoa_supp.model import Instance,validate,digest
from mqhoa_supp.protocol import pool_parameters
from recent_baseline.flexdo import solve
from workflow import configuration,specification,check_numeric


def write_new(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(data,indent=2,allow_nan=False)+'\n')


def append(path,data):
    with Path(path).open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(data,allow_nan=False)+'\n')


def build(out):
    cfg=configuration()
    cpu=json.loads((out/'mac_profile.json').read_text())
    gpu=json.loads((out/'gpu_profile.json').read_text())
    transport=json.loads((out/'network.json').read_text())
    if cpu['environment']['config_sha256']!=gpu['environment']['config_sha256']:
        raise RuntimeError('CPU/GPU config mismatch')
    if cpu['environment']['source_sha256']!=gpu['environment']['source_sha256']:
        raise RuntimeError('CPU/GPU source mismatch')
    delays={}
    for row in transport['rows']:
        delays.setdefault(row['nbytes'],[]).append(row)
    delay={size:float((np.median([r['upload_ack_seconds'] for r in rows])+np.median([r['download_seconds'] for r in rows]))/2)
           for size,rows in delays.items()}
    validations=[]
    for left,right in zip(cpu['profiles'],gpu['profiles']):
        if left['case']!=right['case']:
            raise RuntimeError('profile case mismatch')
        case=left['case']
        for a,b in zip(left['tasks'],right['tasks']):
            if any(a[k]!=b[k] for k in ('id','op','inputs')) or a['output']['nbytes']!=b['output']['nbytes']:
                raise RuntimeError('executed graph mismatch')
            check_numeric(np.asarray(a['output']['sample_real'])+1j*np.asarray(a['output']['sample_imag']),
                          np.asarray(b['output']['sample_real'])+1j*np.asarray(b['output']['sample_imag']),cfg)
            if not b['numerical_validation']['passed']:
                raise RuntimeError('full-array CUDA/CPU validation failed')
        check=check_numeric(left['final_metrics'],right['final_metrics'],cfg)
        validations.append(dict(case_id=case['id'],full_final_vector=check,
                                full_arrays_remote_reference=True,cross_host_intermediates='64 evenly spaced samples per array'))
        edges=[(parent,t['id'],delay[left['tasks'][parent]['output']['nbytes']]) for t in left['tasks'] for parent in t['inputs']]
        instance=Instance(case['id'],[t['median_seconds'] for t in left['tasks']],
                          [t['median_seconds'] for t in right['tasks']],edges,[0,len(left['tasks'])-1],
                          dict(kind='measured-computational-optics',case=case,profile_host=cpu['environment']['host'],
                               edge_host=gpu['environment']['host'],input='analytically generated aperture, not laboratory acquisition',
                               physical_method='scalar Rayleigh-Sommerfeld angular spectrum with 2x zero padding',
                               network_route=transport['route'],communication_model=transport['symmetrization'],
                               unit='seconds',communication='fixed symmetric edge delay, no shared-channel contention',
                               cpu_profile_sha256=hashlib.sha256((out/'mac_profile.json').read_bytes()).hexdigest(),
                               gpu_profile_sha256=hashlib.sha256((out/'gpu_profile.json').read_bytes()).hexdigest(),
                               network_profile_sha256=hashlib.sha256((out/'network.json').read_bytes()).hexdigest()))
        write_new(out/'instances'/f'{case["id"]}.json',instance.data())
    write_new(out/'numerical_validation.json',dict(cases=validations,all_passed=True))
    print('INSTANCES_OK',len(validations),flush=True)


def quality(out):
    cfg=configuration()
    path=out/'quality.jsonl'
    if path.exists():
        raise RuntimeError('quality attempt already exists; preserve it')
    instances=[Instance.load(p) for p in sorted((out/'instances').glob('*.json'))]
    if len(instances)!=3:
        raise RuntimeError('require three measured instances')
    count=0
    for instance in instances:
        size,seed=pool_parameters(instance,cfg)
        pool_start=time.perf_counter()
        pool=Pool(instance,size,seed)
        pool_seconds=time.perf_counter()-pool_start
        for method in ('MQHOA','PSO','HEFT','FlexDO-N2'):
            seeds=cfg['seeds'] if method in ('MQHOA','PSO') else [None]
            for runseed in seeds:
                tic=time.perf_counter()
                if method=='FlexDO-N2':
                    row=solve(instance,'N2',cfg['budget_evals'])
                else:
                    evaluator=Budget(instance,1 if method=='HEFT' else cfg['budget_evals'],100)
                    extra=optimize(method,instance,pool,evaluator,np.random.default_rng(runseed),cfg['mqhoa'] if method=='MQHOA' else cfg['pso'])
                    row={**evaluator.result(),**extra}
                value=row['best_makespan']
                violations=validate(instance,row['best_l'],row['best_pi'],row['starts'],row['finishes'])
                if violations or row['evals_used']>cfg['budget_evals'] or not math.isfinite(value):
                    raise RuntimeError('invalid measured-DAG solution')
                optimizer_seconds=time.perf_counter()-tic
                charged_pool_seconds=pool_seconds if method in ('MQHOA','PSO') else 0.0
                row.update(case_id=instance.identifier,method=method,seed=runseed,instance_sha256=instance.sha256,
                           pool_sha256=pool.sha256 if method in ('MQHOA','PSO') else None,
                           pool_size=size if method in ('MQHOA','PSO') else None,pool_seed=seed if method in ('MQHOA','PSO') else None,
                           t_optimizer=optimizer_seconds,t_pool_build=charged_pool_seconds,
                           t_planning=optimizer_seconds+charged_pool_seconds,
                           planning_time_boundary='single-run-equivalent pool build plus optimizer and validation; profiles excluded',
                           config_sha256=digest(cfg),validated=True)
                append(path,row)
                count+=1
                if count%10==0:
                    print('QUALITY',count,'/126',flush=True)
    write_new(out/'quality_receipt.json',dict(rows=count,expected_rows=126,all_validated=True,
                                              scope='three designed measured computational-optics cases; conditional on frozen median profiles',
                                              pool_build_excluded_from_optimizer_seconds=True,
                                              pool_build_included_in_planning_seconds=True,host=platform.node()))


def actual(out):
    from transport import end_to_end
    cfg=configuration()
    rows=[json.loads(s) for s in (out/'quality.jsonl').read_text().splitlines()]
    selected=[r for r in rows if r['seed'] in (None,cfg['end_to_end_seed'])]
    target=out/'end_to_end.jsonl'
    if target.exists():
        raise RuntimeError('end-to-end attempt already exists')
    for row in selected:
        case=next(c for c in cfg['workloads'] if c['id']==row['case_id'])
        result=end_to_end(case,row['best_l'],row['best_pi'])
        result.update(case_id=case['id'],method=row['method'],seed=row['seed'],predicted_seconds=row['best_makespan'],
                      observed_over_predicted=result['wall_seconds']/row['best_makespan'],
                      planning_seconds=row['t_planning'],planning_plus_execution_seconds=row['t_planning']+result['wall_seconds'],
                      best_l=row['best_l'],best_pi=row['best_pi'])
        append(target,result)
        print('E2E',case['id'],row['method'],row['best_makespan'],result['wall_seconds'],flush=True)
    write_new(out/'end_to_end_receipt.json',dict(rows=len(selected),all_outputs_validated=True,
                                                repetitions_per_schedule=1,claim='observed illustrative executions, no runtime confidence interval'))


def analyze(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cfg=configuration()
    rng=np.random.default_rng(cfg['bootstrap_seed'])
    rows=[json.loads(s) for s in (out/'quality.jsonl').read_text().splitlines()]
    summary=[]
    for case in cfg['workloads']:
        for method in ('MQHOA','PSO','HEFT','FlexDO-N2'):
            group=[r for r in rows if r['case_id']==case['id'] and r['method']==method]
            values=np.asarray([r['best_makespan'] for r in group])
            ci=np.percentile(rng.choice(values,(cfg['bootstrap_samples'],len(values)),replace=True).mean(axis=1),[2.5,97.5]) if len(values)>1 else [values[0],values[0]]
            summary.append(dict(case_id=case['id'],method=method,n_runs=len(values),mean_seconds=float(values.mean()),
                                minimum_seconds=float(values.min()),maximum_seconds=float(values.max()),ci_low=float(ci[0]),ci_high=float(ci[1]),
                                mean_evals=float(np.mean([r['evals_used'] for r in group]))))
    with (out/'summary.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    e2e=[json.loads(s) for s in (out/'end_to_end.jsonl').read_text().splitlines()] if (out/'end_to_end.jsonl').exists() else []
    with (out/'end_to_end_summary.csv').open('x',newline='') as stream:
        fields=['case_id','method','seed','predicted_seconds','wall_seconds','observed_over_predicted','planning_seconds','planning_plus_execution_seconds']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(e2e)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    methods=['MQHOA','PSO','HEFT','FlexDO-N2']
    colors=['#2769a8','#d2872b','#409775','#9369a6']
    x=np.arange(3)
    for index,method in enumerate(methods):
        group=[r for r in summary if r['method']==method]
        axes[0].bar(x+(index-1.5)*0.19,[r['mean_seconds'] for r in group],width=.18,label=method,color=colors[index])
    axes[0].set_ylabel('Predicted workflow makespan (s)')
    axes[0].set_xticks(x,[c['id'] for c in cfg['workloads']])
    axes[0].set_title('Measured-profile scheduling (20 stochastic seeds)')
    axes[0].legend(fontsize=8)
    for index,method in enumerate(methods):
        group=sorted([r for r in e2e if r['method']==method],key=lambda r: [c['id'] for c in cfg['workloads']].index(r['case_id']))
        if group:
            axes[1].bar(x+(index-1.5)*.19,[r['wall_seconds'] for r in group],width=.18,color=colors[index])
    axes[1].set_xticks(x,[c['id'] for c in cfg['workloads']])
    axes[1].set_ylabel('Observed workflow wall time (s)')
    axes[1].set_title('Actual Mac–CUDA execution (one fixed-seed schedule)')
    for ax in axes:
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ('pdf','png'):
        fig.savefig(out/f'optical_comparison.{ext}',dpi=180)
    plt.close(fig)
    write_new(out/'analysis_receipt.json',dict(rows=len(rows),summary_rows=len(summary),end_to_end_rows=len(e2e),
                                              confidence_intervals='95% percentile bootstrap over 20 seeds within each fixed measured workflow; deterministic methods have point values',
                                              no_population_test=True))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['build','quality','actual','analyze'])
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    globals()[args.stage](args.out)
