"""Complete-matrix analysis; immutable engine/source hash is unaffected."""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent


def write_csv(name, rows):
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def ci(x,rng):
    x=np.asarray(x,dtype=float)
    if len(x)==1:return [float(x[0]),float(x[0])]
    return [float(v) for v in np.percentile(np.mean(rng.choice(x,(10000,len(x))),axis=1),[2.5,97.5])]


def main():
    from extended_validation.online_lossy.model import digest,validate
    cfg=json.loads((ROOT/'frozen_config.json').read_text())
    assert json.loads((ROOT/'verification.json').read_text())['verified']
    sources=json.loads((ROOT/'source_manifest.json').read_text())
    receipt=json.loads((ROOT/'execution_receipt.json').read_text())
    assert digest(sources)==receipt['source_sha256']
    for rel,expected in sources.items():
        assert hashlib.sha256((ROOT/'source_snapshot'/rel).read_bytes()).hexdigest()==expected
    assert hashlib.sha256((ROOT/'raw_runs.jsonl.gz').read_bytes()).hexdigest()==receipt['raw_sha256']
    rows=[json.loads(line) for line in gzip.open(ROOT/'raw_runs.jsonl.gz','rt')]
    problems=json.loads((ROOT/'problems.json').read_text())
    counters=Counter();exogenous={};groups=defaultdict(list)
    incremental_running_epochs=0
    same_user_running_epochs=0
    same_user_unfinished_epochs=0
    same_user_prior_started_epochs=0
    for r in rows:
        p=problems[f'online_lossy_incremental-i{r["scenario_index"]}']
        assert not validate(p,r,cfg['kappa_si'],cfg['network_dynamic_power_w'])
        assert digest(p)==r['problem_sha256']
        assert r['source_sha256']==receipt['source_sha256'] and r['config_sha256']==receipt['config_sha256']
        assert abs(r['local_compute_energy_j']+r['server_compute_energy_j']+r['network_energy_j']-r['energy_j'])<1e-6
        assert abs(r['device_energy_j']-r['local_compute_energy_j']-r['device_network_energy_j'])<1e-6
        counters['runs']+=1;counters['tasks']+=len(r['task_schedule'])
        counters['successful_payload_transfers']+=len(r['transfer_schedule'])
        counters['segments']+=len(r['segments']);counters['observations']+=len(r['observations'])
        counters['attempts']+=r['attempts'];counters['failed_attempts']+=r['failed_attempts']
        counters['forecast_calls']+=r['evals_used']
        counters['precedence_edges']+=len(p['edges'])
        counters['started_placement_checks']+=sum(len(o['started_placements']) for o in r['observations'])
        for attempt in r['attempt_schedule']:
            schedule={x['task']:x for x in r['task_schedule']}
            assert attempt['start']>=schedule[attempt['a']]['finish']-1e-6
            assert attempt['source_resource']==schedule[attempt['a']]['resource']
            assert attempt['target_resource']==schedule[attempt['b']]['resource']
            ss=sorted([s for s in r['segments'] if s['kind']=='transfer'
                       and (s['a'],s['b'],s['attempt_index'])==(attempt['a'],attempt['b'],attempt['attempt_index'])],key=lambda s:s['start'])
            cursor=attempt['start']
            for s in ss:
                assert abs(s['start']-cursor)<1e-6
                assert not any(s['start']+1e-8<e['time']<s['finish']-1e-8 for e in p['events'])
                cursor=s['finish']
            assert abs(cursor-attempt['finish'])<1e-6
            key=(r['scenario_index'],attempt['a'],attempt['b'],attempt['attempt_index'])
            value=(attempt['uniform_draw'],attempt['failed'])
            if key in exogenous:assert exogenous[key]==value
            exogenous[key]=value
        assert abs(r['attempt_work']-sum(a['amount'] for a in r['attempt_schedule']))<1e-6
        assert abs(r['useful_payload_work']-sum(a['amount'] for a in r['transfer_schedule']))<1e-6
        assert abs(r['failed_attempt_work']-sum(a['amount'] for a in r['attempt_schedule'] if a['failed']))<1e-6
        assert abs(r['network_airtime']-sum(a['finish']-a['start'] for a in r['attempt_schedule']))<1e-6
        n=cfg['layouts'][r['scenario_index']][2]
        for o in r['observations']:
            is_expansion=any(abs(o['time']-(3*u+cfg['incremental_delay']))<1e-8 for u in range(p['users']))
            if is_expansion:
                running=[x for x in r['task_schedule'] if x['start']<o['time']-1e-8 and x['finish']>o['time']+1e-8]
                incremental_running_epochs+=bool(running)
                u=next(u for u in range(p['users']) if abs(o['time']-(3*u+cfg['incremental_delay']))<1e-8)
                same_user_running_epochs+=any(p['tasks'][x['task']]['owner']==u for x in running)
                old=[x for x in r['task_schedule'] if p['tasks'][x['task']]['owner']==u
                     and p['task_release'][str(x['task'])]<o['time']-1e-8]
                same_user_unfinished_epochs+=any(x['finish']>o['time']+1e-8 for x in old)
                same_user_prior_started_epochs+=any(x['start']<o['time']-1e-8 for x in old)
        groups[r['scenario_index'],r['method']].append(r)
    assert len(rows)==252 and len(groups)==24
    for (i,m),v in groups.items():assert len(v)==(20 if m in ('MQHOA','PSO') else 1)
    for i in range(6):
        a=sorted(r['evals_used'] for r in groups[i,'MQHOA'])
        b=sorted(r['evals_used'] for r in groups[i,'PSO'])
        assert a==b
        expected=800 if cfg['layouts'][i][0]==2 else 1000
        assert all(v==expected for v in a)
    rng=np.random.default_rng(cfg['bootstrap_seed'])
    metrics=['makespan','mean_response_time','max_response_time','energy_j','device_energy_j','network_energy_j',
             'attempts','failed_attempts','useful_payload_work','attempt_work','failed_attempt_work','network_airtime',
             'planning_wall_seconds','wall_seconds','evals_used']
    summary=[];lookup={}
    for (i,m),v in sorted(groups.items()):
        out={'scenario_index':i,'method':m,'algorithm_runs':len(v)}
        for metric in metrics:
            values=[r[metric] for r in v]
            out[metric+'_mean']=float(np.mean(values))
            out[metric+'_ci95_low'],out[metric+'_ci95_high']=ci(values,rng)
        summary.append(out);lookup[i,m]=out
    write_csv('per_scenario_summary.csv',summary)
    tests=[]
    for opponent in ['PSO','HEFT-List']:
        x=np.asarray([lookup[i,'MQHOA']['mean_response_time_mean']/lookup[i,opponent]['mean_response_time_mean']-1
                      for i in range(6)])
        signs=np.asarray(list(itertools.product([-1.,1.],repeat=6)))
        bounds=ci(x,rng)
        p=float(np.mean(np.abs(np.mean(signs*x,axis=1))>=abs(x.mean())-1e-14))
        tests.append({'reference':'MQHOA','opponent':opponent,'endpoint':'mean_full_application_response_time',
                      'independent_scenarios':6,'relative_difference_mean':float(x.mean()),
                      'ci95_low':bounds[0],'ci95_high':bounds[1],'p_raw':p,
                      'scenario_differences':json.dumps(x.tolist()),'direction':'negative favors MQHOA'})
    running=0.
    for rank,r in enumerate(sorted(tests,key=lambda x:x['p_raw'])):
        running=max(running,min(1.,r['p_raw']*(len(tests)-rank)));r['p_holm_two']=running
    write_csv('primary_paired_tests.csv',tests)
    write_csv('run_metrics.csv',[{k:r[k] for k in ['run_id','scenario_index','method','seed',*metrics]} for r in rows])
    keys=[{'scenario_index':k[0],'source_task':k[1],'target_task':k[2],'attempt_index':k[3],
           'uniform_draw':v[0],'failed':v[1]} for k,v in sorted(exogenous.items())]
    write_csv('unique_exogenous_attempt_draws.csv',keys)
    output={'verified':True,'counts':dict(counters),'independent_scenarios':6,
            'unique_exogenous_draws':len(exogenous),'unique_failed_draws':sum(v[1] for v in exogenous.values()),
            'incremental_reveal_epochs_while_prior_compute_running':incremental_running_epochs,
            'incremental_reveal_epochs_same_user_compute_running':same_user_running_epochs,
            'incremental_reveal_epochs_same_user_prior_tasks_unfinished':same_user_unfinished_epochs,
            'incremental_reveal_epochs_same_user_prior_tasks_already_started':same_user_prior_started_epochs,
            'useful_payload_work':sum(r['useful_payload_work'] for r in rows),
            'attempt_work':sum(r['attempt_work'] for r in rows),
            'failed_attempt_work':sum(r['failed_attempt_work'] for r in rows),
            'network_airtime_s':sum(r['network_airtime'] for r in rows),
            'network_dynamic_energy_j':sum(r['network_energy_j'] for r in rows),
            'all_common_draws_identical':True,'primary_tests':tests,
            'equal_actual_metaheuristic_budgets':True,
            'raw_sha256':receipt['raw_sha256']}
    (ROOT/'detailed_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    colors={'MQHOA':'#d45500','PSO':'#0072b2','HEFT-List':'#009e73','FIFO-Local':'#777777'}
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,metric,label in zip(axes,['mean_response_time','network_airtime','energy_j'],
                              ['Full-application mean response (s)','Shared-link airtime (s)','System dynamic energy (J)']):
        for method in colors:
            ax.plot(range(1,7),[lookup[i,method][metric+'_mean'] for i in range(6)],'o-',color=colors[method],label=method)
        ax.set_xlabel('Independent scenario');ax.set_ylabel(label);ax.set_xticks(range(1,7));ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout()
    for suffix in ['pdf','png']:fig.savefig(ROOT/('lossy_incremental_comparison.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)
    exemplar=next(r for r in rows if r['scenario_index']==5 and r['method']=='MQHOA' and r['seed']==1)
    p=problems['online_lossy_incremental-i5'];palette=['#0072b2','#d45500','#009e73']
    fig,ax=plt.subplots(figsize=(12,5))
    for task in exemplar['task_schedule']:
        owner=p['tasks'][task['task']]['owner'];y=task['resource']
        ax.broken_barh([(task['start'],task['finish']-task['start'])],(y-.35,.7),facecolors=palette[owner],edgecolors='white',linewidth=.6)
        ax.text((task['start']+task['finish'])/2,y,str(task['task']),ha='center',va='center',fontsize=8,color='white')
    network_y=len(p['frequency_ghz'])
    for attempt in exemplar['attempt_schedule']:
        color='#cc3311' if attempt['failed'] else '#555555'
        ax.broken_barh([(attempt['start'],attempt['finish']-attempt['start'])],(network_y-.35,.7),facecolors=color,edgecolors='white',linewidth=.4)
    for u in range(p['users']):
        ax.axvline(3*u+cfg['incremental_delay'],linestyle=':',color=palette[u],alpha=.8)
    ax.set_yticks(range(network_y+1),[f'Local {j}' if j<p['users'] else f'Server {j-p["users"]}' for j in range(network_y)]+['Shared link'])
    ax.set_xlabel('Simulated time (s); colored dotted lines: incremental reveal; red link intervals: failed attempts')
    ax.set_title('Online lossy execution: scenario 6, MQHOA seed 1')
    fig.tight_layout()
    for suffix in ['pdf','png']:fig.savefig(ROOT/('lossy_incremental_execution.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(output),flush=True)


if __name__=='__main__':main()
