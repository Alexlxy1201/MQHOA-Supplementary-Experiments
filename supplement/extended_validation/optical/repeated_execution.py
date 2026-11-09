"""Matched, interleaved real execution after the initial descriptive pilot."""
import argparse
import json
import random
import time
from pathlib import Path
from workflow import configuration,specification
from transport import Worker,end_to_end

def main(out):
    cfg=configuration()
    target=out/'e2e_repeated.jsonl'
    warmups=out/'e2e_repeated_warmups.jsonl'
    receipt=out/'e2e_repeated_protocol.json'
    if any(p.exists() for p in (target,warmups,receipt)):
        raise RuntimeError('matched E2E attempt already exists')
    protocol=dict(phase='secondary descriptive real-execution control added after inspecting initial 12 executions',
                  frozen_before='all warmups and measurements of this repeated stage',
                  cases=cfg['workloads'],methods=['MQHOA','PSO','HEFT','FlexDO-N2','AllLocal'],
                  repetitions=5,warmups_per_condition=1,interleave_seed=20261109,
                  schedule_selection='unchanged seed-0 solutions from original 126 quality rows; deterministic methods unchanged',
                  worker='one persistent authenticated remote CUDA process across this entire stage; reset task cache before each run',
                  boundary='two serial workers ready after reset to final task completion; SSH startup and initialize/reset excluded for every condition',
                  planning='original measured pool-build plus optimizer+validation at selected seed; zero for fixed AllLocal placement',
                  combined_cost='sum of separately measured standalone-equivalent planning cost and actual execution wall time',
                  independence='five timing repeats per fixed schedule, not five independent workflows or search seeds',
                  expected_measurement_rows=75,expected_warmup_rows=15)
    receipt.write_text(json.dumps(protocol,indent=2)+'\n')
    quality=[json.loads(s) for s in (out/'quality.jsonl').read_text().splitlines()]
    chosen=[r for r in quality if r['seed'] in (None,0)]
    for case in cfg['workloads']:
        chosen.append(dict(case_id=case['id'],method='AllLocal',seed=None,best_l=[0]*len(specification(case)),
                           best_pi=list(range(len(specification(case)))),best_makespan=sum(json.loads((out/'instances'/f'{case["id"]}.json').read_text())['tl']),t_planning=0.0))
    rng=random.Random(protocol['interleave_seed'])
    worker=Worker()
    begin=time.perf_counter()
    def run(row,repetition,kind):
        case=next(c for c in cfg['workloads'] if c['id']==row['case_id'])
        result=end_to_end(case,row['best_l'],row['best_pi'],worker=worker)
        result.update(case_id=case['id'],method=row['method'],seed=row['seed'],repetition=repetition,kind=kind,
                      predicted_seconds=row['best_makespan'],planning_seconds=row['t_planning'],
                      planning_plus_execution_seconds=row['t_planning']+result['wall_seconds'],
                      observed_over_predicted=result['wall_seconds']/row['best_makespan'],
                      best_l=row['best_l'],best_pi=row['best_pi'])
        path=warmups if kind=='warmup' else target
        with path.open('a') as stream:
            stream.write(json.dumps(result,allow_nan=False)+'\n')
        return result
    try:
        warm_order=chosen.copy()
        rng.shuffle(warm_order)
        for index,row in enumerate(warm_order):
            run(row,-1,'warmup')
            print('WARMUP',index+1,'/15',row['case_id'],row['method'],flush=True)
        count=0
        for rep in range(protocol['repetitions']):
            order=chosen.copy()
            rng.shuffle(order)
            for row in order:
                result=run(row,rep,'measurement')
                count+=1
                print('MATCHED_E2E',count,'/75',row['case_id'],row['method'],result['wall_seconds'],flush=True)
    finally:
        worker.close()
    (out/'e2e_repeated_receipt.json').write_text(json.dumps(dict(rows=count,warmups=15,wall_seconds=time.perf_counter()-begin,
                                                               all_outputs_validated=True,worker_lifecycle='one persistent process, identical reset boundary'),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
