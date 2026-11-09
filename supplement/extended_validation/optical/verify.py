"""Independent arithmetic/constraint audit of stored optical evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def main(out):
    root=Path(__file__).resolve().parent
    cfg=json.loads((root/'config.json').read_text())
    cpu=json.loads((out/'mac_profile.json').read_text())
    gpu=json.loads((out/'gpu_profile.json').read_text())
    net=json.loads((out/'network.json').read_text())
    assert len(cpu['profiles'])==len(gpu['profiles'])==3
    assert len(net['rows'])==24
    assert all(r['valid'] and r['upload_ack_seconds']>0 and r['download_seconds']>0 for r in net['rows'])
    assert cpu['environment']['config_sha256']==gpu['environment']['config_sha256']==hashlib.sha256((root/'config.json').read_bytes()).hexdigest()
    assert cpu['environment']['source_sha256']==gpu['environment']['source_sha256']==hashlib.sha256((root/'workflow.py').read_bytes()).hexdigest()
    for left,right in zip(cpu['profiles'],gpu['profiles']):
        n,p=left['case']['n'],left['case']['planes']
        assert len(left['tasks'])==len(right['tasks'])==3*p+3
        for profile in (left,right):
            for task in profile['tasks']:
                assert len(task['seconds'])==cfg['profile_repetitions']
                assert all(t>0 and math.isfinite(t) for t in task['seconds'])
                assert task['median_seconds']==float(np.median(task['seconds']))
                assert task['output']['nbytes']==int(np.prod(task['output']['shape']))*np.dtype(task['output']['dtype']).itemsize
                assert all(i<task['id'] for i in task['inputs'])
        for task in right['tasks']:
            check=task['numerical_validation']
            assert check['passed']
            assert check['max_absolute_error']<=cfg['absolute_tolerance']
            assert check['relative_l2_error']<=cfg['relative_l2_tolerance']
        np.testing.assert_allclose(left['final_metrics'],right['final_metrics'],rtol=cfg['relative_l2_tolerance'],atol=cfg['absolute_tolerance'])
    instances={p.stem:json.loads(p.read_text()) for p in (out/'instances').glob('*.json')}
    quality=[json.loads(s) for s in (out/'quality.jsonl').read_text().splitlines()]
    assert len(instances)==3 and len(quality)==126
    for row in quality:
        inst=instances[row['case_id']]
        n=len(inst['tl'])
        loc,order=row['best_l'],row['best_pi']
        starts,finish=row['starts'],row['finishes']
        assert len(loc)==n and all(r in (0,1) for r in loc)
        assert sorted(order)==list(range(n))
        assert loc[0]==loc[n-1]==0
        pos={j:i for i,j in enumerate(order)}
        assert all(pos[a]<pos[b] for a,b,c in inst['edges'])
        for j in range(n):
            duration=inst['ts'][j] if loc[j] else inst['tl'][j]
            assert starts[j]>=0 and abs(finish[j]-starts[j]-duration)<1e-10
        for a,b,c in inst['edges']:
            assert starts[b]+1e-10>=finish[a]+(c if loc[a]!=loc[b] else 0)
        for resource in (0,1):
            calendar=sorted((starts[j],finish[j]) for j in range(n) if loc[j]==resource)
            assert all(b[0]+1e-10>=a[1] for a,b in zip(calendar,calendar[1:]))
        assert abs(max(finish)-row['best_makespan'])<1e-10
        assert row['t_planning']==row['t_optimizer']+row['t_pool_build']
        assert 1<=row['evals_used']<=10000
    for name in instances:
        for method in ('MQHOA','PSO'):
            group=[r for r in quality if r['case_id']==name and r['method']==method]
            assert sorted(r['seed'] for r in group)==list(range(20))
        for method in ('HEFT','FlexDO-N2'):
            assert len([r for r in quality if r['case_id']==name and r['method']==method])==1
    e2e=[json.loads(s) for s in (out/'end_to_end.jsonl').read_text().splitlines()]
    assert len(e2e)==12
    for row in e2e:
        inst=instances[row['case_id']]
        events=row['events']
        n=len(inst['tl'])
        assert sorted(e['task'] for e in events)==list(range(n))
        byid={e['task']:e for e in events}
        for a,b,c in inst['edges']:
            assert byid[b]['start_seconds']+1e-8>=byid[a]['finish_seconds']
        for resource in (0,1):
            calendar=sorted((e['start_seconds'],e['finish_seconds']) for e in events if e['resource']==resource)
            assert all(b[0]+1e-8>=a[1] for a,b in zip(calendar,calendar[1:]))
        assert row['numerical_validation']['passed']
        assert row['planning_plus_execution_seconds']==row['planning_seconds']+row['wall_seconds']
        assert row['wall_seconds']>=max(e['finish_seconds'] for e in events)
    receipt=dict(status='VERIFIED',profiled_cases=3,profiled_task_rows=sum(len(c['tasks']) for c in cpu['profiles'])*2,
                 acknowledged_bidirectional_transfer_rows=24,valid_schedules=126,actual_executions=12,
                 old_code_imported_but_unmodified=True,scope='independent stored arithmetic/constraints; no population extrapolation',
                 checks=['source/config hashes','repeat counts and medians','actual byte sizes','CPU/CUDA errors','seed matrix',
                         'precedence and resource intervals','planning cost accounting','real-execution intervals and final outputs'])
    (out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
