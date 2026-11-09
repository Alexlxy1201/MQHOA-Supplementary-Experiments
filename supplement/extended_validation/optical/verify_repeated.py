"""Check matched actual execution independently of worker implementation."""
import argparse
import json
from pathlib import Path
import numpy as np

def main(out):
    rows=[json.loads(s) for s in (out/'e2e_repeated.jsonl').read_text().splitlines()]
    warmups=[json.loads(s) for s in (out/'e2e_repeated_warmups.jsonl').read_text().splitlines()]
    assert len(rows)==75 and len(warmups)==15
    original=[json.loads(s) for s in (out/'quality.jsonl').read_text().splitlines()]
    cpu=json.loads((out/'mac_profile.json').read_text())
    finals={p['case']['id']:p['final_metrics'] for p in cpu['profiles']}
    for row in rows+warmups:
        n=len(row['best_l'])
        assert len(row['events'])==n
        events={event['task']:event for event in row['events']}
        assert sorted(events)==list(range(n))
        for j in range(n):
            assert events[j]['resource']==row['best_l'][j]
            assert 0<=events[j]['start_seconds']<=events[j]['finish_seconds']<=row['wall_seconds']
        instance=json.loads((out/'instances'/f'{row["case_id"]}.json').read_text())
        for a,b,c in instance['edges']:
            assert events[b]['start_seconds']+1e-8>=events[a]['finish_seconds']
        for resource in (0,1):
            tasks=[j for j in row['best_pi'] if row['best_l'][j]==resource]
            assert all(events[b]['start_seconds']+1e-8>=events[a]['finish_seconds'] for a,b in zip(tasks,tasks[1:]))
        assert row['planning_plus_execution_seconds']==row['planning_seconds']+row['wall_seconds']
        np.testing.assert_allclose(row['final_metrics'],finals[row['case_id']],atol=1e-8,rtol=1e-8)
        assert row['numerical_validation']['passed']
        if row['method']=='AllLocal':
            assert not any(row['best_l']) and row['planning_seconds']==0
        else:
            selection=next(r for r in original if r['case_id']==row['case_id'] and r['method']==row['method'] and r['seed']==row['seed'])
            assert selection['best_l']==row['best_l'] and selection['best_pi']==row['best_pi']
            assert selection['t_planning']==row['planning_seconds']
    cases=['asm256-p3','asm512-p5','asm1024-p8']
    methods=['MQHOA','PSO','HEFT','FlexDO-N2','AllLocal']
    for case in cases:
        for method in methods:
            group=[r for r in rows if r['case_id']==case and r['method']==method]
            assert sorted(r['repetition'] for r in group)==list(range(5))
            assert len([r for r in warmups if r['case_id']==case and r['method']==method])==1
    receipt=dict(status='VERIFIED',measurement_rows=75,warmup_rows=15,unchanged_original_schedules=True,
                 all_precedence_resource_checks_passed=True,all_full_final_vectors_valid=True,
                 planning_costs_unchanged=True,all_local_uses_same_worker_timing_boundary=True)
    with (out/'verification_repeated.json').open('x') as stream:
        stream.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
