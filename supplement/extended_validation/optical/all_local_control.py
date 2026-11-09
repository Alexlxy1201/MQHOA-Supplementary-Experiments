"""Secondary actual all-local control; no re-selection of comparative schedules."""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
from workflow import configuration,execute

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    out=p.parse_args().out
    target=out/'all_local_control.json'
    if target.exists():
        raise RuntimeError('all-local control attempt already exists')
    amendment=out/'all_local_control_protocol.md'
    with amendment.open('x') as stream:
        stream.write('# Secondary control protocol, fixed before these measurements\n\n'
                     'After completion of the original 126 profile-driven comparisons and 12 fixed seed-0 real executions, '
                     'measure direct all-local execution for each of the same three optical workflows. '
                     'Two full-workflow warmups, seven complete-workflow timing repetitions; all actual operators execute. '
                     'The boundary is Operators construction to final result and synchronization, excluding optimizer and SSH worker initialization. '
                     'This is a secondary reference control, not a new member of the original comparison family. '
                     'No previous scientific sample is rerun, replaced, selected or erased. '
                     'Conditional timing variation on this Mac is reported descriptively.\n')
    cfg=configuration()
    rows=[]
    for case in cfg['workloads']:
        for _ in range(2):
            execute(case,cfg)
        times=[]
        for repeat in range(7):
            start=time.perf_counter()
            engine,values=execute(case,cfg)
            engine.sync()
            times.append(time.perf_counter()-start)
        rows.append(dict(case_id=case['id'],seconds=times,median_seconds=float(np.median(times)),
                         final_metrics=values[max(values)].tolist(),host=platform.node()))
        print(case['id'],rows[-1]['median_seconds'],flush=True)
    with target.open('x') as stream:
        stream.write(json.dumps(dict(control='direct actual all-local execution; separate secondary protocol',rows=rows),indent=2)+'\n')
