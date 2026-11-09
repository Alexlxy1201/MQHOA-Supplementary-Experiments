from __future__ import annotations
from engine import *
import argparse,subprocess,random,os
from concurrent.futures import ProcessPoolExecutor

def selected(rows):return [r for r in rows if (r['method']=='MQHOA' and r['variant']=='full') or r['method'] in ['PSO','GA']]
def jobs_data():
 inst,old,config,frozen=load();base=selected(old);assert len(base)==840
 assert all(r['params']==(config['mqhoa'] if r['method']=='MQHOA' else frozen[r['method']]) for r in base)
 return inst,base,config,frozen

def gate():
 frozen=json.loads((OUT/'freeze.json').read_text());now=freeze();assert frozen['files']==now['files'],'frozen source changed'

def archived_endpoint(row,budget):
 matches=[x[2] for x in row['trace'] if x[0]==budget]
 assert len(matches)==1
 return matches[0]

def pilot():
 OUT.mkdir(parents=True,exist_ok=True)
 assert not (OUT/'freeze.json').exists(),'already attempted'
 (OUT/'freeze.json').write_text(canonical(freeze())+'\n')
 p=subprocess.run([sys.executable,str(Path(__file__).with_name('tests.py'))],capture_output=True,text=True)
 (OUT/'test_report.txt').write_text(p.stdout+p.stderr);assert p.returncode==0,'unit tests failed'
 inst,base,config,frozen=jobs_data();ids=['archive10','eval-n50-wide-r0','eval-n100-wide-r0'];measure=[]
 for identifier in ids:
  task=next(r for r in base if r['instance_id']==identifier and r['method']=='MQHOA' and r['seed']==0)
  for arm in ['MQHOA','Conjugate',*CONFIG['arms']]:
   row=run_task((task,inst[identifier].data(),arm,10000));record(OUT/'pilot.jsonl',row);measure.append(row)
  group=measure[-4:];assert group[0]['best_makespan']==task['best_makespan'];assert group[0]['stream_sha256']==group[1]['stream_sha256'];assert group[0]['initial_sha256']==group[2]['initial_sha256']
 # Rough hardware-specific scaling, no settings are altered in response.
 average=np.mean([r['t_total'] for r in measure]);receipt={'tests_passed':True,'pilot_runs':12,'mean_full_run_s':float(average),'projected_new_quality_serial_s':float(560*average),'projected_prefix_checks_serial_s':float(210*average*.38),'projected_timing_serial_s':float(630*average*.38),'solver_max_s':600,'formal_workers':2,'timing_workers':1,'note':'timing excludes scheduling/controls overhead; projections only'}
 (OUT/'pilot_receipt.json').write_text(canonical(receipt)+'\n');print(canonical(receipt),flush=True)

def run():
 gate();assert json.loads((OUT/'pilot_receipt.json').read_text())['tests_passed']
 assert not (OUT/'new_quality.jsonl').exists(),'existing attempt preserved'
 inst,base,cfg,frozen=jobs_data();mq=[r for r in base if r['method']=='MQHOA']
 # Archival endpoint matrix: all20 seeds and14 instances, no rerun substitutes.
 for task in base:
  for b in CONFIG['budget_endpoints']:record(OUT/'budget_archival.jsonl',{'instance_id':task['instance_id'],'seed':task['seed'],'method':task['method'],'budget':b,'best_makespan':archived_endpoint(task,b),'source_run_id':task['run_id'],'instance_sha256':task['instance_sha256'],'pool_sha256':task['pool_sha256']})
 # seed0 reruns every endpoint including10000; full stream prefixes checked.
 payloads=[(t,inst[t['instance_id']].data(),t['method'],b) for t in base if t['seed']==0 for b in CONFIG['budget_endpoints']]
 results=[]
 with ProcessPoolExecutor(max_workers=CONFIG['workers']) as ex:
  for n,row in enumerate(ex.map(run_task,payloads),1):
   old=next(t for t in base if t['instance_id']==row['instance_id'] and t['method']==row['method'] and t['seed']==row['seed']);assert abs(row['best_makespan']-archived_endpoint(old,row['budget']))<1e-9
   if row['budget']==10000:assert row['best_l']==old['best_l'] and row['best_pi']==old['best_pi']
   record(OUT/'prefix_checks.jsonl',row);results.append(row)
   if n%10==0:print('prefix',n,'/',len(payloads),flush=True)
 for row in results:
  full=next(r for r in results if r['instance_id']==row['instance_id'] and r['method']==row['method'] and r['budget']==10000)
  assert row['stream_sha256']==full['prefix_hashes'][str(row['budget'])],'prefix candidate mismatch'
 # Extra14×20 relabeled and direct-geometry arm conditions, randomized order.
 payloads=[(t,inst[t['instance_id']].data(),arm,10000) for t in mq for arm in CONFIG['arms']];random.Random(CONFIG['timing_seed']).shuffle(payloads)
 with ProcessPoolExecutor(max_workers=CONFIG['workers']) as ex:
  for n,row in enumerate(ex.map(run_task,payloads),1):
   assert row['validator_ok'] and row['evals_used']==10000;record(OUT/'new_quality.jsonl',row)
   if n%10==0:print('quality',n,'/',len(payloads),flush=True)
 # Exact proof block, all5 targets even if one is only FEASIBLE.
 for i in sorted(inst):
  if inst[i].n==10:
   row=exact(inst[i],CONFIG['solver_time_limit_s']);record(OUT/'exact.jsonl',row);print('exact',i,row['status'],row['lb'],row['ub'],flush=True)
 # Serial fresh runtime across all budgets/methods (seed0),3 repeats.
 timing_tasks=[(t,inst[t['instance_id']].data(),t['method'],b,repeat) for t in base if t['seed']==0 for b in CONFIG['budget_endpoints'] for repeat in range(CONFIG['timing_repeats'])]
 random.Random(CONFIG['timing_seed']).shuffle(timing_tasks)
 anchor=inst['archive10'];warm=time.perf_counter()
 while time.perf_counter()-warm<10:schedule(anchor,[0]*anchor.n,anchor.topo,False)
 def sentinel(position):
  samples=[]
  for _ in range(3):
   tic=time.perf_counter()
   for _ in range(10000):schedule(anchor,[0]*anchor.n,anchor.topo,False)
   samples.append(time.perf_counter()-tic)
  row={'after_rows':position,'samples':samples,'median':float(np.median(samples))};record(OUT/'timing_controls.jsonl',row);return row['median']
 controls=[sentinel(0)]
 for n,(t,data,mode,b,repeat) in enumerate(timing_tasks,1):
  row=run_task((t,data,mode,b));row.update(repeat=repeat,workers=1)
  assert abs(row['best_makespan']-archived_endpoint(t,b))<1e-9;record(OUT/'timing.jsonl',row)
  if n%50==0:print('timing',n,'/',len(timing_tasks),flush=True);controls.append(sentinel(n))
 controls.append(sentinel(len(timing_tasks)));ratio=max(controls)/min(controls)
 receipt={'new_quality':560,'archival_endpoints':4200,'prefix_checks':210,'timing':630,'exact_targets':5,'timing_stable':ratio<=1.5,'sentinel_ratio':ratio}
 (OUT/'run_receipt.json').write_text(canonical(receipt)+'\n');assert receipt['timing_stable'],'timing stability failed; preserve raw attempt'
 print(canonical(receipt),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['pilot','run','analyze','verify']);args=parser.parse_args()
 if args.stage in ['pilot','run']:globals()[args.stage]()
 else:
  from report import analyze,verify
  globals()[args.stage]()
