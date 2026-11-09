from __future__ import annotations
import sys,json,hashlib,time,platform,zipfile,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'supplement'))
from mqhoa_supp.model import Instance,canonical,digest,schedule,validate
from mqhoa_supp.instances import Pool
from mqhoa_supp.methods import Budget,optimize,heft
from mqhoa_supp.protocol import pool_parameters
from recent_baseline.flexdo import solve as flexdo
from extended_validation.search.engine import direct_geometry
CONFIG=json.loads(Path(__file__).with_name('config.json').read_text())
OUT=Path(os.environ.get('MQHOA_TRACE_OUT',str(ROOT/'supplement/results/20261109_trace_validation_v1'))).resolve()
assert OUT.is_relative_to(ROOT/'supplement/results/20261109_trace_validation_v1'), 'new attempts must stay inside the dedicated result tree'
INPUT=ROOT/CONFIG['input_relative']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,obj):Path(path).write_text(canonical(obj)+'\n',encoding='utf8')
def rows(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l] if Path(path).exists() else []
def append(path,row):
 with Path(path).open('a',encoding='utf8') as f:f.write(canonical(row)+'\n');f.flush();os.fsync(f.fileno())

def original_parameters():
 archive=ROOT/'supplement/results/20261108_v1/raw/quality_dataset.zip'
 with zipfile.ZipFile(archive) as z:cfg=json.loads(z.read('protocol.json'));frozen=json.loads(z.read('frozen_defaults.json'))
 return cfg,frozen

def source_files():
 paths=sorted((ROOT/'supplement/mqhoa_supp').glob('*.py'))+sorted(Path(__file__).parent.glob('*.py'))
 paths += [Path(__file__).with_name('config.json'),Path(__file__).with_name('PLAN.md'),Path(__file__).with_name('AMENDMENT_DRAFT.md'),ROOT/'supplement/recent_baseline/flexdo.py',ROOT/'supplement/extended_validation/search/engine.py',ROOT/'supplement/extended_validation/search/config.json',ROOT/'supplement/extended_validation/provenance/parser.py',ROOT/'supplement/extended_validation/provenance/PROTOCOL.md']
 return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def load_inputs():
 gate=read(OUT/'input_gate.json');assert gate['authorized'] is True
 report_path=INPUT/'extraction_report.json';assert sha(report_path)==gate['extraction_report_sha256']
 primary=read(report_path);assert primary['official_sha256_verified'] and primary['source_sha256']==CONFIG['official_sha256']
 secondary_path=ROOT/CONFIG['secondary_report_relative'];assert gate['secondary_authorized'] is True and sha(secondary_path)==gate['secondary_report_sha256']
 secondary=read(secondary_path);assert secondary['source_sha256']==CONFIG['official_sha256'] and secondary['primary_group_unchanged']
 selected=[(r,INPUT/'instances','primary','small' if r['cell']=='small' else r['cell']) for r in primary['instances']]
 selected += [(r,ROOT/CONFIG['secondary_input_relative'],'secondary_large','large') for r in secondary['selected']]
 items={};manifest=[]
 for r,folder,cohort,stratum in selected:
  p=folder/(r['identifier']+'.json');assert sha(p)==r['sha256'];x=Instance(**read(p));md=x.metadata
  assert md['official_archive_verified'] and md['source_sha256']==CONFIG['official_sha256']
  assert md['speedup_is_synthetic'] and md['communication_is_synthetic'] and md['resource_speedup']==3 and md['ccr']==1
  assert r['n']==x.n
  lo,hi={'small':(8,20),'medium':(40,60),'large':(80,100)}[stratum];assert lo<=x.n<=hi
  roots=[i for i,z in enumerate(x.pre) if not z];sinks=[i for i,z in enumerate(x.succ) if not z];assert set(x.pinned)==set(roots+sinks)
  if cohort=='primary':assert len(roots)==len(sinks)==1
  else:assert len(roots)!=1 or len(sinks)!=1;assert md['no_dummy_nodes'] and not md['original_SESE_primary_group']
  assert all(v>0 and np.isfinite(v) for v in x.tl+x.ts)
  assert np.allclose(np.asarray(x.ts)*3,x.tl,rtol=1e-12,atol=1e-12)
  delay=float(np.mean((np.asarray(x.tl)+x.ts)/2));assert all(abs(c-delay)<1e-7 for a,b,c in x.edges)
  topo=digest({'n':x.n,'edges':sorted((a,b) for a,b,c in x.edges)})
  weighted=digest({'n':x.n,'tl':x.tl,'ts':x.ts,'edges':sorted(x.edges),'pinned':sorted(x.pinned)})
  assert x.identifier not in items;items[x.identifier]=x
  manifest.append({'identifier':x.identifier,'job_name':md['job_name'],'cohort':cohort,'stratum':stratum,'n':x.n,'roots':len(roots),'sinks':len(sinks),'input_file_sha256':r['sha256'],'instance_sha256':x.sha256,'weighted_instance_sha256':weighted,'labeled_topology_sha256':topo})
 assert sum(r['cohort']=='primary' for r in manifest)==sum(primary['selected_counts'].values())
 assert sum(r['cohort']=='secondary_large' for r in manifest)==secondary['selected_count']<=10
 for st in ['small','medium','large']:assert sum(r['stratum']==st and r['cohort']=='primary' for r in manifest)==primary['selected_counts'][st]<=10
 report={'source_sha256':primary['source_sha256'],'selected_counts':primary['selected_counts'],'missing_counts':primary['missing_counts'],'secondary_count':secondary['selected_count'],'primary_report_sha256':sha(report_path),'secondary_report_sha256':sha(secondary_path)}
 return items,manifest,report

def make_tasks(items):
 cfg,frozen=original_parameters();tasks=[]
 for identifier,x in sorted(items.items()):
  size,seed=pool_parameters(x,cfg)
  for method in CONFIG['stochastic_methods']+CONFIG['deterministic_methods']:
   for s in CONFIG['seeds'] if method in CONFIG['stochastic_methods'] else [None]:
    params=cfg['mqhoa'] if method in ['MQHOA','DirectGeometryMQHOA'] else frozen['PSO'] if method=='PSO' else {}
    t={'instance_id':identifier,'method':method,'seed':s,'params':params,'budget':1 if method=='HEFT' else CONFIG['budget_evals'],'pool_size':size,'pool_seed':seed}
    t['run_id']=digest(t)[:24];tasks.append(t)
 return tasks

def run_one(payload):
 task,data,provenance=payload;x=Instance(**data);started=time.perf_counter();pool=None
 if task['method'] in ['MQHOA','PSO']:pool=Pool(x,task['pool_size'],task['pool_seed'])
 tpool=time.perf_counter()-started;method=task['method']
 if method=='FlexDO-N2':result=flexdo(x,'N2',task['budget'])
 else:
  ev=Budget(x,task['budget'],100);rng=np.random.default_rng(task['seed'])
  if method=='DirectGeometryMQHOA':details=direct_geometry(x,ev,rng,task['params'])
  elif method=='HEFT':details=heft(x,ev)
  else:details=optimize(method,x,pool,ev,rng,task['params'])
  result={**ev.result(),'details':details}
 assert result['validator_ok'] and not validate(x,result['best_l'],result['best_pi'],result['starts'],result['finishes'])
 return {**result,**task,**provenance,'instance_sha256':x.sha256,'pool_sha256':pool.sha256 if pool else None,'t_pool_s':tpool,'parallel_quality_wall_s':time.perf_counter()-started,'timing_scope':'parallel quality metadata; not dedicated timing','host':platform.node(),'python':sys.version.split()[0],'numpy':np.__version__,'workers':2,'status':'completed'}
