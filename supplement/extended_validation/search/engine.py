from __future__ import annotations
import sys,json,hashlib,math,time,zipfile,platform
from pathlib import Path
from decimal import Decimal
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'supplement'))
from mqhoa_supp.model import Instance,canonical,digest,schedule,validate
from mqhoa_supp.instances import Pool
from mqhoa_supp.methods import Budget,optimize,mqhoa

SOURCE=ROOT/'supplement/results/20261108_v1/raw/quality_dataset.zip'
CONFIG=json.loads(Path(__file__).with_name('config.json').read_text())
OUT=ROOT/'supplement/results/20261109_search_v1'

def load():
 with zipfile.ZipFile(SOURCE) as z:
  cfg=json.loads(z.read('protocol.json'));frozen=json.loads(z.read('frozen_defaults.json'))
  inst={x.identifier:x for n in z.namelist() if n.startswith('instances/') and n.endswith('.json') for x in [Instance(**json.loads(z.read(n)))] if x.metadata.get('split')!='tuning'}
  rows=[json.loads(l) for n in z.namelist() if n.startswith('results/evaluation-shard') and n.endswith('.jsonl') for l in z.read(n).decode().splitlines() if l]
 return inst,rows,cfg,frozen

def record(path,row):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a',encoding='utf8') as f:f.write(canonical(row)+'\n');f.flush()

def rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x] if path.exists() else []

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def freeze():
 files=[SOURCE,*sorted((ROOT/'supplement/mqhoa_supp').glob('*.py')),*sorted(Path(__file__).parent.glob('*.py')),Path(__file__).with_name('config.json'),Path(__file__).with_name('PLAN.md')]
 return {'files':{str(p.relative_to(ROOT)):sha(p) for p in files},'config':CONFIG,'host':platform.node(),'python':sys.version,'numpy':np.__version__}

class StreamBudget(Budget):
 def __init__(self,*args,**kw):super().__init__(*args,**kw);self.stream=hashlib.sha256();self.initial=hashlib.sha256();self.checkpoints={}
 def evaluate(self,loc,order):
  data=canonical([list(map(int,loc)),list(map(int,order))]).encode();self.stream.update(data)
  if self.used<20:self.initial.update(data)
  value=super().evaluate(loc,order)
  if self.used in CONFIG['budget_endpoints']:self.checkpoints[str(self.used)]=self.stream.hexdigest()
  return value

class PermutedPool:
 def __init__(self,pool,seed):
  self.pool=pool;self.size=pool.size;self.sha256=pool.sha256
  r=np.random.default_rng(seed);self.perm=np.asarray([r.permutation(self.size),r.permutation(self.size)])
  self.inverse=np.argsort(self.perm,axis=1)
 def decode(self,pair):return self.pool.decode([self.perm[d,int(pair[d])] for d in range(2)])
 def encode(self,pair):return [int(self.inverse[d,int(pair[d])]) for d in range(2)]
 def latent(self,pair):return [int(self.perm[d,int(pair[d])]) for d in range(2)]

class InitialMappedRNG:
 def __init__(self,seed,pool):self.r=np.random.default_rng(seed);self.pool=pool;self.first=True
 def integers(self,*args,**kwargs):
  x=self.r.integers(*args,**kwargs)
  if self.first:self.first=False;return np.asarray([self.pool.encode(p) for p in x])
  return x
 def random(self,*a,**kw):return self.r.random(*a,**kw)

class ConjugatePool:
 # Latent implementation of conjugate: encode state, inverse before original
 # Gaussian update, encode again; decoder maps encoded state back to candidate.
 def __init__(self,p):self.p=p;self.size=p.size
 def decode(self,pair):return self.p.decode(self.p.encode(pair))

def direct_geometry(inst,ev,rng,params):
 k,m=int(params['k']),int(params['m']);lam=params['lambda'];d=2*inst.n
 def decode(x):
  loc=(x[:inst.n]>=.5).astype(int).tolist()
  for p in inst.pinned:loc[p]=0
  order=[]
  for layer in inst.layers:order.extend(sorted(layer,key=lambda j:(x[inst.n+j],j)))
  return loc,order
 initial=rng.random((k*m,d));fit=[ev.evaluate(*decode(x)) for x in initial];centers=initial[np.argsort(fit,kind='stable')[:k]].copy();sigma=.5;iterations=reductions=0
 while ev.remaining:
  cand=[]
  for c in range(k):
   group=[]
   for _ in range(min(m,ev.remaining)):
    u=rng.random((2,d));z=np.sqrt(-2*np.log(np.maximum(u[0],np.finfo(float).tiny)))*np.sin(2*np.pi*u[1]);x=np.clip(centers[c]+sigma*z,0,1)
    loc,order=decode(x);v=ev.evaluate(loc,order);item=(v,x,loc,order);cand.append(item);group.append(item)
   if group:centers[c]=min(group,key=lambda x:x[0])[1]
   if not ev.remaining:break
  best=min(cand,key=lambda x:x[0]);h=max(sum(a!=b for a,b in zip(x[2],best[2]))+sum(a!=b for a,b in zip(x[3],best[3])) for x in cand)/(2*inst.n)
  if h<sigma:sigma/=lam;reductions+=1
  iterations+=1
 return {'iterations':iterations,'scale_reductions':reductions,'sigma_final':sigma,'interpretation':'joint candidate-space/neighborhood geometry change'}

def run_task(payload):
 task,data,mode,budget=payload;inst=Instance(**data);tic=time.perf_counter();pool=Pool(inst,task['pool_size'],task['pool_seed']);tpool=time.perf_counter()-tic
 ev=StreamBudget(inst,budget,100);rng=np.random.default_rng(task['seed']);params=task['params'];layoutseed=CONFIG['relabel_seed_base']+int(hashlib.sha256(inst.identifier.encode()).hexdigest()[:8],16)
 if mode=='SamePoolRelabeledMQHOA':
  p=PermutedPool(pool,layoutseed);details=mqhoa(p,ev,InitialMappedRNG(task['seed'],p),params)
 elif mode=='Conjugate':details=mqhoa(ConjugatePool(PermutedPool(pool,layoutseed)),ev,rng,params)
 elif mode=='DirectGeometryMQHOA':details=direct_geometry(inst,ev,rng,params)
 else:details=optimize(task['method'],inst,pool,ev,rng,params)
 result=ev.result()
 return {**result,'instance_id':inst.identifier,'method':mode,'seed':task['seed'],'budget':budget,'params':params,'pool_sha256':pool.sha256,'instance_sha256':inst.sha256,'stream_sha256':ev.stream.hexdigest(),'initial_sha256':ev.initial.hexdigest(),'prefix_hashes':ev.checkpoints,'t_pool':tpool,'t_total':time.perf_counter()-tic,'details':details,'host':platform.node(),'python':sys.version.split()[0]}

def exact(inst,limit=120):
 from ortools.sat.python import cp_model
 scale=100;values=inst.tl+inst.ts+[c for a,b,c in inst.edges];err=max(abs(float(v)*scale-round(float(v)*scale)) for v in values)
 if err>1e-7:raise ValueError('nonexact integerization')
 tl=[round(x*scale) for x in inst.tl];ts=[round(x*scale) for x in inst.ts];h=sum(tl)+sum(round(c*scale) for _,_,c in inst.edges)+1
 model=cp_model.CpModel();loc=[model.new_bool_var('x'+str(i)) for i in range(inst.n)];start=[model.new_int_var(0,h,'s'+str(i)) for i in range(inst.n)];end=[model.new_int_var(0,h,'e'+str(i)) for i in range(inst.n)]
 intervals=[[],[]]
 for i in range(inst.n):
  for r,duration,present in [(0,tl[i],loc[i].Not()),(1,ts[i],loc[i])]:intervals[r].append(model.new_optional_interval_var(start[i],duration,end[i],present,f'I{i}_{r}'))
  if i in inst.pinned:model.add(loc[i]==0)
 for a,b,c in inst.edges:
  diff=model.new_bool_var(f'd{a}_{b}');model.add(loc[a]!=loc[b]).only_enforce_if(diff);model.add(loc[a]==loc[b]).only_enforce_if(diff.Not());model.add(start[b]>=end[a]+round(c*scale)*diff)
 for ints in intervals:model.add_no_overlap(ints)
 span=model.new_int_var(0,h,'makespan');model.add_max_equality(span,end);model.minimize(span)
 solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=limit;solver.parameters.num_search_workers=1;solver.parameters.random_seed=20261109
 status=solver.solve(model);name=solver.status_name(status);row={'instance_id':inst.identifier,'instance_sha256':inst.sha256,'solver':'OR-Tools CP-SAT','status':name,'proven':status==cp_model.OPTIMAL,'lb':solver.best_objective_bound/scale,'ub':None,'scale':scale,'integerization_max_error':err,'solver_seconds':solver.wall_time,'branches':solver.num_branches,'conflicts':solver.num_conflicts,'model_domain':'all binary placements/topological orders; not restricted to finite pools or depth-layer orders'}
 if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
  l=[solver.value(x) for x in loc];pi=sorted(range(inst.n),key=lambda i:(solver.value(start[i]),inst.topo.index(i)));v,s,f=schedule(inst,l,pi);assert not validate(inst,l,pi,s,f)
  row.update(ub=solver.objective_value/scale,best_l=l,best_pi=pi,decoded_makespan=v,validator_ok=True)
  if row['proven']:assert abs(v-row['ub'])<1e-7
 return row
