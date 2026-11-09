from engine import *
from collections import defaultdict,Counter
from mqhoa_supp.statistics import paired_analysis,bootstrap_mean,holm,write_csv
from scipy.stats import binomtest

def verify():
 from campaign import gate,jobs_data,archived_endpoint
 gate();inst,base,cfg,frozen=jobs_data();q=rows(OUT/'new_quality.jsonl');b=rows(OUT/'budget_archival.jsonl');p=rows(OUT/'prefix_checks.jsonl');t=rows(OUT/'timing.jsonl');e=rows(OUT/'exact.jsonl')
 assert len(q)==560 and len({(x['instance_id'],x['method'],x['seed']) for x in q})==560
 assert len(b)==4200 and len({(x['instance_id'],x['method'],x['seed'],x['budget']) for x in b})==4200
 assert len(p)==210 and len({(x['instance_id'],x['method'],x['budget']) for x in p})==210
 assert len(t)==630 and len({(x['instance_id'],x['method'],x['budget'],x['repeat']) for x in t})==630
 assert len(e)==5 and len({x['instance_id'] for x in e})==5
 for row in q+p+t:
  x=inst[row['instance_id']];assert x.sha256==row['instance_sha256'];assert row['evals_used']==row['budget'];assert not validate(x,row['best_l'],row['best_pi'],row['starts'],row['finishes'])
  v=schedule(x,row['best_l'],row['best_pi'])[0];assert abs(v-row['best_makespan'])<1e-8
 for x in b:
  old=next(y for y in base if y['instance_id']==x['instance_id'] and y['method']==x['method'] and y['seed']==x['seed']);assert x['best_makespan']==archived_endpoint(old,x['budget'])
 for x in e:
  if x['proven']:assert x['status']=='OPTIMAL' and abs(x['lb']-x['ub'])<1e-8 and abs(x['decoded_makespan']-x['ub'])<1e-8
 receipt=json.loads((OUT/'run_receipt.json').read_text());assert receipt['timing_stable'];assert len({x['host'] for x in t})==1 and all(x['workers']==1 for x in t)
 report={'status':'passed','new_quality_unique':560,'archival_endpoints_unique':4200,'prefix_checks_unique':210,'runtime_unique':630,'proven_exact_instances':sum(x['proven'] for x in e),'schedule_validation_rows':len(q+p+t),'frozen_input_source_sha256':sha(SOURCE),'old_data_modified':False}
 (OUT/'verification.json').write_text(canonical(report)+'\n');print(canonical(report),flush=True);return report

def analyze():
 from campaign import jobs_data
 verification=verify();inst,base,cfg,frozen=jobs_data();q=rows(OUT/'new_quality.jsonl');arch=rows(OUT/'budget_archival.jsonl');time_rows=rows(OUT/'timing.jsonl');exact_rows=rows(OUT/'exact.jsonl');dest=OUT/'analysis';assert not dest.exists();dest.mkdir()
 rng=np.random.default_rng(CONFIG['bootstrap_seed']);allbudget=arch+[{k:r[k] for k in ['instance_id','method','seed']}|{'budget':b,'best_makespan':next(x[2] for x in r['trace'] if x[0]==b)} for r in q for b in CONFIG['budget_endpoints']]
 groups=defaultdict(list)
 for r in allbudget:groups[r['instance_id'],r['method'],r['budget']].append(r['best_makespan'])
 summaries=[];lookup={}
 for (i,m,b),v in sorted(groups.items()):
  assert len(v)==20;lo,hi=bootstrap_mean(v,rng,CONFIG['bootstrap_samples']);row={'instance_id':i,'kind':inst[i].metadata['kind'],'method':m,'budget':b,'runs':20,'mean':float(np.mean(v)),'sd':float(np.std(v,ddof=1)),'ci95_low':lo,'ci95_high':hi,'best_observed':min(v)};summaries.append(row);lookup[i,m,b]=row['mean']
 write_csv(dest/'budget_and_encoding_summary.csv',summaries)
 synthetic=sorted(i for i,x in inst.items() if x.metadata['kind']=='synthetic');comparisons=[]
 for arm in CONFIG['arms']:
  d=[(lookup[i,'MQHOA',10000]-lookup[i,arm,10000])/lookup[i,arm,10000] for i in synthetic];comparisons.append({'family':'encoding-two','reference':'MQHOA','opponent':arm,**paired_analysis(d,rng,CONFIG['bootstrap_samples'])})
 holm(comparisons)
 for m in CONFIG['methods_budget']:
  family=[]
  for low,high in zip(CONFIG['budget_endpoints'],CONFIG['budget_endpoints'][1:]):
   d=[(lookup[i,m,high]-lookup[i,m,low])/lookup[i,m,low] for i in synthetic];family.append({'family':'budget-increments-'+m,'reference':str(low),'opponent':str(high),**paired_analysis(d,rng,CONFIG['bootstrap_samples'])})
  holm(family);comparisons.extend(family)
 write_csv(dest/'paired_instance_tests.csv',comparisons)
 hits=[]
 for e in exact_rows:
  for m in CONFIG['methods_budget']+CONFIG['arms']:
   for b in CONFIG['budget_endpoints']:
    vals=groups[e['instance_id'],m,b];proven=e['proven'];count=sum(abs(v-e['ub'])<1e-7 for v in vals) if proven else None;ci=binomtest(count,20).proportion_ci(confidence_level=.95,method='exact') if proven else None
    hits.append({'instance_id':e['instance_id'],'method':m,'budget':b,'solver_status':e['status'],'proven':proven,'lb':e['lb'],'ub':e['ub'],'hits':count,'denominator':20 if proven else 0,'hit_rate':count/20 if proven else None,'binomial_ci95_low':ci.low if ci else None,'binomial_ci95_high':ci.high if ci else None,'mean_relative_gap':float(np.mean([(v-e['ub'])/e['ub'] for v in vals])) if proven else None})
 write_csv(dest/'optimal_hit_rates.csv',hits);write_csv(dest/'exact_solver_summary.csv',[{k:e.get(k) for k in ['instance_id','status','proven','lb','ub','scale','integerization_max_error','solver_seconds','branches','conflicts']} for e in exact_rows])
 runtime=[]
 for i in sorted(inst):
  for m in CONFIG['methods_budget']:
   for b in CONFIG['budget_endpoints']:
    rs=[r for r in time_rows if r['instance_id']==i and r['method']==m and r['budget']==b];assert len(rs)==3
    runtime.append({'instance_id':i,'method':m,'budget':b,'repeats':3,'mean_total_s':float(np.mean([r['t_total'] for r in rs])),'mean_pool_s':float(np.mean([r['t_pool'] for r in rs])),'sd_total_s':float(np.std([r['t_total'] for r in rs],ddof=1))})
 write_csv(dest/'runtime_by_budget.csv',runtime)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(11,4))
 for m in CONFIG['methods_budget']+CONFIG['arms']:
  y=[np.mean([lookup[i,m,b]/lookup[i,'MQHOA',10000] for i in synthetic]) for b in CONFIG['budget_endpoints']];ax[0].plot(CONFIG['budget_endpoints'],y,'o-',label=m)
 ax[0].set(xlabel='Evaluations',ylabel='Mean normalized makespan (12 DAGs)',xscale='log');ax[0].legend(fontsize=7)
 for m in CONFIG['methods_budget']:
  y=[np.mean([r['mean_total_s'] for r in runtime if r['method']==m and r['budget']==b]) for b in CONFIG['budget_endpoints']];ax[1].plot(CONFIG['budget_endpoints'],y,'o-',label=m)
 ax[1].set(xlabel='Evaluations',ylabel='Serial runtime including pool (s)',xscale='log');ax[1].legend();fig.tight_layout();fig.savefig(dest/'budget_encoding.png',dpi=180);fig.savefig(dest/'budget_encoding.pdf');plt.close(fig)
 fig,axs=plt.subplots(1,5,figsize=(16,3.4),sharey=True)
 for ax,e in zip(axs,exact_rows):
  for m in ['MQHOA',*CONFIG['arms']]:
   subset=[h for h in hits if h['instance_id']==e['instance_id'] and h['method']==m];ax.plot([h['budget'] for h in subset],[h['hit_rate'] for h in subset],'o-',label=m)
  ax.set_title(e['instance_id'],fontsize=8);ax.set_xscale('log');ax.set_ylim(-.03,1.03);ax.set_xlabel('Evaluations')
 axs[0].set_ylabel('Proven optimum hit fraction');axs[-1].legend(fontsize=6);fig.tight_layout();fig.savefig(dest/'optimal_hits.png',dpi=180);fig.savefig(dest/'optimal_hits.pdf');plt.close(fig)
 result={'status':'completed','verification':verification,'encoding_comparisons':comparisons[:2],'timing':json.loads((OUT/'run_receipt.json').read_text()),'new_results_only':True}
 (OUT/'completion.json').write_text(canonical(result)+'\n')
 lines=['# Additional search validation (new evidence)','',f'All 14 old DAGs and20 seeds were retained. Prefix checks:210, new encoding/layout runs:560, serial runtime runs:630. Proven optimum instances: {verification["proven_exact_instances"]}/5.','', 'The conjugate representation is an invariance check, not contribution evidence. SamePoolRelabeledMQHOA isolates fixed index-layout effects with identical initial physical candidates. DirectGeometryMQHOA changes both candidate space and neighborhood geometry; its contrast cannot isolate a pure representational cause.','']
 for c in comparisons[:2]:lines.append(f'{c["opponent"]}: mean (original-new)/new={c["mean_relative_difference"]:.4%},95%CI [{c["ci95_low"]:.4%},{c["ci95_high"]:.4%}], exact sign-flip p={c["primary_p"]:.6g}, Holm p={c["holm_p"]:.6g}; independent n=12 DAGs.')
 lines+=['','Budget trajectories are archived20-run traces;210 fresh endpoint checks reproduced candidate prefixes and scores. New timing is same-Mac serial including common pool generation, with3 repeats and no cross-machine speed claims. Small-instance optimal hits use only CP-SAT OPTIMAL proofs and exact integer scale100; unrestricted topological order optimum may be outside finite pool/layer search spaces.','', 'Reviewer response: Added fixed-budget trajectories, layout-controlled and direct-geometry contrasts, plus solver-certified small-instance hit rates. Tables disclose all outcomes, confidence intervals, effect sizes and correction families. Old results/statistics/manuscripts are unchanged.','', 'Reproduce: `.venv/bin/python campaign.py pilot`, then `run`, `analyze`, `verify` from this package directory. Remove no evidence; use a separate new result version for reruns.','', 'Solver reference: https://developers.google.com/optimization/cp/cp_solver and https://developers.google.com/optimization/scheduling/job_shop.']
 (dest/'ADDITIONAL_SECTION_AND_RESPONSE_EN.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 zh=['# 新增搜索验证与审稿回复','',f'保留14个旧实例和每例20种子；新增560次编码/布局对照、210次预算前缀复核、630次同机串行计时。5个小实例中{verification["proven_exact_instances"]}例获得CP-SAT最优证明。','', '同候选池重排仅改变索引布局，并映射初始候选以保持物理解相同；共轭表示仅作数学与实现不变性校验。直接二进制加层内随机键同时改变候选空间和算子邻域，应解释为去索引的搜索几何总体效应，不能声称纯表示因果。','']
 for c in comparisons[:2]:zh.append(f'{c["opponent"]}：原MQHOA相对新臂差值{c["mean_relative_difference"]:.2%}，95%CI [{c["ci95_low"]:.2%},{c["ci95_high"]:.2%}]，Holm p={c["holm_p"]:.6g}。正值表示新臂makespan较低，负值表示原算法较低。独立统计单位为12个合成DAG。')
 zh+=['','预算图来自原20次运行的真实500/1000/2500/5000/10000评价端点，新运行核对相同候选前缀。计时为本Mac单进程串行、含候选池生成，3次重复；不与旧Windows时间混合比较。最优命中率只使用OPTIMAL证明，报告LB/UB、百分之一精度与有限池/层内顺序限制。','', '逐条回复：已新增等预算轨迹、同候选空间索引布局对照、去索引搜索几何对照及小实例最优命中率；全部结果和不利结果均保留。新增统计以实例为独立单位，并报告95%CI、效应量和预设族内Holm校正。旧论文、数据、代码、结果和旧统计均未改写。']
 (dest/'ADDITIONAL_SECTION_AND_RESPONSE_ZH.md').write_text('\n'.join(zh)+'\n',encoding='utf8')
 manifest={'files':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}};(OUT/'manifest.json').write_text(canonical(manifest)+'\n');print(canonical(result),flush=True)
