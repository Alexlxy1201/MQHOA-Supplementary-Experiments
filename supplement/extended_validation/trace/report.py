from __future__ import annotations
from engine import *
from collections import defaultdict,Counter
from mqhoa_supp.statistics import paired_analysis,bootstrap_mean,holm,write_csv


def expected_and_rows():
 from campaign import inputs_and_meta
 items,manifest,report,meta=inputs_and_meta();tasks=make_tasks(items);data=rows(OUT/'quality.jsonl')
 assert {r['run_id'] for r in data}=={t['run_id'] for t in tasks} and len(data)==len(tasks),'incomplete/duplicate condition set'
 return items,manifest,report,meta,tasks,data

def verify():
 items,manifest,report,meta,tasks,data=expected_and_rows();lookup={t['run_id']:t for t in tasks};poolhash={}
 for r in data:
  t=lookup[r['run_id']];x=items[r['instance_id']]
  for key in ['method','seed','params','budget','pool_seed','pool_size']:assert r[key]==t[key]
  for key,val in meta[x.identifier].items():assert r[key]==val,(x.identifier,key)
  assert r['instance_sha256']==x.sha256 and r['validator_ok'];assert 0<r['evals_used']<=r['budget']
  if r['method'] in CONFIG['stochastic_methods']:assert r['evals_used']==10000
  assert not validate(x,r['best_l'],r['best_pi'],r['starts'],r['finishes']);assert abs(schedule(x,r['best_l'],r['best_pi'])[0]-r['best_makespan'])<1e-7
  assert np.isfinite(r['best_makespan']) and r['best_makespan']>0
  if r['method'] in ['MQHOA','PSO']:
   key=(x.identifier,r['pool_seed'],r['pool_size']);poolhash.setdefault(key,r['pool_sha256']);assert poolhash[key]==r['pool_sha256']
  else:assert r['pool_sha256'] is None
 forbidden=[]
 for p in OUT.rglob('*'):
  if p.is_file() and (p.suffix.lower() in ['.csv','.sqlite','.tgz'] and 'analysis' not in p.parts or p.name.endswith('.tar.gz')):forbidden.append(str(p))
 # Analysis CSVs contain newly computed results, never source CSV rows.
 assert not forbidden,forbidden
 value={'status':'passed','jobs':len(items),'primary_jobs':sum(m['cohort']=='primary' for m in meta.values()),'secondary_jobs':sum(m['cohort']=='secondary_large' for m in meta.values()),'unique_conditions':len(data),'schedule_rows_recomputed':len(data),'stochastic_conditions':sum(r['method'] in CONFIG['stochastic_methods'] for r in data),'deterministic_conditions':sum(r['method'] in CONFIG['deterministic_methods'] for r in data),'selected_counts':report['selected_counts'],'missing_counts':report['missing_counts'],'same_input_pool_hashes':True,'raw_trace_or_input_instances_copied':False,'source_freeze_sha256':sha(OUT/'source_freeze.json'),'quality_log_sha256':sha(OUT/'quality.jsonl')};write(OUT/'verification.json',value);print(canonical(value),flush=True);return value

def statistics(differences,seed):
 rng=np.random.default_rng(seed);r=paired_analysis(differences,rng,CONFIG['bootstrap_samples']);n=len(differences)
 if 16<n<=20:
  d=np.asarray(differences);observed=abs(float(d.mean()));count=0;total=1<<n
  for start in range(0,total,4096):
   codes=np.arange(start,min(start+4096,total),dtype=np.uint64);signs=2*((codes[:,None]>>np.arange(n,dtype=np.uint64))&1).astype(float)-1
   count+=int(np.sum(np.abs((signs*d).mean(axis=1))>=observed-1e-12))
  r['primary_p']=count/total;r['primary_test']='exact sign flips'
 r['analysis_seed']=seed;r['pairing_unit']='selected trace job; repeated topology/shared environment acknowledged'
 return r

def analyze():
 verified=verify();items,manifest,report,meta,tasks,data=expected_and_rows();dest=OUT/'analysis';assert not dest.exists(),'existing analysis preserved';dest.mkdir()
 groups=defaultdict(list)
 for r in data:groups[r['instance_id'],r['method']].append(r)
 rng=np.random.default_rng(CONFIG['bootstrap_seed']);summaries=[];means={}
 for (i,m),group in sorted(groups.items()):
  expected=20 if m in CONFIG['stochastic_methods'] else 1;assert len(group)==expected
  scores=[r['best_makespan'] for r in group];lo,hi=bootstrap_mean(scores,rng,CONFIG['bootstrap_samples']);means[i,m]=float(np.mean(scores))
  summaries.append({'instance_id':i,'job_name':meta[i]['job_name'],'stratum':meta[i]['stratum'],'cohort':meta[i]['cohort'],'n':items[i].n,'method':m,'runs':expected,'mean_makespan_proxy_s':means[i,m],'sd':float(np.std(scores,ddof=1)) if expected>1 else 0.,'ci95_low_conditional_seeds':lo,'ci95_high_conditional_seeds':hi,'best_observed':min(scores),'median_evals_used':float(np.median([r['evals_used'] for r in group]))})
 write_csv(dest/'per_job_method_summary.csv',summaries)
 main=[];ids=sorted(items);primary_ids=[i for i in ids if meta[i]['cohort']=='primary'];secondary_ids=[i for i in ids if meta[i]['cohort']=='secondary_large']
 if len(primary_ids)>=2:
  for index,opponent in enumerate(CONFIG['main_opponents']):
   d=[(means[i,'MQHOA']-means[i,opponent])/means[i,opponent] for i in primary_ids];main.append({'family':'main-three','scope':'primary-SESE-only','reference':'MQHOA','opponent':opponent,'direction':'negative favors MQHOA',**statistics(d,CONFIG['bootstrap_seed']+index)})
  holm(main)
 write_csv(dest/'main_three_job_tests.csv',main)
 descriptive=[]
 scopes=[('primary20',primary_ids),('secondary-large',secondary_ids),('combined-fixed-cohorts-descriptive',ids)]+[('primary-'+s,[i for i in primary_ids if meta[i]['stratum']==s]) for s in ['small','medium','large']]
 for scope,selected in scopes:
  if not selected:continue
  for opponent in CONFIG['main_opponents']+['DirectGeometryMQHOA']:
   d=[(means[i,'MQHOA']-means[i,opponent])/means[i,opponent] for i in selected];ci=bootstrap_mean(d,rng,CONFIG['bootstrap_samples'])
   descriptive.append({'scope':scope,'reference':'MQHOA','opponent':opponent,'selected_jobs':len(selected),'mean_relative_difference':float(np.mean(d)),'ci95_low':ci[0],'ci95_high':ci[1],'mqhoa_lower_count':sum(v<-1e-10 for v in d),'tie_count':sum(abs(v)<=1e-10 for v in d),'mqhoa_higher_count':sum(v>1e-10 for v in d),'inference_role':'descriptive; DirectGeometry excluded from main family'})
 write_csv(dest/'stratified_relative_descriptive.csv',descriptive)
 features=[]
 for i in ids:
  x=items[i];features.append({'instance_id':i,'job_name':meta[i]['job_name'],'stratum':meta[i]['stratum'],'cohort':meta[i]['cohort'],'tasks':x.n,'edges':len(x.edges),'depth':max(x.depth),'max_layer_width':max(map(len,x.layers)),'directed_dag_density':len(x.edges)/(x.n*(x.n-1)/2),'wall_span_proxy_mean_s':float(np.mean(x.tl)),'synthetic_speedup':3,'synthetic_ccr':1,'input_file_sha256':meta[i]['input_file_sha256'],'instance_sha256':x.sha256,'weighted_instance_sha256':meta[i]['weighted_instance_sha256'],'labeled_topology_sha256':meta[i]['labeled_topology_sha256']})
 write_csv(dest/'derived_job_features.csv',features)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 topology_details=[];topology_sensitivity=[]
 for cohort,selected in [('primary',primary_ids),('secondary_large',secondary_ids)]:
  groups_topo=defaultdict(list)
  for i in selected:groups_topo[meta[i]['labeled_topology_sha256']].append(i)
  for topo,members in sorted(groups_topo.items()):topology_details.append({'cohort':cohort,'labeled_topology_sha256':topo,'jobs_in_group':len(members),'job_ids':';'.join(members)})
  for opponent in CONFIG['main_opponents']+['DirectGeometryMQHOA']:
   values=[float(np.mean([(means[i,'MQHOA']-means[i,opponent])/means[i,opponent] for i in members])) for members in groups_topo.values()]
   ci=bootstrap_mean(values,rng,CONFIG['bootstrap_samples']);topology_sensitivity.append({'cohort':cohort,'opponent':opponent,'jobs':len(selected),'labeled_topology_groups':len(groups_topo),'mean_relative_difference_equal_topology_weight':float(np.mean(values)),'ci95_low_descriptive':ci[0],'ci95_high_descriptive':ci[1],'p_value':'not tested','interpretation':'descriptive repeated-layout sensitivity; no independent-random-topology or population claim'})
 write_csv(dest/'topology_groups.csv',topology_details);write_csv(dest/'equal_topology_group_sensitivity.csv',topology_sensitivity)
 methods=['MQHOA','PSO','HEFT','FlexDO-N2','DirectGeometryMQHOA'];fig,axs=plt.subplots(1,3,figsize=(13,4),sharey=True)
 panels=[('Primary small',[i for i in primary_ids if meta[i]['stratum']=='small']),('Primary medium',[i for i in primary_ids if meta[i]['stratum']=='medium']),('Secondary large (non-SESE)',secondary_ids)]
 for ax,(label,selected) in zip(axs,panels):
  for m in methods:
   if selected:ax.plot(range(len(selected)),[means[i,m]/means[i,'MQHOA'] for i in selected],'o-',label=m)
  ax.set_title(label+f' ({len(selected)} jobs)',fontsize=10);ax.set_xlabel('Lexicographic selected job rank');ax.set_xticks(range(len(selected)),range(1,len(selected)+1));ax.axhline(1,color='black',ls='--',lw=.7)
 axs[0].set_ylabel('Makespan / MQHOA job mean');axs[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(dest/'trace_comparison.png',dpi=180);fig.savefig(dest/'trace_comparison.pdf');plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for ax,(label,selected) in zip(axs,[('Primary SESE',primary_ids),('Secondary non-SESE large',secondary_ids)]):
  for m in methods:
   values=[means[i,m]/means[i,'MQHOA'] for i in selected];ax.plot(sorted(values),np.arange(1,len(values)+1)/len(values),label=m)
  ax.set(xlabel='Per-job mean makespan / MQHOA',ylabel='ECDF over selected jobs',title=label)
 axs[-1].legend(fontsize=8);fig.tight_layout();fig.savefig(dest/'trace_ecdf.png',dpi=180);fig.savefig(dest/'trace_ecdf.pdf');plt.close(fig)
 completion={'status':'completed','verification':verified,'main_three_comparisons':main,'quality_search_evaluations':sum(r['evals_used'] for r in data),'primary_strata':report['selected_counts'],'primary_missing':report['missing_counts'],'secondary_count':report['secondary_count'],'topology_group_counts':{c:sum(r['cohort']==c for r in topology_details) for c in ['primary','secondary_large']},'selection_is_representative_random_sample':False,'device_performance_measured':False,'historical_job_identity_recovered':False,'parallel_times_not_speed_evidence':True,'source_freeze_sha256':sha(OUT/'source_freeze.json')};write(OUT/'completion.json',completion)
 report_text(dest,completion,descriptive,report)
 print(canonical(completion),flush=True)

def report_text(dest,completion,descriptive,report):
 number=completion['verification']['jobs'];counts=report['selected_counts'];summary=[]
 for r in completion['main_three_comparisons']:summary.append(f'MQHOA vs {r["opponent"]}: mean (MQHOA-opponent)/opponent={r["mean_relative_difference"]:.4%},95%CI[{r["ci95_low"]:.4%},{r["ci95_high"]:.4%}], primary p={r["primary_p"]:.6g}, Holm p={r["holm_p"]:.6g}; n={r["instances"]} selected jobs.')
 base=['# New trace-derived workload supplement','',f'Official Alibaba batch_task.tar.gz checksum verified: {CONFIG["official_sha256"]}. Primary selection yields20SESE jobs: {counts}; primary shortages={report["missing_counts"]}. A separately declared secondary cohort contributes{report["secondary_count"]} large non-SESE jobs, without filling the primary missing cell. Total computed jobs={number}. Each stochastic method uses20 seeds and10000 evaluations, while HEFT and FlexDO-N² each run once. All schedules are independently recomputed and validated.','', 'The DAGs are new trace-derived graphs. Durations are observed batch-task wall-span proxies; server speedup3 and communicationCCR1 are synthetic. This is neither measured device performance nor a reconstruction of historical archive10. Selection takes the first eligible names rather than a representative random sample. No holdout/population generalization claim is made. Parallel quality wall times are not dedicated speed benchmarks.','',*summary,'', 'The main three-comparison Holm family uses primary20 only; secondary10 and combined30 summaries are descriptive. DirectGeometryMQHOA is descriptive and excluded from the main family. Selected trace jobs, not seeds, are pairing units; repeated layouts and shared cluster wall-span conditions limit independent-topology inference. An equal-weight labeled-topology-group bootstrap sensitivity is descriptive only, with no p claims. Group counts are recorded in completion.json and topology_groups.csv.','', '## R3.2 — dependency accuracy','The new experiment uses the official numeric task-name dependency rule, validated parser fixtures, missing-parent/self-edge/cycle checks and connectivity; primary requires single entry/exit, while the separately labeled secondary cohort preserves multiple terminals and pins all of them. Input file and semantic graph hashes plus selected job IDs are retained. New scheduling feasibility validation is distinct from source parsing validation; the extraction report supplies the full filtering funnel. No claim of recovering the archived job is added.','', '## R4.8 — justify trace extraction','The official archive hash and pinned source/parser/config hashes identify the source and procedure. Exact selection rules, shortages and job IDs are recorded without performance selection. Raw CSV/tar/SQLite and per-task inputJSON remain local and are excluded from evidence/Git packages.','', '## R4.9 — workload generation and characteristics','This extension supplements the earlier synthetic grid with selected real task-dependency graphs. Derived task counts, edge counts, depth, width and density are tabulated in derived_job_features.csv. Wall-span proxy means and explicit synthetic speedup3/CCR1 describe the numerical transformation. The previous synthetic generator/results and original manuscript are unchanged.','', '## Reproducibility and provenance','Use the frozen local inputs and original repository with the commands in PLAN.md; verify reconstructs every schedule and checks the complete condition matrix and hashes. Raw input is deliberately not redistributed. Source: https://github.com/alibaba/clusterdata/blob/3858992ca41bf92082e6d6522ddf0bebd14e838b/cluster-trace-v2018/trace_2018.md. Procedural design credit: Kassis et al.(2026), Scientific Agent Skills, https://doi.org/10.48550/arXiv.2609.00065.']
 (dest/'NEW_SECTION_AND_RESPONSES_EN.md').write_text('\n'.join(base)+'\n',encoding='utf8')
 zh=['# 新增真实trace派生DAG补充实验与逐条回复','',f'官方batch_task.tar.gz通过SHA核验，primary严格SESE选得20job，分层数量为{counts}，缺口{report["missing_counts"]}。单独事前声明的secondary非SESE大图有{report["secondary_count"]}job，不能填称primary缺额；总计算{number}job。MQHOA、PSO、直接几何变体各20种子和一万评价；HEFT及FlexDO-N²各一次，全部结果经独立调度校验。','', '依赖拓扑来自trace；任务耗时是batch-task实际墙钟跨度的均值代理，不是两类设备上的CPU服务时间。服务器加速比3与通信CCR1属于合成设定。该实验既不是原历史job复现，也不是实测设备性能；按job名称选取首批合格样本，不作全库代表性或独立holdout泛化声明。并行质量运行wall时间仅作执行元数据，不作速度优越性证据。','']
 for r in completion['main_three_comparisons']:zh.append(f'MQHOA相对{r["opponent"]}差值{r["mean_relative_difference"]:.3%}，95%CI[{r["ci95_low"]:.3%},{r["ci95_high"]:.3%}]，主检验p={r["primary_p"]:.6g}，三比较Holm p={r["holm_p"]:.6g}。负值有利MQHOA；统计独立单位为job。')
 zh+=['', '三个主要比较及Holm仅使用primary20；secondary10与合并30仅作描述。直接几何变体改变候选空间与邻域，也只作描述。统计配对单位称所选trace job，不能当作独立随机拓扑样本；重复布局和共享集群壁钟环境限制独立性。另加按节点数与规范索引边集分组、组等权的bootstrap敏感性，仅给描述区间，不作小组数p强结论。拓扑组数、全weighted图hash与输入hash均已记录。','', '## R3.2：依赖准确性','采用官方task_name数字依赖规则，并通过测试夹具、缺失父节点、自环、循环和连通性检查；primary要求单入口出口，secondary单独标记多入口出口并全部固定在本地。job选择ID、输入文件hash和语义图hash保留。调度可行性核验与来源解析正确性分开报告，完整筛选漏斗由提取报告记录；不声称恢复原归档job。','', '## R4.8：提取依据','官方数据SHA与固定parser/config/source hash构成来源链；明确选取规则与缺口，不按算法成绩挑选。原CSV、tar、SQLite和输入Instance JSON仅本地保留，不进入结果或Git包。','', '## R4.9：工作负载构造细节','在原合成网格之外新增trace依赖图，逐job报告节点、边、深度、宽度和密度，区分wall-span代理与合成速度/通信设定。既有合成生成器、实验结果和原稿均不改。','', '复现命令见新PLAN.md；独立verify重建每条调度并核对完整矩阵、输入及源码hash。完整统计、派生特征和图表均位于此新目录。']
 (dest/'NEW_SECTION_AND_RESPONSES_ZH.md').write_text('\n'.join(zh)+'\n',encoding='utf8')
