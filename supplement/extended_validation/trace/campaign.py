from __future__ import annotations
from engine import *
from concurrent.futures import ProcessPoolExecutor
import argparse,subprocess,random

def freeze_sources():
 assert not (OUT/'source_freeze.json').exists(),'freeze already exists; preserve attempt'
 cfg,frozen=original_parameters();archive=ROOT/'supplement/results/20261108_v1/raw/quality_dataset.zip'
 value={'files':source_files(),'config':CONFIG,'old_quality_archive_sha256':sha(archive),'mqhoa_parameters':cfg['mqhoa'],'pso_parameters':frozen['PSO'],'old_protocol_sha256':digest(cfg),'old_frozen_sha256':digest(frozen),'trace_selection':'first10 per frozen stratum, lexicographic job name; no performance selection','source_frozen_before_input_access':False,'source_freeze_stage':'amendment1-before-any-formal-results; original freeze and primary pilot preserved','input_gate_sha256':sha(OUT/'input_gate.json')}
 write(OUT/'source_freeze.json',value);print('source freeze',sha(OUT/'source_freeze.json'),flush=True)

def check_sources():
 f=read(OUT/'source_freeze.json');assert sha(OUT/'input_gate.json')==f['input_gate_sha256'];assert f['files']==source_files(),'frozen source/config changed'
 archive=ROOT/'supplement/results/20261108_v1/raw/quality_dataset.zip';assert sha(archive)==f['old_quality_archive_sha256']
 return sha(OUT/'source_freeze.json')

def unit_tests():
 check_sources();assert not (OUT/'test_report.txt').exists(),'test attempt exists'
 p=subprocess.run([sys.executable,str(Path(__file__).with_name('tests.py'))],capture_output=True,text=True);(OUT/'test_report.txt').write_text(p.stdout+p.stderr)
 assert p.returncode==0,'fixture tests failed';write(OUT/'tests_receipt.json',{'tests_passed':True,'fixture_only':True,'source_freeze_sha256':sha(OUT/'source_freeze.json')});print('fixture tests passed',flush=True)

def inputs_and_meta():
 sourcehash=check_sources();items,manifest,report=load_inputs();lookup={r['identifier']:r for r in manifest};reporthash=sha(INPUT/'extraction_report.json')
 meta={i:{'source_freeze_sha256':sourcehash,'trace_config_sha256':sha(Path(__file__).with_name('config.json')),'extraction_report_sha256':reporthash,'secondary_report_sha256':report['secondary_report_sha256'],'official_source_archive_sha256':CONFIG['official_sha256'],'input_file_sha256':lookup[i]['input_file_sha256'],'job_name':lookup[i]['job_name'],'stratum':lookup[i]['stratum'],'cohort':lookup[i]['cohort'],'labeled_topology_sha256':lookup[i]['labeled_topology_sha256'],'weighted_instance_sha256':lookup[i]['weighted_instance_sha256'],'n':lookup[i]['n'],'duration_source':'observed batch-task wall-span proxy','synthetic_server_speedup':3,'synthetic_communication_ccr':1,'historical_job_reconstruction':False} for i in items}
 return items,manifest,report,meta

def pilot():
 assert read(OUT/'tests_receipt.json')['tests_passed'];assert not (OUT/'pilot.jsonl').exists(),'pilot exists; no retry'
 items,manifest,report,meta=inputs_and_meta();write(OUT/'input_manifest.json',{'extraction_report_sha256':sha(INPUT/'extraction_report.json'),'source_archive_sha256':CONFIG['official_sha256'],'selected_counts':report['selected_counts'],'missing_counts':report['missing_counts'],'jobs':manifest})
 tasks=make_tasks(items);chosen=[]
 for stratum in ['small','medium','large']:
  ids=sorted(x['identifier'] for x in manifest if x['stratum']==stratum)
  if ids:chosen.extend(t for t in tasks if t['instance_id']==ids[0] and (t['seed'] in [0,None]))
 durations={};results=[]
 for t in chosen:
  row=run_one((t,items[t['instance_id']].data(),meta[t['instance_id']]));append(OUT/'pilot.jsonl',row);durations[row['stratum'],row['method']]=row['parallel_quality_wall_s'];results.append(row)
  print('pilot',row['stratum'],row['method'],round(row['parallel_quality_wall_s'],4),row['evals_used'],flush=True)
 projected=sum(durations[meta[t['instance_id']]['stratum'],t['method']] for t in tasks)
 value={'status':'passed','pilot_runs':len(results),'formal_runs':len(tasks),'projected_serial_s':projected,'projected_workers2_s':projected/2,'workers':2,'note':'projection from one job per stratum; actual topology/duration variations may change runtime; no settings tuned','selected_counts':report['selected_counts'],'missing_counts':report['missing_counts'],'source_freeze_sha256':sha(OUT/'source_freeze.json')};write(OUT/'pilot_receipt.json',value);print(canonical(value),flush=True)

def pilot_secondary():
 assert read(OUT/'tests_receipt.json')['tests_passed'];assert not (OUT/'pilot_secondary.jsonl').exists(),'secondary pilot already attempted'
 items,manifest,report,meta=inputs_and_meta();tasks=make_tasks(items);secondary=sorted(i for i in items if meta[i]['cohort']=='secondary_large');chosen=[t for t in tasks if secondary and t['instance_id']==secondary[0] and t['seed'] in [0,None]]
 durations={('primary',r['stratum'],r['method']):r['parallel_quality_wall_s'] for r in rows(OUT/'pilot.jsonl')}
 for t in chosen:
  r=run_one((t,items[t['instance_id']].data(),meta[t['instance_id']]));append(OUT/'pilot_secondary.jsonl',r);durations[('secondary_large',r['stratum'],r['method'])]=r['parallel_quality_wall_s'];print('secondary pilot',r['method'],round(r['parallel_quality_wall_s'],4),r['evals_used'],flush=True)
 projection=sum(durations[(meta[t['instance_id']]['cohort'],meta[t['instance_id']]['stratum'],t['method'])] for t in tasks)
 write(OUT/'formal_input_manifest.json',{'jobs':manifest,'primary_counts':report['selected_counts'],'primary_missing':report['missing_counts'],'secondary_count':report['secondary_count'],'primary_report_sha256':report['primary_report_sha256'],'secondary_report_sha256':report['secondary_report_sha256']})
 value={'status':'passed','new_secondary_pilot_runs':len(chosen),'formal_runs':len(tasks),'projected_serial_s':projection,'projected_workers2_s':projection/2,'workers':2,'source_freeze_sha256':sha(OUT/'source_freeze.json'),'primary_pilot_preserved':True,'primary_counts':report['selected_counts'],'primary_missing':report['missing_counts'],'secondary_count':report['secondary_count']};write(OUT/'pilot_secondary_receipt.json',value);print(canonical(value),flush=True)

def run():
 assert read(OUT/'pilot_receipt.json')['status']=='passed' and read(OUT/'pilot_secondary_receipt.json')['status']=='passed';assert not (OUT/'quality.jsonl').exists(),'formal attempt exists; no automatic retry'
 items,manifest,report,meta=inputs_and_meta();tasks=make_tasks(items);random.Random(CONFIG['shuffle_seed']).shuffle(tasks);payload=[(t,items[t['instance_id']].data(),meta[t['instance_id']]) for t in tasks]
 tic=time.perf_counter();seen=set()
 with ProcessPoolExecutor(max_workers=CONFIG['workers']) as ex:
  for n,row in enumerate(ex.map(run_one,payload),1):
   assert row['run_id'] not in seen;seen.add(row['run_id']);append(OUT/'quality.jsonl',row)
   if n%20==0 or n==len(tasks):print('quality',n,'/',len(tasks),flush=True)
 receipt={'status':'completed','runs':len(tasks),'jobs':len(items),'selected_counts':report['selected_counts'],'missing_counts':report['missing_counts'],'secondary_count':report['secondary_count'],'parallel_workers':2,'elapsed_quality_parallel_s':time.perf_counter()-tic,'search_evaluations':sum(r['evals_used'] for r in rows(OUT/'quality.jsonl')),'dedicated_runtime_comparison':False,'source_freeze_sha256':sha(OUT/'source_freeze.json')};write(OUT/'run_receipt.json',receipt);print(canonical(receipt),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','tests','pilot','pilot_secondary','run','analyze','verify']);a=p.parse_args()
 if a.stage=='freeze':freeze_sources()
 elif a.stage=='tests':unit_tests()
 elif a.stage in ['pilot','pilot_secondary','run']:globals()[a.stage]()
 else:
  from report import analyze,verify
  globals()[a.stage]()
