from pathlib import Path
import os,sys,time,json,subprocess,signal
root=Path(__file__).resolve().parents[3];out=Path(os.environ.get('MQHOA_TRACE_OUT',str(root/'supplement/results/20261109_trace_validation_v1'))).resolve();stage=sys.argv[1];limit=int(sys.argv[2]);cwd=Path(__file__).parent
receipt=out/(stage+'_native_receipt.json');assert not receipt.exists(),'attempt exists; no automatic retry'
env=dict(os.environ,PYTHONUTF8='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(out/'matplotlib_cache'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
with (out/(stage+'.stdout.log')).open('wb') as so,(out/(stage+'.stderr.log')).open('wb') as se:
 p=subprocess.Popen([sys.executable,str(cwd/'campaign.py'),stage],cwd=root,env=env,stdout=so,stderr=se,start_new_session=True)
 r={'status':'running','stage':stage,'pid':p.pid,'started_unix':time.time(),'timeout_s':limit,'command':[sys.executable,str(cwd/'campaign.py'),stage],'env':{k:env[k] for k in ['PYTHONUTF8','MPLBACKEND','MPLCONFIGDIR','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PYTHONDONTWRITEBYTECODE']}};receipt.write_text(json.dumps(r,indent=2)+'\n');print('START',stage,p.pid,flush=True)
 try:code=p.wait(timeout=limit)
 except subprocess.TimeoutExpired:
  print('HARD TIMEOUT: stopping only own process group in5s',flush=True);time.sleep(5);os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=30);code=124
 r.update(status='completed' if code==0 else 'failed',exit_code=code,finished_unix=time.time());receipt.write_text(json.dumps(r,indent=2)+'\n');print('EXIT',stage,code,flush=True);sys.exit(code)
