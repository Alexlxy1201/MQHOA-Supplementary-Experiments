"""Persistent measured SSH transport and an actual two-endpoint task worker."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pickle
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
import numpy as np
from workflow import Operators,configuration,execute,specification,check_numeric

HOST='amst-4060'
REMOTE_PYTHON='E:\\AMST_runs\\python312_crop_core_20261104\\python.exe'
REMOTE_ROOT='E:\\MQHOA_supp\\optical_20261109_v1'


def read_exact(stream,size):
    chunks=[]
    while size:
        chunk=stream.read(min(size,1024*1024))
        if not chunk:
            raise EOFError('SSH transport ended inside a frame')
        chunks.append(chunk)
        size-=len(chunk)
    return b''.join(chunks)


def send(stream,obj):
    data=pickle.dumps(obj,protocol=5)
    stream.write(struct.pack('!Q',len(data)))
    stream.write(data)
    stream.flush()


def receive(stream):
    size=struct.unpack('!Q',read_exact(stream,8))[0]
    if size>1024**3:
        raise ValueError('oversized trusted-worker frame')
    return pickle.loads(read_exact(stream,size))


def server():
    # SSH endpoint is a dedicated authenticated worker for this experiment.
    if os.name=='nt':
        import msvcrt
        msvcrt.setmode(sys.stdin.fileno(),os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(),os.O_BINARY)
    cfg=configuration()
    engine=None
    values={}
    payload=None
    while True:
        message=receive(sys.stdin.buffer)
        op=message['op']
        if op=='quit':
            send(sys.stdout.buffer,dict(ok=True))
            break
        if op=='upload_payload':
            payload=message['payload']
            response=dict(sha256=hashlib.sha256(payload).hexdigest(),nbytes=len(payload))
        elif op=='download_payload':
            response=dict(payload=payload)
        elif op=='initialize':
            engine=Operators(message['n'],cfg,'cuda')
            values={}
            engine.sync()
            response=dict(ok=True)
        elif op=='put':
            values[message['id']]=engine.array(message['array'])
            engine.sync()
            response=dict(ok=True)
        elif op=='get':
            response=dict(array=engine.numpy(values[message['id']]))
        elif op=='run':
            task=message['task']
            engine.sync()
            start=time.perf_counter()
            value=engine.run(task,[values[i] for i in task['inputs']])
            engine.sync()
            elapsed=time.perf_counter()-start
            values[task['id']]=value
            response=dict(ok=True,compute_seconds=elapsed)
        else:
            raise ValueError(op)
        send(sys.stdout.buffer,response)


class Worker:
    def __init__(self):
        command=f'{REMOTE_PYTHON} -u {REMOTE_ROOT}\\transport.py server'
        self.process=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',HOST,command],
                                      stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        self.lock=threading.RLock()

    def request(self,**message):
        with self.lock:
            send(self.process.stdin,message)
            return receive(self.process.stdout)

    def close(self):
        if self.process.poll() is None:
            self.request(op='quit')
            self.process.stdin.close()
        code=self.process.wait(timeout=30)
        err=self.process.stderr.read().decode(errors='replace')
        if code:
            raise RuntimeError(f'optical worker exit {code}: {err[-2000:]}')


def network(out):
    cfg=configuration()
    out=Path(out)
    raw=out.with_suffix('.jsonl')
    if out.exists() or raw.exists():
        raise RuntimeError('network attempt already exists')
    payloads={}
    for case in cfg['workloads']:
        engine,outputs=execute(case,cfg)
        specs=specification(case)
        consumed={i for t in specs for i in t['inputs']}
        for i in consumed:
            array=np.ascontiguousarray(outputs[i])
            payloads.setdefault(array.nbytes,(case['id'],i,array.tobytes()))
    worker=Worker()
    rows=[]
    try:
        for size,(case_id,task_id,payload) in sorted(payloads.items()):
            expected=hashlib.sha256(payload).hexdigest()
            # Warm the persistent encrypted route, excluded from recorded reps.
            ack=worker.request(op='upload_payload',payload=payload)
            if ack['sha256']!=expected:
                raise RuntimeError('upload content mismatch')
            worker.request(op='download_payload')
            for repetition in range(cfg['network_repetitions']):
                start=time.perf_counter()
                ack=worker.request(op='upload_payload',payload=payload)
                upload=time.perf_counter()-start
                start=time.perf_counter()
                received=worker.request(op='download_payload')['payload']
                download=time.perf_counter()-start
                if ack['sha256']!=expected or hashlib.sha256(received).hexdigest()!=expected:
                    raise RuntimeError('transport content hash mismatch')
                row=dict(nbytes=size,case_id=case_id,task_id=task_id,repetition=repetition,
                         upload_ack_seconds=upload,download_seconds=download,sha256=expected,valid=True)
                rows.append(row)
                with raw.open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
            print('TRANSFER_OK',size,flush=True)
    finally:
        worker.close()
    out.write_text(json.dumps(dict(route='existing SSH amst-4060 / existing SOCKS 127.0.0.1:1082; encrypted application payload',
                                   symmetrization='mean(median(upload_ack_seconds),median(download_seconds))',
                                   includes='serialization, SSH, acknowledgments; no one-way clock claim',rows=rows),indent=2)+'\n')


def end_to_end(case,loc,order,worker=None):
    cfg=configuration()
    local=Operators(case['n'],cfg)
    owns_worker=worker is None
    worker=Worker() if worker is None else worker
    worker.request(op='initialize',n=case['n'])
    tasks=specification(case)
    values={}
    remote_copies=set()
    local_copies=set()
    ready=[threading.Event() for _ in tasks]
    events=[]
    failures=[]
    lock=threading.Lock()
    started=time.perf_counter()

    def run_resource(resource):
        try:
            for index in order:
                if loc[index]!=resource:
                    continue
                task=tasks[index]
                for parent in task['inputs']:
                    if not ready[parent].wait(timeout=240):
                        raise TimeoutError('dependency wait exceeded 240s')
                begin=time.perf_counter()-started
                transfers=[]
                if resource==0:
                    for parent in task['inputs']:
                        if parent not in local_copies:
                            tic=time.perf_counter()
                            values[parent]=worker.request(op='get',id=parent)['array']
                            transfers.append(dict(parent=parent,direction='download',seconds=time.perf_counter()-tic,nbytes=values[parent].nbytes))
                            local_copies.add(parent)
                    tic=time.perf_counter()
                    values[index]=local.run(task,[values[i] for i in task['inputs']])
                    compute=time.perf_counter()-tic
                    local_copies.add(index)
                else:
                    for parent in task['inputs']:
                        if parent not in remote_copies:
                            tic=time.perf_counter()
                            worker.request(op='put',id=parent,array=values[parent])
                            transfers.append(dict(parent=parent,direction='upload',seconds=time.perf_counter()-tic,nbytes=values[parent].nbytes))
                            remote_copies.add(parent)
                    compute=worker.request(op='run',task=task)['compute_seconds']
                    remote_copies.add(index)
                with lock:
                    events.append(dict(task=index,resource=resource,start_seconds=begin,
                                       finish_seconds=time.perf_counter()-started,compute_seconds=compute,transfers=transfers))
                ready[index].set()
        except Exception as error:
            failures.append(repr(error))
            for event in ready:
                event.set()
    threads=[threading.Thread(target=run_resource,args=(r,)) for r in (0,1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=300)
    wall=time.perf_counter()-started
    if any(thread.is_alive() for thread in threads):
        raise TimeoutError('task executor exceeded 300s')
    if owns_worker:
        worker.close()
    if failures:
        raise RuntimeError(failures)
    _,reference=execute(case,cfg)
    validation=check_numeric(values[len(tasks)-1],reference[len(tasks)-1],cfg)
    return dict(wall_seconds=wall,events=events,final_metrics=values[len(tasks)-1].tolist(),numerical_validation=validation,
                timing_boundary='persistent worker ready to final task completion; excludes SSH startup and optimizer time',
                dispatch='two serial workers in returned per-resource order; actual dependency-triggered transfers; remote input cache')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['server','network'])
    parser.add_argument('--out')
    args=parser.parse_args()
    server() if args.stage=='server' else network(args.out)
