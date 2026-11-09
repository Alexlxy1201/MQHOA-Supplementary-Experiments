"""Full timing matrix on one fixed CPU of this process, with stability gates.

Does not change power plans, other processes, algorithm settings or raw attempts.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mqhoa_supp.instances import load_instances
from mqhoa_supp.model import canonical, digest, schedule
from mqhoa_supp.protocol import read_config, tasks
from mqhoa_supp.runner import append_json, provenance, run_one


def pin_current_process(index):
    if sys.platform!="win32":raise RuntimeError("this recorded protocol uses Windows process affinity")
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    kernel.GetProcessAffinityMask.argtypes=[wintypes.HANDLE,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    kernel.SetProcessAffinityMask.argtypes=[wintypes.HANDLE,ctypes.c_size_t]
    handle=kernel.GetCurrentProcess()
    current,system=ctypes.c_size_t(),ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(handle,ctypes.byref(current),ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    selected=1<<index
    if not current.value & selected:raise RuntimeError("requested CPU is outside the allowed process mask")
    if not kernel.SetProcessAffinityMask(handle,selected):raise ctypes.WinError(ctypes.get_last_error())
    return {"original_process_affinity_mask":current.value,"system_affinity_mask":system.value,
            "selected_cpu_index":index,"process_affinity_mask":selected,"pid":os.getpid()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root")
    parser.add_argument("output_root")
    parser.add_argument("--cpu-index",type=int,default=2)
    args=parser.parse_args()
    original,out=Path(args.input_root),Path(args.output_root)
    if out.exists():raise ValueError("output exists; preserve the previous attempt")
    affinity=pin_current_process(args.cpu_index)
    source=provenance()
    config=read_config(original/"protocol.json")
    frozen=json.loads((original/"frozen_defaults.json").read_text(encoding="utf-8"))
    if source["code_sha256"]!=frozen["code_sha256"] or digest(config)!=frozen["protocol_hash"]:
        raise ValueError("frozen source/protocol mismatch")
    instances=load_instances(original)
    by_id={i.identifier:i for i in instances}
    jobs=tasks(instances,config,"timing",frozen)
    if len(jobs)!=1260:raise ValueError("this recorded full protocol expects 1260 timing conditions")
    out.mkdir(parents=True)
    start=time.perf_counter()
    warmup_count=warmup_evaluations=0
    while time.perf_counter()-start<60:
        job=dict(jobs[warmup_count%len(jobs)])
        job.update({"phase":"warmup","variant":"warmup","seed":9100000+warmup_count,
                    "budget_evals":1 if job["method"]=="HEFT" else 1000})
        job["run_id"]=digest({"warmup":warmup_count,"source":source["code_sha256"],"host":source["host"]})[:24]
        row=run_one((job,by_id[job["instance_id"]].data(),source))
        append_json(out/"warmup.jsonl",row)
        warmup_count+=1
        warmup_evaluations+=row["evals_used"]
    reference=by_id["archive10"]
    controls=[]
    def sentinel(position):
        samples=[]
        for _ in range(3):
            tick=time.perf_counter()
            for _ in range(10000):schedule(reference,[0]*reference.n,reference.topo,check=False)
            samples.append(time.perf_counter()-tick)
        control={"after_timing_runs":position,"samples_seconds":samples,"median_seconds":statistics.median(samples)}
        controls.append(control)
        append_json(out/"stability_controls.jsonl",control)
        ratio=max(c["median_seconds"] for c in controls)/min(c["median_seconds"] for c in controls)
        return ratio
    ratio=sentinel(0)
    order=np.random.default_rng(20261108).permutation(len(jobs)).tolist()
    completed=0
    stable=True
    for position,index in enumerate(order):
        job=jobs[index]
        row=run_one((job,by_id[job["instance_id"]].data(),source))
        row.update({"parallel_workers":1,"frozen_sha256":digest(frozen),"timing_attempt":"fixed-cpu-full-matrix",
                    "process_affinity_mask":affinity["process_affinity_mask"],"measurement_order":position,
                    "measurement_finished":datetime.now(timezone.utc).isoformat()})
        append_json(out/"results/timing-shard0.jsonl",row)
        completed+=1
        if completed%25==0:print(f"fixed-CPU timing {completed}/{len(jobs)}",flush=True)
        if completed%100==0 or completed==len(jobs):
            ratio=sentinel(completed)
            if ratio>1.5:
                stable=False
                break
    receipt={**source,**affinity,"completed_runs":completed,"planned_runs":len(jobs),
             "warmup_minimum_seconds":60,"warmup_runs":warmup_count,"warmup_evaluations":warmup_evaluations,
             "sentinel_calls_total":len(controls)*30000,"sentinel_ratio":ratio,"stability_limit":1.5,"stable":stable,
             "shuffle_seed":20261108,"controls":controls,"elapsed_seconds":time.perf_counter()-start,
             "tool_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "policy":"use the entire stable same-host matrix; retain earlier host attempts separately",
             "finished":datetime.now(timezone.utc).isoformat()}
    (out/"measurement_receipt.json").write_text(canonical(receipt)+"\n",encoding="utf-8")
    print(canonical(receipt),flush=True)
    if not stable or completed!=len(jobs):raise SystemExit(2)


if __name__=="__main__":main()
