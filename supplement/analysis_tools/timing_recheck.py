"""Warm, interleaved remeasurement of the complete archive10 timing block.

The first timing attempt is preserved. No optimisation parameters are changed.
The replacement policy applies to all 90 conditions, never selected fast rows.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

from mqhoa_supp.instances import Instance
from mqhoa_supp.model import canonical, digest, schedule
from mqhoa_supp.protocol import read_config, tasks
from mqhoa_supp.runner import append_json, provenance, run_one


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("output")
    args=parser.parse_args()
    root,out=Path(args.root),Path(args.output)
    if out.exists():raise ValueError("recheck output already exists; preserve it")
    config=read_config(root/"protocol.json")
    frozen=json.loads((root/"frozen_defaults.json").read_text(encoding="utf-8"))
    source=provenance()
    if source["code_sha256"]!=frozen["code_sha256"] or digest(config)!=frozen["protocol_hash"]:
        raise ValueError("frozen code/protocol mismatch")
    instance=Instance.load(root/"instances"/"archive10.json")
    jobs=tasks([instance],config,"timing",frozen)
    if len(jobs)!=90:raise ValueError("expected the entire 90-run timing block")
    out.mkdir(parents=True)
    started=time.perf_counter()
    warmup_count=0
    while time.perf_counter()-started<10.0:
        job=dict(jobs[0])
        job.update({"phase":"warmup","variant":"warmup","seed":9000000+warmup_count,"budget_evals":1000})
        job["run_id"]=digest({"warmup":warmup_count,"source":source["code_sha256"]})[:24]
        row=run_one((job,instance.data(),source))
        append_json(out/"warmup.jsonl",row)
        warmup_count+=1
    def sentinel():
        tic=time.perf_counter()
        loc=[0]*instance.n
        for _ in range(10000):schedule(instance,loc,instance.topo,check=False)
        return time.perf_counter()-tic
    before=sentinel()
    order=np.random.default_rng(20261108).permutation(len(jobs)).tolist()
    for position,index in enumerate(order):
        row=run_one((jobs[index],instance.data(),source))
        row.update({"parallel_workers":1,"frozen_sha256":digest(frozen),"timing_attempt":2,
                    "measurement_order":position,"warmup_seconds_minimum":10.0,
                    "timing_recheck_reason":"complete-block remeasurement after an initial speed transition"})
        append_json(out/"timing_archive10_attempt2.jsonl",row)
        if (position+1)%10==0:print(f"archive10 recheck {position+1}/90",flush=True)
    after=sentinel()
    ratio=max(before,after)/min(before,after)
    report={**source,"completed_runs":len(jobs),"warmup_runs":warmup_count,"warmup_evaluations":1000*warmup_count,
            "sentinel_calls_each":10000,"sentinel_before_seconds":before,"sentinel_after_seconds":after,
            "sentinel_ratio":ratio,"predeclared_stability_limit":1.5,"stable":ratio<=1.5,
            "shuffle_seed":20261108,"replacement_policy":"replace the complete archive10 block, not individual rows",
            "original_timing_log_sha256":hashlib.sha256((root/"results/timing-shard0.jsonl").read_bytes()).hexdigest(),
            "tool_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "finished":datetime.now(timezone.utc).isoformat()}
    (out/"recheck_receipt.json").write_text(canonical(report)+"\n",encoding="utf-8")
    print(canonical(report),flush=True)
    if not report["stable"]:raise SystemExit(2)


if __name__=="__main__":main()
