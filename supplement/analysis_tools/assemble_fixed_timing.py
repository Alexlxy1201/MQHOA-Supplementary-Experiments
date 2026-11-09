"""New final analysis view with a complete stable same-host timing matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mqhoa_supp.instances import load_instances
from mqhoa_supp.model import canonical, digest
from mqhoa_supp.protocol import read_config, tasks
from mqhoa_supp.runner import append_json, read_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_root")
    parser.add_argument("measurement_root")
    parser.add_argument("new_root")
    args=parser.parse_args()
    original,measurement,target=map(Path,(args.original_root,args.measurement_root,args.new_root))
    if target.exists():raise ValueError("final analysis view already exists; preserve it")
    receipt=json.loads((measurement/"measurement_receipt.json").read_text(encoding="utf-8"))
    if not receipt["stable"] or receipt["completed_runs"]!=1260 or receipt["sentinel_ratio"]>1.5:
        raise ValueError("complete stable measurement required")
    config=read_config(original/"protocol.json")
    frozen=json.loads((original/"frozen_defaults.json").read_text(encoding="utf-8"))
    rows=read_rows(measurement/"results/timing-shard0.jsonl")
    expected=tasks(load_instances(original),config,"timing",frozen)
    if len(rows)!=1260 or len({r["run_id"] for r in rows})!=1260 or {r["run_id"] for r in rows}!={t["run_id"] for t in expected}:
        raise ValueError("timing matrix mismatch")
    quality=read_rows(original/"results/evaluation-shard0.jsonl")
    lookup={(r["instance_id"],r["method"],r["variant"],r["seed"]):r for r in quality}
    for row in rows:
        key=(row["instance_id"],row["method"],row["variant"],0 if row["method"]=="HEFT" else row["seed"])
        other=lookup[key]
        for field in ["best_l","best_pi","best_makespan","code_sha256","protocol_hash","pool_sha256","frozen_sha256"]:
            if row[field]!=other[field]:raise ValueError(f"quality/timing mismatch: {field}")
        if row["host"]!=receipt["host"] or row["parallel_workers"]!=1 or row["evals_used"]!=row["budget_evals"]:
            raise ValueError("timing host/worker/budget mismatch")
    target.mkdir(parents=True)
    shutil.copytree(original/"instances",target/"instances")
    for name in ["protocol.json","frozen_defaults.json","preflight.json","test_report.txt"]:
        if (original/name).exists():shutil.copy2(original/name,target/name)
    (target/"results").mkdir()
    for path in (original/"results").glob("*.jsonl"):
        if path.name!="timing-shard0.jsonl":shutil.copy2(path,target/"results"/path.name)
    for row in rows:append_json(target/"results/timing-shard0.jsonl",row)
    (target/"timing_selection_manifest.json").write_text(canonical({
        "policy":"use the complete fixed-CPU stable matrix, without row-wise selection",
        "selected_timing_runs":len(rows),"selected_rows_sha256":digest(rows),"quality_changed":False,
        "prior_timing_attempts":"retained separately; initial transition and failed recheck are excluded from final timing summaries",
        "measurement_receipt":receipt})+"\n",encoding="utf-8")
    print(canonical({"new_root":str(target),"selected_timing_runs":len(rows),"stable":True}))


if __name__=="__main__":main()
