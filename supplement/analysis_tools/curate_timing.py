"""Build a new analysis input view; preserve both timing attempts unchanged."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mqhoa_supp.model import canonical, digest
from mqhoa_supp.runner import append_json, read_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_root")
    parser.add_argument("recheck_root")
    parser.add_argument("new_root")
    args=parser.parse_args()
    original,recheck,target=map(Path,(args.original_root,args.recheck_root,args.new_root))
    if target.exists():raise ValueError("new analysis input already exists")
    receipt=json.loads((recheck/"recheck_receipt.json").read_text(encoding="utf-8"))
    if not receipt["stable"] or receipt["completed_runs"]!=90:
        raise ValueError("complete stable recheck required")
    old=read_rows(original/"results/timing-shard0.jsonl")
    new=read_rows(recheck/"timing_archive10_attempt2.jsonl")
    selected={r["run_id"]:r for r in new}
    affected={r["run_id"] for r in old if r["instance_id"]=="archive10"}
    if len(old)!=1260 or len(new)!=90 or len(selected)!=90 or set(selected)!=affected:
        raise ValueError("entire block identity mismatch")
    for row in old:
        if row["run_id"] in selected:
            other=selected[row["run_id"]]
            for key in ["best_l","best_pi","best_makespan","code_sha256","protocol_hash","pool_sha256","frozen_sha256","host"]:
                if row[key]!=other[key]:raise ValueError(f"recheck mismatch: {key}")
    target.mkdir(parents=True)
    shutil.copytree(original/"instances",target/"instances")
    for name in ["protocol.json","frozen_defaults.json","preflight.json","test_report.txt"]:
        if (original/name).exists():shutil.copy2(original/name,target/name)
    (target/"results").mkdir()
    for path in (original/"results").glob("*.jsonl"):
        if path.name!="timing-shard0.jsonl":shutil.copy2(path,target/"results"/path.name)
    combined=[selected.get(r["run_id"],r) for r in old]
    for row in combined:append_json(target/"results/timing-shard0.jsonl",row)
    (target/"curation_manifest.json").write_text(canonical({"reason":"initial archive10 speed transition",
        "policy":receipt["replacement_policy"],"original_timing_rows_sha256":digest(old),
        "attempt2_rows_sha256":digest(new),"selected_rows_sha256":digest(combined),
        "replaced_block_runs":90,"selected_matrix_runs":1260,"quality_changed":False,
        "recheck_receipt":receipt})+"\n",encoding="utf-8")
    print(canonical({"new_root":str(target),"selected_timing_runs":1260,"replaced_complete_block":90}))


if __name__=="__main__":main()
