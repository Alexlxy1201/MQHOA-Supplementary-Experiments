"""Analyse a COMPLETE quality matrix while separate timing is still running.

Uses the same predeclared statistical functions as the final package report.
This is a provisional presentation, not a new experiment or an extra test set.
"""
from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np

from mqhoa_supp.instances import load_instances
from mqhoa_supp.model import canonical, digest
from mqhoa_supp.protocol import read_config, tasks
from mqhoa_supp.runner import read_rows
from mqhoa_supp.statistics import bootstrap_mean, holm, label, paired_analysis, write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--config",default=str(Path(__file__).resolve().parents[1]/"configs"/"five_experiments.json"))
    args=parser.parse_args()
    root=Path(args.root)
    config=read_config(args.config)
    frozen=json.loads((root/"frozen_defaults.json").read_text(encoding="utf-8"))
    rows=[r for p in sorted((root/"results").glob("evaluation-shard*.jsonl")) for r in read_rows(p)]
    expected=tasks(load_instances(root),config,"evaluation",frozen)
    if len(rows)!=len(expected) or len({r["run_id"] for r in rows})!=len(rows) or {r["run_id"] for r in rows}!={t["run_id"] for t in expected}:
        raise ValueError("complete unique quality matrix required")
    for row in rows:
        if not row["validator_ok"] or row["evals_used"]!=row["budget_evals"]:
            raise ValueError("invalid schedule or evaluation count")
        if row["frozen_sha256"]!=digest(frozen) or row["code_sha256"]!=frozen["code_sha256"]:
            raise ValueError("quality provenance mismatch")
    out=root/"quality_preview"
    if out.exists():
        raise ValueError("quality preview already exists; preserve it")
    out.mkdir()
    rng=np.random.default_rng(config["statistics"]["seed"])
    grouped=defaultdict(list)
    for row in rows:grouped[row["instance_id"],label(row)].append(row)
    summaries=[]
    lookup={}
    for (identifier,name),items in sorted(grouped.items()):
        values=[r["best_makespan"] for r in items]
        lo,hi=bootstrap_mean(values,rng,config["statistics"]["bootstrap_samples"])
        summary={"instance_id":identifier,"kind":items[0]["instance_kind"],"n":items[0]["n"],"method":name,
                 "runs":len(values),"mean":float(np.mean(values)),"sd":float(np.std(values,ddof=1)) if len(values)>1 else 0,
                 "median":float(np.median(values)),"best_observed":min(values),"ci95_low":lo,"ci95_high":hi}
        summaries.append(summary)
        lookup[identifier,name]=summary
    write_csv(out/"per_instance_quality.csv",summaries)
    ids=sorted({r["instance_id"] for r in rows if r["instance_kind"]=="synthetic"})
    comparisons=[]
    for family,opponents in (("comparison",["PSO","GA","HEFT","RandomPool","DirectRandom"]),
                             ("ablation",["NoHamming","DirectRandom","RandomPool"])):
        family_rows=[]
        for opponent in opponents:
            d=[(lookup[i,"MQHOA"]["mean"]-lookup[i,opponent]["mean"])/lookup[i,opponent]["mean"] for i in ids]
            family_rows.append({"family":family,"reference":"MQHOA","opponent":opponent,
                                "direction":"negative favours MQHOA",**paired_analysis(d,rng,config["statistics"]["bootstrap_samples"])})
        holm(family_rows)
        comparisons.extend(family_rows)
    write_csv(out/"paired_instance_tests.csv",comparisons)
    (out/"provenance.json").write_text(canonical({"status":"quality complete; final timing/report pending",
        "runs":len(rows),"protocol_hash":digest(config),"frozen_sha256":digest(frozen),
        "evaluation_rows_sha256":digest(sorted(rows,key=lambda r:r["run_id"])),
        "tool_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})+"\n",encoding="utf-8")
    print(canonical({"quality_runs":len(rows),"instance_summaries":len(summaries),"output":str(out)}))


if __name__=="__main__":
    main()
