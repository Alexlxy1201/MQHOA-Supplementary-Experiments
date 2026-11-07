"""Predeclared analyses; independent DAGs, not seeds, are inferential units."""
from __future__ import annotations

from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

from .instances import load_instances
from .model import canonical, digest
from .protocol import tasks
from .runner import read_rows


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean(values, rng, samples=10000):
    x=np.asarray(values,dtype=float)
    if len(x)==0 or not np.all(np.isfinite(x)):
        raise ValueError("finite nonempty bootstrap sample required")
    if len(x)==1:
        return float(x[0]),float(x[0])
    means=np.mean(x[rng.integers(0,len(x),(samples,len(x)))],axis=1)
    return tuple(float(v) for v in np.quantile(means,[0.025,0.975]))


def paired_analysis(differences,rng,samples=10000):
    """Primary sign-flip randomisation; t/Wilcoxon are secondary diagnostics."""
    d=np.asarray(differences,dtype=float)
    if len(d)<2:
        raise ValueError("at least two independent instances required")
    observed=abs(float(d.mean()))
    if len(d)<=16:
        signs=np.asarray(list(itertools.product((-1,1),repeat=len(d))))
        null=np.abs(np.mean(signs*d,axis=1))
        p=float(np.mean(null>=observed-1e-12))
        permutation="exact sign flips"
    else:
        signs=rng.choice([-1,1],size=(100000,len(d)))
        null=np.abs(np.mean(signs*d,axis=1))
        p=float((np.sum(null>=observed-1e-12)+1)/(len(null)+1))
        permutation="100000 sign flips, plus-one p"
    lo,hi=bootstrap_mean(d,rng,samples)
    sd=float(np.std(d,ddof=1))
    if sd<1e-12:
        tstat,tp=(0.0,1.0) if observed<1e-12 else (None,0.0)
        dz=None
    else:
        result=stats.ttest_1samp(d,0)
        tstat,tp=float(result.statistic),float(result.pvalue)
        dz=float(d.mean()/sd)
    rounded=np.round(d,12)
    nonzero=rounded[rounded!=0]
    if not len(nonzero):
        wp,rank_effect=1.0,0.0
    else:
        wp=float(stats.wilcoxon(rounded,zero_method="pratt",method="auto").pvalue)
        ranks=stats.rankdata(abs(nonzero))
        rank_effect=float((ranks[nonzero>0].sum()-ranks[nonzero<0].sum())/ranks.sum())
    normality=float(stats.shapiro(d).pvalue) if len(d)>=3 and sd>1e-12 else None
    return {"instances":len(d),"mean_relative_difference":float(d.mean()),"ci95_low":lo,"ci95_high":hi,
            "primary_p":p,"primary_test":permutation,"paired_t":tstat,"paired_t_p_secondary":tp,
            "wilcoxon_p_secondary":wp,"cohen_dz":dz,"rank_biserial":rank_effect,
            "shapiro_difference_p_diagnostic":normality}


def holm(rows):
    ordered=sorted(range(len(rows)),key=lambda i:rows[i]["primary_p"])
    maximum=0.0
    for rank,index in enumerate(ordered):
        maximum=max(maximum,min(1.0,(len(rows)-rank)*rows[index]["primary_p"]))
        rows[index]["holm_p"]=maximum


def label(row):
    return row["method"] if row["variant"] in {"full","tuned","one-shot","fixed-sigma","direct-level","same-pool"} else f"MQHOA:{row['variant']}"


def analyze(root,config):
    root=Path(root)
    frozen=json.loads((root/"frozen_defaults.json").read_text(encoding="utf-8"))
    rows=[r for p in sorted((root/"results").glob("evaluation-shard*.jsonl")) for r in read_rows(p)]
    expected=tasks(load_instances(root),config,"evaluation",frozen)
    ids=[r["run_id"] for r in rows]
    if len(set(ids))!=len(ids) or set(ids)!={t["run_id"] for t in expected}:
        raise ValueError("complete unique evaluation matrix required; missing/extra runs cannot be silently dropped")
    if any(not r["validator_ok"] or r["protocol_hash"]!=digest(config) or r["frozen_sha256"]!=digest(frozen)
           or r["code_sha256"]!=frozen["code_sha256"] for r in rows):
        raise ValueError("inconsistent evaluation provenance")
    instances={i.identifier:i for i in load_instances(root)}
    timing=[r for p in sorted((root/"results").glob("timing-shard*.jsonl")) for r in read_rows(p)]
    expected_timing=tasks(list(instances.values()),config,"timing",frozen)
    if {r["run_id"] for r in timing}!={t["run_id"] for t in expected_timing} or len(timing)!=len(expected_timing):
        raise ValueError("complete unique sequential timing matrix required")
    if any(r["parallel_workers"]!=1 or r["frozen_sha256"]!=digest(frozen)
           or r["code_sha256"]!=frozen["code_sha256"] or r["protocol_hash"]!=digest(config)
           or not r["validator_ok"] for r in timing):
        raise ValueError("timing must be sequential and use the shared frozen defaults")
    # Runtime cannot be pooled across different machines.
    if len({r["host"] for r in timing})!=1:
        raise ValueError("all timing comparisons must be from the same host")
    out=root/"analysis"
    if out.exists():
        raise ValueError("analysis already exists; preserve it and use a separate output version")
    out.mkdir()
    rng=np.random.default_rng(config["statistics"]["seed"])
    groups=defaultdict(list)
    for row in rows:
        groups[row["instance_id"],label(row)].append(row)
    summaries=[]
    lookup={}
    for (identifier,name),items in sorted(groups.items()):
        values=[r["best_makespan"] for r in items]
        ci=bootstrap_mean(values,rng,config["statistics"]["bootstrap_samples"])
        matching=[r for r in timing if r["instance_id"]==identifier and label(r)==name]
        summary={"instance_id":identifier,"kind":instances[identifier].metadata["kind"],"n":instances[identifier].n,
                 "method":name,"runs":len(values),"mean":float(np.mean(values)),"sd":float(np.std(values,ddof=1)) if len(values)>1 else 0.0,
                 "median":float(np.median(values)),"best_observed":float(min(values)),"ci95_low":ci[0],"ci95_high":ci[1],
                 "mean_runtime_s":float(np.mean([r["t_total"] for r in matching])),
                 "mean_pool_s":float(np.mean([r["t_pool"] for r in matching])),
                 "mean_evaluation_s":float(np.mean([r["t_eval"] for r in matching]))}
        summaries.append(summary)
        lookup[identifier,name]=summary
    write_csv(out/"per_instance_summary.csv",summaries)
    main=["PSO","GA","HEFT","RandomPool","DirectRandom"]
    ablations=["NoHamming","DirectRandom","RandomPool"]
    synthetic_ids=sorted(i for i,x in instances.items() if x.metadata["kind"]=="synthetic")
    comparisons=[]
    for family,opponents in (("comparison",main),("ablation",ablations)):
        family_rows=[]
        for opponent in opponents:
            differences=[(lookup[i,"MQHOA"]["mean"]-lookup[i,opponent]["mean"])/lookup[i,opponent]["mean"] for i in synthetic_ids]
            family_rows.append({"family":family,"reference":"MQHOA","opponent":opponent,
                                "direction":"negative favours MQHOA",**paired_analysis(differences,rng,config["statistics"]["bootstrap_samples"])})
        holm(family_rows)
        comparisons.extend(family_rows)
    write_csv(out/"paired_instance_tests.csv",comparisons)
    write_csv(out/"all_completed_runs.csv",[{k:r[k] for k in ("run_id","phase","instance_id","method","variant","seed","best_makespan","evals_used","t_total","t_pool","t_eval","host","code_sha256")} for r in rows+timing])
    from .plots import make_plots
    make_plots(out,rows,timing,lookup,instances,config)
    from .report import write_report
    write_report(out,rows,timing,summaries,comparisons,config,frozen)
    (out/"provenance.json").write_text(canonical({"protocol_hash":digest(config),"frozen_sha256":digest(frozen),
                "evaluation_rows_sha256":digest(sorted(rows,key=lambda r:r["run_id"])),
                "timing_rows_sha256":digest(sorted(timing,key=lambda r:r["run_id"])),
                "historical_results_modified":False})+"\n",encoding="utf-8")
    return {"evaluation_runs":len(rows),"timing_runs":len(timing),"analysis":str(out)}
