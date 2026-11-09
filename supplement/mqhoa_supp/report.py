"""Evidence-led draft fragments; original manuscript remains untouched."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .model import canonical
from .statistics import bootstrap_mean, write_csv


def write_report(out,rows,timing,summaries,comparisons,config,frozen):
    out=Path(out)
    summary_lines=["# MQHOA: five supplementary experiments", "", "## Material Passport", "",
        "- Mode: run + statistical analysis", "- Status: computed from complete new run logs",
        "- Historical manuscript, figures, code and results: preserved",
        f"- Protocol: {config['protocol']}",f"- Evaluation runs: {len(rows)}",f"- Sequential timing runs: {len(timing)}", "",
        "## Scope", "",
        "These are supplementary results from a separate implementation of the manuscript's stated scheduling model and Algorithm 2. "
        "They are not a rerun or replacement of the historical experiment campaign. All stochastic methods receive the same "
        "fitness-evaluation budget; tuning uses separate instances and seeds. Transfers are uncapacitated, each resource "
        "executes one task at a time, and entry/exit tasks stay local. Synthetic DAGs are generic workloads, not optical measurements.", "",
        "## New comparison results", "",
        "Differences below are mean relative differences across independent synthetic DAGs: "
        "(MQHOA minus comparator) / comparator. Negative values favour MQHOA. "
        "CIs resample instances. Archived DAGs are described separately in per_instance_summary.csv.", "",
        "| Family | Comparator | Mean difference | 95% CI | Holm p |", "|---|---|---:|---|---:|"]
    for r in comparisons:
        summary_lines.append(f"| {r['family']} | {r['opponent']} | {100*r['mean_relative_difference']:.2f}% | "
                             f"[{100*r['ci95_low']:.2f}%, {100*r['ci95_high']:.2f}%] | {r['holm_p']:.6g} |")
    summary_lines += ["", "## Parameter sensitivity", "",
        "Seven one-factor settings were evaluated: baseline (k=4, m=5, lambda=1.3), k=2/6, m=3/8, and lambda=1.2/1.5. "
        "See parameter_sensitivity.pdf for quality and sequential runtime at fixed evaluation budgets. "
        "These evaluation results describe sensitivity; they are not used to retune and rerun the reported primary method.", "",
        "## Ablation interpretation", "",
        "NoHamming retains the indexed search and holds sigma at M/2. DirectRandom draws placements and level-respecting "
        "orders without a pool; it changes both representation and search dynamics. Its contrast therefore describes "
        "whole-search-strategy differences. RandomPool supplies an additional uniform-search diagnostic over the identical pool. "
        "Neither a weak contrast nor an interval crossing zero is evidence that a component is beneficial.", "",
        "## Files and traceability", "",
        "- per_instance_summary.csv: 20-run descriptives and seed-bootstrap CIs (HEFT is one-shot).",
        "- paired_instance_tests.csv: primary two-sided paired sign-flip tests with Holm correction; "
        "paired t and Wilcoxon outputs are predeclared secondary analyses, not alternative routes to significance.",
        "- all_completed_runs.csv and provenance.json: run identifiers and source hashes.",
        "- figures/: PDF and PNG comparison, convergence, ablation, sensitivity and runtime figures.", "",
        "No exact solver was used in this five-task scope. Best values are labelled best observed, never proven optimal."]
    (out/"RESULTS_SUMMARY.md").write_text("\n".join(summary_lines)+"\n",encoding="utf-8")
    # Keep the archival instances as separate rows; never pool historical and new values.
    table=[r for r in summaries if r["kind"]=="archived" and r["method"] in {"MQHOA","PSO","GA","HEFT","NoHamming","DirectRandom","RandomPool"}]
    tex=[r"% NEW supplementary results only; original tables are unchanged.",
         r"\begin{tabular}{llrrrr}",r"\hline",r"Instance & Method & Mean & SD & CI low & CI high \\",r"\hline"]
    for r in table:
        tex.append(f"{r['instance_id']} & {r['method']} & {r['mean']:.4f} & {r['sd']:.4f} & {r['ci95_low']:.4f} & {r['ci95_high']:.4f} " + r"\\")
    tex += [r"\hline",r"\end{tabular}"]
    (out/"supplementary_comparison.tex").write_text("\n".join(tex)+"\n",encoding="utf-8")
    # Aggregate independent synthetic instances after normalising each by its
    # own HEFT result; absolute times from differently sized graphs are not pooled.
    lookup={(r["instance_id"],r["method"]):r for r in summaries}
    ids=sorted({r["instance_id"] for r in summaries if r["kind"]=="synthetic"})
    rng=np.random.default_rng(config["statistics"]["seed"]+1)
    aggregate=[]
    for name in ("MQHOA","PSO","GA","HEFT","NoHamming","DirectRandom","RandomPool"):
        values=[lookup[i,name]["mean"]/lookup[i,"HEFT"]["mean"] for i in ids]
        lo,hi=bootstrap_mean(values,rng,config["statistics"]["bootstrap_samples"])
        aggregate.append({"method":name,"instances":len(ids),"mean_ratio_to_heft":float(np.mean(values)),
                          "ci95_low":lo,"ci95_high":hi})
    write_csv(out/"synthetic_relative_comparison.csv",aggregate)
    tex=[r"% Independent synthetic instances; normalised by each instance's HEFT.",
         r"\begin{tabular}{lrrr}",r"\hline",r"Method & Mean ratio & CI low & CI high \\",r"\hline"]
    for r in aggregate:
        tex.append(f"{r['method']} & {r['mean_ratio_to_heft']:.4f} & {r['ci95_low']:.4f} & {r['ci95_high']:.4f} " + r"\\")
    tex += [r"\hline",r"\end{tabular}"]
    (out/"synthetic_relative_comparison.tex").write_text("\n".join(tex)+"\n",encoding="utf-8")
    def result_text(family,opponent):
        r=next(x for x in comparisons if x["family"]==family and x["opponent"]==opponent)
        return (f"The mean relative makespan difference (MQHOA minus {opponent}) across {r['instances']} independent "
                f"synthetic DAGs was {100*r['mean_relative_difference']:.2f}% "
                f"(95% CI [{100*r['ci95_low']:.2f}%, {100*r['ci95_high']:.2f}%]; "
                f"Holm-adjusted two-sided paired randomisation p={r['holm_p']:.4g}).")
    response=["# Draft responses based on the supplementary runs", "",
        "These paragraphs address only the five supplementary tasks; they are not a complete response to every reviewer comment.", "",
        "## R1 baseline concern; R2 baseline concern", "",
        "We added a global-best PSO comparator using the same two-index representation, candidate pools and scheduling evaluator as MQHOA. "
        "PSO and the supplementary GA were tuned on a separate set before the evaluation configuration was frozen. "
        "All stochastic methods received 10,000 objective evaluations, including initialisation. HEFT was evaluated as a one-shot heuristic. "
        +result_text("comparison","PSO"), "",
        "PSO is an additional classical metaheuristic baseline; adding it does not by itself satisfy the separate request for a recent method.", "",
        "## R1 hyperparameter sensitivity", "",
        "We added one-factor sensitivity experiments for k={2,4,6}, m={3,5,8}, and lambda={1.2,1.3,1.5}, "
        "with 20 independent runs per instance and setting. The quality and runtime panels show the resulting trade-offs under equal evaluation budgets.", "",
        "## R1 Hamming mechanism; R2 encoding ablation", "",
        "We compared the full method with a fixed-scale variant and a direct random-generation control, and added uniform random search over "
        "the same indexed pool. "+result_text("ablation","NoHamming")+" "+result_text("ablation","RandomPool"), "",
        "The direct random-generation contrast changes both encoding and search operators; we therefore interpret it as a comparison of search strategies.", "",
        "## R3 equivalent computational budgets", "",
        "The new supplementary curves use fitness evaluations rather than generations. Every call, including duplicate candidates and "
        "initialisation, counts toward the budget; the final partial generation stops exactly at the shared budget. "
        "One-shot HEFT is shown separately. This describes the new protocol and does not reconstruct the historical figure's generation count.", "",
        "## R4 uncertainty and code availability", "",
        "We added 95% confidence intervals, instance-level paired comparisons with Holm correction, and effect sizes. "
        "Random seeds are nested within instances. The code release contains configuration files and scripts that regenerate the tables and figures. "
        "[INSERT VERIFIED PUBLIC CODE URL].", "",
        "## Remaining non-experimental responses", "",
        "Optical applicability, trace provenance, mathematical formulation, assumptions, theoretical complexity, and recent related work "
        "still require their corresponding manuscript explanations; no new evidence for these topics is implied by this experiment block."]
    (out/"REVIEWER_RESPONSE_DRAFT.md").write_text("\n".join(response)+"\n",encoding="utf-8")
    (out/"frozen_choices.json").write_text(canonical(frozen)+"\n",encoding="utf-8")
