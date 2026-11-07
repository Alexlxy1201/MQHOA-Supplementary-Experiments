"""Final presentation only: CSV-derived Chinese summary and raw distributions.

Run AFTER the frozen campaign and package analysis are complete. This script
does not evaluate schedules, tune parameters or modify raw experiment logs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",help="completed campaign output directory")
    parser.add_argument("--public-code-url",default="https://github.com/Alexlxy1201/MQHOA-Supplementary-Experiments")
    args=parser.parse_args()
    root=Path(args.root)
    out=root/"analysis"
    if (out/"final_presentation_manifest.json").exists():
        raise ValueError("final presentation already exists; preserve it before producing another version")
    summaries=read_csv(out/"per_instance_summary.csv")
    tests=read_csv(out/"paired_instance_tests.csv")
    aggregates=read_csv(out/"synthetic_relative_comparison.csv")
    raw=[json.loads(line) for p in sorted((root/"results").glob("evaluation-shard*.jsonl"))
         for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    timing=[json.loads(line) for p in sorted((root/"results").glob("timing-shard*.jsonl"))
            for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(r["validator_ok"] for r in raw+timing):
        raise ValueError("cannot finalise invalid schedules")
    instance_kinds={r["instance_id"]:r["instance_kind"] for r in raw}
    archived_count=sum(kind=="archived" for kind in instance_kinds.values())
    synthetic_count=sum(kind=="synthetic" for kind in instance_kinds.values())
    budgets=sorted({r["budget_evals"] for r in raw if r["method"]!="HEFT"})
    methods=["MQHOA","PSO","GA","HEFT","NoHamming","DirectRandom","RandomPool"]
    palette=["#0072B2","#D55E00","#009E73","#333333","#CC79A7","#E69F00","#56B4E9"]
    def label(row):
        return row["method"] if row["variant"] in {"full","tuned","one-shot","fixed-sigma","direct-level","same-pool"} else f"MQHOA:{row['variant']}"
    plt.rcParams.update({"font.size":10,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    rng=np.random.default_rng(20261108)
    cases=[("archive10","Archived 10-task DAG: seed-level distribution"),
           ("archive50","Archived 50-task DAG: seed-level distribution"),
           ("synthetic","Synthetic DAGs: independent-instance means")]
    cases=[(case,title) for case,title in cases if case=="synthetic" or any(r["instance_id"]==case for r in raw)]
    lookup={(r["instance_id"],r["method"]):float(r["mean"]) for r in summaries}
    synthetic=sorted({r["instance_id"] for r in summaries if r["kind"]=="synthetic"})
    for case,title in cases:
        fig,ax=plt.subplots(figsize=(7.2,4.5))
        values=[]
        for method in methods:
            if case=="synthetic":
                v=[lookup[i,method]/lookup[i,"HEFT"] for i in synthetic]
            else:
                v=[r["best_makespan"] for r in raw if r["instance_id"]==case and label(r)==method]
            values.append(v)
        artists=ax.boxplot(values,tick_labels=methods,showfliers=False,patch_artist=True,
                           medianprops={"color":"black"},widths=.55)
        for i,(box,v,color) in enumerate(zip(artists["boxes"],values,palette),1):
            box.set_facecolor(color)
            box.set_alpha(.25)
            ax.scatter(i+rng.uniform(-.12,.12,len(v)),v,s=12,alpha=.65,color=color,zorder=3)
        ax.set_title(title)
        ax.set_ylabel("Mean makespan / HEFT" if case=="synthetic" else "Makespan (archived time units)")
        ax.tick_params(axis="x",labelrotation=35)
        ax.grid(axis="y",alpha=.2)
        fig.tight_layout()
        for extension in ("pdf","png"):
            fig.savefig(out/"figures"/f"distribution_{case}.{extension}",dpi=180,bbox_inches="tight")
        plt.close(fig)
    lines=["# MQHOA 五项补充实验结果", "",
        "本次在独立目录和分支完成补充工作，原论文、图表、代码压缩包和历史实验记录保留。以下数字均来自新增运行日志。", "",
        "## 完成范围", "",
        f"- 正式比较、参数敏感性与消融：{len(raw):,} 次运行。",
        f"- 单独的同机串行计时：{len(timing):,} 次运行。",
        "- 正式调参：160 次；参数在正式比较前冻结。",
        f"- {archived_count} 个原有 DAG 和 {synthetic_count} 个独立生成的合成 DAG；每个随机算法/配置/实例运行 20 个种子。",
        f"- PSO 与 MQHOA 使用相同索引编码及候选池；随机方法的评价预算为 {budgets}。",
        "- 参数组共 7 个：基准 k=4、m=5、lambda=1.3，以及 k=2/6、m=3/8、lambda=1.2/1.5。",
        "- 消融/对照：固定尺度 NoHamming、直接随机生成 DirectRandom、同候选池随机搜索 RandomPool。", "",
        "## 新增合成 DAG 上的比较", "",
        f"下表以每个实例的 HEFT makespan 为基准，再对 {synthetic_count} 个实例等权汇总；低于 1 表示低于 HEFT。置信区间重采样实例。", "",
        "| 方法 | 平均 makespan 比值 | 95% CI |", "|---|---:|---|"]
    for r in aggregates:
        lines.append(f"| {r['method']} | {float(r['mean_ratio_to_heft']):.4f} | [{float(r['ci95_low']):.4f}, {float(r['ci95_high']):.4f}] |")
    lines += ["", "## 组件与算法差异", "",
        "差异定义为（MQHOA－比较方法）/比较方法；负值有利于 MQHOA。主检验以实例为单位，并在预先定义的比较组内进行 Holm 校正。", "",
        "| 分组 | 对照 | 平均相对差异 | 95% CI | 校正 p |", "|---|---|---:|---|---:|"]
    for r in tests:
        lines.append(f"| {r['family']} | {r['opponent']} | {100*float(r['mean_relative_difference']):.2f}% | "
                     f"[{100*float(r['ci95_low']):.2f}%, {100*float(r['ci95_high']):.2f}%] | {float(r['holm_p']):.6g} |")
    lines += ["", "## 使用这些材料", "",
        "- `per_instance_summary.csv`：每个实例的均值、标准差、中位数和 95% CI，以及同机计时结果。",
        "- `paired_instance_tests.csv`：实例层面的主检验、校正 p 值、效应量及次要 t/Wilcoxon 分析。",
        "- `figures/`：对比、收敛、参数敏感性、参数收敛、消融、运行时间及原始分布的 PDF/PNG。",
        "- 两份 `.tex` 表格片段可作为新增补充表；不覆盖原论文表格。",
        "- `REVIEWER_RESPONSE_DRAFT.md`：根据新增数据生成的英文逐条回复草稿。", "",
        "DirectRandom 同时改变表示和搜索机制，其对比反映整体策略差异。原有两个 DAG 的种子分布单独描述，不充当两个独立外部验证集。",
        "本轮没有运行精确求解器，最好值统一标为观测最好值。参数图用于描述质量、收敛与开销的权衡，不据此重新选择主参数并重跑结果。", "",
        f"公开代码：{args.public_code_url}"]
    (out/"RESULTS_SUMMARY_ZH.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    response=out/"REVIEWER_RESPONSE_DRAFT.md"
    response.write_text(response.read_text(encoding="utf-8").replace("[INSERT VERIFIED PUBLIC CODE URL]",args.public_code_url),encoding="utf-8")
    manifest={"tool_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "evaluation_runs":len(raw),"timing_runs":len(timing),
              "analysis_type":"presentation from completed CSV/JSONL; no new optimisation",
              "inputs":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/"per_instance_summary.csv",out/"paired_instance_tests.csv",out/"synthetic_relative_comparison.csv")}}
    (out/"final_presentation_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"finalised":str(out),"new_distribution_figures":len(cases)},ensure_ascii=False))


if __name__=="__main__":
    main()
