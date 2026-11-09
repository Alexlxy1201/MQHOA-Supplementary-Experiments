"""Standalone publication figures, regenerated only from completed run logs."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .statistics import bootstrap_mean, label


COLORS = {"MQHOA":"#0072B2","PSO":"#D55E00","GA":"#009E73","HEFT":"#333333",
          "NoHamming":"#CC79A7","DirectRandom":"#E69F00","RandomPool":"#56B4E9"}


def save(fig, folder, name):
    fig.savefig(Path(folder)/f"{name}.pdf",bbox_inches="tight")
    fig.savefig(Path(folder)/f"{name}.png",dpi=180,bbox_inches="tight")
    plt.close(fig)


def curve_at(row, grid):
    trace=np.asarray(row["trace"],dtype=float)
    index=np.searchsorted(trace[:,0],grid,side="right")-1
    return trace[np.maximum(index,0),2]


def make_plots(out,rows,timing,lookup,instances,config):
    folder=Path(out)/"figures"
    folder.mkdir()
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42,"figure.dpi":120})
    rng=np.random.default_rng(config["statistics"]["seed"])
    synthetic=sorted(i for i,x in instances.items() if x.metadata["kind"]=="synthetic")
    panels=[("archive10",["archive10"]),("archive50",["archive50"]),("Synthetic DAGs",synthetic)]
    panels=[(title,ids) for title,ids in panels if all(i in instances for i in ids)]
    methods=["MQHOA","PSO","GA","HEFT","RandomPool","DirectRandom"]
    fig,axes=plt.subplots(1,len(panels),figsize=(5*len(panels),4),squeeze=False)
    for ax,(title,ids) in zip(axes[0],panels):
        for x,name in enumerate(methods):
            values=np.array([lookup[i,name]["mean"]/lookup[i,"HEFT"]["mean"] for i in ids])
            if len(ids)==1:
                s=lookup[ids[0],name]
                mean=values[0]
                lo,hi=s["ci95_low"]/lookup[ids[0],"HEFT"]["mean"],s["ci95_high"]/lookup[ids[0],"HEFT"]["mean"]
            else:
                mean=float(values.mean())
                lo,hi=bootstrap_mean(values,rng)
            # Percentile intervals can lie to one side of the observed mean.
            ax.vlines(x,lo,hi,color=COLORS[name],linewidth=1.5)
            ax.plot(x,mean,"o",color=COLORS[name],markersize=6)
        ax.set_xticks(range(len(methods)),methods,rotation=35,ha="right")
        ax.axhline(1,color="#888888",linestyle="--",linewidth=.8)
        ax.set_title(title)
        ax.set_ylabel("Mean makespan / HEFT makespan")
        ax.grid(axis="y",alpha=.2)
    fig.tight_layout()
    save(fig,folder,"comparison")

    for filename,names in (("convergence",["MQHOA","PSO","GA","RandomPool"]),
                           ("ablation_convergence",["MQHOA","NoHamming","DirectRandom","RandomPool"])):
        fig,axes=plt.subplots(1,len(panels),figsize=(5*len(panels),4),squeeze=False)
        grid=np.unique(np.r_[1,np.arange(config["trace_every"],config["budget_evals"]+1,config["trace_every"]),config["budget_evals"]])
        for ax,(title,ids) in zip(axes[0],panels):
            for name in names:
                by_instance=[]
                for i in ids:
                    selected=[r for r in rows if r["instance_id"]==i and label(r)==name]
                    matrix=np.asarray([curve_at(r,grid) for r in selected])/lookup[i,"HEFT"]["mean"]
                    by_instance.append(matrix.mean(axis=0))
                matrix=np.asarray(by_instance)
                mean=matrix.mean(axis=0)
                if len(ids)>1:
                    sampled=matrix[rng.integers(0,len(ids),(2000,len(ids)))].mean(axis=1)
                else:
                    selected=[r for r in rows if r["instance_id"]==ids[0] and label(r)==name]
                    seed_curves=np.asarray([curve_at(r,grid) for r in selected])/lookup[ids[0],"HEFT"]["mean"]
                    sampled=seed_curves[rng.integers(0,len(seed_curves),(2000,len(seed_curves)))].mean(axis=1)
                lo,hi=np.quantile(sampled,[.025,.975],axis=0)
                ax.plot(grid,mean,label=name,color=COLORS[name])
                ax.fill_between(grid,lo,hi,color=COLORS[name],alpha=.12)
            ax.set_title(title)
            ax.set_xlabel("Fitness evaluations (initialisation included)")
            ax.set_ylabel("Best-so-far makespan / HEFT")
            ax.grid(alpha=.2)
        axes[0,-1].legend(fontsize=8)
        fig.tight_layout()
        save(fig,folder,filename)

    fig,axes=plt.subplots(3,2,figsize=(11,11))
    for row,(parameter,values) in enumerate(config["sensitivity"].items()):
        for size in sorted({instances[i].n for i in synthetic}):
            ids=[i for i in synthetic if instances[i].n==size]
            quality,time_ratio=[],[]
            for value in values:
                name="MQHOA" if value==config["mqhoa"][parameter] else f"MQHOA:{parameter}={value}"
                quality.append(float(np.mean([lookup[i,name]["mean"]/lookup[i,"MQHOA"]["mean"] for i in ids])))
                time_ratio.append(float(np.mean([lookup[i,name]["mean_runtime_s"]/lookup[i,"MQHOA"]["mean_runtime_s"] for i in ids])))
            axes[row,0].plot(values,quality,"o-",label=f"n={size}")
            axes[row,1].plot(values,time_ratio,"o-",label=f"n={size}")
        for col in range(2):
            axes[row,col].set_xlabel(parameter)
            axes[row,col].set_xticks(values)
            axes[row,col].axhline(1,color="#888888",linewidth=.7,linestyle="--")
            axes[row,col].grid(alpha=.2)
            axes[row,col].legend()
        axes[row,0].set_ylabel("Mean makespan / baseline setting")
        axes[row,1].set_ylabel("Sequential runtime / baseline setting")
    fig.tight_layout()
    save(fig,folder,"parameter_sensitivity")

    fig,axes=plt.subplots(1,3,figsize=(15,4))
    grid=np.unique(np.r_[1,np.arange(config["trace_every"],config["budget_evals"]+1,config["trace_every"]),config["budget_evals"]])
    for ax,(parameter,values) in zip(axes,config["sensitivity"].items()):
        for value in values:
            name="MQHOA" if value==config["mqhoa"][parameter] else f"MQHOA:{parameter}={value}"
            instance_curves=[]
            for i in synthetic:
                selected=[r for r in rows if r["instance_id"]==i and label(r)==name]
                matrix=np.asarray([curve_at(r,grid) for r in selected])/lookup[i,"HEFT"]["mean"]
                instance_curves.append(matrix.mean(axis=0))
            ax.plot(grid,np.mean(instance_curves,axis=0),label=f"{parameter}={value}")
        ax.set_title(f"Sensitivity: {parameter}")
        ax.set_xlabel("Fitness evaluations")
        ax.set_ylabel("Mean best-so-far makespan / HEFT")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig,folder,"parameter_convergence")

    fig,axes=plt.subplots(1,2,figsize=(12,4))
    timing_groups=defaultdict(list)
    for r in timing:
        if label(r) in methods and instances[r["instance_id"]].metadata["kind"]=="synthetic":
            timing_groups[label(r),r["n"]].append(r)
    for name in methods:
        sizes=sorted(n for m,n in timing_groups if m==name)
        times=[np.median([r["t_total"] for r in timing_groups[name,n]]) for n in sizes]
        axes[0].plot(sizes,times,"o-",color=COLORS[name],label=name)
    axes[0].set_xlabel("Tasks")
    axes[0].set_ylabel("Median sequential end-to-end time (s)")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    base=np.zeros(len(methods))
    for component,name in (("t_pool","Pool generation"),("t_eval","Fitness evaluation"),("t_search","Search and validation")):
        values=[np.mean([r[component] for r in timing if label(r)==m and r["n"]==100]) for m in methods]
        axes[1].bar(range(len(methods)),values,bottom=base,label=name)
        base+=values
    axes[1].set_xticks(range(len(methods)),methods,rotation=35,ha="right")
    axes[1].set_ylabel("Mean runtime components at n=100 (s)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig,folder,"runtime")
