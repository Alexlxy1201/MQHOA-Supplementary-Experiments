"""Explicit stages: preparation/tests/estimate precede any formal run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .instances import load_instances, prepare
from .model import canonical, digest
from .protocol import read_config, tasks
from .runner import execute, freeze, preflight, read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare","plan","preflight","tune","freeze","run","timing","status","analyze"])
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "five_experiments.json"))
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--out", default="runs/five-v1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--frozen", help="shared frozen_defaults.json from the tuning host")
    args = parser.parse_args()
    config = read_config(args.config)
    root = Path(args.out).resolve()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.command == "prepare":
        rows = prepare(args.repo, config, root)
        protocol_path = root / "protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != config:
            raise ValueError("existing output folder has another protocol")
        protocol_path.write_text(canonical(config) + "\n", encoding="utf-8")
        print(canonical({"prepared_instances":len(rows),"protocol_hash":digest(config),"out":str(root)}))
        return
    if not (root / "protocol.json").exists():
        raise ValueError("run prepare first")
    if json.loads((root / "protocol.json").read_text(encoding="utf-8")) != config:
        raise ValueError("configuration differs from prepared protocol")
    if args.command == "plan":
        proxy = {"PSO":config["pso_candidates"][0],"GA":config["ga_candidates"][0]}
        summary = {}
        for phase in ("tuning","evaluation","timing"):
            jobs = tasks(load_instances(root,tuning=phase=="tuning"),config,phase,proxy)
            summary[phase] = {"runs":len(jobs),"evaluations":sum(t["budget_evals"] for t in jobs)}
        print(json.dumps(summary,indent=2))
    elif args.command == "preflight":
        report = preflight(root,config,args.repo)
        print(canonical({"tests_passed":report["tests_passed"],"tests_run":report["tests_run"],
                         "estimated_serial_hours":report["estimated_serial_seconds"]/3600,
                         "report":str(root/"preflight.json")}))
    elif args.command == "tune":
        print(canonical(execute(root,config,"tuning",args.workers)))
    elif args.command == "freeze":
        report=freeze(root,config)
        print(canonical({"PSO":report["PSO"],"GA":report["GA"],"path":str(root/"frozen_defaults.json")}))
    elif args.command in {"run","timing"}:
        frozen_path=Path(args.frozen) if args.frozen else root/"frozen_defaults.json"
        frozen=json.loads(frozen_path.read_text(encoding="utf-8"))
        phase="timing" if args.command=="timing" else "evaluation"
        print(canonical(execute(root,config,phase,args.workers,frozen,args.shard_index,args.shard_count)))
    elif args.command=="status":
        rows=[r for p in (root/"results").glob("*.jsonl") for r in read_rows(p)]
        print(canonical({"completed":len(rows),"phases":{phase:sum(r["phase"]==phase for r in rows)
                                                       for phase in ("tuning","evaluation","timing")}}))
    elif args.command=="analyze":
        from .statistics import analyze
        print(canonical(analyze(root,config)))


if __name__=="__main__":
    main()
