from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from .instances import Pool, load_instances
from .methods import Budget, optimize
from .model import Instance, canonical, digest
from .protocol import tasks


def code_hash():
    root = Path(__file__).resolve().parent.parent
    h = hashlib.sha256()
    files = list((root / "mqhoa_supp").rglob("*.py")) + list((root / "tests").rglob("*.py"))
    for p in sorted(files):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def provenance():
    receipt = Path(__file__).resolve().parents[2] / "SOURCE_COMMIT"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = receipt.read_text(encoding="utf-8").strip() if receipt.exists() else "source-export"
    return {"git_commit": commit, "code_sha256": code_hash(), "host": platform.node(),
            "cpu": platform.processor(), "os": platform.platform(), "python": sys.version.split()[0],
            "numpy": np.__version__, "timestamp": datetime.now(timezone.utc).isoformat()}


def append_json(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(canonical(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"incomplete/corrupt log {path}:{number}; preserve it and recover separately") from e
    return rows


def run_one(payload):
    task, instance_data, source = payload
    instance = Instance(**instance_data)
    started = time.perf_counter()
    pool = None
    if task["method"] not in {"DirectRandom", "HEFT"}:
        pool = Pool(instance, task["pool_size"], task["pool_seed"])
    t_pool = time.perf_counter() - started
    evaluator = Budget(instance, task["budget_evals"], task["trace_every"])
    details = optimize(task["method"], instance, pool, evaluator,
                       np.random.default_rng(task["seed"]), task["params"])
    result = evaluator.result()
    total = time.perf_counter() - started
    return {**task, **source, **result, "details": details,
            "pool_sha256": pool.sha256 if pool else None,
            "instance_kind": instance.metadata["kind"], "n": instance.n,
            "shape": instance.metadata.get("shape"), "t_pool": t_pool,
            "t_search": max(0.0, total - t_pool - result["t_eval"]), "t_total": total,
            "status": "completed", "historical": False}


def require_preflight(root, config):
    path = Path(root) / "preflight.json"
    if not path.exists():
        raise ValueError("run preflight first")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if not gate.get("tests_passed") or gate["code_sha256"] != code_hash() or gate["protocol_hash"] != digest(config):
        raise ValueError("preflight does not match current code and protocol; rerun it")
    return gate


def execute(root, config, phase, workers=1, frozen=None, shard_index=0, shard_count=1):
    require_preflight(root, config)
    if phase == "timing" and workers != 1:
        raise ValueError("timing phase must be sequential on an otherwise idle machine")
    if not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard")
    source = provenance()
    if frozen is not None:
        if frozen["protocol_hash"] != digest(config) or frozen["code_sha256"] != source["code_sha256"]:
            raise ValueError("frozen choices do not match current protocol/code")
    instances = load_instances(root, tuning=phase == "tuning")
    by_id = {i.identifier: i for i in instances}
    expected = tasks(instances, config, phase, frozen)
    selected = [t for i, t in enumerate(expected) if i % shard_count == shard_index]
    log = Path(root) / "results" / f"{phase}-shard{shard_index}.jsonl"
    previous = read_rows(log)
    if len({r["run_id"] for r in previous}) != len(previous):
        raise ValueError("duplicate completed run IDs in existing log")
    for row in previous:
        if row["code_sha256"] != source["code_sha256"] or row["protocol_hash"] != digest(config):
            raise ValueError("existing results from different code/config; use a new output root")
    done = {r["run_id"] for r in previous}
    pending = [t for t in selected if t["run_id"] not in done]
    print(f"{phase}: {len(done)}/{len(selected)} already complete; {len(pending)} pending", flush=True)
    payloads = [(t, by_id[t["instance_id"]].data(), source) for t in pending]
    def consume(iterator):
        for row in iterator:
            row["parallel_workers"] = workers
            row["frozen_sha256"] = digest(frozen) if frozen else None
            append_json(log, row)
            done.add(row["run_id"])
            if len(done) % 10 == 0 or len(done) == len(selected):
                print(f"{phase}: {len(done)}/{len(selected)} completed", flush=True)
    # Workers return results; the parent is the sole append-only log writer.
    try:
        if workers == 1:
            consume(map(run_one, payloads))
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                consume(executor.map(run_one, payloads, chunksize=1))
    except BaseException as error:
        append_json(Path(root) / "failures.jsonl", {**source, "phase": phase,
                    "error": repr(error), "completed": len(done), "planned": len(selected)})
        raise
    return {"expected": len(selected), "completed": len(done), "log": str(log)}


def freeze(root, config):
    source = provenance()
    rows = [r for p in sorted((Path(root) / "results").glob("tuning-shard*.jsonl")) for r in read_rows(p)]
    expected = tasks(load_instances(root, tuning=True), config, "tuning")
    lookup = {r["run_id"]: r for r in rows}
    if len(lookup) != len(rows) or set(lookup) != {t["run_id"] for t in expected}:
        raise ValueError("complete, unique tuning results required before freezing")
    if any(r["code_sha256"] != source["code_sha256"] or not r["validator_ok"] for r in rows):
        raise ValueError("tuning provenance/validation mismatch")
    # Average per-instance performance normalised by the best candidate mean
    # within that method/instance. No evaluation-instance results are consulted.
    selected, scores = {}, {}
    for method in ("PSO", "GA"):
        subset = [r for r in rows if r["method"] == method]
        instance_ids = sorted({r["instance_id"] for r in subset})
        variants = sorted({r["variant"] for r in subset})
        means = {(i, v): float(np.mean([r["best_makespan"] for r in subset if r["instance_id"] == i and r["variant"] == v]))
                 for i in instance_ids for v in variants}
        score = {v: float(np.mean([means[i, v] / max(1e-12, min(means[i, q] for q in variants)) for i in instance_ids])) for v in variants}
        winner = min(variants, key=lambda v: (score[v], v))
        selected[method] = next(r["params"] for r in subset if r["variant"] == winner)
        scores[method] = {"winner": winner, "normalised_scores": score}
    result = {**source, **selected, "protocol_hash": digest(config), "scores": scores,
              "tuning_results_sha256": digest(sorted(rows, key=lambda r: r["run_id"])),
              "evaluation_results_used": False}
    path = Path(root) / "frozen_defaults.json"
    if path.exists():
        raise ValueError("frozen defaults already exist; refusing to overwrite")
    path.write_text(canonical(result) + "\n", encoding="utf-8")
    return result


def preflight(root, config, repo):
    """Remote-only tests and short timing pilots; not a scientific result block."""
    import unittest
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent.parent / "tests"))
    started = time.perf_counter()
    Path(root).mkdir(parents=True, exist_ok=True)
    output = Path(root) / "test_report.txt"
    with output.open("w", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError(f"preflight failed; see {output}; no campaign started")
    source = provenance()
    from .protocol import task
    instances = load_instances(root)
    pilots = []
    for size in sorted({i.n for i in instances}):
        instance = next(i for i in instances if i.n == size)
        for method, params in (("MQHOA", config["mqhoa"]), ("PSO", config["pso_candidates"][0]),
                               ("GA", config["ga_candidates"][0]), ("NoHamming", config["mqhoa"]),
                               ("RandomPool", {}), ("DirectRandom", {}), ("HEFT", {})):
            t = task(instance, method, "pilot", params, 991991, "pilot", ["pilot"], config)
            t["budget_evals"] = 500 if method != "HEFT" else 1
            r = run_one((t, instance.data(), source))
            pilots.append({"n": size, "method": method, "t_total": r["t_total"],
                           "t_pool": r["t_pool"], "t_eval": r["t_eval"], "evals": r["evals_used"]})
    proxy = {"PSO": config["pso_candidates"][0], "GA": config["ga_candidates"][0]}
    estimate = 0.0
    for phase in ("tuning", "evaluation", "timing"):
        for t in tasks(load_instances(root, tuning=phase == "tuning"), config, phase, proxy):
            n = next(i.n for i in load_instances(root, tuning=phase == "tuning") if i.identifier == t["instance_id"])
            closest = min((p for p in pilots if p["method"] == t["method"]), key=lambda p: abs(p["n"] - n))
            estimate += closest["t_pool"] + (closest["t_total"] - closest["t_pool"]) * t["budget_evals"] / closest["evals"]
    gate = {**source, "protocol_hash": digest(config), "tests_passed": True,
            "tests_run": result.testsRun, "pilots": pilots, "estimated_serial_seconds": estimate,
            "preflight_seconds": time.perf_counter() - started, "repo_path": str(Path(repo).resolve())}
    path = Path(root) / "preflight.json"
    if path.exists():
        backup = Path(root) / f"preflight-{time.time_ns()}.json"
        path.rename(backup)
    path.write_text(canonical(gate) + "\n", encoding="utf-8")
    return gate
