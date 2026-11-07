from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .model import digest


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(config["seeds"]) != len(set(config["seeds"])) or len(config["seeds"]) < 20:
        raise ValueError("at least 20 distinct evaluation seeds required")
    if set(config["seeds"]) & set(config["tuning_seeds"]):
        raise ValueError("tuning and evaluation seeds must be disjoint")
    return config


def pool_parameters(instance, config):
    size = config["pool_large"] if instance.n >= config["large_pool_threshold"] else config["pool_small"]
    seed = config["pool_seed_base"] + int(hashlib.sha256(instance.identifier.encode()).hexdigest()[:8], 16)
    return int(size), int(seed)


def sensitivity(config):
    base = config["mqhoa"]
    variants = [("full", deepcopy(base), ["comparison", "ablation", "sensitivity"])]
    for parameter, values in config["sensitivity"].items():
        for value in values:
            if value != base[parameter]:
                params = deepcopy(base)
                params[parameter] = value
                variants.append((f"{parameter}={value}", params, ["sensitivity"]))
    return variants


def task(instance, method, variant, params, seed, phase, groups, config):
    limit = config["tuning_budget"] if phase == "tuning" else config["budget_evals"]
    if method == "HEFT":
        limit = 1
    size, pool_seed = pool_parameters(instance, config)
    data = {"instance_id": instance.identifier, "instance_sha256": instance.sha256,
            "method": method, "variant": variant, "params": params, "seed": seed,
            "phase": phase, "groups": groups, "budget_evals": int(limit),
            "pool_size": size, "pool_seed": pool_seed, "protocol_hash": digest(config),
            "trace_every": config["trace_every"]}
    data["run_id"] = digest(data)[:24]
    return data


def tasks(instances, config, phase, frozen=None):
    output = []
    for instance in instances:
        if phase == "tuning":
            for method in ("PSO", "GA"):
                for index, params in enumerate(config[f"{method.lower()}_candidates"]):
                    for seed in config["tuning_seeds"]:
                        output.append(task(instance, method, f"candidate-{index}", params, seed, phase, ["tuning"], config))
            continue
        if frozen is None:
            raise ValueError("frozen tuning choices required")
        variants = [("MQHOA", name, params, groups) for name, params, groups in sensitivity(config)]
        variants += [("PSO", "tuned", frozen["PSO"], ["comparison"]),
                     ("GA", "tuned", frozen["GA"], ["comparison"]),
                     ("NoHamming", "fixed-sigma", config["mqhoa"], ["ablation"]),
                     ("DirectRandom", "direct-level", {}, ["ablation", "comparison"]),
                     ("RandomPool", "same-pool", {}, ["comparison", "ablation"])]
        seeds = config["timing_seeds"] if phase == "timing" else config["seeds"]
        for method, name, params, groups in variants:
            for seed in seeds:
                output.append(task(instance, method, name, params, seed, phase, groups, config))
        heft_repeats = range(config["timing_heft_repeats"]) if phase == "timing" else [0]
        for seed in heft_repeats:
            output.append(task(instance, "HEFT", "one-shot", {}, seed, phase, ["comparison"], config))
    return output
