"""Read literal archive data without importing or running archived scripts."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .model import Instance, canonical


def archived(repo):
    archive = Path(repo) / "代码 数据等.zip"
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    output = []
    with ZipFile(archive) as z:
        for suffix, identifier in (("/MQHOA/test2.py", "archive10"), ("/MQHOA/DAG50.py", "archive50")):
            matches = [n for n in z.namelist() if n.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"expected exactly one archive member ending in {suffix}")
            values = {}
            for node in ast.parse(z.read(matches[0]).decode("utf-8")).body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in {"edges", "tl", "ts"}:
                            values[target.id] = ast.literal_eval(node.value)
            n = len(values["tl"])
            output.append(Instance(identifier, values["tl"], values["ts"], values["edges"], [0, n - 1],
                                   {"kind": "archived", "source_archive_sha256": checksum,
                                    "source_member_suffix": suffix, "historical_results_reused": False,
                                    "origin": "archived constants; no trace-provenance claim"}))
    return output


def synthetic(n, shape, seed, identifier, split, ccr=1.0):
    rng = np.random.default_rng(seed)
    inner = list(range(1, n - 1))
    width = max(1, int(round((n - 2) ** (0.35 if shape == "deep" else 0.75))))
    layers = [[0]] + [inner[i:i + width] for i in range(0, len(inner), width)] + [[n - 1]]
    edges = set()
    for left, right in zip(layers, layers[1:]):
        # Every task has a path from the entry and to the exit.
        for b in right:
            edges.add((int(rng.choice(left)), b))
        for a in left:
            edges.add((a, int(rng.choice(right))))
        for a in left:
            for b in right:
                if rng.random() < 0.35:
                    edges.add((a, b))
    tl = np.round(rng.uniform(3, 12, n), 2)
    ratio = rng.uniform(2, 5, n)
    ts = np.round(tl / ratio, 2)
    costs = rng.uniform(0.5, 1.5, len(edges))
    costs *= ccr * float(np.mean((tl + ts) / 2)) / float(costs.mean())
    weighted = [(a, b, float(round(c, 2))) for (a, b), c in zip(sorted(edges), costs)]
    return Instance(identifier, tl.tolist(), ts.tolist(), weighted, [0, n - 1],
                    {"kind": "synthetic", "split": split, "seed": seed, "shape": shape,
                     "ccr_target": ccr, "width_parameter": width, "extra_edge_probability": 0.35,
                     "generator": "layered-v1", "optical_workload": False})


def build_instances(repo, config):
    items = archived(repo) if config.get("include_archived", True) else []
    workload = config["workloads"]
    for n in workload["sizes"]:
        for shape in workload["shapes"]:
            for repetition in range(workload["instances_per_cell"]):
                seed = workload["evaluation_seed_base"] + n * 100 + (shape == "wide") * 10 + repetition
                name = f"eval-n{n}-{shape}-r{repetition}"
                items.append(synthetic(n, shape, seed, name, "evaluation", workload["ccr"]))
    for i, (n, shape) in enumerate([(20, "deep"), (20, "wide"), (50, "deep"), (50, "wide")]):
        seed = workload["tuning_seed_base"] + i
        items.append(synthetic(n, shape, seed, f"tune-n{n}-{shape}", "tuning", workload["ccr"]))
    return items


def prepare(repo, config, root):
    folder = Path(root) / "instances"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for instance in build_instances(repo, config):
        path = folder / f"{instance.identifier}.json"
        content = canonical(instance.data()) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"refusing to overwrite a different instance: {path}")
        if not path.exists():
            path.write_text(content, encoding="utf-8")
        rows.append({"instance_id": instance.identifier, "sha256": instance.sha256,
                     "n": instance.n, "edges": len(instance.edges), "depth": max(instance.depth),
                     "max_layer_width": max(map(len, instance.layers)),
                     "kind": instance.metadata["kind"], "split": instance.metadata.get("split", "archived"),
                     "seed": instance.metadata.get("seed", ""), "shape": instance.metadata.get("shape", "")})
    with (folder / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_instances(root, tuning=False):
    items = [Instance.load(p) for p in sorted((Path(root) / "instances").glob("*.json"))]
    return [i for i in items if (i.metadata.get("split") == "tuning") == tuning]


def random_solution(instance, rng):
    loc = rng.integers(0, 2, instance.n).tolist()
    for i in instance.pinned:
        loc[i] = 0
    order = []
    for layer in instance.layers:
        order.extend(int(i) for i in rng.permutation(layer))
    return loc, order


class Pool:
    def __init__(self, instance, size, seed):
        rng = np.random.default_rng(seed)
        self.loc = rng.integers(0, 2, (size, instance.n), dtype=np.int8)
        self.loc[:, instance.pinned] = 0
        self.order = np.empty((size, instance.n), dtype=np.int32)
        offset = 0
        for layer in instance.layers:
            # Each row independently permutes the same level; no objective-based sorting.
            keys = rng.random((size, len(layer)))
            self.order[:, offset:offset + len(layer)] = np.asarray(layer)[np.argsort(keys, axis=1)]
            offset += len(layer)
        self.size = size
        h = hashlib.sha256(self.loc.tobytes())
        h.update(self.order.tobytes())
        self.sha256 = h.hexdigest()

    def decode(self, pair):
        a, b = map(int, pair)
        if not (0 <= a < self.size and 0 <= b < self.size):
            raise ValueError("index outside pool")
        return self.loc[a].tolist(), self.order[b].tolist()
