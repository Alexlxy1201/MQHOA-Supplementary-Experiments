"""Scheduling semantics shared by all NEW comparisons, not historical results."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import heapq
import json
import math
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass
class Instance:
    identifier: str
    tl: list[float]
    ts: list[float]
    edges: list[tuple[int, int, float]]
    pinned: list[int]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.n = len(self.tl)
        if not self.n or len(self.ts) != self.n:
            raise ValueError("nonempty, equally sized local and edge durations required")
        if any(not math.isfinite(x) or x < 0 for x in self.tl + self.ts):
            raise ValueError("durations must be finite and nonnegative")
        if len(set(self.pinned)) != len(self.pinned) or any(i not in range(self.n) for i in self.pinned):
            raise ValueError("invalid pinned node")
        self.pre = [[] for _ in range(self.n)]
        self.succ = [[] for _ in range(self.n)]
        seen = set()
        for a, b, c in self.edges:
            if a not in range(self.n) or b not in range(self.n) or a == b:
                raise ValueError("invalid edge")
            if not math.isfinite(c) or c < 0 or (a, b) in seen:
                raise ValueError("invalid cost or duplicate edge")
            seen.add((a, b))
            self.pre[b].append((a, c))
            self.succ[a].append((b, c))
        indegree = [len(p) for p in self.pre]
        ready = [i for i, d in enumerate(indegree) if d == 0]
        heapq.heapify(ready)
        self.topo = []
        self.depth = [0] * self.n
        while ready:
            i = heapq.heappop(ready)
            self.topo.append(i)
            for j, _ in self.succ[i]:
                self.depth[j] = max(self.depth[j], self.depth[i] + 1)
                indegree[j] -= 1
                if not indegree[j]:
                    heapq.heappush(ready, j)
        if len(self.topo) != self.n:
            raise ValueError("instance must be acyclic")
        self.layers = [[i for i in self.topo if self.depth[i] == d] for d in range(max(self.depth) + 1)]

    def data(self):
        return {"identifier": self.identifier, "tl": self.tl, "ts": self.ts,
                "edges": self.edges, "pinned": self.pinned, "metadata": self.metadata}

    @property
    def sha256(self):
        return digest(self.data())

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def check_solution(instance, loc, order):
    n = instance.n
    if len(loc) != n or any(x not in (0, 1) for x in loc):
        raise ValueError("placement must contain n binary values")
    if len(order) != n or sorted(order) != list(range(n)):
        raise ValueError("order must be a permutation")
    if any(loc[i] != 0 for i in instance.pinned):
        raise ValueError("pinned tasks must remain local")
    position = {j: i for i, j in enumerate(order)}
    if any(position[a] >= position[b] for a, b, _ in instance.edges):
        raise ValueError("order violates precedence")


def schedule(instance, loc, order, check=True):
    if check:
        check_solution(instance, loc, order)
    starts, finishes = [0.0] * instance.n, [0.0] * instance.n
    available = [0.0, 0.0]
    for j in order:
        r = int(loc[j])
        ready = max((finishes[i] + (c if loc[i] != r else 0.0) for i, c in instance.pre[j]), default=0.0)
        starts[j] = max(ready, available[r])
        finishes[j] = starts[j] + (instance.ts[j] if r else instance.tl[j])
        available[r] = finishes[j]
    return float(max(finishes)), starts, finishes


def validate(instance, loc, order, starts, finishes, eps=1e-8):
    """Independent constraint check; does not call schedule()."""
    check_solution(instance, loc, order)
    bad = []
    if len(starts) != instance.n or len(finishes) != instance.n:
        return ["schedule length"]
    for i, (s, f) in enumerate(zip(starts, finishes)):
        duration = instance.ts[i] if loc[i] else instance.tl[i]
        if not math.isfinite(s) or not math.isfinite(f) or s < -eps or abs(f - s - duration) > eps:
            bad.append(f"duration/start:{i}")
    for a, b, c in instance.edges:
        if starts[b] + eps < finishes[a] + (c if loc[a] != loc[b] else 0):
            bad.append(f"precedence:{a}->{b}")
    # Check the promised non-insertion, per-resource pi order as well as overlap.
    for resource in (0, 1):
        previous = None
        for i in order:
            if loc[i] == resource:
                if previous is not None and starts[i] + eps < finishes[previous]:
                    bad.append(f"resource:{previous}->{i}")
                previous = i
    return bad
