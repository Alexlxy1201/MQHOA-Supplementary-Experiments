"""Algorithm-1 candidate generation with the adaptations frozen in the plan."""
from __future__ import annotations
import heapq
import math
from mqhoa_supp.methods import Budget


def critical_order(instance, placement):
    ranks = [0.0] * instance.n
    for u in reversed(instance.topo):
        ranks[u] = (instance.ts[u] if placement[u] else instance.tl[u]) + max(
            (ranks[v] + (cost if placement[u] != placement[v] else 0)
             for v, cost in instance.succ[u]), default=0.0)
    degree = [len(x) for x in instance.pre]
    ready = [(-ranks[i], i) for i in instance.topo if not degree[i]]
    heapq.heapify(ready)
    order = []
    while ready:
        _, u = heapq.heappop(ready)
        order.append(u)
        for v, _ in instance.succ[u]:
            degree[v] -= 1
            if not degree[v]:
                heapq.heappush(ready, (-ranks[v], v))
    return order


def candidates(instance, variant):
    if variant not in ('0', 'N', 'N2'):
        raise ValueError('unknown FlexDO variant')
    free = sorted(set(range(instance.n)) - set(instance.pinned))
    q = len(free)
    bits = 0 if not q or variant == '0' else min(q, int(math.floor(
        (1 if variant == 'N' else 2) * math.log2(q))))
    target = q - bits
    pinned = set(instance.pinned)
    anchors = [(u, v, c) for u, v, c in instance.edges if u in pinned or v in pinned]
    edges = [(u, v, c - sum(d for a, b, d in anchors if a in (u, v) or b in (u, v)))
             for u, v, c in instance.edges if u not in pinned and v not in pinned]
    gain = {u: instance.tl[u] - instance.ts[u]
            - sum(c for a, b, c in instance.edges if u in (a, b)) for u in free}
    placement = [0] * instance.n
    seen = set()
    generated = duplicates = greedy_count = 0
    output = []

    def retain(stage):
        nonlocal generated, duplicates, greedy_count
        generated += 1
        key = tuple(placement)
        if key in seen:
            duplicates += 1
            return
        seen.add(key)
        output.append((key, stage))
        greedy_count += stage == 'greedy'

    retain('greedy')
    while edges and sum(placement) < target:
        edges = [e for e in edges if not (placement[e[0]] and placement[e[1]])]
        if not edges:
            break
        crossing = [e for e in edges if placement[e[0]] != placement[e[1]]]
        u, v, _ = min(crossing or edges, key=lambda e: (-e[2], e[0], e[1]))
        placement[u] = placement[v] = 1
        edges = [e for e in edges if e[:2] != (u, v)]
        retain('greedy')
    for u in sorted(free, key=lambda u: (-gain[u], u)):
        if sum(placement) >= target:
            break
        if not placement[u]:
            placement[u] = 1
            retain('greedy')
    remaining = [u for u in free if not placement[u]]
    for mask in range(1 << len(remaining)):
        for j, u in enumerate(remaining):
            placement[u] = (mask >> j) & 1
        retain('enumeration')
    return output, {'offloadable_tasks': q, 'target_free_bits': bits,
                    'actual_free_bits': len(remaining), 'greedy_distinct': greedy_count,
                    'enumeration_attempted': 1 << len(remaining),
                    'attempted_candidates': generated, 'duplicates_removed': duplicates,
                    'distinct_candidates': len(output)}


def solve(instance, variant='N2', limit=10000):
    placements, details = candidates(instance, variant)
    if len(placements) > limit:
        raise ValueError('complete FlexDO decision set exceeds evaluation cap')
    evaluator = Budget(instance, limit, trace_every=100)
    for placement, _stage in placements:
        evaluator.evaluate(placement, critical_order(instance, placement))
    return {**evaluator.result(), 'details': details, 'method': 'FlexDO', 'variant': variant,
            'instance_id': instance.identifier, 'instance_sha256': instance.sha256,
            'pool_sha256': None, 'deterministic': True}
