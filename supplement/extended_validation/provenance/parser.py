"""Explicit local-file Alibaba v2018 extraction; never fetches external data."""
from __future__ import annotations
import argparse
import ast
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sqlite3
import tarfile
from zipfile import ZipFile

OFFICIAL_SHA256 = '7c4b32361bd1ec2083647a8f52a6854a03bc125ca5c202652316c499fbf978c6'
NAME = re.compile(r'[A-Za-z](\d+)((?:_\d+)*)\Z')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def parse_name(name):
    if name.startswith('task_') and len(name) > 5:
        return name, ()
    match = NAME.fullmatch(name)
    if not match:
        raise ValueError('unknown task name: ' + name)
    return int(match[1]), tuple(int(x) for x in match[2].split('_') if x)


def graph(names):
    ids, dependencies = {}, {}
    for name in names:
        key, parents = parse_name(name)
        if key in ids:
            raise ValueError('duplicate task identifier')
        ids[key], dependencies[key] = name, parents
    if any(p not in ids for ps in dependencies.values() for p in ps):
        raise ValueError('missing parent')
    if any(k in ps for k, ps in dependencies.items()):
        raise ValueError('self edge')
    keys = sorted(ids, key=lambda x: (isinstance(x, str), x))
    index = {k: i for i, k in enumerate(keys)}
    edges = sorted({(index[p], index[k]) for k, ps in dependencies.items() for p in ps})
    pre, succ = [set() for _ in keys], [set() for _ in keys]
    for a, b in edges:
        pre[b].add(a)
        succ[a].add(b)
    remaining = [len(p) for p in pre]
    ready = collections.deque(i for i, n in enumerate(remaining) if not n)
    order = []
    while ready:
        i = ready.popleft()
        order.append(i)
        for j in sorted(succ[i]):
            remaining[j] -= 1
            if not remaining[j]:
                ready.append(j)
    if len(order) != len(keys):
        raise ValueError('cycle')
    visited, pending = set(), ([0] if keys else [])
    while pending:
        i = pending.pop()
        if i not in visited:
            visited.add(i)
            pending.extend(pre[i] | succ[i])
    sources = [i for i in range(len(keys)) if not pre[i]]
    sinks = [i for i in range(len(keys)) if not succ[i]]
    return {'names': [ids[k] for k in keys], 'keys': keys, 'edges': edges,
            'topo': order, 'connected': len(visited) == len(keys),
            'sources': sources, 'sinks': sinks, 'numeric_only': all(isinstance(k, int) for k in keys)}


def isomorphism(n, left, right):
    """Directed exact backtracking with degree partitions; returns node mapping."""
    if len(left) != len(right):
        return None
    def adjacency(edges):
        a, b = [set() for _ in range(n)], [set() for _ in range(n)]
        for u, v in edges:
            a[u].add(v)
            b[v].add(u)
        return a, b
    ls, lp = adjacency(left)
    rs, rp = adjacency(right)
    candidates = {i: [j for j in range(n) if (len(ls[i]), len(lp[i])) == (len(rs[j]), len(rp[j]))] for i in range(n)}
    if any(not c for c in candidates.values()):
        return None
    mapping, used = {}, set()
    def visit():
        if len(mapping) == n:
            return dict(mapping)
        i = min((k for k in range(n) if k not in mapping),
                key=lambda k: (sum(j not in used for j in candidates[k]), -sum(u in mapping for u in ls[k] | lp[k]), k))
        for j in candidates[i]:
            if j in used:
                continue
            if any(((u in ls[i]) != (v in rs[j])) or ((u in lp[i]) != (v in rp[j])) for u, v in mapping.items()):
                continue
            mapping[i], _ = j, used.add(j)
            result = visit()
            if result is not None:
                return result
            del mapping[i]
            used.remove(j)
        return None
    return visit()


def archive_audit(repo, out):
    repo, out = Path(repo), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'archive_audit.json'
    if target.exists():
        raise FileExistsError(target)
    archive = repo / '代码 数据等.zip'
    with ZipFile(archive) as z:
        members = {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if not n.endswith('/')}
        values = {}
        name = '代码/MQHOA/test2.py'
        for node in ast.parse(z.read(name).decode('utf-8')).body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id in {'edges', 'tl', 'ts'}:
                        values[t.id] = ast.literal_eval(node.value)
        paths = {}
        for n in ('代码/DAG(test)/data.py', '代码/DAG(test)/dataset.py'):
            paths[n] = re.findall(r'[DE]:/[^\n\r\'\"]+', z.read(n).decode('utf-8'))
    result = {'source_archive_sha256': sha(archive), 'members_sha256': members,
              'archived_constants': values, 'hard_coded_paths': paths,
              'job_identifier_recovered': False, 'trace_mapping_recovered': False,
              'timing_or_edge_weight_source_recovered': False,
              'interpretation': 'No raw trace, job ID, or extraction mapping is included in this archive; this does not establish that the original extraction did not occur.'}
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def extract(local, out, repo, allow_nonofficial_csv=False):
    local, out = Path(local), Path(out)
    if out.exists():
        raise FileExistsError('fresh attempt output required: ' + str(out))
    checksum = sha(local)
    is_tar = local.name.endswith(('.tar.gz', '.tgz'))
    if is_tar and checksum != OFFICIAL_SHA256:
        raise ValueError('official archive SHA256 mismatch')
    if not is_tar and not allow_nonofficial_csv:
        raise ValueError('unverified CSV requires explicit --allow-nonofficial-csv')
    out.mkdir(parents=True)
    db = sqlite3.connect(out / 'task_aggregation.sqlite')
    db.execute('CREATE TABLE t(job TEXT,name TEXT,n INTEGER,total REAL,valid INTEGER,PRIMARY KEY(job,name))')
    handle = tarfile.open(local, 'r:gz') if is_tar else None
    if handle:
        members = [m for m in handle.getmembers() if m.isfile() and Path(m.name).name == 'batch_task.csv']
        if len(members) != 1:
            raise ValueError('one batch_task.csv member required')
        stream = io.TextIOWrapper(handle.extractfile(members[0]), encoding='utf-8')
    else:
        stream = local.open(encoding='utf-8')
    records, invalid = 0, 0
    query = ('INSERT INTO t VALUES(?,?,1,?,?) ON CONFLICT(job,name) DO UPDATE SET '
             'n=n+1,total=total+excluded.total,valid=valid+excluded.valid')
    batch = []
    for row in csv.reader(stream):
        records += 1
        if len(row) != 9 or not row[0] or not row[2]:
            invalid += 1
            continue
        try:
            span = float(row[6]) - float(row[5])
            good = math.isfinite(span) and span > 0
        except ValueError:
            span, good = 0, False
        batch.append((row[2], row[0], span if good else 0, int(good)))
        if len(batch) >= 20000:
            db.executemany(query, batch)
            db.commit()
            batch.clear()
            if records % 200000 == 0:
                print('raw rows', records, flush=True)
    if batch:
        db.executemany(query, batch)
        db.commit()
    stream.close()
    if handle:
        handle.close()
    audit = archive_audit(repo, out / 'archive')
    old_edges = [(a, b) for a, b, _ in audit['archived_constants']['edges']]
    funnel = collections.Counter(raw_rows=records, malformed_rows=invalid)
    selected = collections.Counter()
    matches = []
    exported = []
    current, rows = None, []
    def process(job, records):
        if job is None:
            return
        funnel['jobs'] += 1
        try:
            g = graph([r[0] for r in records])
        except ValueError as e:
            funnel['rejected:' + str(e).split(':')[0]] += 1
            return
        funnel['acyclic_valid'] += 1
        n = len(g['names'])
        if n == 10:
            mapping = isomorphism(10, old_edges, g['edges'])
            if mapping is not None:
                matches.append({'job_name': job, 'archive_node_to_job_index': mapping,
                                'names': g['names'], 'historical_identity_proven': False})
        if not g['numeric_only'] or not g['connected']:
            return
        funnel['numeric_connected'] += 1
        if len(g['sources']) != 1 or len(g['sinks']) != 1:
            return
        funnel['single_entry_exit'] += 1
        spans = {r[0]: r[2] / r[3] for r in records if r[3]}
        if len(spans) != n:
            return
        funnel['positive_span_all_tasks'] += 1
        cell = next((label for lo, hi, label in [(8, 20, 'small'), (40, 60, 'medium'), (80, 100, 'large')] if lo <= n <= hi), None)
        if cell is None or selected[cell] >= 10:
            return
        tl = [spans[name] for name in g['names']]
        ts = [x / 3 for x in tl]
        delay = sum((a + b) / 2 for a, b in zip(tl, ts)) / n
        identifier = 'trace-' + cell + '-' + job
        data = {'identifier': identifier, 'tl': tl, 'ts': ts,
                'edges': [(a, b, delay) for a, b in g['edges']],
                'pinned': sorted(set(g['sources'] + g['sinks'])),
                'metadata': {'kind': 'new-trace-derived', 'job_name': job, 'original_task_names': g['names'],
                             'duration_source': 'mean positive observed batch-task wall span (seconds)',
                             'resource_speedup': 3, 'speedup_is_synthetic': True,
                             'communication_is_synthetic': True, 'ccr': 1,
                             'source_sha256': checksum, 'source_rows': sum(r[1] for r in records),
                             'not_a_recovered_historical_input': True}}
        p = out / 'instances' / (identifier + '.json')
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps(data, sort_keys=True, ensure_ascii=False) + '\n')
        selected[cell] += 1
        exported.append({'identifier': identifier, 'n': n, 'cell': cell, 'sha256': sha(p)})
    for job, name, count, total, valid in db.execute('SELECT job,name,n,total,valid FROM t ORDER BY job,name'):
        if current is not None and job != current:
            process(current, rows)
            rows = []
        current = job
        rows.append((name, count, total, valid))
    process(current, rows)
    db.close()
    report = {'source_sha256': checksum, 'official_sha256_verified': is_tar,
              'funnel': dict(funnel), 'selected_counts': dict(selected), 'instances': exported,
              'archived_topology_candidates': matches,
              'historical_job_or_timing_provenance_recovered': False}
    (out / 'extraction_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['funnel', 'selected_counts']}, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--repo', default='.')
    p.add_argument('--out', required=True)
    p.add_argument('--local')
    p.add_argument('--allow-nonofficial-csv', action='store_true')
    a = p.parse_args()
    if a.local:
        extract(a.local, a.out, a.repo, a.allow_nonofficial_csv)
    else:
        print(json.dumps(archive_audit(a.repo, a.out), ensure_ascii=False, indent=2))
