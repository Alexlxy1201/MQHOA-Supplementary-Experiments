"""Independently stream the official archive to verify derived selected inputs.

This verifier does not read the extraction SQLite database or call the parser.
It never writes original records, times, task rows, or derived input JSON.
"""
from collections import defaultdict, deque
import hashlib
import io
import csv
import json
import math
from pathlib import Path
import re
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / 'supplement/runs/extended_trace_20261109'
OFFICIAL = '7c4b32361bd1ec2083647a8f52a6854a03bc125ca5c202652316c499fbf978c6'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def near(actual, expected):
    assert math.isfinite(actual) and math.isfinite(expected)
    assert abs(actual-expected) <= 1e-10 * max(1., abs(actual), abs(expected)), (actual, expected)


def main():
    raw = LOCAL / 'source/batch_task.tar.gz'
    assert raw.stat().st_size == 130310549 and sha(raw) == OFFICIAL
    groups = [('primary', LOCAL / 'extraction_v1/instances', 20),
              ('secondary_large', LOCAL / 'source/projection_audit_v1/secondary_large_instances', 10)]
    selected = {}
    for group, directory, count in groups:
        files = sorted(directory.glob('*.json'))
        assert len(files) == count
        for file in files:
            data = json.loads(file.read_text())
            job = data['metadata']['job_name']
            assert job not in selected
            selected[job] = (group, file, data)
    spans = {job: defaultdict(list) for job in selected}
    selected_records = defaultdict(int)
    all_names = {job: set() for job in selected}
    raw_records = 0
    with tarfile.open(raw, 'r:gz') as archive:
        member, = [m for m in archive.getmembers() if m.isfile() and Path(m.name).name == 'batch_task.csv']
        with io.TextIOWrapper(archive.extractfile(member), encoding='utf-8') as stream:
            for row in csv.reader(stream):
                raw_records += 1
                assert len(row) == 9
                job = row[2]
                if job not in selected:
                    continue
                selected_records[job] += 1
                all_names[job].add(row[0])
                try:
                    span = float(row[6]) - float(row[5])
                except ValueError:
                    continue
                if math.isfinite(span) and span > 0:
                    spans[job][row[0]].append(span)
    assert raw_records == 14295731
    checked = []
    for job, (group, file, data) in sorted(selected.items()):
        assert set(data['metadata']['original_task_names']) == all_names[job]
        numeric = {}
        for name in all_names[job]:
            match = re.fullmatch(r'[A-Za-z](\d+)((?:_\d+)*)', name)
            assert match
            node = int(match[1])
            assert node not in numeric
            numeric[node] = (name, [int(x) for x in match[2].split('_') if x])
        ordered = sorted(numeric)
        n = len(ordered)
        assert n == len(data['tl']) == len(data['ts'])
        assert [numeric[node][0] for node in ordered] == data['metadata']['original_task_names']
        index = {node: i for i, node in enumerate(ordered)}
        edges = {(index[p], index[node]) for node, (_, parents) in numeric.items() for p in parents}
        assert edges == {(a, b) for a, b, _ in data['edges']}
        local = [statistics.mean(spans[job][numeric[node][0]]) for node in ordered]
        for actual, expected in zip(data['tl'], local):
            near(actual, expected)
        for actual, expected in zip(data['ts'], local):
            near(actual, expected / 3)
        delay = statistics.mean(2 * x / 3 for x in local)
        for _, _, cost in data['edges']:
            near(cost, delay)
        parents, children = [set() for _ in ordered], [set() for _ in ordered]
        for a, b in edges:
            parents[b].add(a)
            children[a].add(b)
        sources = [i for i in range(n) if not parents[i]]
        sinks = [i for i in range(n) if not children[i]]
        assert data['pinned'] == sorted(set(sources + sinks))
        degree = [len(p) for p in parents]
        ready = deque(sources)
        visited = []
        while ready:
            node = ready.popleft()
            visited.append(node)
            for child in children[node]:
                degree[child] -= 1
                if not degree[child]:
                    ready.append(child)
        assert len(visited) == n
        pending, connected = [0], set()
        while pending:
            node = pending.pop()
            if node not in connected:
                connected.add(node)
                pending.extend(parents[node] | children[node])
        assert len(connected) == n
        if group == 'primary':
            assert len(sources) == len(sinks) == 1 and (8 <= n <= 20 or 40 <= n <= 60)
        else:
            assert 80 <= n <= 100 and (len(sources) != 1 or len(sinks) != 1)
            assert data['metadata']['no_dummy_nodes']
        assert data['metadata']['source_rows'] == selected_records[job]
        assert data['metadata']['source_sha256'] == OFFICIAL
        assert data['metadata']['communication_is_synthetic'] and data['metadata']['speedup_is_synthetic']
        checked.append({'job_name': job, 'group': group, 'nodes': n, 'edges': len(edges),
                        'source_count': len(sources), 'sink_count': len(sinks),
                        'original_records_checked': selected_records[job], 'input_file_sha256': sha(file),
                        'all_original_job_nodes_preserved': True, 'duration_and_communication_checks_passed': True})
    out = ROOT / 'supplement/results/20261109_provenance_actual_v1/selected_inputs_independent_verification.json'
    assert not out.exists(), 'preserve previous verification receipt'
    report = {'status': 'verified', 'source_sha256': OFFICIAL, 'full_raw_rows_streamed_again': raw_records,
              'jobs_checked': len(checked), 'primary_jobs': 20, 'secondary_large_jobs': 10,
              'raw_records_or_derived_input_files_exported': False, 'verified_metadata_only': checked,
              'verifier_source_sha256': sha(Path(__file__))}
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'verified', 'full_raw_rows': raw_records, 'jobs_checked': len(checked)}), flush=True)


if __name__ == '__main__':
    main()
