"""Replay the exact frozen trace search kernel without a private evidence ZIP.

Input graphs must be regenerated locally from the checksum-verified source.
The published parameter JSON contains global settings, not source/run data.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mqhoa_supp.model import Instance, digest, canonical
from mqhoa_supp.protocol import pool_parameters
from extended_validation.trace.engine import run_one, CONFIG


def settings():
    return json.loads((HERE / 'trace_parameters_20261109.json').read_text())


def tasks(items):
    parameters = settings()
    cfg = parameters['pool']
    output = []
    for identifier, instance in sorted(items.items()):
        size, seed = pool_parameters(instance, cfg)
        for method in CONFIG['stochastic_methods'] + CONFIG['deterministic_methods']:
            for run_seed in parameters['seeds'] if method in CONFIG['stochastic_methods'] else [None]:
                params = parameters['mqhoa'] if method in ['MQHOA', 'DirectGeometryMQHOA'] else parameters['PSO'] if method == 'PSO' else {}
                task = {'instance_id': identifier, 'method': method, 'seed': run_seed, 'params': params,
                        'budget': 1 if method == 'HEFT' else parameters['budget_evals'],
                        'pool_size': size, 'pool_seed': seed}
                task['run_id'] = digest(task)[:24]
                output.append(task)
    return output


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--secondary', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    items, cohorts = {}, {}
    for cohort, directory in [('primary', args.primary), ('secondary_large', args.secondary)]:
        if directory is None:
            continue
        for path in sorted(directory.glob('*.json')):
            instance = Instance(**json.loads(path.read_text()))
            md = instance.metadata
            if not md.get('official_archive_verified') or md.get('source_sha256') != CONFIG['official_sha256']:
                raise ValueError('inputs must come from the checksum-verified extraction')
            if instance.identifier in items:
                raise ValueError('duplicate instance')
            items[instance.identifier] = instance
            cohorts[instance.identifier] = cohort
    if not items:
        raise ValueError('no input graphs found')
    matrix = tasks(items)
    if args.out.exists():
        raise FileExistsError('use a fresh output directory; previous evidence is preserved')
    args.out.mkdir(parents=True)
    params_path = HERE / 'trace_parameters_20261109.json'
    provenance = {'reproduction_entry': 'published-global-settings; unchanged frozen run_one kernel',
                  'global_settings_sha256': hashlib.sha256(params_path.read_bytes()).hexdigest(),
                  'search_kernel_source_sha256': hashlib.sha256(Path(run_one.__code__.co_filename).read_bytes()).hexdigest(),
                  'entrypoint_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'source_archive_sha256': CONFIG['official_sha256']}
    receipt = {**provenance, 'validated_input_graphs': len(items), 'expected_runs': len(matrix),
               'private_evidence_zip_required': False, 'raw_inputs_copied_to_results': False,
               'validation_only': args.validate_only,
               'input_sha256': {i: x.sha256 for i, x in items.items()}}
    (args.out / 'input_receipt.json').write_text(canonical(receipt) + '\n')
    if args.validate_only:
        print(canonical({'validated_inputs': len(items), 'planned_runs': len(matrix)}))
        return
    payloads = [(task, items[task['instance_id']].data(), {**provenance, 'cohort': cohorts[task['instance_id']]}) for task in matrix]
    with (args.out / 'quality.jsonl').open('x') as output, ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_one, payload) for payload in payloads]
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            output.write(canonical(result) + '\n')
            output.flush()
            if count % 20 == 0 or count == len(matrix):
                print('quality', count, '/', len(matrix), flush=True)
    (args.out / 'completion.json').write_text(canonical({'status': 'completed', 'runs': len(matrix), **provenance}) + '\n')


if __name__ == '__main__':
    main()
