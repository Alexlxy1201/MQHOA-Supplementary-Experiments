"""Standalone new-source extraction without private historical archives.

The original frozen extractor is reused for source checks, grouping and input
construction. Historical target matching is explicitly disabled and reported
as not assessed; this entry never fabricates an archived target or evidence ZIP.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'provenance'))
import parser as extractor
import projection_audit


def extract(local, primary_out, secondary_out=None, allow_nonofficial_csv=False):
    primary_out = Path(primary_out)
    if secondary_out is not None and Path(secondary_out).exists():
        raise FileExistsError('fresh secondary output required')
    original_audit, original_match = extractor.archive_audit, extractor.isomorphism
    try:
        # Only private historical-target access is disabled. Official SHA,
        # graph validation, duration rules and selection remain unchanged.
        extractor.archive_audit = lambda *_: {'archived_constants': {'edges': []}}
        extractor.isomorphism = lambda *_: None
        report = extractor.extract(local, primary_out, '.', allow_nonofficial_csv)
    finally:
        extractor.archive_audit, extractor.isomorphism = original_audit, original_match
    report.pop('archived_topology_candidates', None)
    report.update(historical_audit_performed=False,
                  historical_audit_scope='Not assessed by this standalone new-workload entry; no private target loaded.')
    (primary_out / 'extraction_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if secondary_out is not None:
        if not report['official_sha256_verified']:
            raise ValueError('official checksum required for secondary source export')
        dbpath = (primary_out / 'task_aggregation.sqlite').resolve()
        with sqlite3.connect('file:' + quote(str(dbpath)) + '?mode=ro&immutable=1', uri=True) as db:
            db.execute('PRAGMA query_only=ON')
            projection_audit.secondary_coverage(db, Path(secondary_out), report['source_sha256'])
    return report


if __name__ == '__main__':
    args = argparse.ArgumentParser(__doc__)
    args.add_argument('--local', type=Path, required=True)
    args.add_argument('--primary-out', type=Path, required=True)
    args.add_argument('--secondary-out', type=Path)
    options = args.parse_args()
    extract(options.local, options.primary_out, options.secondary_out)
