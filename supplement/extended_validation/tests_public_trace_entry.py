"""Fixtures for the standalone publication entry; no real trace is consumed."""
import json
from pathlib import Path
import tempfile
import unittest
import extract_new_trace as entry


class PublicExtractionTests(unittest.TestCase):
    def test_no_private_archive_needed_and_no_historical_verdict_fabricated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / 'fixture.csv'
            records = []
            for node in range(1, 9):
                name = 'M1' if node == 1 else f'M{node}_{node-1}'
                records.append(f'{name},1,fixture_job,1,Terminated,1,3,100,1')
            raw.write_text('\n'.join(records) + '\n')
            audit, match = entry.extractor.archive_audit, entry.extractor.isomorphism
            report = entry.extract(raw, root / 'out', allow_nonofficial_csv=True)
            self.assertFalse(report['official_sha256_verified'])
            self.assertFalse(report['historical_audit_performed'])
            self.assertNotIn('archived_topology_candidates', report)
            self.assertEqual(report['selected_counts'], {'small': 1, 'medium': 0, 'large': 0})
            self.assertIs(entry.extractor.archive_audit, audit)
            self.assertIs(entry.extractor.isomorphism, match)
            self.assertFalse((root / '代码 数据等.zip').exists())

    def test_source_rejection_restores_callbacks_and_writes_no_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / 'batch_task.tar.gz'
            raw.write_bytes(b'wrong source')
            audit, match = entry.extractor.archive_audit, entry.extractor.isomorphism
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                entry.extract(raw, root / 'out')
            self.assertIs(entry.extractor.archive_audit, audit)
            self.assertIs(entry.extractor.isomorphism, match)
            self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
