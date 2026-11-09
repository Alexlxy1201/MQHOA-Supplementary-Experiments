import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

spec = importlib.util.spec_from_file_location('trace_parser', Path(__file__).parents[1] / 'parser.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ParserTests(unittest.TestCase):
    @staticmethod
    def fixture_repo(root, n=2, edges=None):
        repo = root / 'fixture_repo'
        repo.mkdir()
        if edges is None:
            edges = [(0, 1, 1)]
        with ZipFile(repo / '代码 数据等.zip', 'w') as archive:
            archive.writestr('代码/MQHOA/test2.py', f'edges={edges!r}\ntl={[1]*n!r}\nts={[1]*n!r}\n')
            archive.writestr('代码/DAG(test)/data.py', "path='D:/fixture/batch_task.csv'\n")
            archive.writestr('代码/DAG(test)/dataset.py', "path='E:/fixture/DAGAnalysis.csv'\n")
        return repo

    def test_official_example(self):
        g = m.graph(['M1', 'M2_1', 'M3_1', 'R4_2', 'M5_3_4'])
        self.assertEqual(g['edges'], [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)])
        self.assertTrue(g['connected'])
        self.assertEqual(g['sources'], [0])
        self.assertEqual(g['sinks'], [4])

    def test_opaque_independent_name(self):
        g = m.graph(['task_Nzg3ODAwNDgzMTAwNTc2NTQ2Mw==', 'M1'])
        self.assertFalse(g['connected'])
        self.assertFalse(g['numeric_only'])
        self.assertEqual(g['edges'], [])

    def test_letter_irrelevant_and_ids_normalized(self):
        self.assertEqual(m.parse_name('R5_3_4'), m.parse_name('M5_3_4'))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            m.graph(['M1', 'R01'])

    def test_rejects_unknown_missing_self_and_cycle(self):
        for names, error in [(['MergeTask3'], 'unknown'), (['M2_1'], 'missing'),
                             (['M1_1'], 'self'), (['M1_2', 'M2_1'], 'cycle')]:
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, error):
                m.graph(names)

    def test_directed_isomorphism_with_mapping(self):
        a = [(0, 1), (0, 2), (2, 3)]
        relabel = [3, 1, 0, 2]
        b = [(relabel[i], relabel[j]) for i, j in a]
        mapping = m.isomorphism(4, a, b)
        self.assertIsNotNone(mapping)
        self.assertEqual({(mapping[i], mapping[j]) for i, j in a}, set(b))
        self.assertIsNone(m.isomorphism(4, a, [(0, 1), (0, 2), (0, 3)]))

    def test_noncontiguous_job_records_aggregation_and_span_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for i in range(1, 9):
                name = 'M1' if i == 1 else f'M{i}_{i-1}'
                rows.append(f'{name},1,j_example,1,Terminated,10,12,100,1')
                rows.append(f'{name},1,j_other,1,Terminated,10,0,100,1')
            rows.append('M1,1,j_example,1,Terminated,10,14,100,1')
            raw = root / 'fixture.csv'
            raw.write_text('\n'.join(rows) + '\n')
            # A synthetic archive makes this integration test runnable in the
            # public code repository without the private historical archive.
            fixture_repo = self.fixture_repo(root)
            report = m.extract(raw, root / 'attempt', fixture_repo, True)
            self.assertEqual(report['funnel']['raw_rows'], 17)
            self.assertEqual(report['funnel']['jobs'], 2)
            self.assertEqual(report['funnel']['positive_span_all_tasks'], 1)
            self.assertEqual(report['selected_counts'], {'small': 1, 'medium': 0, 'large': 0})
            self.assertEqual(report['missing_counts'], {'small': 9, 'medium': 10, 'large': 10})
            self.assertFalse(report['selection_complete'])
            self.assertFalse(report['official_sha256_verified'])
            self.assertEqual(report['archived_topology_candidates'], [])
            inst = json.loads(next((root / 'attempt' / 'instances').glob('*.json')).read_text())
            self.assertEqual(inst['tl'], [3.0] + [2.0] * 7)
            self.assertEqual(inst['metadata']['source_rows'], 9)
            self.assertTrue(inst['metadata']['communication_is_synthetic'])
            self.assertTrue(inst['metadata']['not_a_recovered_historical_input'])
            self.assertEqual(inst['metadata']['kind'], 'unverified-local-csv')
            before = m.sha(raw)
            with self.assertRaises(FileExistsError):
                m.extract(raw, root / 'attempt', fixture_repo, True)
            self.assertEqual(m.sha(raw), before)

    def test_complete_stratification_lexical_selection_and_funnel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self.fixture_repo(root)
            records = []
            for label, n in [('small', 8), ('medium', 40), ('large', 80)]:
                for job in reversed(range(12)):
                    for i in reversed(range(1, n+1)):
                        name = 'M1' if i == 1 else f'R{i}_{i-1}'
                        records.append(f'{name},1,{label}_{job:02d},1,Terminated,10,12,100,1')
            # These rejected records stay in the corresponding denominators.
            records += ['M1_2,1,cyclic,1,Terminated,10,12,100,1',
                        'M2_1,1,cyclic,1,Terminated,10,12,100,1', 'malformed']
            raw = root / 'fixture.csv'
            raw.write_text('\n'.join(records) + '\n')
            report = m.extract(raw, root / 'attempt', repo, True)
            self.assertEqual(report['funnel']['raw_rows'], len(records))
            self.assertEqual(report['funnel']['malformed_rows'], 1)
            self.assertEqual(report['funnel']['jobs'], 37)
            self.assertEqual(report['funnel']['rejected:cycle'], 1)
            self.assertEqual(report['funnel']['positive_span_all_tasks'], 36)
            self.assertEqual(report['selected_counts'], dict.fromkeys(['small', 'medium', 'large'], 10))
            self.assertTrue(report['selection_complete'])
            self.assertEqual(report['missing_counts'], dict.fromkeys(['small', 'medium', 'large'], 0))
            self.assertEqual({x['identifier'] for x in report['instances']},
                             {f'trace-{cell}-{cell}_{j:02d}' for cell in ['small', 'medium', 'large'] for j in range(10)})

    def test_structural_match_is_only_candidate_and_job_names_cannot_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            edges = [(0, i, 1) for i in range(1, 9)] + [(i, 9, 1) for i in range(1, 9)]
            repo = self.fixture_repo(root, 10, edges)
            records = []
            for i in reversed(range(10)):
                parents = [a+1 for a, b, _ in edges if b == i]
                name = 'M' + str(i+1) + ''.join('_' + str(p) for p in parents)
                records.append(f'{name},1,../job,1,Terminated,10,12,100,1')
            raw = root / 'fixture.csv'
            raw.write_text('\n'.join(records) + '\n')
            report = m.extract(raw, root / 'attempt', repo, True)
            self.assertEqual(len(report['archived_topology_candidates']), 1)
            candidate = report['archived_topology_candidates'][0]
            mapping = candidate['archive_node_to_job_index']
            actual = m.graph([line.split(',')[0] for line in records])
            self.assertEqual({(mapping[a], mapping[b]) for a, b, _ in edges}, set(actual['edges']))
            self.assertFalse(candidate['historical_identity_proven'])
            self.assertFalse(report['historical_job_or_timing_provenance_recovered'])
            paths = list((root / 'attempt' / 'instances').glob('*.json'))
            self.assertEqual(len(paths), 1)
            self.assertEqual(json.loads(paths[0].read_text())['metadata']['job_name'], '../job')

    def test_wrong_official_archive_hash_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / 'batch_task.tar.gz'
            raw.write_bytes(b'this is deliberately not the official archive')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                m.extract(raw, root / 'attempt', root)
            self.assertFalse((root / 'attempt').exists())


if __name__ == '__main__':
    unittest.main()
