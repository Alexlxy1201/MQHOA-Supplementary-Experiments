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
            fixture_repo = root / 'fixture_repo'
            fixture_repo.mkdir()
            with ZipFile(fixture_repo / '代码 数据等.zip', 'w') as archive:
                archive.writestr('代码/MQHOA/test2.py', 'edges=[(0,1,1)]\ntl=[1,2]\nts=[1,1]\n')
                archive.writestr('代码/DAG(test)/data.py', "path='D:/fixture/batch_task.csv'\n")
                archive.writestr('代码/DAG(test)/dataset.py', "path='E:/fixture/DAGAnalysis.csv'\n")
            report = m.extract(raw, root / 'attempt', fixture_repo, True)
            self.assertEqual(report['funnel']['raw_rows'], 17)
            self.assertEqual(report['funnel']['jobs'], 2)
            self.assertEqual(report['funnel']['positive_span_all_tasks'], 1)
            self.assertEqual(report['selected_counts'], {'small': 1})
            self.assertFalse(report['official_sha256_verified'])
            self.assertEqual(report['archived_topology_candidates'], [])
            inst = json.loads(next((root / 'attempt' / 'instances').glob('*.json')).read_text())
            self.assertEqual(inst['tl'], [3.0] + [2.0] * 7)
            self.assertEqual(inst['metadata']['source_rows'], 9)
            self.assertTrue(inst['metadata']['communication_is_synthetic'])
            self.assertTrue(inst['metadata']['not_a_recovered_historical_input'])


if __name__ == '__main__':
    unittest.main()
