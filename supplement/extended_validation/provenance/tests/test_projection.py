import importlib.util
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from projection_audit import historical_projection,search,secondary_coverage

class ProjectionTests(unittest.TestCase):
    def test_historical_case_sensitive_filter(self):
        g=historical_projection(['M1','M2_1','task_opaque','MergeTask_1'])
        self.assertTrue(g['strict_valid'])
        self.assertEqual(g['edges'],[(0,1)])
        self.assertEqual(g['discarded'],dict(task=1,MergeTask=1))
        # Upper-case Task is not discarded by the historical case-sensitive regex.
        self.assertEqual(historical_projection(['M1','Task_1'])['status'],'unknown-retained-name')
        self.assertEqual(historical_projection(['task_x','MergeTask_3'])['status'],'empty')

    def test_literal_strings_missing_parents_and_aliases(self):
        g=historical_projection(['M01','M1','M2_01_1'])
        self.assertEqual(g['keys'],['01','1','2'])
        self.assertEqual(g['edges'],[(0,2),(1,2)])
        self.assertTrue(g['strict_valid'])
        g=historical_projection(['M2_1'])
        self.assertEqual(g['keys'],['1','2'])
        self.assertEqual(g['missing_parents'],['1'])
        self.assertFalse(g['strict_valid'])
        g=historical_projection(['M1','J1','M2_1'])
        self.assertEqual(g['collapsed_ids'],{'1':['M1','J1']})
        self.assertFalse(g['strict_valid'])

    def test_cycles_self_edges_and_disconnected(self):
        self.assertFalse(historical_projection(['M1_2','M2_1'])['acyclic'])
        self.assertEqual(historical_projection(['M1_1'])['self_edges'],['1'])
        self.assertFalse(historical_projection(['M1','M2'])['connected'])

    def test_whole_job_would_miss_projection_candidate(self):
        db=sqlite3.connect(':memory:')
        db.execute('CREATE TABLE t(job TEXT,name TEXT,n INTEGER,total REAL,valid INTEGER)')
        names=['M1']+[f'M{i}_{i-1}' for i in range(2,11)]+['task_opaque','MergeTask_1']
        db.executemany('INSERT INTO t VALUES(?,?,1,2,1)',[('job-candidate',name) for name in names])
        db.execute('INSERT INTO t VALUES(?,?,1,2,1)',('job-unknown','Unknown'))
        with tempfile.TemporaryDirectory() as tmp:
            result=search(db,list(zip(range(9),range(1,10))),Path(tmp))
            self.assertEqual(result['funnel']['jobs_all'],2)
            self.assertEqual(result['funnel']['strict_isomorphic_jobs'],1)
            self.assertEqual(result['whole_job_funnel'].get('isomorphic_jobs',0),0)
            self.assertEqual(result['safe_candidates'][0]['job_name'],'job-candidate')
            self.assertFalse(result['safe_candidates'][0]['historical_identity_proven'])
            self.assertEqual(len((Path(tmp)/'projection_candidates_full_local.jsonl').read_text().splitlines()),1)

    def test_secondary_non_sese_all_nodes_preserved_lexicographic(self):
        db=sqlite3.connect(':memory:')
        db.execute('CREATE TABLE t(job TEXT,name TEXT,n INTEGER,total REAL,valid INTEGER)')
        # Connected 80-node fork with two sources and one sink.
        names=['M1','M2','M3_1_2']+[f'M{i}_{i-1}' for i in range(4,81)]
        for job in ['job-z','job-a']:
            db.executemany('INSERT INTO t VALUES(?,?,1,6,1)',[(job,name) for name in names])
        # All nodes exist but one has no positive span: coverage, not selected.
        db.executemany('INSERT INTO t VALUES(?,?,1,6,1)',[('job-bad',name) for name in names[:-1]])
        db.execute('INSERT INTO t VALUES(?,?,1,0,0)',('job-bad',names[-1]))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            result=secondary_coverage(db,root,'fixture')
            self.assertEqual(result['coverage']['non_sese_valid'],3)
            self.assertEqual(result['coverage']['non_sese_all_tasks_positive_span'],2)
            self.assertEqual([r['job_name'] for r in result['selected']],['job-a','job-z'])
            import json
            data=json.loads(next((root/'secondary_large_instances').glob('*job-a.json')).read_text())
            self.assertEqual(len(data['tl']),80)
            self.assertEqual(len(data['edges']),79)
            self.assertEqual(data['pinned'],[0,1,79])
            self.assertTrue(all(t==2 for t in data['ts']))
            self.assertTrue(all(c==4 for a,b,c in data['edges']))
            self.assertTrue(data['metadata']['no_dummy_nodes'])
            self.assertFalse(data['metadata']['original_SESE_primary_group'])

if __name__=='__main__':
    unittest.main(verbosity=2)
