"""Additive historical numeric-projection search, never changes the main parser.

Frozen rules: exact case-sensitive historical task|MergeTask exclusion; literal
string ID semantics of x.replace(x[0], '') from the preserved dataset.py;
unknown retained names rejected, never silently removed. The literal graph adds
referenced parents as nodes and merges repeated IDs, as nx.DiGraph did. Report
strict no-missing/no-collapsed-ID matches separately. All matches are structural
candidates, never proofs of historical job identity or numerical-weight origin.

Secondary coverage is frozen before opening the completed SQLite: among all
whole-job numeric-only, valid, weakly connected 80--100-node DAGs count SESE and
non-SESE and availability of positive mean wall-span for every task. Select the
lexicographically first ten duration-eligible non-SESE cases, preserving every
node/edge, pinning all roots and sinks without dummy nodes. Keep them separate
from the original main sample; never relabel the main 20 as a completed 30.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import quote
from zipfile import ZipFile

from parser import graph as whole_graph,isomorphism,OFFICIAL_SHA256

NUMERIC=re.compile(r'[A-Za-z]\d+(?:_\d+)*\Z')
HISTORICAL_FILTER=re.compile('task|MergeTask')
PROTOCOL={
    'protocol':'historical-projection-and-secondary-large-v1',
    'scope':'additive structural source search; preserve original inputs and all experiments',
    'filter':'case-sensitive regex task|MergeTask applied to the full task name, exactly as data.py',
    'tokenization':"x.replace(x[0], '') followed by split('_'); strings are not normalized to integers",
    'numeric_retention':'all retained names must match [A-Za-z]digits(_digits)*; unknown retained names reject the projection',
    'literal_graph':'directed simple graph of string identifiers; implicit referenced parents and collapsed IDs retained and flagged',
    'strict_projection':'no implicit parents, no collapsed distinct-name IDs, acyclic; connectivity and SESE reported',
    'target':'unweighted directed archive10 graph from the previously preserved supplementary quality input',
    'search':'all valid ten-node projected graphs; no performance filtering; all structural candidates retained locally',
    'whole_comparison':'unchanged main parser graph() semantics; report whole-job matches separately',
    'identity':'topology alone does not identify historical job or recover archived tl/ts/communication weights',
    'secondary_large':{
        'size':[80,100], 'graph':'whole job, numeric only, valid DAG, weakly connected',
        'coverage':'count SESE/non-SESE and per-task positive mean wall-span availability before selection',
        'selection':'lexicographically first ten non-SESE jobs with at least one positive valid span for every task',
        'duration':'arithmetic mean of positive finite observed wall spans for each task, seconds; not CPU service time',
        'edge_duration':'local duration / 3, synthetic speedup',
        'communication':'constant mean((tl+ts)/2), synthetic CCR=1',
        'pinning':'all original sources and sinks, no dummy nodes, all original nodes and edges preserved',
        'group':'secondary multi-terminal coverage extension, not original prespecified SESE primary group'
    },
    'publication':'safe summaries only counts, candidate job IDs, graph/source/file hashes and assumptions; raw mappings and instances local only'
}


def dump_new(path,obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(obj,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def hashed(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def historical_projection(names):
    discarded=collections.Counter()
    retained=[]
    for name in names:
        if HISTORICAL_FILTER.search(name):
            discarded['MergeTask' if 'MergeTask' in name else 'task']+=1
        else:
            retained.append(name)
    info=dict(discarded=dict(discarded),retained_names=retained,unknown_names=[s for s in retained if NUMERIC.fullmatch(s) is None])
    if not retained:
        return {**info,'status':'empty'}
    if info['unknown_names']:
        return {**info,'status':'unknown-retained-name'}
    definitions=collections.defaultdict(list)
    arcs=set()
    for name in retained:
        tokens=name.replace(name[0],'').split('_')
        key=tokens[0]
        definitions[key].append(name)
        arcs.update((parent,key) for parent in tokens[1:])
    keys=sorted(set(definitions)|{p for p,c in arcs})
    index={key:i for i,key in enumerate(keys)}
    edges=sorted((index[a],index[b]) for a,b in arcs)
    pre,succ=[set() for _ in keys],[set() for _ in keys]
    for a,b in edges:
        succ[a].add(b)
        pre[b].add(a)
    indegree=[len(ps) for ps in pre]
    ready=collections.deque(i for i,d in enumerate(indegree) if not d)
    order=[]
    while ready:
        i=ready.popleft()
        order.append(i)
        for j in sorted(succ[i]):
            indegree[j]-=1
            if not indegree[j]:
                ready.append(j)
    seen,pending=set(),([0] if keys else [])
    while pending:
        i=pending.pop()
        if i not in seen:
            seen.add(i)
            pending.extend(pre[i]|succ[i])
    missing=sorted({p for p,c in arcs if p not in definitions})
    collapsed={key:values for key,values in definitions.items() if len(values)>1}
    self_edges=sorted(a for a,b in arcs if a==b)
    return {**info,'status':'parsed','keys':keys,'edges':edges,'topo':order,
            'missing_parents':missing,'collapsed_ids':collapsed,'self_edges':self_edges,
            'acyclic':len(order)==len(keys),'connected':len(seen)==len(keys),
            'sources':[i for i,ps in enumerate(pre) if not ps],
            'sinks':[i for i,cs in enumerate(succ) if not cs],
            'strict_valid':not missing and not collapsed and len(order)==len(keys),
            'labeled_graph_sha256':hashed(dict(keys=keys,edges=edges))}


def secondary_coverage(db,out,source_sha):
    counts=collections.Counter({key:0 for key in ('whole_jobs_with_80_to_100_distinct_task_names',
        'numeric_connected_valid_dags','sese_valid','non_sese_valid','sese_missing_positive_span',
        'non_sese_missing_positive_span','sese_all_tasks_positive_span','non_sese_all_tasks_positive_span')})
    selected=[]
    query='SELECT job,COUNT(*) FROM t GROUP BY job HAVING COUNT(*) BETWEEN 80 AND 100 ORDER BY job'
    for job,count in db.execute(query):
        counts['whole_jobs_with_80_to_100_distinct_task_names']+=1
        rows=list(db.execute('SELECT name,n,total,valid FROM t WHERE job=? ORDER BY name',(job,)))
        try:
            g=whole_graph([r[0] for r in rows])
        except ValueError as error:
            counts['rejected:'+str(error).split(':')[0]]+=1
            continue
        if not g['numeric_only']:
            counts['rejected:nonnumeric']+=1
            continue
        if not g['connected']:
            counts['rejected:disconnected']+=1
            continue
        n=len(g['names'])
        if not 80<=n<=100:
            raise RuntimeError('whole-parser task count mismatch')
        counts['numeric_connected_valid_dags']+=1
        sese=len(g['sources'])==1 and len(g['sinks'])==1
        label='sese' if sese else 'non_sese'
        counts[label+'_valid']+=1
        spans={r[0]:r[2]/r[3] for r in rows if r[3] and math.isfinite(r[2]/r[3]) and r[2]/r[3]>0}
        if len(spans)!=n:
            counts[label+'_missing_positive_span']+=1
            continue
        counts[label+'_all_tasks_positive_span']+=1
        if sese or len(selected)>=10:
            continue
        tl=[spans[name] for name in g['names']]
        ts=[value/3 for value in tl]
        delay=sum((a+b)/2 for a,b in zip(tl,ts))/n
        identifier='trace-secondary-large-'+quote(job,safe='-_')
        data=dict(identifier=identifier,tl=tl,ts=ts,edges=[(a,b,delay) for a,b in g['edges']],
                  pinned=sorted(set(g['sources']+g['sinks'])),
                  metadata=dict(kind='new-trace-derived-secondary-multi-terminal',group='secondary_large_nonSESE',
                                job_name=job,original_task_names=g['names'],official_archive_verified=True,
                                source_sha256=source_sha,source_rows=sum(r[1] for r in rows),
                                duration_source='mean positive observed batch-task wall span (seconds), not CPU service time',
                                resource_speedup=3,speedup_is_synthetic=True,communication_is_synthetic=True,ccr=1,
                                source_count=len(g['sources']),sink_count=len(g['sinks']),
                                selection='lex first ten duration-eligible numeric weakly connected non-SESE whole jobs, N=80..100',
                                original_SESE_primary_group=False,not_a_recovered_historical_input=True,no_dummy_nodes=True))
        path=out/'secondary_large_instances'/f'{identifier}.json'
        dump_new(path,data)
        selected.append(dict(identifier=identifier,job_name=job,n=n,edges=len(g['edges']),
                             roots=len(g['sources']),sinks=len(g['sinks']),sha256=sha(path),
                             graph_sha256=hashed(dict(keys=g['keys'],edges=g['edges']))))
    report=dict(coverage=dict(counts),selected=selected,selected_count=len(selected),requested_secondary_count=10,
                primary_group_unchanged=True,secondary_selection_complete=len(selected)==10)
    dump_new(out/'secondary_large_coverage.json',report)
    print('SECONDARY_LARGE_READY',json.dumps(report,ensure_ascii=False),flush=True)
    return report


def search(db,target_edges,out):
    funnel=collections.Counter({key:0 for key in ('jobs_all','aggregated_unique_task_names','raw_rows_represented',
        'excluded_names:task','excluded_names:MergeTask','jobs_with_excluded_names','retained_unique_task_names',
        'projection:empty','projection:unknown-retained-name','unknown_retained_names','projection_parsed_jobs',
        'projection_missing_parent_jobs','projection_missing_parent_identifiers','projection_collapsed_identifier_jobs',
        'projection_self_edge_jobs','projection_cyclic_jobs','projection_disconnected_jobs','strict_valid_projection_jobs',
        'strict_connected_projection_jobs','strict_connected_SESE_projection_jobs','literal_ten_node_projection_jobs',
        'literal_acyclic_ten_node_projection_jobs','literal_isomorphic_jobs','strict_isomorphic_jobs',
        'isomorphic_jobs_with_filtered_names')})
    whole_counts=collections.Counter(dict(valid_acyclic_jobs=0,valid_acyclic_ten_node_jobs=0,isomorphic_jobs=0))
    safe_candidates=[]
    current,names,counts=None,[],[]
    candidate_path=out/'projection_candidates_full_local.jsonl'
    def process(job,names,counts):
        if job is None:
            return
        funnel['jobs_all']+=1
        funnel['aggregated_unique_task_names']+=len(names)
        funnel['raw_rows_represented']+=sum(counts)
        whole_match=False
        try:
            whole=whole_graph(names)
            whole_counts['valid_acyclic_jobs']+=1
            if len(whole['names'])==10:
                whole_counts['valid_acyclic_ten_node_jobs']+=1
                whole_match=isomorphism(10,target_edges,whole['edges']) is not None
                if whole_match:
                    whole_counts['isomorphic_jobs']+=1
        except ValueError as error:
            whole_counts['rejected:'+str(error).split(':')[0]]+=1
        projected=historical_projection(names)
        for kind,value in projected['discarded'].items():
            funnel['excluded_names:'+kind]+=value
        if projected['discarded']:
            funnel['jobs_with_excluded_names']+=1
        funnel['retained_unique_task_names']+=len(projected['retained_names'])
        if projected['status']!='parsed':
            funnel['projection:'+projected['status']]+=1
            funnel['unknown_retained_names']+=len(projected['unknown_names'])
            return
        funnel['projection_parsed_jobs']+=1
        if projected['missing_parents']:
            funnel['projection_missing_parent_jobs']+=1
            funnel['projection_missing_parent_identifiers']+=len(projected['missing_parents'])
        if projected['collapsed_ids']:
            funnel['projection_collapsed_identifier_jobs']+=1
        if projected['self_edges']:
            funnel['projection_self_edge_jobs']+=1
        if not projected['acyclic']:
            funnel['projection_cyclic_jobs']+=1
        if not projected['connected']:
            funnel['projection_disconnected_jobs']+=1
        if projected['strict_valid']:
            funnel['strict_valid_projection_jobs']+=1
            if projected['connected']:
                funnel['strict_connected_projection_jobs']+=1
                if len(projected['sources'])==1 and len(projected['sinks'])==1:
                    funnel['strict_connected_SESE_projection_jobs']+=1
        n=len(projected['keys'])
        if n!=10:
            return
        funnel['literal_ten_node_projection_jobs']+=1
        if not projected['acyclic']:
            return
        funnel['literal_acyclic_ten_node_projection_jobs']+=1
        mapping=isomorphism(10,target_edges,projected['edges'])
        if mapping is None:
            return
        funnel['literal_isomorphic_jobs']+=1
        if projected['strict_valid']:
            funnel['strict_isomorphic_jobs']+=1
        if projected['discarded']:
            funnel['isomorphic_jobs_with_filtered_names']+=1
        full=dict(job_name=job,whole_job_unique_names=len(names),retained_unique_names=len(projected['retained_names']),
                  excluded_name_counts=projected['discarded'],strict_valid=projected['strict_valid'],whole_job_isomorphic=whole_match,
                  projected_keys=projected['keys'],projected_edges=projected['edges'],
                  retained_names=projected['retained_names'],missing_parents=projected['missing_parents'],collapsed_ids=projected['collapsed_ids'],
                  archive_node_to_projected_index=mapping,
                  archive_node_to_projected_key={i:projected['keys'][j] for i,j in mapping.items()},
                  labeled_graph_sha256=projected['labeled_graph_sha256'],historical_identity_proven=False,
                  archived_timing_or_edge_weight_provenance_proven=False)
        with candidate_path.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(full,ensure_ascii=False,sort_keys=True)+'\n')
        safe_candidates.append({key:full[key] for key in ('job_name','labeled_graph_sha256','strict_valid','whole_job_isomorphic','historical_identity_proven','archived_timing_or_edge_weight_provenance_proven')})
        safe_candidates[-1].update(projected_nodes=n,whole_job_unique_names=len(names),retained_unique_names=len(projected['retained_names']),
                                   excluded_name_count=sum(projected['discarded'].values()),
                                   missing_parent_count=len(projected['missing_parents']),collapsed_identifier_count=len(projected['collapsed_ids']))
    for job,name,count in db.execute('SELECT job,name,n FROM t ORDER BY job,name'):
        if current is not None and job!=current:
            process(current,names,counts)
            names,counts=[],[]
            if funnel['jobs_all']%200000==0:
                print('PROJECTION_JOBS',funnel['jobs_all'],'MATCHES',funnel['strict_isomorphic_jobs'],flush=True)
        current=job
        names.append(name)
        counts.append(count)
    process(current,names,counts)
    if not candidate_path.exists():
        candidate_path.touch()
    return dict(funnel=dict(funnel),whole_job_funnel=dict(whole_counts),safe_candidates=safe_candidates,
                candidate_mapping_file_sha256=sha(candidate_path),candidate_mapping_file_local_only=True,
                historical_job_identity_proven=False,archived_tl_ts_or_communication_origin_proven=False)


def run(repo,extraction,out,safe):
    repo,extraction,out,safe=map(Path,(repo,extraction,out,safe))
    if out.exists():
        raise FileExistsError('fresh local audit output required')
    report=json.loads((extraction/'extraction_report.json').read_text())
    if report['source_sha256']!=OFFICIAL_SHA256 or report['official_sha256_verified'] is not True:
        raise RuntimeError('official completed extraction report required')
    out.mkdir(parents=True)
    dump_new(out/'projection_frozen_protocol.json',PROTOCOL)
    oldarchive=repo/'代码 数据等.zip'
    with ZipFile(oldarchive) as archive:
        old_scripts={name:hashlib.sha256(archive.read(name)).hexdigest() for name in ('代码/DAG(test)/data.py','代码/DAG(test)/dataset.py')}
    with ZipFile(repo/'supplement/results/20261108_v1/raw/quality_dataset.zip') as archived_inputs:
        target=json.loads(archived_inputs.read('instances/archive10.json'))
    target_edges=[(a,b) for a,b,c in target['edges']]
    dbpath=(extraction/'task_aggregation.sqlite').resolve()
    db=sqlite3.connect('file:'+quote(str(dbpath))+'?mode=ro&immutable=1',uri=True)
    db.execute('PRAGMA query_only=ON')
    start=time.perf_counter()
    coverage=secondary_coverage(db,out,report['source_sha256'])
    coverage_safe={**coverage,'source_sha256':report['source_sha256'],'protocol_sha256':hashed(PROTOCOL),
                   'publication_boundary':'safe counts and identifiers/hashes only; no duration/row fields'}
    dump_new(safe/'projection_secondary_large_coverage.json',coverage_safe)
    search_report=search(db,target_edges,out)
    db.close()
    meta=dict(**search_report,secondary_large_coverage=coverage,source_sha256=report['source_sha256'],official_source_verified=True,
              protocol=PROTOCOL,protocol_sha256=hashed(PROTOCOL),archived_scripts_sha256=old_scripts,
              archive10_unweighted_graph_sha256=hashed(dict(n=10,edges=target_edges)),
              target_preexisting_input_sha256=hashed(target),projection_code_sha256=sha(Path(__file__)),
              main_parser_code_sha256=sha(Path(__file__).with_name('parser.py')),
              original_archive_sha256=sha(oldarchive),elapsed_seconds=time.perf_counter()-start,
              raw_trace_fields_in_safe_report=False)
    dump_new(out/'projection_audit_report.json',meta)
    dump_new(safe/'projection_audit_summary.json',meta)
    print('PROJECTION_AUDIT_COMPLETE',json.dumps({k:meta[k] for k in ('funnel','whole_job_funnel','elapsed_seconds')},ensure_ascii=False),flush=True)
    return meta


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--repo',default='.')
    p.add_argument('--extraction',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--safe',required=True)
    a=p.parse_args()
    run(a.repo,a.extraction,a.out,a.safe)
