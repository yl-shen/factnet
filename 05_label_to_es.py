#!/usr/bin/env python3
"""
nohup python 05_label_to_es.py \
--parquet-dir 01_factstatement \
--parquet-file labels_part_1.parquet \
--es-index-prefix factnet \
--batch-size 2000 \
--chunk-size 2000 \
--parallel-workers 32 \
--progress-dir 04_index_statement_labels/progress \
--log-dir 04_index_statement_labels/logs \
> 04_index_statement_labels/runing.log &
"""

import argparse
import json
import logging
import os
import sys
import time
import urllib3
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

# third-party
try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except Exception as e:
    print("请先安装依赖：pip install pyarrow")
    raise

from concurrent.futures import ProcessPoolExecutor, as_completed
from elasticsearch import Elasticsearch, helpers
from es_config import ES_IP_LIST, ES_PORT, ES_USER, ES_PASSWARD
import ssl
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ---------------- logging ----------------
def setup_logging(log_dir: Path, file_name: str, name: str = "es_indexer") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove old handlers
    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(ch)

    fh = logging.FileHandler(str(log_dir / f"{name}_{file_name}.log"), encoding='utf-8')
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    return logger


# ---------------- small helpers ----------------
def sha1_short(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:12]


def load_json_safe(p: Path) -> Dict:
    if not p.exists():
        return {}
    try:
        with p.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_json_atomic(p: Path, data: Dict):
    tmp = p.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


# ---------------- ES factory ----------------
class ElasticFactory:
    def __init__(self, host: list, port: str, username: str, password: str, maxsize: int = 25):
        self.port = port
        self.host = host
        self.username = username
        self.password = password
        self.maxsize = maxsize

    def create(self) -> Elasticsearch:
        context = ssl._create_unverified_context()
        addrs = []
        for h in self.host:
            addrs.append({'host': h, 'port': self.port})

        if self.username and self.password:
            es = Elasticsearch(
                addrs,
                http_auth=(self.username, self.password),
                scheme="https",
                ssl_context=context,
                sniff_on_start=False,
                sniff_on_connection_fail=False,
                sniffer_timeout=30,
                maxsize=self.maxsize
            )
        else:
            es = Elasticsearch(addrs, maxsize=self.maxsize)
        return es


# ---------------- index creation (keep yours) ----------------
def create_indices_if_missing(es: Elasticsearch, prefix: str, logger: logging.Logger):
    labels_index = f"{prefix}_labels_v1"
    facts_index = f"{prefix}_factstatements_v1"

    settings = {
        "settings": {
            "analysis": {
                "filter": {
                    "edge_ngram_filter": {"type": "edge_ngram", "min_gram": 2, "max_gram": 20}
                },
                "analyzer": {
                    "edge_ngram_analyzer": {"type": "custom", "tokenizer": "standard",
                                           "filter": ["lowercase", "edge_ngram_filter"]},
                    "lowercase_analyzer": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase"]}
                }
            },
            "index": {"number_of_shards": 30, "number_of_replicas": 1}
        }
    }

    labels_mapping = {
        **settings,
        "mappings": {
            "properties": {
                "subject_qid": {"type": "keyword"},
                "language": {"type": "keyword"},
                "label": {"type": "text", "analyzer": "lowercase_analyzer",
                          "fields": {"keyword": {"type": "keyword"}, "ngram": {"type": "text", "analyzer": "edge_ngram_analyzer"}}},
                "label_keyword": {"type": "keyword"},
                "label_suggest": {"type": "completion"},
                "aliases": {"type": "text", "analyzer": "lowercase_analyzer"}
            }
        }
    }

    facts_mapping = {
        **settings,
        "mappings": {
            "properties": {
                "core_id": {"type": "keyword"},
                "statement_id": {"type": "keyword"},
                "subject_qid": {"type": "keyword"},
                "property_pid": {"type": "keyword"},
                "value": {"type": "text", "analyzer": "lowercase_analyzer", "fields": {"keyword": {"type": "keyword"}, "ngram": {"type": "text", "analyzer": "edge_ngram_analyzer"}}},
                "value_keyword": {"type": "keyword"},
                "rank": {"type": "keyword"},
                "claim_hash": {"type": "keyword"},
                "claim_hash_prefix": {"type": "keyword"},
                "subject_prefix": {"type": "keyword"},
                "confidence": {"type": "float"},
                "provenance": {"type": "object", "enabled": False},
                "page_title": {"type": "text", "analyzer": "lowercase_analyzer", "fields": {"keyword": {"type": "keyword"}}}
            }
        }
    }

    if not es.indices.exists(index=labels_index):
        try:
            es.indices.create(index=labels_index, body=labels_mapping)
            logger.info(f"Created index: {labels_index}")
        except Exception as e:
            logger.error(f"Failed to create index {labels_index}: {e}")

    if not es.indices.exists(index=facts_index):
        try:
            es.indices.create(index=facts_index, body=facts_mapping)
            logger.info(f"Created index: {facts_index}")
        except Exception as e:
            logger.error(f"Failed to create index {facts_index}: {e}")

    return labels_index, facts_index


# ---------------- document preparation ----------------
def prepare_label_docs(records: List[Dict]) -> List[Dict]:
    docs = []
    for r in records:
        subject = r.get('subject_qid') or r.get('subject') or ''
        lang = r.get('language') or r.get('lang') or 'und'
        label = None
        if 'label' in r:
            lf = r['label']
            if isinstance(lf, (list, tuple)) and lf:
                label = str(lf[0])
            elif lf is not None:
                label = str(lf)
        if not label or not subject:
            continue
        aliases = []
        if 'aliases' in r and r['aliases'] is not None:
            if isinstance(r['aliases'], (list, tuple)):
                aliases = [str(a) for a in r['aliases'] if a is not None]
            else:
                try:
                    alias_str = str(r['aliases'])
                    if alias_str:
                        aliases = [alias_str]
                except:
                    pass
        doc_id = f"{subject}_{lang}_{sha1_short(label)}"
        src = {
            'subject_qid': subject,
            'language': lang,
            'label': label,
            'label_keyword': label,
            'aliases': aliases,
            'label_suggest': {'input': [label] + aliases, 'weight': 1}
        }
        docs.append({'_id': doc_id, '_source': src})
    return docs


def prepare_factstatement_docs(records: List[Dict]) -> List[Dict]:
    docs = []
    for r in records:
        core_id = r.get('core_id') or r.get('id') or r.get('statement_id') or sha1_short(json.dumps(r, ensure_ascii=False))
        subject = r.get('subject_qid')
        prop = r.get('property_pid')
        if not (core_id and subject and prop):
            continue
        value = r.get('value')
        value_str = None
        if isinstance(value, (list, tuple)) and value:
            value_str = str(value[0])
        elif value is not None:
            try:
                value_str = str(value)
            except:
                pass
        claim_hash = r.get('claim_hash') or sha1_short(f"{subject}|{prop}|{value_str}")
        claim_hash_prefix = claim_hash[:4] if claim_hash else None
        subject_prefix = subject[:4] if subject else None
        src = {
            'core_id': core_id,
            'statement_id': r.get('statement_id', core_id),
            'subject_qid': subject,
            'property_pid': prop,
            'value': value_str,
            'value_keyword': value_str,
            'rank': r.get('rank', 'normal'),
            'claim_hash': claim_hash,
            'claim_hash_prefix': claim_hash_prefix,
            'subject_prefix': subject_prefix,
            'confidence': float(r.get('confidence', 1.0)),
            'provenance': r.get('provenance', {})
        }
        if 'page_title' in r:
            src['page_title'] = r['page_title']
        docs.append({'_id': core_id, '_source': src})
    return docs


# ---------------- bulk index (worker-side) ----------------
def bulk_index_docs(es, index_name: str, docs: List[Dict], chunk_size: int = 500, logger=None, skip_existing: bool = False, op_type_create_if_skip: bool = True) -> Tuple[int, int]:
    """
    Returns (success_count, failed_count)
    If skip_existing=True, does an mget pre-check (expensive).
    If skip_existing=False, uses op_type 'index' (overwrite). If you prefer create-only, set op_type_create_if_skip accordingly.
    """
    if not docs:
        return 0, 0

    docs_to_index = docs

    if skip_existing:
        # mget check in chunks to avoid huge request
        ids = [d['_id'] for d in docs]
        existing_ids = set()
        try:
            chunk = 1000
            for i in range(0, len(ids), chunk):
                chunk_ids = ids[i:i + chunk]
                resp = es.mget(body={'ids': chunk_ids}, index=index_name)
                for doc in resp.get('docs', []):
                    if doc.get('found'):
                        existing_ids.add(doc.get('_id'))
        except Exception:
            existing_ids = set()
        docs_to_index = [d for d in docs if d['_id'] not in existing_ids]

    if not docs_to_index:
        return 0, 0

    if skip_existing and op_type_create_if_skip:
        op_type = 'create'  # will fail if exists
    else:
        op_type = 'index'   # overwrite / upsert

    actions = ({
        '_op_type': op_type,
        '_index': index_name,
        '_id': d['_id'],
        '_source': d['_source']
    } for d in docs_to_index)

    success = 0
    failed = 0
    error_counter = Counter()

    # streaming_bulk is memory-efficient
    for ok, resp in helpers.streaming_bulk(
            es,
            actions,
            chunk_size=chunk_size,
            max_retries=3,
            raise_on_error=False,
            request_timeout=120
    ):
        if ok:
            success += 1
        else:
            failed += 1
            # try to extract error
            try:
                err = resp.get('create', {}) if isinstance(resp, dict) else resp
                err_type = err.get('error', {}).get('type') or str(err)
            except Exception:
                err_type = str(resp)
            error_counter[err_type] += 1

    if logger and error_counter:
        logger.info(f"Bulk errors summary (top10): {error_counter.most_common(10)}")

    return success, failed


# ---------------- worker function (top-level so picklable) ----------------
def process_batch_worker(batch_tuple: Tuple[int, List[Dict], str], es_settings: Dict, index_name: str, chunk_size: int, skip_existing: bool, maxsize: int) -> Dict:
    batch_idx, records, doc_type = batch_tuple
    try:
        # Each worker creates its own ES client
        es = ElasticFactory(es_settings['hosts'], es_settings['port'], es_settings['user'], es_settings['password'], maxsize=maxsize).create()

        if doc_type == 'label':
            docs = prepare_label_docs(records)
        else:
            docs = prepare_factstatement_docs(records)

        if not docs:
            return {'batch_idx': batch_idx, 'status': 'success', 'records_processed': len(records), 'docs_prepared': 0, 'success_count': 0, 'error_count': 0}

        success_count, failed_count = bulk_index_docs(es, index_name, docs, chunk_size=chunk_size, logger=None, skip_existing=skip_existing)
        return {'batch_idx': batch_idx, 'status': 'success', 'records_processed': len(records), 'docs_prepared': len(docs), 'success_count': success_count, 'error_count': failed_count}
    except Exception as e:
        return {'batch_idx': batch_idx, 'status': 'error', 'error': str(e)}


# ---------------- main processing loop (streaming submit) ----------------
def process_parquet_file_parallel(file_path: Path, index_name: str, doc_type: str, batch_size: int, progress_dir: Path, logger: logging.Logger, workers: int = 4, chunk_size: int = 500, skip_existing: bool = False, maxsize: int = 25):
    progress_file = progress_dir / (file_path.name + '.progress.json')
    progress = load_json_safe(progress_file)
    processed_rows = progress.get('processed_rows', 0)

    try:
        es = ElasticFactory(ES_IP_LIST, ES_PORT, ES_USER, ES_PASSWARD, maxsize=maxsize).create()
        try:
            es.info()
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            return {'file': str(file_path), 'status': 'error', 'error': f"ES connection failed: {e}", 'processed_rows': processed_rows}

        pf = pq.ParquetFile(file_path)
        total_rows = pf.metadata.num_rows
        logger.info(f"File {file_path.name} rows={total_rows}")

        if progress.get('status') == 'success' and processed_rows >= total_rows:
            logger.info(f"{file_path.name} already processed - skipping")
            return progress

        # Stream batches from parquet and submit to process pool on-the-fly to avoid storing all batches
        es_settings = {'hosts': ES_IP_LIST, 'port': ES_PORT, 'user': ES_USER, 'password': ES_PASSWARD}

        max_outstanding = max(2, workers * 2)
        futures = []
        results = []
        batch_idx = 0
        total_success = progress.get('success_count', 0)
        total_error = progress.get('error_count', 0)
        outstanding = 0

        logger.info(f"Start processing with workers={workers}, batch_size={batch_size}, chunk_size={chunk_size}, skip_existing={skip_existing}")

        with ProcessPoolExecutor(max_workers=workers) as executor:
            batch_iter = pf.iter_batches(batch_size=batch_size)
            # submit loop
            for batch in batch_iter:
                # convert batch (RecordBatch) to python list of dicts without going through pandas
                tbl = pa.Table.from_batches([batch])
                records = tbl.to_pylist()
                batch_tuple = (batch_idx, records, doc_type)

                future = executor.submit(process_batch_worker, batch_tuple, es_settings, index_name, chunk_size, skip_existing, maxsize)
                futures.append(future)
                outstanding += 1
                batch_idx += 1

                # If too many outstanding futures, wait for at least one to finish
                if outstanding >= max_outstanding:
                    # wait for any to complete
                    done, not_done = [], []
                    for f in as_completed(futures, timeout=None):
                        # process the first completed one and break
                        res = f.result()
                        results.append(res)
                        if res.get('status') == 'success':
                            processed_rows += res.get('records_processed', 0)
                            total_success += res.get('success_count', 0)
                            total_error += res.get('error_count', 0)
                        else:
                            logger.error(f"Batch {res.get('batch_idx')} failed: {res.get('error')}")
                        futures.remove(f)
                        outstanding -= 1
                        break  # handle one completed future then continue submitting

            # after all batches submitted, collect remaining futures
            for f in tqdm(as_completed(futures), total=len(futures), desc="Finishing batches"):
                try:
                    res = f.result()
                    results.append(res)
                    if res.get('status') == 'success':
                        processed_rows += res.get('records_processed', 0)
                        total_success += res.get('success_count', 0)
                        total_error += res.get('error_count', 0)
                    else:
                        logger.error(f"Batch {res.get('batch_idx')} failed: {res.get('error')}")
                except Exception as e:
                    logger.exception(f"Future exception: {e}")

        # finalize progress
        final_status = 'success' if total_error == 0 else 'in_progress'
        progress.update({
            'processed_rows': processed_rows,
            'last_updated': time.time(),
            'status': final_status,
            'total_rows': total_rows,
            'success_count': total_success,
            'error_count': total_error
        })
        save_json_atomic(progress_file, progress)
        logger.info(f"Completed {file_path.name}: success={total_success}, error={total_error}")

        try:
            es.indices.refresh(index=index_name)
        except Exception:
            pass

        return {'file': str(file_path), 'status': final_status, 'processed_rows': processed_rows, 'total_rows': total_rows, 'success_count': total_success, 'error_count': total_error}

    except Exception as e:
        logger.exception(f"Failed processing {file_path.name}: {e}")
        progress.update({'processed_rows': processed_rows, 'last_updated': time.time(), 'status': 'error', 'error': str(e)})
        save_json_atomic(progress_file, progress)
        return {'file': str(file_path), 'status': 'error', 'error': str(e), 'processed_rows': processed_rows}


# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description='Faster indexer for parquet -> Elasticsearch')
    parser.add_argument('--parquet-dir', required=True, type=Path)
    parser.add_argument('--parquet-file', required=True, type=str)
    parser.add_argument('--es-index-prefix', default='factnet')
    parser.add_argument('--batch-size', type=int, default=2000, help='pyarrow batch size (rows per batch read)')
    parser.add_argument('--parallel-workers', type=int, default=max(1, (os.cpu_count() or 2) // 2), help='number of processes')
    parser.add_argument('--progress-dir', type=Path, default=Path('./progress'))
    parser.add_argument('--log-dir', type=Path, default=Path('./logs'))
    parser.add_argument('--recreate-indices', action='store_true')
    parser.add_argument('--chunk-size', type=int, default=500, help='ES bulk chunk size')
    parser.add_argument('--skip-existing', action='store_true', help='If set, do an mget pre-check to skip existing docs (slower)')
    parser.add_argument('--es-connection-pool', type=int, default=16, help='ES client maxsize (connection pool per process)')
    args = parser.parse_args()

    args.progress_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(args.log_dir, args.parquet_file, "es_indexer_fast")

    es = ElasticFactory(ES_IP_LIST, ES_PORT, ES_USER, ES_PASSWARD, maxsize=args.es_connection_pool).create()
    labels_index, facts_index = create_indices_if_missing(es, args.es_index_prefix, logger)

    if args.recreate_indices:
        # handle user typo
        args.recreate_indices = True

    if args.recreate_indices:
        try:
            if es.indices.exists(index=labels_index):
                es.indices.delete(index=labels_index)
                logger.info(f"Deleted {labels_index}")
            if es.indices.exists(index=facts_index):
                es.indices.delete(index=facts_index)
                logger.info(f"Deleted {facts_index}")
            labels_index, facts_index = create_indices_if_missing(es, args.es_index_prefix, logger)
        except Exception as e:
            logger.error(f"Failed to recreate indices: {e}")

    parquet_file = args.parquet_dir / args.parquet_file
    if not parquet_file.exists():
        logger.error(f"Parquet file not found: {parquet_file}")
        return 1

    if "labels" in args.parquet_file.lower():
        doc_type = "label"
        index_name = labels_index
    elif "factstatements" in args.parquet_file.lower() or "fact" in args.parquet_file.lower():
        doc_type = "fact"
        index_name = facts_index
    else:
        logger.error("Unknown file type: filename must contain 'labels' or 'factstatements' / 'fact'")
        return 1

    result = process_parquet_file_parallel(
        parquet_file,
        index_name,
        doc_type,
        args.batch_size,
        args.progress_dir,
        logger,
        workers=args.parallel_workers,
        chunk_size=args.chunk_size,
        skip_existing=args.skip_existing,
        maxsize=args.es_connection_pool
    )

    if result.get('status') == 'success':
        logger.info(f"Done: {parquet_file.name} rows={result.get('processed_rows')} success={result.get('success_count')}")
        return 0
    else:
        logger.error(f"Processing failed: {result.get('error')}")
        return 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print("Fatal:", e)
        raise