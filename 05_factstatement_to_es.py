#!/usr/bin/env python3
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
def setup_logging(log_dir: Path, name: str = "factstatement_indexer") -> logging.Logger:
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

    fh = logging.FileHandler(str(log_dir / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log"), encoding='utf-8')
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    return logger


# ---------------- helpers ----------------
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


# ---------------- index creation ----------------
def create_factstatement_index(es: Elasticsearch, index_name: str, logger: logging.Logger):
    """Create FactStatement ES index with optimized mapping"""
    if es.indices.exists(index=index_name):
        logger.info(f"Index {index_name} already exists")
        return
    
    settings = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "lowercase_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase"]
                    }
                }
            },
            "index": {
                "number_of_shards": 30,
                "number_of_replicas": 1,
                "refresh_interval": "60s"
            }
        },
        "mappings": {
            "properties": {
                "core_id": {"type": "keyword"},
                "subject_qid": {"type": "keyword"},
                "property_pid": {"type": "keyword"},
                "value": {"type": "text", "analyzer": "lowercase_analyzer", 
                         "fields": {"keyword": {"type": "keyword"}}},
                "rank": {"type": "keyword"},
                "claim_hash": {"type": "keyword"},
                "claim_hash_prefix": {"type": "keyword"},
                "subject_prefix": {"type": "keyword"},
                "confidence": {"type": "float"},
                # Large fields stored but not indexed for search
                "qualifiers": {"type": "object", "enabled": False},
                "references": {"type": "object", "enabled": False},
                "sitelinks": {"type": "object", "enabled": False},
                "provenance": {"type": "object", "enabled": False},
                "labels_present": {"type": "keyword"},
                "last_edit": {"type": "date", "format": "strict_date_optional_time||epoch_millis"}
            }
        }
    }
    
    try:
        es.indices.create(index=index_name, body=settings)
        logger.info(f"Created index: {index_name}")
    except Exception as e:
        logger.error(f"Failed to create index {index_name}: {e}")
        raise


# ---------------- document preparation ----------------
def prepare_factstatement_docs(records: List[Dict]) -> List[Dict]:
    """Prepare FactStatement documents for ES indexing"""
    docs = []
    for r in records:
        try:
            core_id = r.get('core_id')
            if not core_id:
                continue
            
            # Parse JSON fields if they are strings
            qualifiers = r.get('qualifiers')
            if isinstance(qualifiers, str):
                try:
                    qualifiers = json.loads(qualifiers)
                except:
                    qualifiers = {}
            
            references = r.get('references')
            if isinstance(references, str):
                try:
                    references = json.loads(references)
                except:
                    references = []
            
            sitelinks = r.get('sitelinks')
            if isinstance(sitelinks, str):
                try:
                    sitelinks = json.loads(sitelinks)
                except:
                    sitelinks = {}
            
            provenance = r.get('provenance')
            if isinstance(provenance, str):
                try:
                    provenance = json.loads(provenance)
                except:
                    provenance = {}
            
            labels_present = r.get('labels_present')
            if isinstance(labels_present, str):
                try:
                    labels_present = json.loads(labels_present)
                except:
                    labels_present = []
            
            # Get value as string
            value = r.get('value')
            if isinstance(value, str):
                value_str = value
            else:
                try:
                    value_str = json.dumps(value, ensure_ascii=False) if value else None
                except:
                    value_str = str(value) if value else None
            
            src = {
                'core_id': core_id,
                'subject_qid': r.get('subject_qid'),
                'property_pid': r.get('property_pid'),
                'value': value_str,
                'rank': r.get('rank', 'normal'),
                'claim_hash': r.get('claim_hash'),
                'claim_hash_prefix': r.get('claim_hash_prefix'),
                'subject_prefix': r.get('subject_prefix'),
                'confidence': float(r.get('confidence', 0.7)),
                'qualifiers': qualifiers,
                'references': references,
                'sitelinks': sitelinks,
                'provenance': provenance,
                'labels_present': labels_present if isinstance(labels_present, list) else [],
                'last_edit': r.get('last_edit')
            }
            
            docs.append({'_id': core_id, '_source': src})
            
        except Exception as e:
            # Skip problematic records
            continue
    
    return docs


# ---------------- bulk index ----------------
def bulk_index_docs(es, index_name: str, docs: List[Dict], chunk_size: int = 500, 
                    logger=None, skip_existing: bool = False) -> Tuple[int, int]:
    """Bulk index documents to ES"""
    if not docs:
        return 0, 0

    docs_to_index = docs

    if skip_existing:
        # Check existing documents
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

    actions = ({
        '_op_type': 'index',
        '_index': index_name,
        '_id': d['_id'],
        '_source': d['_source']
    } for d in docs_to_index)

    success = 0
    failed = 0
    error_counter = Counter()

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
            try:
                err = resp.get('index', {}) if isinstance(resp, dict) else resp
                err_type = err.get('error', {}).get('type') or str(err)
            except Exception:
                err_type = str(resp)
            error_counter[err_type] += 1

    if logger and error_counter:
        logger.info(f"Bulk errors summary (top10): {error_counter.most_common(10)}")

    return success, failed


# ---------------- worker function ----------------
def process_batch_worker(batch_tuple: Tuple[int, List[Dict]], es_settings: Dict, 
                        index_name: str, chunk_size: int, skip_existing: bool, 
                        maxsize: int) -> Dict:
    """Worker function to process a batch of records"""
    batch_idx, records = batch_tuple
    try:
        # Create ES client for this worker
        es = ElasticFactory(es_settings['hosts'], es_settings['port'], 
                           es_settings['user'], es_settings['password'], 
                           maxsize=maxsize).create()

        docs = prepare_factstatement_docs(records)

        if not docs:
            return {
                'batch_idx': batch_idx,
                'status': 'success',
                'records_processed': len(records),
                'docs_prepared': 0,
                'success_count': 0,
                'error_count': 0
            }

        success_count, failed_count = bulk_index_docs(
            es, index_name, docs, chunk_size=chunk_size, 
            logger=None, skip_existing=skip_existing
        )
        
        return {
            'batch_idx': batch_idx,
            'status': 'success',
            'records_processed': len(records),
            'docs_prepared': len(docs),
            'success_count': success_count,
            'error_count': failed_count
        }
    except Exception as e:
        return {
            'batch_idx': batch_idx,
            'status': 'error',
            'error': str(e)
        }


# ---------------- main processing ----------------
def process_parquet_files(parquet_dir: Path, index_name: str, batch_size: int, 
                         progress_dir: Path, logger: logging.Logger, 
                         workers: int = 32, chunk_size: int = 2000, 
                         skip_existing: bool = False, maxsize: int = 25):
    """Process all factstatement parquet files"""
    
    # Find all factstatement files
    files = sorted(parquet_dir.glob('factstatements_part_*.parquet'))
    
    if not files:
        logger.error(f"No factstatement files found in {parquet_dir}")
        return
    
    logger.info(f"Found {len(files)} factstatement files")
    
    # ES settings for workers
    es_settings = {
        'hosts': ES_IP_LIST,
        'port': ES_PORT,
        'user': ES_USER,
        'password': ES_PASSWARD
    }
    
    total_success = 0
    total_error = 0
    total_processed = 0
    
    # Process each file
    for file_idx, file_path in enumerate(files):
        logger.info(f"Processing file {file_idx + 1}/{len(files)}: {file_path.name}")
        
        # Check progress
        progress_file = progress_dir / (file_path.name + '.progress.json')
        progress = load_json_safe(progress_file)
        
        if progress.get('status') == 'success':
            logger.info(f"{file_path.name} already processed - skipping")
            total_success += progress.get('success_count', 0)
            total_processed += progress.get('total_rows', 0)
            continue
        
        try:
            pf = pq.ParquetFile(file_path)
            total_rows = pf.metadata.num_rows
            logger.info(f"File {file_path.name} has {total_rows} rows")
            
            file_success = 0
            file_error = 0
            
            # Process in parallel batches
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = []
                batch_idx = 0
                
                for batch in pf.iter_batches(batch_size=batch_size):
                    tbl = pa.Table.from_batches([batch])
                    records = tbl.to_pylist()
                    batch_tuple = (batch_idx, records)
                    
                    future = executor.submit(
                        process_batch_worker,
                        batch_tuple,
                        es_settings,
                        index_name,
                        chunk_size,
                        skip_existing,
                        maxsize
                    )
                    futures.append(future)
                    batch_idx += 1
                
                # Collect results
                for f in tqdm(as_completed(futures), total=len(futures), 
                            desc=f"Processing {file_path.name}"):
                    try:
                        res = f.result()
                        if res.get('status') == 'success':
                            file_success += res.get('success_count', 0)
                            file_error += res.get('error_count', 0)
                        else:
                            logger.error(f"Batch {res.get('batch_idx')} failed: {res.get('error')}")
                    except Exception as e:
                        logger.exception(f"Future exception: {e}")
            
            # Save progress
            progress.update({
                'total_rows': total_rows,
                'success_count': file_success,
                'error_count': file_error,
                'status': 'success' if file_error == 0 else 'partial',
                'last_updated': time.time()
            })
            save_json_atomic(progress_file, progress)
            
            total_success += file_success
            total_error += file_error
            total_processed += total_rows
            
            logger.info(f"Completed {file_path.name}: success={file_success}, error={file_error}")
            
        except Exception as e:
            logger.exception(f"Failed processing {file_path.name}: {e}")
            continue
    
    logger.info(f"All files processed: total_processed={total_processed}, "
               f"success={total_success}, error={total_error}")


# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description='Index FactStatements to Elasticsearch')
    parser.add_argument('--parquet-dir', required=True, type=Path,
                       help='Directory containing factstatement parquet files')
    parser.add_argument('--es-index', default='factnet_factstatements_v1',
                       help='Elasticsearch index name')
    parser.add_argument('--batch-size', type=int, default=2000,
                       help='Pyarrow batch size (rows per batch)')
    parser.add_argument('--parallel-workers', type=int, default=32,
                       help='Number of parallel worker processes')
    parser.add_argument('--progress-dir', type=Path, default=Path('./progress'),
                       help='Directory for progress tracking')
    parser.add_argument('--log-dir', type=Path, default=Path('./logs'),
                       help='Directory for logs')
    parser.add_argument('--chunk-size', type=int, default=2000,
                       help='ES bulk chunk size')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip existing documents (slower)')
    parser.add_argument('--es-connection-pool', type=int, default=2,
                       help='ES client maxsize (connection pool per process)')
    
    args = parser.parse_args()
    
    args.progress_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(args.log_dir)
    logger.info(f"Starting FactStatement indexer with args: {args}")
    
    # Create ES client and index
    es = ElasticFactory(ES_IP_LIST, ES_PORT, ES_USER, ES_PASSWARD, 
                       maxsize=args.es_connection_pool).create()
    
    create_factstatement_index(es, args.es_index, logger)
    
    # Process files
    try:
        process_parquet_files(
            args.parquet_dir,
            args.es_index,
            args.batch_size,
            args.progress_dir,
            logger,
            workers=args.parallel_workers,
            chunk_size=args.chunk_size,
            skip_existing=args.skip_existing,
            maxsize=args.es_connection_pool
        )
        
        # Re-enable refresh after indexing
        logger.info("Re-enabling index refresh...")
        es.indices.put_settings(
            index=args.es_index,
            body={'index': {'refresh_interval': '30s'}}
        )
        
        # Force refresh
        es.indices.refresh(index=args.es_index)
        
        logger.info("FactStatement indexing completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Fatal: {e}")
        raise