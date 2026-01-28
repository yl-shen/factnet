import re
import argparse
import json
import logging
import logging.handlers
import sys
import time
import hashlib
import multiprocessing as mp
from multiprocessing import Queue, Process
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime
from collections import defaultdict
import ssl
import urllib3
from dateutil import parser as date_parser
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import gc

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("Please install pyarrow: pip install pyarrow")
    sys.exit(1)

try:
    from elasticsearch import Elasticsearch, helpers
except ImportError:
    print("Please install elasticsearch: pip install elasticsearch")
    sys.exit(1)

from es_config import (
    ES_IP_LIST, ES_PORT, ES_USER, ES_PASSWARD,
    ES_FACTSYNSET_INDEX, ES_FACTSTATEMENT_INDEX, 
    ES_LABELS_INDEX
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -------------------- 扩展的属性映射 --------------------

PROPERTY_RELATION_MAP = {
    'P279': ('hypernym', 'subclass_of', 1.0),
    'P31': ('hypernym', 'instance_of', 0.95),
    'P1542': ('causal', 'has_effect', 0.90),
    'P828': ('causal', 'has_cause', 0.90),
    'P1479': ('causal', 'contributing_factor', 0.85),
    'P1536': ('causal', 'immediate_cause', 0.92),
    'P1537': ('causal', 'ultimate_cause', 0.88),
    'P706': ('geographic_location', 'located_on', 0.88),
    'P131': ('geographic_contains', 'admin_division', 0.90),
    'P17': ('geographic_contains', 'country', 0.92),
    'P30': ('geographic_contains', 'continent', 0.85),
    'P47': ('adjacent', 'shares_border', 0.88),
    'P206': ('adjacent', 'water_border', 0.85),
    'P361': ('part_of', 'part_of', 0.90),
    'P527': ('has_part', 'has_part', 0.90),
    'P463': ('member_of', 'member_of', 0.88),
    'P1416': ('affiliated_with', 'affiliation', 0.82),
    'P108': ('employed_by', 'employer', 0.85),
    'P155': ('follows', 'follows', 0.90),
    'P156': ('followed_by', 'followed_by', 0.90),
    'P1365': ('replaces', 'replaces', 0.88),
    'P1366': ('replaced_by', 'replaced_by', 0.88),
    'P737': ('influenced_by', 'influenced_by', 0.80),
    'P1074': ('influences', 'influenced', 0.80),
    'P144': ('derives_from', 'based_on', 0.83),
    'P2596': ('cultural_origin', 'culture', 0.78),
    'P170': ('created_by', 'creator', 0.90),
    'P50': ('created_by', 'author', 0.92),
    'P178': ('created_by', 'developer', 0.88),
    'P57': ('created_by', 'director', 0.90),
    'P86': ('created_by', 'composer', 0.88),
    'P366': ('used_for', 'use', 0.85),
    'P642': ('of', 'of', 0.75),
    'P461': ('opposite_of', 'opposite', 0.85),
    'P22': ('family', 'father', 0.95),
    'P25': ('family', 'mother', 0.95),
    'P40': ('family', 'child', 0.95),
    'P26': ('family', 'spouse', 0.93),
    'P3373': ('family', 'sibling', 0.90),
}

SYMMETRIC_RELATIONS = {'adjacent', 'opposite_of', 'similar_to'}
TRANSITIVE_RELATIONS = {'hypernym': 2, 'part_of': 2, 'geographic_contains': 2}

ALL_RELATION_TYPES = set()
for rel_type, _, _ in PROPERTY_RELATION_MAP.values():
    ALL_RELATION_TYPES.add(rel_type)

SPECIAL_TYPES = {
    'equivalent', 'contradiction', 'temporal_before', 'temporal_overlap',
    'support', 'refute', 'co_occurrence', 'similar_to', 'inferred_relation'
}
ALL_RELATION_TYPES.update(SPECIAL_TYPES)

# -------------------- Confidence计算 --------------------

def calculate_dynamic_confidence(
    base_confidence: float,
    evidence: Dict[str, Any],
    is_inferred: bool = False
) -> float:
    conf = base_confidence
    multipliers = []
    
    source_count = evidence.get('source_count', 0)
    if source_count >= 10:
        multipliers.append(1.15)
    elif source_count >= 5:
        multipliers.append(1.10)
    elif source_count >= 2:
        multipliers.append(1.05)
    elif source_count == 1:
        multipliers.append(0.95)
    else:
        multipliers.append(0.90)
    
    rank = evidence.get('rank', 'normal')
    if rank == 'preferred':
        multipliers.append(1.10)
    elif rank == 'deprecated':
        multipliers.append(0.70)
    
    qualifier_count = evidence.get('qualifier_count', 0)
    if qualifier_count > 3:
        multipliers.append(1.08)
    elif qualifier_count > 0:
        multipliers.append(1.03)
    
    lang_count = evidence.get('language_count', 0)
    if lang_count >= 10:
        multipliers.append(1.12)
    elif lang_count >= 5:
        multipliers.append(1.06)
    elif lang_count >= 2:
        multipliers.append(1.02)
    
    if is_inferred:
        multipliers.append(0.85)
        chain_length = evidence.get('inference_chain_length', 1)
        multipliers.append(0.95 ** (chain_length - 1))
    
    property_reliability = evidence.get('property_reliability', 1.0)
    multipliers.append(property_reliability)
    
    for m in multipliers:
        conf *= m
    
    return max(0.1, min(1.0, conf))

# -------------------- Data Models --------------------

@dataclass
class RelationEdge:
    relation_id: str
    source_synset_id: str
    target_synset_id: str
    relation_type: str
    confidence: float
    evidence: Dict[str, Any]
    created_at: str

    def to_dict(self) -> Dict:
        return asdict(self)

# -------------------- ES Factory --------------------

class ElasticFactory:
    def __init__(self, host: list, port: str, username: str, password: str, maxsize: int = 25):
        self.port = port
        self.host = host
        self.username = username
        self.password = password
        self.maxsize = maxsize

    def create(self) -> Elasticsearch:
        context = ssl._create_unverified_context()
        addrs = [{"host": h, "port": self.port} for h in self.host]
        
        if self.username and self.password:
            es = Elasticsearch(
                addrs,
                http_auth=(self.username, self.password),
                scheme="https",
                ssl_context=context,
                maxsize=self.maxsize,
                timeout=180,
                max_retries=5,
                retry_on_timeout=True
            )
        else:
            es = Elasticsearch(addrs, maxsize=self.maxsize, timeout=180)
        return es

# -------------------- Logging --------------------

def setup_logging(log_dir: Path, name: str = "synset_relations_optimized") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(ch)
    
    log_file = log_dir / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.handlers.RotatingFileHandler(
        str(log_file),
        maxBytes=100*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    
    return logger

# -------------------- Utilities --------------------

def sha256_short(s: str, length: int = 16) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:length]

def extract_entity_value(value_repr: Any) -> Optional[str]:
    if isinstance(value_repr, str):
        try:
            value_repr = json.loads(value_repr)
        except:
            if value_repr and value_repr.startswith('Q'):
                return value_repr
            return None
    
    if isinstance(value_repr, dict):
        return value_repr.get('id')
    
    return None

def extract_time_from_qualifiers(qualifiers: Any) -> Optional[str]:
    if not qualifiers:
        return None
    
    if isinstance(qualifiers, str):
        try:
            qualifiers = json.loads(qualifiers)
        except:
            return None
    
    if not isinstance(qualifiers, dict):
        return None
    
    for prop in ['P580', 'P582', 'P585']:
        if prop in qualifiers:
            qual_list = qualifiers[prop]
            if isinstance(qual_list, list) and len(qual_list) > 0:
                first_qual = qual_list[0]
                if isinstance(first_qual, dict):
                    datavalue = first_qual.get('datavalue', {})
                    if isinstance(datavalue, dict):
                        value = datavalue.get('value', {})
                        if isinstance(value, dict):
                            time_str = value.get('time')
                            if time_str:
                                return time_str
    
    return None

def parse_wikidata_time(time_str: str) -> Optional[str]:
    if not time_str:
        return None
    
    try:
        time_str = time_str.lstrip('+')
        is_negative = time_str.startswith('-')
        if is_negative:
            time_str = time_str[1:]
        
        time_str = re.sub(r'^(\d{4})-00-', r'\1-01-', time_str)
        time_str = re.sub(r'^(\d{4}-\d{2})-00', r'\1-01', time_str)
        
        try:
            dt = date_parser.parse(time_str)
            result = dt.isoformat()
        except Exception:
            year_match = re.match(r'^(\d{4})', time_str)
            if year_match:
                year = year_match.group(1)
                result = f"{year}-01-01T00:00:00"
            else:
                return None
        
        if is_negative:
            result = '-' + result
        
        return result
        
    except Exception:
        return None

def safe_parse_datetime(time_str: str) -> Optional[datetime]:
    if not time_str:
        return None
    
    try:
        if time_str.startswith('-'):
            return None
        
        time_str = re.sub(r'^(\d{4})-00-', r'\1-01-', time_str)
        time_str = re.sub(r'^(\d{4}-\d{2})-00', r'\1-01', time_str)
        
        return date_parser.parse(time_str)
    except Exception:
        return None

def calculate_time_gap_days(time_str_a: str, time_str_b: str) -> Optional[int]:
    try:
        dt_a = safe_parse_datetime(time_str_a)
        dt_b = safe_parse_datetime(time_str_b)
        
        if dt_a is None or dt_b is None:
            return None
        
        return abs((dt_b - dt_a).days)
    except Exception:
        return None

# -------------------- Checkpoint --------------------

@dataclass
class CheckpointData:
    processed_synsets: int = 0
    total_relations: int = 0
    output_file_index: int = 0
    current_file_rows: int = 0
    last_sort_value: List = None
    last_synset_id: str = None
    processed_synset_ids: Set[str] = field(default_factory=set)
    last_update: float = 0
    completed: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'processed_synsets': self.processed_synsets,
            'total_relations': self.total_relations,
            'output_file_index': self.output_file_index,
            'current_file_rows': self.current_file_rows,
            'last_sort_value': self.last_sort_value,
            'last_synset_id': self.last_synset_id,
            'recent_synset_ids': list(self.processed_synset_ids)[-1000:],
            'total_processed_ids': len(self.processed_synset_ids),
            'last_update': self.last_update,
            'completed': self.completed
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CheckpointData':
        cp = cls()
        cp.processed_synsets = data.get('processed_synsets', 0)
        cp.total_relations = data.get('total_relations', 0)
        cp.output_file_index = data.get('output_file_index', 0)
        cp.current_file_rows = data.get('current_file_rows', 0)
        cp.last_sort_value = data.get('last_sort_value')
        cp.last_synset_id = data.get('last_synset_id')
        cp.processed_synset_ids = set(data.get('recent_synset_ids', []))
        cp.last_update = data.get('last_update', 0)
        cp.completed = data.get('completed', False)
        return cp

def load_checkpoint_enhanced(checkpoint_path: Path, logger: logging.Logger) -> CheckpointData:
    if not checkpoint_path.exists():
        logger.info("No checkpoint found, starting fresh")
        return CheckpointData()
    
    try:
        with checkpoint_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        
        cp = CheckpointData.from_dict(data)
        logger.info(f"Loaded checkpoint: processed={cp.processed_synsets}, "
                   f"relations={cp.total_relations}, last_synset={cp.last_synset_id}")
        
        if cp.completed:
            logger.warning("Previous job was marked as completed!")
        
        return cp
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return CheckpointData()

def save_checkpoint_enhanced(
    checkpoint_path: Path, 
    checkpoint: CheckpointData,
    logger: logging.Logger
):
    checkpoint.last_update = time.time()
    
    tmp_path = checkpoint_path.with_suffix('.tmp')
    backup_path = checkpoint_path.with_suffix('.bak')
    
    try:
        with tmp_path.open('w', encoding='utf-8') as f:
            json.dump(checkpoint.to_dict(), f, ensure_ascii=False, indent=2)
        
        if checkpoint_path.exists():
            checkpoint_path.replace(backup_path)
        
        tmp_path.replace(checkpoint_path)
        logger.debug(f"Checkpoint saved: synsets={checkpoint.processed_synsets}")
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if backup_path.exists() and not checkpoint_path.exists():
            backup_path.replace(checkpoint_path)

# -------------------- ES Index Creation --------------------

def create_relations_index(es: Elasticsearch, index_name: str, logger: logging.Logger):
    if es.indices.exists(index=index_name):
        logger.info(f"Index {index_name} already exists")
        return
    
    mapping = {
        "settings": {
            "number_of_shards": 30,
            "number_of_replicas": 1,
            "refresh_interval": "30s"
        },
        "mappings": {
            "properties": {
                "relation_id": {"type": "keyword"},
                "source_synset_id": {"type": "keyword"},
                "target_synset_id": {"type": "keyword"},
                "relation_type": {"type": "keyword"},
                "confidence": {"type": "float"},
                "evidence": {"type": "object", "enabled": False},
                "created_at": {"type": "date"}
            }
        }
    }
    
    try:
        es.indices.create(index=index_name, body=mapping)
        logger.info(f"Successfully created index: {index_name}")
    except Exception as e:
        logger.error(f"Failed to create index {index_name}: {e}")
        raise

# -------------------- ES Query Functions (优化版) --------------------

def scroll_all_synsets_with_search_after(
    es: Elasticsearch, 
    index_name: str, 
    batch_size: int = 1000, 
    max_synsets: int = None,
    last_sort_value: List = None
):
    """使用 search_after 遍历所有 synsets"""
    query = {
        "query": {"match_all": {}},
        "sort": [
            {"synset_id": "asc"},
            {"_id": "asc"}
        ],
        "size": batch_size
    }
    
    if last_sort_value:
        query["search_after"] = last_sort_value
    
    total_fetched = 0
    
    try:
        while True:
            resp = es.search(index=index_name, body=query, request_timeout=300)
            hits = resp['hits']['hits']
            
            if not hits:
                break
            
            batch = [hit['_source'] for hit in hits]
            last_sort = hits[-1]['sort']
            
            yield batch, last_sort
            
            total_fetched += len(batch)
            if max_synsets and total_fetched >= max_synsets:
                break
            
            query["search_after"] = last_sort
            
    except Exception as e:
        logging.error(f"Error in search_after: {e}")
        raise

def query_statements_batch_parallel(
    es: Elasticsearch, 
    index_name: str, 
    statement_ids: List[str],
    num_threads: int = 8
) -> Dict[str, Dict]:
    """并行查询statements"""
    if not statement_ids:
        return {}
    
    unique_ids = list(set(statement_ids))
    results = {}
    results_lock = threading.Lock()
    chunk_size = 20000  # 每个线程处理的chunk大小
    
    def query_chunk(chunk_ids: List[str]) -> Dict[str, Dict]:
        chunk_results = {}
        # 分批查询，每批最多10000
        for i in range(0, len(chunk_ids), 10000):
            batch = chunk_ids[i:i+10000]
            query = {
                "query": {
                    "terms": {
                        "core_id": batch
                    }
                },
                "size": len(batch)
            }
            
            try:
                resp = es.search(index=index_name, body=query, request_timeout=120)
                for hit in resp.get('hits', {}).get('hits', []):
                    src = hit['_source']
                    sid = src.get('core_id')
                    if sid:
                        chunk_results[sid] = src
            except Exception as e:
                logging.warning(f"Failed to query statements batch: {e}")
        
        return chunk_results
    
    # 分割成多个chunks
    chunks = [unique_ids[i:i+chunk_size] for i in range(0, len(unique_ids), chunk_size)]
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(query_chunk, chunk) for chunk in chunks]
        
        for future in as_completed(futures):
            try:
                chunk_results = future.result()
                with results_lock:
                    results.update(chunk_results)
            except Exception as e:
                logging.warning(f"Thread failed: {e}")
    
    return results

def query_synsets_by_subject_qids_parallel(
    es: Elasticsearch, 
    synset_index: str, 
    statement_index: str,
    qids: List[str],
    num_threads: int = 8
) -> Dict[str, str]:
    """并行查询 QID 到 synset 的映射"""
    if not qids:
        return {}
    
    qid_to_synset = {}
    qid_lock = threading.Lock()
    chunk_size = 20000  # 每个线程处理的QID数量
    
    def process_qid_chunk(chunk_qids: List[str]) -> Dict[str, str]:
        local_results = {}
        
        try:
            # 步骤1: 查询statements
            stmt_query = {
                "query": {
                    "terms": {
                        "subject_qid": chunk_qids
                    }
                },
                "_source": ["core_id", "subject_qid"],
                "size": len(chunk_qids) * 10
            }
            
            stmt_resp = es.search(index=statement_index, body=stmt_query, request_timeout=120)
            statement_to_qid = {}
            statement_ids = []
            
            for hit in stmt_resp.get('hits', {}).get('hits', []):
                src = hit['_source']
                stmt_id = src.get('core_id')
                subj_qid = src.get('subject_qid')
                if stmt_id and subj_qid:
                    statement_to_qid[stmt_id] = subj_qid
                    statement_ids.append(stmt_id)
            
            if not statement_ids:
                return local_results
            
            # 步骤2: 查询synsets
            synset_query = {
                "query": {
                    "terms": {
                        "member_statement_ids": statement_ids
                    }
                },
                "_source": ["synset_id", "member_statement_ids"],
                "size": 10000
            }
            
            synset_resp = es.search(index=synset_index, body=synset_query, scroll='2m', request_timeout=120)
            scroll_id = synset_resp.get('_scroll_id')
            hits = synset_resp['hits']['hits']
            
            while hits:
                for hit in hits:
                    src = hit['_source']
                    synset_id = src.get('synset_id')
                    member_ids = src.get('member_statement_ids', [])
                    
                    if isinstance(member_ids, str):
                        try:
                            member_ids = json.loads(member_ids)
                        except:
                            member_ids = []
                    
                    for stmt_id in member_ids:
                        if stmt_id in statement_to_qid:
                            qid = statement_to_qid[stmt_id]
                            if qid not in local_results:
                                local_results[qid] = synset_id
                
                if scroll_id:
                    try:
                        synset_resp = es.scroll(scroll_id=scroll_id, scroll='2m')
                        hits = synset_resp['hits']['hits']
                        scroll_id = synset_resp.get('_scroll_id')
                    except:
                        break
                else:
                    break
                    
        except Exception as e:
            logging.warning(f"Failed to query synsets by subject QIDs: {e}")
        
        return local_results
    
    # 分割成多个chunks
    unique_qids = list(set(qids))
    chunks = [unique_qids[i:i+chunk_size] for i in range(0, len(unique_qids), chunk_size)]
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_qid_chunk, chunk) for chunk in chunks]
        
        for future in as_completed(futures):
            try:
                chunk_results = future.result()
                with qid_lock:
                    qid_to_synset.update(chunk_results)
            except Exception as e:
                logging.warning(f"Thread failed: {e}")
    
    return qid_to_synset

# -------------------- 保持原有的关系检测函数不变 --------------------

def detect_equivalent_relations(synsets: List[Dict], logger: logging.Logger) -> List[RelationEdge]:
    """等价关系检测"""
    relations = []
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    for synset in synsets:
        synset_id = synset.get('synset_id')
        member_ids = synset.get('member_statement_ids', [])
        
        if isinstance(member_ids, str):
            try:
                member_ids = json.loads(member_ids)
            except:
                member_ids = []
        
        if not synset_id or len(member_ids) < 2:
            continue
        
        for i in range(len(member_ids)):
            for j in range(i+1, len(member_ids)):
                rel_id = sha256_short(f"{member_ids[i]}_{member_ids[j]}_equivalent")
                
                evidence = {
                    "synset_id": synset_id,
                    "member_count": len(member_ids),
                    "source_count": synset.get('source_count', 0),
                    "language_count": len(synset.get('language_coverage', {}))
                }
                
                conf = calculate_dynamic_confidence(1.0, evidence, is_inferred=False)
                
                relations.append(RelationEdge(
                    relation_id=rel_id,
                    source_synset_id=synset_id,
                    target_synset_id=synset_id,
                    relation_type='equivalent',
                    confidence=conf,
                    evidence=evidence,
                    created_at=timestamp
                ))
    
    return relations

def detect_property_based_relations(
    synsets: List[Dict],
    statements_map: Dict[str, Dict],
    es_client: Elasticsearch,
    synset_index: str,
    statement_index: str,
    logger: logging.Logger
) -> List[RelationEdge]:
    """统一的基于属性的关系检测"""
    relations = []
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    property_edges = defaultdict(list)
    
    for synset in synsets:
        synset_id = synset.get('synset_id')
        if not synset_id:
            continue
        
        member_ids = synset.get('member_statement_ids', [])
        if isinstance(member_ids, str):
            try:
                member_ids = json.loads(member_ids)
            except:
                member_ids = []
        
        for stmt_id in member_ids:
            stmt = statements_map.get(stmt_id)
            if not stmt:
                continue
            
            property_pid = stmt.get('property_pid')
            if property_pid not in PROPERTY_RELATION_MAP:
                continue
            
            value_repr = stmt.get('value')
            target_qid = extract_entity_value(value_repr)
            
            if target_qid:
                rel_type, subtype, base_conf = PROPERTY_RELATION_MAP[property_pid]
                
                qualifiers = stmt.get('qualifiers', '{}')
                if isinstance(qualifiers, str):
                    try:
                        quals_dict = json.loads(qualifiers)
                        qual_count = len(quals_dict)
                    except:
                        qual_count = 0
                else:
                    qual_count = len(qualifiers) if isinstance(qualifiers, dict) else 0
                
                edge_info = {
                    'synset_id': synset_id,
                    'target_qid': target_qid,
                    'relation_type': rel_type,
                    'base_confidence': base_conf,
                    'source_count': synset.get('source_count', 0),
                    'rank': stmt.get('rank', 'normal'),
                    'qualifier_count': qual_count,
                    'language_count': len(synset.get('language_coverage', {})),
                    'property': property_pid,
                    'subtype': subtype
                }
                
                property_edges[property_pid].append(edge_info)
    
    if not property_edges:
        return relations
    
    # 批量查询目标QIDs - 使用并行版本
    all_target_qids = []
    for edges in property_edges.values():
        all_target_qids.extend([e['target_qid'] for e in edges])
    
    unique_qids = list(set(all_target_qids))
    
    qid_to_synset = query_synsets_by_subject_qids_parallel(
        es_client, synset_index, statement_index, unique_qids, num_threads=8
    )
    
    # 创建关系
    for property_pid, edges in property_edges.items():
        for edge_info in edges:
            source_synset_id = edge_info['synset_id']
            target_qid = edge_info['target_qid']
            target_synset_id = qid_to_synset.get(target_qid)
            
            if not target_synset_id or source_synset_id == target_synset_id:
                continue
            
            evidence = {
                'target_qid': target_qid,
                'property': edge_info['property'],
                'subtype': edge_info['subtype'],
                'source_count': edge_info['source_count'],
                'rank': edge_info['rank'],
                'qualifier_count': edge_info['qualifier_count'],
                'language_count': edge_info['language_count']
            }
            
            conf = calculate_dynamic_confidence(
                edge_info['base_confidence'],
                evidence,
                is_inferred=False
            )
            
            rel_id = sha256_short(f"{source_synset_id}_{target_synset_id}_{edge_info['relation_type']}_{property_pid}")
            
            relations.append(RelationEdge(
                relation_id=rel_id,
                source_synset_id=source_synset_id,
                target_synset_id=target_synset_id,
                relation_type=edge_info['relation_type'],
                confidence=conf,
                evidence=evidence,
                created_at=timestamp
            ))
            
            if edge_info['relation_type'] in SYMMETRIC_RELATIONS:
                rel_id_sym = sha256_short(f"{target_synset_id}_{source_synset_id}_{edge_info['relation_type']}_{property_pid}")
                relations.append(RelationEdge(
                    relation_id=rel_id_sym,
                    source_synset_id=target_synset_id,
                    target_synset_id=source_synset_id,
                    relation_type=edge_info['relation_type'],
                    confidence=conf,
                    evidence=evidence,
                    created_at=timestamp
                ))
    
    return relations

def detect_contradiction_relations(synsets: List[Dict], statements_map: Dict[str, Dict], 
                                   logger: logging.Logger) -> List[RelationEdge]:
    """矛盾关系检测"""
    relations = []
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    subject_prop_groups = defaultdict(list)
    
    for synset in synsets:
        member_ids = synset.get('member_statement_ids', [])
        if isinstance(member_ids, str):
            try:
                member_ids = json.loads(member_ids)
            except:
                member_ids = []
        
        for sid in member_ids[:1]:
            stmt = statements_map.get(sid, {})
            subject_qid = stmt.get('subject_qid')
            property_pid = stmt.get('property_pid')
            
            if subject_qid and property_pid:
                key = f"{subject_qid}_{property_pid}"
                subject_prop_groups[key].append(synset)
                break
    
    for key, group_synsets in subject_prop_groups.items():
        if len(group_synsets) < 2:
            continue
        
        for i in range(len(group_synsets)):
            for j in range(i+1, len(group_synsets)):
                synset_a = group_synsets[i]
                synset_b = group_synsets[j]
                
                if synset_a.get('aggregation_key') != synset_b.get('aggregation_key'):
                    synset_a_id = synset_a.get('synset_id')
                    synset_b_id = synset_b.get('synset_id')
                    
                    if not synset_a_id or not synset_b_id:
                        continue
                    
                    evidence = {
                        'subject_property': key,
                        'source_count_a': synset_a.get('source_count', 0),
                        'source_count_b': synset_b.get('source_count', 0),
                        'language_count': max(
                            len(synset_a.get('language_coverage', {})),
                            len(synset_b.get('language_coverage', {}))
                        )
                    }
                    
                    source_ratio = min(evidence['source_count_a'], evidence['source_count_b']) / max(evidence['source_count_a'], evidence['source_count_b'], 1)
                    base_conf = 0.6 + 0.2 * source_ratio
                    
                    conf = calculate_dynamic_confidence(base_conf, evidence, is_inferred=False)
                    
                    rel_id = sha256_short(f"{synset_a_id}_{synset_b_id}_contradiction")
                    
                    relations.append(RelationEdge(
                        relation_id=rel_id,
                        source_synset_id=synset_a_id,
                        target_synset_id=synset_b_id,
                        relation_type='contradiction',
                        confidence=conf,
                        evidence=evidence,
                        created_at=timestamp
                    ))
    
    return relations

def detect_temporal_relations(
    synsets: List[Dict], 
    statements_map: Dict[str, Dict],
    logger
) -> List[RelationEdge]:
    """时间关系检测"""
    relations = []
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    synsets_with_time = []
    
    for synset in synsets:
        synset_id = synset.get('synset_id')
        if not synset_id:
            continue
        
        member_ids = synset.get('member_statement_ids', [])
        if isinstance(member_ids, str):
            try:
                member_ids = json.loads(member_ids)
            except:
                member_ids = []
        
        time_values = []
        for stmt_id in member_ids:
            stmt = statements_map.get(stmt_id)
            if not stmt:
                continue
            
            qualifiers = stmt.get('qualifiers')
            time_str = extract_time_from_qualifiers(qualifiers)
            if time_str:
                parsed_time = parse_wikidata_time(time_str)
                if parsed_time:
                    time_values.append(parsed_time)
        
        if time_values:
            time_values.sort()
            synsets_with_time.append({
                'synset_id': synset_id,
                'earliest': time_values[0],
                'latest': time_values[-1],
                'time_count': len(time_values),
                'source_count': synset.get('source_count', 0),
                'language_count': len(synset.get('language_coverage', {}))
            })
    
    for i in range(len(synsets_with_time)):
        for j in range(i+1, len(synsets_with_time)):
            item_a = synsets_with_time[i]
            item_b = synsets_with_time[j]
            
            synset_a_id = item_a['synset_id']
            synset_b_id = item_b['synset_id']
            
            if (item_a['earliest'].startswith('-') or 
                item_a['latest'].startswith('-') or
                item_b['earliest'].startswith('-') or 
                item_b['latest'].startswith('-')):
                continue
            
            try:
                if item_a['latest'] < item_b['earliest']:
                    time_gap = calculate_time_gap_days(item_a['latest'], item_b['earliest'])
                    if time_gap is None:
                        time_gap = 0
                    
                    evidence = {
                        'time_a_latest': item_a['latest'],
                        'time_b_earliest': item_b['earliest'],
                        'ordering': 'before',
                        'source_count': item_a['source_count'] + item_b['source_count'],
                        'language_count': max(item_a['language_count'], item_b['language_count'])
                    }
                    
                    base_conf = min(0.95, 0.75 + 0.0001 * time_gap)
                    conf = calculate_dynamic_confidence(base_conf, evidence, is_inferred=False)
                    
                    rel_id = sha256_short(f"{synset_a_id}_{synset_b_id}_temporal_before")
                    relations.append(RelationEdge(
                        relation_id=rel_id,
                        source_synset_id=synset_a_id,
                        target_synset_id=synset_b_id,
                        relation_type='temporal_before',
                        confidence=conf,
                        evidence=evidence,
                        created_at=timestamp
                    ))
                
                elif item_b['latest'] < item_a['earliest']:
                    time_gap = calculate_time_gap_days(item_b['latest'], item_a['earliest'])
                    if time_gap is None:
                        time_gap = 0
                    
                    evidence = {
                        'time_a_latest': item_b['latest'],
                        'time_b_earliest': item_a['earliest'],
                        'ordering': 'before',
                        'source_count': item_a['source_count'] + item_b['source_count'],
                        'language_count': max(item_a['language_count'], item_b['language_count'])
                    }
                    
                    base_conf = min(0.95, 0.75 + 0.0001 * time_gap)
                    conf = calculate_dynamic_confidence(base_conf, evidence, is_inferred=False)
                    
                    rel_id = sha256_short(f"{synset_b_id}_{synset_a_id}_temporal_before")
                    relations.append(RelationEdge(
                        relation_id=rel_id,
                        source_synset_id=synset_b_id,
                        target_synset_id=synset_a_id,
                        relation_type='temporal_before',
                        confidence=conf,
                        evidence=evidence,
                        created_at=timestamp
                    ))
                
                else:
                    evidence = {
                        'time_overlap': True,
                        'source_count': item_a['source_count'] + item_b['source_count'],
                        'language_count': max(item_a['language_count'], item_b['language_count'])
                    }
                    
                    conf = calculate_dynamic_confidence(0.65, evidence, is_inferred=False)
                    
                    rel_id = sha256_short(f"{synset_a_id}_{synset_b_id}_temporal_overlap")
                    relations.append(RelationEdge(
                        relation_id=rel_id,
                        source_synset_id=synset_a_id,
                        target_synset_id=synset_b_id,
                        relation_type='temporal_overlap',
                        confidence=conf,
                        evidence=evidence,
                        created_at=timestamp
                    ))
                    
            except Exception as e:
                continue
    
    return relations

def detect_support_refute_relations(synsets: List[Dict], statements_map: Dict[str, Dict], 
                                    logger: logging.Logger) -> List[RelationEdge]:
    """支持/反驳关系检测"""
    relations = []
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    subject_groups = defaultdict(list)
    
    for synset in synsets:
        member_ids = synset.get('member_statement_ids', [])
        if isinstance(member_ids, str):
            try:
                member_ids = json.loads(member_ids)
            except:
                member_ids = []
        
        for sid in member_ids[:1]:
            stmt = statements_map.get(sid, {})
            subject_qid = stmt.get('subject_qid')
            if subject_qid:
                subject_groups[subject_qid].append(synset)
                break
    
    for subject_qid, group_synsets in subject_groups.items():
        if len(group_synsets) < 2:
            continue
        
        for i in range(len(group_synsets)):
            for j in range(i+1, len(group_synsets)):
                synset_a = group_synsets[i]
                synset_b = group_synsets[j]
                
                synset_a_id = synset_a.get('synset_id')
                synset_b_id = synset_b.get('synset_id')
                
                if not synset_a_id or not synset_b_id:
                    continue
                
                source_count_a = synset_a.get('source_count', 0)
                source_count_b = synset_b.get('source_count', 0)
                
                if source_count_a > source_count_b * 1.5:
                    ratio = source_count_a / max(source_count_b, 1)
                    base_conf = min(0.80, 0.50 + 0.05 * ratio)
                    
                    evidence = {
                        'source_count_ratio': ratio,
                        'subject_qid': subject_qid,
                        'source_count': source_count_a + source_count_b,
                        'language_count': max(
                            len(synset_a.get('language_coverage', {})),
                            len(synset_b.get('language_coverage', {}))
                        )
                    }
                    
                    conf = calculate_dynamic_confidence(base_conf, evidence, is_inferred=False)
                    
                    rel_id = sha256_short(f"{synset_a_id}_{synset_b_id}_support")
                    relations.append(RelationEdge(
                        relation_id=rel_id,
                        source_synset_id=synset_a_id,
                        target_synset_id=synset_b_id,
                        relation_type='support',
                        confidence=conf,
                        evidence=evidence,
                        created_at=timestamp
                    ))
                
                elif source_count_b > source_count_a * 1.5:
                    ratio = source_count_b / max(source_count_a, 1)
                    base_conf = min(0.80, 0.50 + 0.05 * ratio)
                    
                    evidence = {
                        'source_count_ratio': ratio,
                        'subject_qid': subject_qid,
                        'source_count': source_count_a + source_count_b,
                        'language_count': max(
                            len(synset_a.get('language_coverage', {})),
                            len(synset_b.get('language_coverage', {}))
                        )
                    }
                    
                    conf = calculate_dynamic_confidence(base_conf, evidence, is_inferred=False)
                    
                    rel_id = sha256_short(f"{synset_b_id}_{synset_a_id}_support")
                    relations.append(RelationEdge(
                        relation_id=rel_id,
                        source_synset_id=synset_b_id,
                        target_synset_id=synset_a_id,
                        relation_type='support',
                        confidence=conf,
                        evidence=evidence,
                        created_at=timestamp
                    ))
    
    return relations

def detect_cooccurrence_relations(synsets: List[Dict], logger: logging.Logger) -> List[RelationEdge]:
    """共现关系检测"""
    relations = []
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    page_to_synsets = defaultdict(list)
    
    for synset in synsets:
        canonical_mentions = synset.get('canonical_mentions', {})
        if isinstance(canonical_mentions, str):
            try:
                canonical_mentions = json.loads(canonical_mentions)
            except:
                canonical_mentions = {}
        
        pages = set()
        for lang, mention_data in canonical_mentions.items():
            if isinstance(mention_data, dict):
                page_title = mention_data.get('page_title')
                if page_title:
                    pages.add(page_title)
        
        for page in pages:
            page_to_synsets[page].append(synset)
    
    processed_pairs = set()
    
    for page, co_synsets in page_to_synsets.items():
        if len(co_synsets) < 2:
            continue
        
        for i in range(len(co_synsets)):
            for j in range(i+1, len(co_synsets)):
                synset_a = co_synsets[i]
                synset_b = co_synsets[j]
                
                synset_a_id = synset_a.get('synset_id')
                synset_b_id = synset_b.get('synset_id')
                
                if not synset_a_id or not synset_b_id:
                    continue
                
                pair_key = tuple(sorted([synset_a_id, synset_b_id]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                
                evidence = {
                    'page': page,
                    'source_count': synset_a.get('source_count', 0) + synset_b.get('source_count', 0),
                    'language_count': max(
                        len(synset_a.get('language_coverage', {})),
                        len(synset_b.get('language_coverage', {}))
                    )
                }
                
                base_conf = 0.45
                conf = calculate_dynamic_confidence(base_conf, evidence, is_inferred=False)
                
                rel_id = sha256_short(f"{synset_a_id}_{synset_b_id}_cooccur")
                relations.append(RelationEdge(
                    relation_id=rel_id,
                    source_synset_id=synset_a_id,
                    target_synset_id=synset_b_id,
                    relation_type='co_occurrence',
                    confidence=conf,
                    evidence=evidence,
                    created_at=timestamp
                ))
    
    return relations

# -------------------- 全局Worker状态 --------------------

_es_client: Optional[Elasticsearch] = None
_es_settings: Dict[str, Any] = None

def worker_init(es_settings: Dict[str, Any]):
    """Initialize global ES client for worker process"""
    global _es_client, _es_settings
    _es_settings = es_settings
    try:
        _es_client = ElasticFactory(
            es_settings['hosts'],
            es_settings['port'],
            es_settings['user'],
            es_settings['password'],
            maxsize=20
        ).create()
    except Exception as e:
        logging.error(f"Worker init failed: {e}")

def get_worker_es():
    """获取worker的ES client，如果不存在则创建"""
    global _es_client, _es_settings
    if _es_client is None and _es_settings is not None:
        worker_init(_es_settings)
    return _es_client

# -------------------- Worker Processing --------------------

def worker_process_batch(
    synsets_batch: List[Dict], 
    es_settings: Dict,
    synset_index: str,
    statement_index: str,
    enabled_types: Set[str]
) -> Tuple[List[Dict], Dict]:
    """Worker function to process a batch of synsets"""
    global _es_client, _es_settings
    
    if _es_client is None:
        _es_settings = es_settings
        worker_init(es_settings)
    
    es = _es_client
    logger = logging.getLogger("worker")
    
    # Collect all statement IDs
    all_statement_ids = []
    for synset in synsets_batch:
        member_ids = synset.get('member_statement_ids', [])
        if isinstance(member_ids, str):
            try:
                member_ids = json.loads(member_ids)
            except:
                member_ids = []
        all_statement_ids.extend(member_ids)
    
    # Query statements - 使用并行版本
    statements_map = query_statements_batch_parallel(es, statement_index, all_statement_ids, num_threads=4)
    
    # Detect different types of relations
    all_relations = []
    
    if 'equivalent' in enabled_types:
        all_relations.extend(detect_equivalent_relations(synsets_batch, logger))
    
    if any(t in enabled_types for t in ['hypernym', 'causal', 'geographic_location', 'geographic_contains', 
                                          'part_of', 'has_part', 'member_of', 'follows', 'influenced_by', 
                                          'created_by', 'used_for', 'opposite_of', 'family', 'adjacent',
                                          'affiliated_with', 'employed_by', 'followed_by', 'replaces',
                                          'replaced_by', 'influences', 'derives_from', 'cultural_origin']):
        all_relations.extend(detect_property_based_relations(
            synsets_batch, statements_map, es,
            synset_index, statement_index, logger
        ))
    
    if 'contradiction' in enabled_types:
        all_relations.extend(detect_contradiction_relations(synsets_batch, statements_map, logger))
    
    if any(t in enabled_types for t in ['temporal_before', 'temporal_overlap']):
        all_relations.extend(detect_temporal_relations(synsets_batch, statements_map, logger))
    
    if any(t in enabled_types for t in ['support', 'refute']):
        all_relations.extend(detect_support_refute_relations(synsets_batch, statements_map, logger))
    
    if 'co_occurrence' in enabled_types:
        all_relations.extend(detect_cooccurrence_relations(synsets_batch, logger))
    
    stats = {
        'synsets_processed': len(synsets_batch),
        'statements_queried': len(statements_map),
        'relations_generated': len(all_relations)
    }
    
    return [r.to_dict() for r in all_relations], stats

# -------------------- 异步写入器 --------------------

class AsyncBatchWriter:
    """异步批量写入器"""
    
    def __init__(self, outdir: Path, job_prefix: str, es: Elasticsearch, 
                 es_index: str, logger: logging.Logger, 
                 max_relations_per_file: int = 5_000_000,
                 num_writer_threads: int = 4):
        self.outdir = outdir
        self.job_prefix = job_prefix
        self.es = es
        self.es_index = es_index
        self.logger = logger
        self.max_relations_per_file = max_relations_per_file
        
        self.write_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        self.file_index = 0
        self.current_file_rows = 0
        self.parquet_writer = None
        self.writer_lock = threading.Lock()
        
        self.schema = pa.schema([
            ('relation_id', pa.string()),
            ('source_synset_id', pa.string()),
            ('target_synset_id', pa.string()),
            ('relation_type', pa.string()),
            ('confidence', pa.float32()),
            ('evidence', pa.string()),
            ('created_at', pa.string())
        ])
        
        # 启动写入线程
        self.writer_threads = []
        for i in range(num_writer_threads):
            t = threading.Thread(target=self._writer_worker, daemon=True)
            t.start()
            self.writer_threads.append(t)
    
    def _writer_worker(self):
        """写入工作线程"""
        while not self.stop_event.is_set():
            try:
                task = self.write_queue.get(timeout=1)
                if task is None:
                    break
                
                buffer, task_id = task
                written = self._do_write(buffer)
                self.result_queue.put((task_id, written))
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Writer worker error: {e}")
    
    def _do_write(self, buffer: List[Dict]) -> int:
        """执行实际写入"""
        if not buffer:
            return 0
        
        # 去重
        seen_ids = set()
        unique_buffer = []
        for item in buffer:
            rel_id = item.get('relation_id')
            if rel_id and rel_id not in seen_ids:
                seen_ids.add(rel_id)
                unique_buffer.append(item)
        
        # Parquet写入
        with self.writer_lock:
            try:
                pq_buffer = []
                for item in unique_buffer:
                    pq_item = item.copy()
                    pq_item['evidence'] = json.dumps(item['evidence'], ensure_ascii=False)
                    pq_buffer.append(pq_item)
                
                table = pa.Table.from_pylist(pq_buffer, schema=self.schema)
                
                if self.parquet_writer is None or self.current_file_rows >= self.max_relations_per_file:
                    if self.parquet_writer:
                        self.parquet_writer.close()
                    
                    self.file_index += 1
                    self.current_file_rows = 0
                    out_path = self.outdir / f"synset_relations_{self.job_prefix}{self.file_index:04d}.parquet"
                    
                    while out_path.exists():
                        self.file_index += 1
                        out_path = self.outdir / f"synset_relations_{self.job_prefix}{self.file_index:04d}.parquet"
                    
                    self.parquet_writer = pq.ParquetWriter(str(out_path), self.schema, compression='snappy')
                    self.logger.info(f"Created new parquet file: {out_path}")
                
                self.parquet_writer.write_table(table)
                self.current_file_rows += len(unique_buffer)
                
            except Exception as e:
                self.logger.error(f"Parquet write failed: {e}")
        
        # ES写入
        try:
            actions = [
                {
                    '_op_type': 'index',
                    '_index': self.es_index,
                    '_id': rec['relation_id'],
                    '_source': rec
                }
                for rec in unique_buffer
            ]
            
            success, errors = helpers.bulk(
                self.es, actions, chunk_size=20000, request_timeout=180, raise_on_error=False
            )
            
            if errors:
                self.logger.warning(f"ES bulk had {len(errors)} errors")
                
        except Exception as e:
            self.logger.error(f"ES bulk index failed: {e}")
        
        return len(unique_buffer)
    
    def submit(self, buffer: List[Dict], task_id: int = 0):
        """提交写入任务"""
        self.write_queue.put((buffer.copy(), task_id))
    
    def wait_all(self):
        """等待所有写入完成"""
        self.write_queue.join()
    
    def get_file_info(self) -> Tuple[int, int]:
        """获取当前文件信息"""
        with self.writer_lock:
            return self.file_index, self.current_file_rows
    
    def close(self):
        """关闭写入器"""
        self.stop_event.set()
        
        # 发送停止信号
        for _ in self.writer_threads:
            self.write_queue.put(None)
        
        # 等待线程结束
        for t in self.writer_threads:
            t.join(timeout=5)
        
        # 关闭parquet writer
        with self.writer_lock:
            if self.parquet_writer:
                self.parquet_writer.close()

# -------------------- 主处理流程 --------------------

def process_pipeline_optimized(
    outdir: Path,
    es_synset_index: str,
    es_statement_index: str,
    es_relations_index: str,
    batch_size: int,
    workers: int,
    resume: bool,
    logger: logging.Logger,
    job_id: Optional[str] = None,
    max_synsets: Optional[int] = None,
    max_relations_per_file: int = 5_000_000,
    enabled_types: Set[str] = None,
    checkpoint_interval: int = 1
):
    """优化后的主处理流程"""
    
    if enabled_types is None:
        enabled_types = ALL_RELATION_TYPES
    
    outdir.mkdir(parents=True, exist_ok=True)
    
    if job_id:
        checkpoint_path = outdir / f'checkpoint_{job_id}.json'
        job_prefix = f"{job_id}_"
    else:
        checkpoint_path = outdir / 'checkpoint.json'
        job_prefix = ""
    
    # 加载checkpoint
    if resume:
        checkpoint = load_checkpoint_enhanced(checkpoint_path, logger)
        if checkpoint.completed:
            logger.warning("Job already completed. Delete checkpoint to restart.")
            return
    else:
        checkpoint = CheckpointData()
    
    logger.info(f"Starting from: processed={checkpoint.processed_synsets}, "
               f"last_sort={checkpoint.last_sort_value}")
    logger.info(f"Enabled relation types ({len(enabled_types)}): {sorted(enabled_types)}")
    
    # ES客户端
    es = ElasticFactory(ES_IP_LIST, ES_PORT, ES_USER, ES_PASSWARD, maxsize=50).create()
    create_relations_index(es, es_relations_index, logger)
    
    es_settings = {
        'hosts': ES_IP_LIST,
        'port': ES_PORT,
        'user': ES_USER,
        'password': ES_PASSWARD
    }
    
    # 异步写入器
    async_writer = AsyncBatchWriter(
        outdir, job_prefix, es, es_relations_index, logger,
        max_relations_per_file=max_relations_per_file,
        num_writer_threads=4
    )
    async_writer.file_index = checkpoint.output_file_index
    async_writer.current_file_rows = checkpoint.current_file_rows
    
    buffer = []
    batch_counter = 0
    write_threshold = 20000
    start_time = time.time()
    last_log_time = start_time
    
    # 创建持久进程池
    pool = mp.Pool(processes=workers, initializer=worker_init, initargs=(es_settings,))
    
    try:
        for synsets_batch, last_sort in scroll_all_synsets_with_search_after(
            es, es_synset_index, batch_size, max_synsets,
            last_sort_value=checkpoint.last_sort_value
        ):
            batch_counter += 1
            batch_start_time = time.time()
            
            # 过滤已处理的
            if checkpoint.processed_synset_ids:
                synsets_batch = [
                    s for s in synsets_batch 
                    if s.get('synset_id') not in checkpoint.processed_synset_ids
                ]
            
            if not synsets_batch:
                continue
            
            # 分割为多个chunks并行处理
            chunk_size = max(100, len(synsets_batch) // workers)
            chunks = [synsets_batch[i:i+chunk_size] 
                     for i in range(0, len(synsets_batch), chunk_size)]
            
            # 异步提交任务
            async_results = [
                pool.apply_async(worker_process_batch, 
                               (chunk, es_settings, es_synset_index, es_statement_index, enabled_types))
                for chunk in chunks
            ]
            
            # 收集结果
            batch_relations = 0
            for async_result in async_results:
                try:
                    relations, stats = async_result.get(timeout=600)
                    buffer.extend(relations)
                    batch_relations += len(relations)
                except Exception as e:
                    logger.error(f"Worker failed: {e}")
            
            # 更新checkpoint
            checkpoint.processed_synsets += len(synsets_batch)
            checkpoint.total_relations += batch_relations
            checkpoint.last_sort_value = last_sort
            checkpoint.last_synset_id = synsets_batch[-1].get('synset_id') if synsets_batch else None
            
            for s in synsets_batch:
                sid = s.get('synset_id')
                if sid:
                    checkpoint.processed_synset_ids.add(sid)
            
            # 限制内存中的ID数量
            if len(checkpoint.processed_synset_ids) > 100000:
                checkpoint.processed_synset_ids = set(
                    list(checkpoint.processed_synset_ids)[-50000:]
                )
            
            # 异步写入
            if len(buffer) >= write_threshold:
                async_writer.submit(buffer, batch_counter)
                buffer = []
            
            # 保存checkpoint
            if batch_counter % checkpoint_interval == 0:
                file_idx, file_rows = async_writer.get_file_info()
                checkpoint.output_file_index = file_idx
                checkpoint.current_file_rows = file_rows
                save_checkpoint_enhanced(checkpoint_path, checkpoint, logger)
            
            # 日志
            batch_time = time.time() - batch_start_time
            current_time = time.time()
            
            if current_time - last_log_time >= 30:  # 每30秒输出一次详细日志
                total_time = current_time - start_time
                rate = checkpoint.processed_synsets / total_time if total_time > 0 else 0
                
                if max_synsets:
                    remaining = max_synsets - checkpoint.processed_synsets
                    eta_seconds = remaining / rate if rate > 0 else 0
                    eta_hours = eta_seconds / 3600
                else:
                    eta_hours = 0
                
                logger.info(f"Batch {batch_counter}: {len(synsets_batch)} synsets, "
                           f"{batch_relations} relations in {batch_time:.1f}s | "
                           f"Total: {checkpoint.processed_synsets:,} synsets, "
                           f"{checkpoint.total_relations:,} relations | "
                           f"Rate: {rate:.0f}/s | "
                           f"Buffer: {len(buffer)} | "
                           f"ETA: {eta_hours:.1f}h")
                last_log_time = current_time
            
            # 定期GC
            if batch_counter % 20 == 0:
                gc.collect()
        
        # 最终写入
        if buffer:
            async_writer.submit(buffer, batch_counter + 1)
            buffer = []
        
        # 等待所有写入完成
        async_writer.wait_all()
        
        # 更新最终checkpoint
        file_idx, file_rows = async_writer.get_file_info()
        checkpoint.output_file_index = file_idx
        checkpoint.current_file_rows = file_rows
        checkpoint.completed = True
        save_checkpoint_enhanced(checkpoint_path, checkpoint, logger)
        
        total_time = time.time() - start_time
        logger.info(f"Completed! Synsets: {checkpoint.processed_synsets:,}, "
                   f"Relations: {checkpoint.total_relations:,}, "
                   f"Time: {total_time/3600:.2f}h")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        
        # 保存checkpoint
        file_idx, file_rows = async_writer.get_file_info()
        checkpoint.output_file_index = file_idx
        checkpoint.current_file_rows = file_rows
        save_checkpoint_enhanced(checkpoint_path, checkpoint, logger)
        raise
    
    finally:
        pool.close()
        pool.join()
        async_writer.close()

# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser(description="Optimized FactSynset Relations Builder")
    parser.add_argument('--outdir', type=Path, required=True, help="Output directory")
    parser.add_argument('--es-synset-index', default=ES_FACTSYNSET_INDEX, help="ES synset index")
    parser.add_argument('--es-statement-index', default=ES_FACTSTATEMENT_INDEX, help="ES statement index")
    parser.add_argument('--es-relations-index', default="factnet_synset_relations_v3", help="ES relations index")
    parser.add_argument('--workers', type=int, default=32, help="Number of worker processes")
    parser.add_argument('--batch-size', type=int, default=5000000, help="Batch size for scrolling synsets")
    parser.add_argument('--max-relations-per-file', type=int, default=5_000_000, help="Max relations per file")
    parser.add_argument('--max-synsets', type=int, default=None, help="Max synsets to process (for testing)")
    parser.add_argument('--resume', action='store_true', help="Resume from checkpoint")
    parser.add_argument('--log-dir', type=Path, default=Path('logs'), help="Log directory")
    parser.add_argument('--job-id', type=str, default=None, help="Job ID for checkpointing")
    parser.add_argument('--relation-types', type=str, default=None, 
                       help="Comma-separated relation types to detect (default: all)")
    
    args = parser.parse_args()
    
    logger = setup_logging(args.log_dir)
    logger.info(f"Arguments: {args}")
    
    # Parse relation types
    if args.relation_types:
        enabled_types = set(args.relation_types.split(','))
        invalid = enabled_types - ALL_RELATION_TYPES
        if invalid:
            logger.error(f"Invalid relation types: {invalid}")
            logger.error(f"Valid types: {sorted(ALL_RELATION_TYPES)}")
            sys.exit(1)
    else:
        enabled_types = ALL_RELATION_TYPES
    
    logger.info(f"Total available relation types: {len(ALL_RELATION_TYPES)}")
    logger.info(f"Enabled relation types: {len(enabled_types)}")
    
    process_pipeline_optimized(
        args.outdir,
        args.es_synset_index,
        args.es_statement_index,
        args.es_relations_index,
        args.batch_size,
        args.workers,
        args.resume,
        logger,
        args.job_id,
        args.max_synsets,
        args.max_relations_per_file,
        enabled_types
    )

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()