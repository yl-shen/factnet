#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nohup python 07_build_factsynset_final.py \
--outdir /path/to/output \
--es-index factnet_factsynset_final \
--workers 32 \
--batch-size 5000 \
--log-dir /path/to/logs \
> /path/to/running.log 2>&1 &
"""

import argparse
import json
import logging
import logging.handlers
import os
import sys
import time
import hashlib
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Iterator
from dataclasses import dataclass, asdict
from datetime import datetime
import ssl
import urllib3
import re
from collections import defaultdict, Counter

# Third-party imports
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

from es_config import ES_IP_LIST, ES_PORT, ES_USER, ES_PASSWARD

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -------------------- Data Models --------------------

@dataclass
class FactSynset:
    synset_id: str
    aggregation_key: str
    member_statement_ids: List[str]
    member_factsense_ids: List[str]
    canonical_statement_id: str
    canonical_mentions: Dict[str, Dict]
    aggregate_confidence: float
    language_coverage: Dict[str, int]
    source_count: int
    time_span: Dict[str, str]
    subject_qid: str
    property_pid: str
    normalized_value: str
    value_variants: List[str]  # 记录原始值的变体
    qualifier_variants: List[str]  # 记录qualifiers的变体
    aggregation_reason: str  # 聚合原因（便于调试）
    updated_at: str

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
                timeout=60,
                max_retries=3,
                retry_on_timeout=True
            )
        else:
            es = Elasticsearch(addrs, maxsize=self.maxsize)
        return es

# -------------------- Global Worker State --------------------

_es_client: Optional[Elasticsearch] = None

def worker_init(es_settings: Dict[str, Any]):
    """Initialize global ES client for worker process"""
    global _es_client
    try:
        _es_client = ElasticFactory(
            es_settings['hosts'],
            es_settings['port'],
            es_settings['user'],
            es_settings['password'],
            maxsize=10
        ).create()
    except Exception as e:
        logging.error(f"Worker init failed: {e}")

# -------------------- Logging --------------------

def setup_logging(log_dir: Path, name: str = "factsynset_final") -> logging.Logger:
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
        maxBytes=50*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    
    return logger

# -------------------- Utilities --------------------

def sha256_short(s: str, length: int = 16) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:length]

def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    if checkpoint_path.exists():
        try:
            with checkpoint_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_checkpoint(checkpoint_path: Path, data: Dict[str, Any]):
    tmp_path = checkpoint_path.with_suffix('.tmp')
    try:
        with tmp_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp_path.replace(checkpoint_path)
    except Exception as e:
        logging.warning(f"Failed to save checkpoint: {e}")

# -------------------- Value Normalization --------------------

def normalize_time_value(time_str: str) -> Tuple[str, str]:
    """
    归一化时间值
    返回：(normalized_value, precision)
    
    例子：
    - +1990-01-15T00:00:00Z → (+1990-01-15, day)
    - +1990-01-00T00:00:00Z → (+1990-01, month)
    - +1990-00-00T00:00:00Z → (+1990, year)
    """
    if not isinstance(time_str, str):
        return str(time_str), 'unknown'
    
    # Wikidata时间格式: +YYYY-MM-DDT00:00:00Z
    if not (time_str.startswith('+') or time_str.startswith('-')):
        return time_str, 'unknown'
    
    # 提取日期部分
    date_part = time_str.split('T')[0]
    
    # 判断精度并归一化
    if date_part.endswith('-00-00'):
        # 只精确到年
        return date_part[:-6], 'year'
    elif date_part.endswith('-00'):
        # 精确到月
        return date_part[:-3], 'month'
    else:
        # 精确到日
        return date_part, 'day'

def normalize_quantity_value(quantity_dict: Dict) -> str:
    """
    归一化数量值
    TODO: 未来可以实现单位转换
    
    当前：保持原样，只标准化格式
    """
    if not isinstance(quantity_dict, dict):
        return str(quantity_dict)
    
    amount = quantity_dict.get('amount', '')
    unit = quantity_dict.get('unit', '1')
    
    # 标准化格式
    return f"{amount}@{unit}"

def normalize_value_for_synset(value_str: str) -> Tuple[str, str]:
    """
    归一化值用于FactSynset聚合
    返回：(normalized_value, value_type)
    
    支持的归一化：
    1. 时间：统一精度
    2. 数量：标准化格式（未来可扩展单位转换）
    3. QID：保持不变
    4. 字符串：保持不变
    """
    try:
        # 尝试解析JSON
        value = json.loads(value_str) if isinstance(value_str, str) else value_str
        
        # QID类型
        if isinstance(value, str) and value.startswith('Q') and len(value) > 1:
            if value[1:].isdigit() or (value[1] == '-' and len(value) > 2):
                return value, 'qid'
        
        # 时间类型
        if isinstance(value, str) and (value.startswith('+') or value.startswith('-')):
            normalized, precision = normalize_time_value(value)
            return normalized, f'time_{precision}'
        
        # 数量类型
        if isinstance(value, dict) and 'amount' in value:
            normalized = normalize_quantity_value(value)
            return normalized, 'quantity'
        
        # 地理坐标
        if isinstance(value, dict) and 'lat' in value and 'lon' in value:
            lat = value.get('lat', '')
            lon = value.get('lon', '')
            # 保留2位小数（约1km精度）
            try:
                lat_norm = f"{float(lat):.2f}"
                lon_norm = f"{float(lon):.2f}"
                return f"{lat_norm},{lon_norm}", 'coordinate'
            except:
                return str(value), 'coordinate'
        
        # 多语言文本
        if isinstance(value, dict) and 'text' in value:
            return str(value.get('text', '')), 'monolingualtext'
        
        # 其他类型
        return str(value), 'other'
        
    except Exception as e:
        # 解析失败，返回原始字符串
        return str(value_str), 'string'

# -------------------- Qualifier Normalization --------------------

# 定义应该忽略的临时性qualifiers
IGNORED_QUALIFIERS = {
    'P585',  # point in time - 时间点（通常是临时信息）
    'P813',  # retrieved - 检索日期
    'P577',  # publication date - 有时是冗余的
    # 可以根据需要添加更多
}

# 定义核心qualifiers（应该保留的）
CORE_QUALIFIERS = {
    'P580',  # start time
    'P582',  # end time
    'P276',  # location
    'P17',   # country
    'P1365', # replaces
    'P1366', # replaced by
    # 可以添加更多认为重要的qualifiers
}

def normalize_qualifiers_for_synset(qualifiers_str: str, mode: str = 'core') -> Tuple[str, List[str]]:
    """
    归一化qualifiers用于聚合
    
    Args:
        qualifiers_str: JSON字符串格式的qualifiers
        mode: 'core' - 只保留核心qualifiers, 'ignore_temporal' - 忽略临时性qualifiers
    
    Returns:
        (normalized_qualifiers_json, qualifier_property_list)
    """
    try:
        quals = json.loads(qualifiers_str) if isinstance(qualifiers_str, str) else qualifiers_str
        
        if not isinstance(quals, dict):
            return "{}", []
        
        if mode == 'ignore_temporal':
            # 忽略临时性qualifiers
            filtered_quals = {k: v for k, v in quals.items() if k not in IGNORED_QUALIFIERS}
        elif mode == 'core':
            # 只保留核心qualifiers（更宽松，聚合度更高）
            filtered_quals = {}
        else:
            # 保留所有
            filtered_quals = quals
        
        # 标准化输出
        normalized = json.dumps(filtered_quals, sort_keys=True, ensure_ascii=False)
        qual_list = sorted(quals.keys())
        
        return normalized, qual_list
        
    except Exception:
        return "{}", []

# -------------------- Aggregation Strategy --------------------

def build_hybrid_aggregation_key(
    statement: Dict,
    normalize_values: bool = True,
    normalize_qualifiers: bool = True,
    qualifier_mode: str = 'core'
) -> Tuple[str, Dict[str, Any]]:
    """
    构建混合聚合键
    
    返回：
    - aggregation_key: 用于分组的键
    - metadata: 归一化过程的元数据（用于调试）
    """
    subject_qid = statement.get('subject_qid')
    property_pid = statement.get('property_pid')
    
    if not subject_qid or not property_pid:
        return "", {}
    
    metadata = {
        'original_value': statement.get('value', ''),
        'original_qualifiers': statement.get('qualifiers', '{}')
    }
    
    # 值归一化
    if normalize_values:
        normalized_value, value_type = normalize_value_for_synset(statement.get('value', ''))
        metadata['normalized_value'] = normalized_value
        metadata['value_type'] = value_type
    else:
        normalized_value = statement.get('value', '')
        metadata['normalized_value'] = normalized_value
        metadata['value_type'] = 'raw'
    
    # Qualifier归一化
    if normalize_qualifiers:
        normalized_quals, qual_list = normalize_qualifiers_for_synset(
            statement.get('qualifiers', '{}'),
            mode=qualifier_mode
        )
        metadata['normalized_qualifiers'] = normalized_quals
        metadata['qualifier_list'] = qual_list
    else:
        normalized_quals = statement.get('qualifiers', '{}')
        metadata['normalized_qualifiers'] = normalized_quals
        metadata['qualifier_list'] = []
    
    # 构建聚合键
    agg_key = f"{subject_qid}||{property_pid}||{normalized_value}||{normalized_quals}"
    
    return agg_key, metadata

# -------------------- ES Operations --------------------

def create_factsynset_index(es: Elasticsearch, index_name: str, logger: logging.Logger):
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
                "synset_id": {"type": "keyword"},
                "aggregation_key": {"type": "keyword"},
                "member_statement_ids": {"type": "keyword"},
                "member_factsense_ids": {"type": "keyword"},
                "canonical_statement_id": {"type": "keyword"},
                "canonical_mentions": {"type": "object", "enabled": False}, 
                "aggregate_confidence": {"type": "float"},
                "language_coverage": {"type": "object"},
                "source_count": {"type": "integer"},
                "time_span": {"type": "object"},
                "subject_qid": {"type": "keyword"},
                "property_pid": {"type": "keyword"},
                "normalized_value": {"type": "keyword"},
                "value_variants": {"type": "keyword"},
                "qualifier_variants": {"type": "keyword"},
                "aggregation_reason": {"type": "keyword"},
                "updated_at": {"type": "date"}
            }
        }
    }
    
    try:
        es.indices.create(index=index_name, body=mapping)
        logger.info(f"Successfully created index: {index_name}")
    except Exception as e:
        logger.error(f"Failed to create index {index_name}: {e}")
        raise

def stream_factstatements_from_es(
    es: Elasticsearch, 
    batch_size: int = 5000,
    max_statements: Optional[int] = None
) -> Iterator[List[Dict]]:
    """从ES流式读取FactStatements"""
    query = {"query": {"match_all": {}}}
    
    try:
        resp = es.search(
            index="factnet_factstatements_v1",
            body=query,
            scroll='5m',
            size=batch_size
        )
        
        scroll_id = resp.get('_scroll_id')
        hits = resp.get('hits', {}).get('hits', [])
        total_processed = 0
        
        while hits:
            batch = [hit['_source'] for hit in hits]
            yield batch
            
            total_processed += len(batch)
            if max_statements and total_processed >= max_statements:
                break
            
            resp = es.scroll(scroll_id=scroll_id, scroll='5m')
            scroll_id = resp.get('_scroll_id')
            hits = resp.get('hits', {}).get('hits', [])
            
        if scroll_id:
            try:
                es.clear_scroll(scroll_id=scroll_id)
            except:
                pass
                
    except Exception as e:
        logging.error(f"Failed to stream from ES: {e}")
        raise

def query_factsenses_batch(es: Elasticsearch, statement_ids: List[str]) -> Dict[str, List[Dict]]:
    """Query FactSenses for statement IDs"""
    if not statement_ids:
        return {}
    
    results = defaultdict(list)
    chunk_size = 500
    
    for i in range(0, len(statement_ids), chunk_size):
        chunk = statement_ids[i:i+chunk_size]
        query = {
            "query": {
                "terms": {
                    "belongs_to_statement_id": chunk
                }
            },
            "size": len(chunk) * 10 
        }
        
        try:
            resp = es.search(index="factnet_factsense_v1", body=query)
            for hit in resp.get('hits', {}).get('hits', []):
                src = hit['_source']
                sid = src.get('belongs_to_statement_id')
                if sid:
                    results[sid].append(src)
        except Exception as e:
            logging.warning(f"Failed to query FactSenses batch: {e}")
            
    return results

# -------------------- Core Logic --------------------

def build_synset_from_group(
    agg_key: str, 
    statements: List[Dict], 
    all_factsenses: Dict[str, List[Dict]],
    metadata_list: List[Dict]
) -> Optional[FactSynset]:
    """从一组statements构建FactSynset"""
    if not statements:
        return None
    
    # Extract subject_qid and property_pid from first statement
    first_stmt = statements[0]
    subject_qid = first_stmt.get('subject_qid', '')
    property_pid = first_stmt.get('property_pid', '')
    
    # Get normalized value from metadata
    normalized_value = metadata_list[0].get('normalized_value', '') if metadata_list else ''
        
    synset_id = sha256_short(f"synset_{agg_key}")
    updated_at = datetime.utcnow().isoformat() + 'Z'
    
    member_statement_ids = [s['core_id'] for s in statements if s.get('core_id')]
    member_factsense_ids = []
    
    # Collect value and qualifier variants
    value_variants = list(set([m.get('original_value', '')[:200] for m in metadata_list]))
    qualifier_variants = list(set([m.get('original_qualifiers', '')[:200] for m in metadata_list]))
    
    # Determine aggregation reason
    aggregation_reasons = []
    if len(value_variants) > 1:
        aggregation_reasons.append('value_normalization')
    if len(qualifier_variants) > 1:
        aggregation_reasons.append('qualifier_difference')
    if len(statements) > len(set(s.get('claim_hash', '') for s in statements)):
        aggregation_reasons.append('duplicate_hash')
    
    aggregation_reason = ','.join(aggregation_reasons) if aggregation_reasons else 'exact_match'
    
    # Collect all factsenses
    factsense_list = []
    for sid in member_statement_ids:
        if sid in all_factsenses:
            fs_list = all_factsenses[sid]
            factsense_list.extend(fs_list)
            member_factsense_ids.extend([fs['factsense_id'] for fs in fs_list if fs.get('factsense_id')])
            
    # Canonical Statement: Preferred rank, highest confidence
    sorted_stmts = sorted(
        statements, 
        key=lambda x: (
            1 if x.get('rank') == 'preferred' else (0 if x.get('rank') == 'normal' else -1),
            x.get('confidence', 0)
        ), 
        reverse=True
    )
    
    canonical_statement = sorted_stmts[0]
    canonical_statement_id = canonical_statement.get('core_id')
    
    # Aggregate attributes
    agg_conf = max([s.get('confidence', 0) for s in statements])
    
    # Count unique references
    references = set()
    for s in statements:
        try:
            refs = s.get('references')
            if isinstance(refs, str):
                refs = json.loads(refs)
            if refs:
                for r in refs:
                    references.add(json.dumps(r, sort_keys=True))
        except:
            pass
    source_count = len(references)
    
    # Time span
    time_span = {}
    
    # Canonical Factsenses
    mentions_by_lang = defaultdict(list)
    for fs in factsense_list:
        lang = fs.get('language')
        if lang:
            mentions_by_lang[lang].append(fs)
            
    canonical_mentions = {}
    language_coverage = {}
    
    for lang, mentions in mentions_by_lang.items():
        sorted_m = sorted(
            mentions, 
            key=lambda x: (x.get('confidence', 0), -(x.get('sentence_index', 999))), 
            reverse=True
        )
        best = sorted_m[0]
        canonical_mentions[lang] = {
            "factsense_id": best.get('factsense_id'),
            "sentence": best.get('sentence'),
            "page_title": best.get('page_title'),
            "confidence": best.get('confidence')
        }
        language_coverage[lang] = len(mentions)
        
    return FactSynset(
        synset_id=synset_id,
        aggregation_key=agg_key,
        member_statement_ids=member_statement_ids,
        member_factsense_ids=member_factsense_ids,
        canonical_statement_id=canonical_statement_id,
        canonical_mentions=canonical_mentions,
        aggregate_confidence=agg_conf,
        language_coverage=language_coverage,
        source_count=source_count,
        time_span=time_span,
        subject_qid=subject_qid,
        property_pid=property_pid,
        normalized_value=normalized_value[:500],  # Limit length
        value_variants=value_variants[:10],  # Top 10 variants
        qualifier_variants=qualifier_variants[:10],
        aggregation_reason=aggregation_reason,
        updated_at=updated_at
    )

def worker_process_batch(
    statements_chunk: List[Dict], 
    es_settings: Dict,
    normalize_values: bool = True,
    normalize_qualifiers: bool = True,
    qualifier_mode: str = 'core'
) -> Tuple[List[Dict], Dict]:
    """Process a batch of statements"""
    if _es_client is None:
        worker_init(es_settings)
        
    grouped = defaultdict(list)
    metadata_grouped = defaultdict(list)
    all_sids = []
    
    for stmt in statements_chunk:
        agg_key, metadata = build_hybrid_aggregation_key(
            stmt, 
            normalize_values=normalize_values,
            normalize_qualifiers=normalize_qualifiers,
            qualifier_mode=qualifier_mode
        )
        if agg_key:
            grouped[agg_key].append(stmt)
            metadata_grouped[agg_key].append(metadata)
            if stmt.get('core_id'):
                all_sids.append(stmt['core_id'])
                
    factsense_map = query_factsenses_batch(_es_client, all_sids)
    
    synsets = []
    stats = {
        'input_statements': len(statements_chunk),
        'unique_agg_keys': len(grouped),
        'generated_synsets': 0,
        'aggregation_distribution': Counter(),
        'aggregation_reasons': Counter()
    }
    
    for agg_key, stmts in grouped.items():
        metadata_list = metadata_grouped[agg_key]
        synset = build_synset_from_group(agg_key, stmts, factsense_map, metadata_list)
        if synset:
            synsets.append(synset.to_dict())
            stats['aggregation_distribution'][len(stmts)] += 1
            stats['aggregation_reasons'][synset.aggregation_reason] += 1
            
    stats['generated_synsets'] = len(synsets)
    return synsets, stats

# -------------------- Persistence --------------------

def write_batch(
    buffer: List[Dict], 
    outdir: Path, 
    job_prefix: str, 
    file_index: int, 
    es: Elasticsearch, 
    es_index: str,
    logger: logging.Logger,
    parquet_writer: Optional[pq.ParquetWriter] = None
) -> Tuple[Optional[pq.ParquetWriter], int]:
    
    simple_schema = pa.schema([
        ('synset_id', pa.string()),
        ('aggregation_key', pa.string()),
        ('member_statement_ids', pa.string()),
        ('member_factsense_ids', pa.string()),
        ('canonical_statement_id', pa.string()),
        ('canonical_mentions', pa.string()),
        ('aggregate_confidence', pa.float32()),
        ('language_coverage', pa.string()),
        ('source_count', pa.int32()),
        ('time_span', pa.string()),
        ('subject_qid', pa.string()),
        ('property_pid', pa.string()),
        ('normalized_value', pa.string()),
        ('value_variants', pa.string()),
        ('qualifier_variants', pa.string()),
        ('aggregation_reason', pa.string()),
        ('updated_at', pa.string())
    ])
    
    pq_buffer = []
    for item in buffer:
        pq_item = item.copy()
        pq_item['member_statement_ids'] = json.dumps(item['member_statement_ids'])
        pq_item['member_factsense_ids'] = json.dumps(item['member_factsense_ids'])
        pq_item['canonical_mentions'] = json.dumps(item['canonical_mentions'], ensure_ascii=False)
        pq_item['language_coverage'] = json.dumps(item['language_coverage'])
        pq_item['time_span'] = json.dumps(item['time_span'])
        pq_item['value_variants'] = json.dumps(item['value_variants'])
        pq_item['qualifier_variants'] = json.dumps(item['qualifier_variants'])
        pq_buffer.append(pq_item)
        
    try:
        table = pa.Table.from_pylist(pq_buffer, schema=simple_schema)
        
        if parquet_writer is None:
            out_path = outdir / f"factsynset_part_{job_prefix}{file_index}.parquet"
            parquet_writer = pq.ParquetWriter(out_path, simple_schema, compression='snappy')
            logger.info(f"Created new parquet file: {out_path}")
            
        parquet_writer.write_table(table)
    except Exception as e:
        logger.error(f"Parquet write failed: {e}")
        
    try:
        actions = [
            {
                '_op_type': 'index',
                '_index': es_index,
                '_id': rec['synset_id'],
                '_source': rec
            }
            for rec in buffer
        ]
        helpers.bulk(es, actions, chunk_size=10000, request_timeout=60)
        logger.info(f"Indexed {len(actions)} synsets to ES")
    except Exception as e:
        logger.error(f"ES bulk index failed: {e}")
        
    return parquet_writer, len(pq_buffer)

# -------------------- Main Pipeline --------------------

def process_pipeline_from_es(
    outdir: Path,
    es_index: str,
    batch_size: int,
    workers: int,
    logger: logging.Logger,
    normalize_values: bool = True,
    normalize_qualifiers: bool = True,
    qualifier_mode: str = 'core',
    max_statements: Optional[int] = None,
    job_id: Optional[str] = None,
    max_synsets_per_file: int = 5_000_000
):
    """Main pipeline"""
    outdir.mkdir(parents=True, exist_ok=True)
    
    job_prefix = f"{job_id}_" if job_id else ""
    
    logger.info(f"Starting FactSynset Final Builder")
    logger.info(f"  Normalize values: {normalize_values}")
    logger.info(f"  Normalize qualifiers: {normalize_qualifiers}")
    logger.info(f"  Qualifier mode: {qualifier_mode}")
    logger.info(f"  Output directory: {outdir}")
    logger.info(f"  ES index: {es_index}")
    
    es = ElasticFactory(ES_IP_LIST, ES_PORT, ES_USER, ES_PASSWARD).create()
    create_factsynset_index(es, es_index, logger)
    
    es_settings = {
        'hosts': ES_IP_LIST, 'port': ES_PORT, 'user': ES_USER, 'password': ES_PASSWARD
    }
    
    buffer = []
    parquet_writer = None
    output_file_index = 0
    current_output_rows = 0
    write_threshold = 5000
    
    total_statements = 0
    total_synsets = 0
    global_stats = {
        'aggregation_distribution': Counter(),
        'aggregation_reasons': Counter(),
        'total_agg_keys': 0
    }
    
    start_time = time.time()
    
    try:
        for batch in stream_factstatements_from_es(es, batch_size, max_statements):
            total_statements += len(batch)
            logger.info(f"Processing batch of {len(batch)} statements (total: {total_statements})")
            
            with mp.Pool(processes=workers, initializer=worker_init, initargs=(es_settings,)) as pool:
                chunk_size = max(1, len(batch) // workers)
                chunks = [batch[i:i+chunk_size] for i in range(0, len(batch), chunk_size)]
                results = pool.starmap(
                    worker_process_batch, 
                    [(c, es_settings, normalize_values, normalize_qualifiers, qualifier_mode) for c in chunks]
                )
                
            for res_synsets, res_stats in results:
                buffer.extend(res_synsets)
                total_synsets += len(res_synsets)
                global_stats['total_agg_keys'] += res_stats['unique_agg_keys']
                global_stats['aggregation_distribution'].update(res_stats['aggregation_distribution'])
                global_stats['aggregation_reasons'].update(res_stats['aggregation_reasons'])
                
            if len(buffer) >= write_threshold:
                parquet_writer, num_written = write_batch(
                    buffer, outdir, job_prefix, output_file_index,
                    es, es_index, logger, parquet_writer
                )
                current_output_rows += num_written
                buffer.clear()
                
                if current_output_rows >= max_synsets_per_file:
                    if parquet_writer:
                        parquet_writer.close()
                        parquet_writer = None
                    output_file_index += 1
                    current_output_rows = 0
                    
                elapsed = time.time() - start_time
                rate = total_statements / elapsed if elapsed > 0 else 0
                agg_factor = total_statements / max(1, total_synsets)
                logger.info(f"Progress: {total_statements:,} statements → {total_synsets:,} synsets "
                           f"(factor: {agg_factor:.2f}x, rate: {rate:.0f} stmt/s)")
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in pipeline: {e}", exc_info=True)
        raise
    finally:
        if buffer:
            parquet_writer, num_written = write_batch(
                buffer, outdir, job_prefix, output_file_index,
                es, es_index, logger, parquet_writer
            )
            total_synsets += len(buffer)
            buffer.clear()
            
        if parquet_writer:
            parquet_writer.close()
            
    # Final statistics
    elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info("FINAL STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total FactStatements: {total_statements:,}")
    logger.info(f"Total FactSynsets: {total_synsets:,}")
    logger.info(f"Aggregation factor: {total_statements/max(1, total_synsets):.2f}x")
    logger.info(f"Total time: {elapsed:.2f}s")
    logger.info(f"Processing rate: {total_statements/elapsed:.0f} stmt/s")
    logger.info("")
    logger.info("Synset size distribution:")
    for size in sorted(global_stats['aggregation_distribution'].keys())[:20]:
        count = global_stats['aggregation_distribution'][size]
        pct = 100 * count / total_synsets
        logger.info(f"  {size:3d} statements: {count:8,} synsets ({pct:5.2f}%)")
    logger.info("")
    logger.info("Aggregation reasons:")
    for reason, count in global_stats['aggregation_reasons'].most_common():
        pct = 100 * count / total_synsets
        logger.info(f"  {reason}: {count:8,} synsets ({pct:5.2f}%)")
    logger.info("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="FactSynset Final Builder (Strategy C)")
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--es-index', default='factnet_factsynset_final')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=5000)
    parser.add_argument('--log-dir', type=Path, default=Path('logs'))
    parser.add_argument('--job-id', type=str, default=None)
    parser.add_argument('--max-synsets-per-file', type=int, default=5_000_000)
    parser.add_argument('--max-statements', type=int, default=None)
    parser.add_argument('--no-normalize-values', action='store_true',
                       help="Disable value normalization")
    parser.add_argument('--no-normalize-qualifiers', action='store_true',
                       help="Disable qualifier normalization")
    parser.add_argument('--qualifier-mode', choices=['core', 'ignore_temporal', 'all'],
                       default='core',
                       help="Qualifier normalization mode")
    
    args = parser.parse_args()
    logger = setup_logging(args.log_dir)
    
    logger.info(f"Starting with arguments: {args}")

    process_pipeline_from_es(
        outdir=args.outdir,
        es_index=args.es_index,
        batch_size=args.batch_size,
        workers=args.workers,
        logger=logger,
        normalize_values=not args.no_normalize_values,
        normalize_qualifiers=not args.no_normalize_qualifiers,
        qualifier_mode=args.qualifier_mode,
        max_statements=args.max_statements,
        job_id=args.job_id,
        max_synsets_per_file=args.max_synsets_per_file
    )

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
