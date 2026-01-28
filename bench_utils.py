#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FactNet-Bench Common Utilities

Common utilities for all benchmark construction scripts:
- Deterministic split assignment based on synset_id
- ES client factory
- Value normalization
- Streaming helpers
"""

import hashlib
import json
import ssl
import urllib3
from typing import Dict, List, Any, Optional, Iterator, Tuple
from datetime import datetime
from elasticsearch import Elasticsearch
from dataclasses import dataclass, asdict

from es_config import ES_IP_LIST, ES_PORT, ES_USER, ES_PASSWARD

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -------------------- Configuration --------------------

# Default build_id for deterministic split assignment
DEFAULT_BUILD_ID = "factnet_bench_v1"

# Target languages (18 languages from the paper)
TARGET_LANGUAGES = [
    'en', 'zh', 'es', 'fr', 'de', 'ru', 'ar', 'hi', 'id', 
    'it', 'ja', 'ko', 'nl', 'pl', 'pt', 'th', 'tr', 'vi'
]

# Split ratios (80/10/10)
TRAIN_THRESHOLD = 80  # h(y) mod 100 < 80 -> Train
DEV_THRESHOLD = 90    # 80 <= h(y) mod 100 < 90 -> Dev
                      # 90 <= h(y) mod 100 -> Test

# -------------------- Split Assignment --------------------

def compute_split_hash(build_id: str, synset_id: str) -> int:
    """
    Compute deterministic split hash for a synset.
    
    Based on SHA1(build_id || synset_id), take first 4 bytes as u32.
    
    Args:
        build_id: Build identifier for reproducibility
        synset_id: FactSynset ID
        
    Returns:
        Unsigned 32-bit integer hash value
    """
    combined = f"{build_id}{synset_id}".encode('utf-8')
    sha1_hash = hashlib.sha1(combined).digest()
    # Take first 4 bytes as unsigned 32-bit integer (big-endian)
    h = int.from_bytes(sha1_hash[:4], byteorder='big', signed=False)
    return h


def get_split(hash_value: int) -> str:
    """
    Determine split based on hash value.
    
    Args:
        hash_value: Hash value from compute_split_hash
        
    Returns:
        "train", "dev", or "test"
    """
    mod_value = hash_value % 100
    if mod_value < TRAIN_THRESHOLD:
        return "train"
    elif mod_value < DEV_THRESHOLD:
        return "dev"
    else:
        return "test"


def get_synset_split(synset_id: str, build_id: str = DEFAULT_BUILD_ID) -> str:
    """
    Get the split assignment for a synset.
    
    Args:
        synset_id: FactSynset ID
        build_id: Build identifier for reproducibility
        
    Returns:
        "train", "dev", or "test"
    """
    h = compute_split_hash(build_id, synset_id)
    return get_split(h)


# -------------------- ES Factory --------------------

class ElasticFactory:
    """Factory for creating Elasticsearch clients."""
    
    def __init__(
        self, 
        host: list = None, 
        port: str = None, 
        username: str = None, 
        password: str = None, 
        maxsize: int = 25
    ):
        self.host = host or ES_IP_LIST
        self.port = port or ES_PORT
        self.username = username or ES_USER
        self.password = password or ES_PASSWARD
        self.maxsize = maxsize

    def create(self) -> Elasticsearch:
        """Create and return an Elasticsearch client."""
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


def get_es_client(maxsize: int = 25) -> Elasticsearch:
    """Convenience function to get an ES client with default settings."""
    return ElasticFactory(maxsize=maxsize).create()


# -------------------- ES Streaming Helpers --------------------

def scroll_all_synsets(
    es: Elasticsearch,
    index_name: str = "factnet_factsynset_v1",
    batch_size: int = 1000,
    max_synsets: Optional[int] = None,
    source_fields: Optional[List[str]] = None
) -> Iterator[List[Dict]]:
    """
    Stream all synsets from ES using search_after.
    
    Args:
        es: Elasticsearch client
        index_name: Synset index name
        batch_size: Batch size for each query
        max_synsets: Optional limit on total synsets
        source_fields: Optional list of fields to include in _source
        
    Yields:
        Batches of synset documents
    """
    query = {
        "query": {"match_all": {}},
        "sort": [
            {"synset_id": "asc"},
            {"_id": "asc"}
        ],
        "size": batch_size
    }
    
    if source_fields:
        query["_source"] = source_fields
    
    total_fetched = 0
    
    try:
        while True:
            resp = es.search(index=index_name, body=query, request_timeout=300)
            hits = resp['hits']['hits']
            
            if not hits:
                break
            
            batch = [hit['_source'] for hit in hits]
            last_sort = hits[-1]['sort']
            
            yield batch
            
            total_fetched += len(batch)
            if max_synsets and total_fetched >= max_synsets:
                break
            
            query["search_after"] = last_sort
            
    except Exception as e:
        raise RuntimeError(f"Error streaming synsets: {e}")


def scroll_all_factsenses(
    es: Elasticsearch,
    index_name: str = "factnet_factsense_v1",
    batch_size: int = 2000,
    max_items: Optional[int] = None,
    language: Optional[str] = None,
    source_fields: Optional[List[str]] = None
) -> Iterator[List[Dict]]:
    """
    Stream all FactSenses from ES.
    
    Args:
        es: Elasticsearch client
        index_name: FactSense index name
        batch_size: Batch size
        max_items: Optional limit
        language: Optional language filter
        source_fields: Optional fields to include
        
    Yields:
        Batches of FactSense documents
    """
    if language:
        query = {
            "query": {"term": {"language": language}},
            "sort": [{"factsense_id": "asc"}],
            "size": batch_size
        }
    else:
        query = {
            "query": {"match_all": {}},
            "sort": [{"factsense_id": "asc"}],
            "size": batch_size
        }
    
    if source_fields:
        query["_source"] = source_fields
    
    total_fetched = 0
    
    try:
        while True:
            resp = es.search(index=index_name, body=query, request_timeout=300)
            hits = resp['hits']['hits']
            
            if not hits:
                break
            
            batch = [hit['_source'] for hit in hits]
            last_sort = hits[-1]['sort']
            
            yield batch
            
            total_fetched += len(batch)
            if max_items and total_fetched >= max_items:
                break
            
            query["search_after"] = last_sort
            
    except Exception as e:
        raise RuntimeError(f"Error streaming FactSenses: {e}")


def query_statements_by_ids(
    es: Elasticsearch,
    statement_ids: List[str],
    index_name: str = "factnet_factstatements_v1",
    source_fields: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Query FactStatements by their IDs.
    
    Args:
        es: Elasticsearch client
        statement_ids: List of statement IDs
        index_name: Statement index name
        source_fields: Optional fields to include
        
    Returns:
        Dict mapping statement_id -> statement document
    """
    if not statement_ids:
        return {}
    
    results = {}
    chunk_size = 10000
    
    for i in range(0, len(statement_ids), chunk_size):
        chunk = statement_ids[i:i+chunk_size]
        query = {
            "query": {
                "terms": {
                    "core_id": chunk
                }
            },
            "size": len(chunk)
        }
        
        if source_fields:
            query["_source"] = source_fields
        
        try:
            resp = es.search(index=index_name, body=query, request_timeout=120)
            for hit in resp.get('hits', {}).get('hits', []):
                src = hit['_source']
                sid = src.get('core_id')
                if sid:
                    results[sid] = src
        except Exception as e:
            print(f"Warning: Failed to query statements batch: {e}")
    
    return results


def query_labels_for_qids(
    es: Elasticsearch,
    qids: List[str],
    languages: Optional[List[str]] = None,
    index_name: str = "factnet_labels_v1"
) -> Dict[str, Dict[str, Dict]]:
    """
    Query labels for multiple QIDs.
    
    Args:
        es: Elasticsearch client
        qids: List of QIDs
        languages: Optional language filter
        index_name: Labels index name
        
    Returns:
        Dict: {qid: {lang: {'label': str, 'aliases': [str]}}}
    """
    if not qids:
        return {}
    
    results = {}
    chunk_size = 5000
    
    for i in range(0, len(qids), chunk_size):
        chunk = qids[i:i+chunk_size]
        
        if languages:
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"terms": {"subject_qid": chunk}},
                            {"terms": {"language": languages}}
                        ]
                    }
                },
                "size": len(chunk) * len(languages)
            }
        else:
            query = {
                "query": {
                    "terms": {
                        "subject_qid": chunk
                    }
                },
                "size": len(chunk) * 50  # Assume max 50 languages per QID
            }
        
        try:
            resp = es.search(index=index_name, body=query, request_timeout=120)
            for hit in resp.get('hits', {}).get('hits', []):
                src = hit['_source']
                qid = src.get('subject_qid')
                lang = src.get('language', 'und')
                label = src.get('label')
                aliases = src.get('aliases', [])
                
                if qid not in results:
                    results[qid] = {}
                results[qid][lang] = {
                    'label': label,
                    'aliases': aliases if isinstance(aliases, list) else []
                }
        except Exception as e:
            print(f"Warning: Failed to query labels batch: {e}")
    
    return results


# -------------------- Value Normalization --------------------

def normalize_value(value: Any) -> str:
    """
    Normalize a value for answer matching.
    
    Args:
        value: Raw value (could be QID, time, quantity, etc.)
        
    Returns:
        Normalized string representation
    """
    if value is None:
        return ""
    
    if isinstance(value, str):
        # Try to parse as JSON
        try:
            parsed = json.loads(value)
            return normalize_value(parsed)
        except (json.JSONDecodeError, TypeError):
            # Already a string
            return value.strip()
    
    if isinstance(value, dict):
        # Time value
        if 'time' in value:
            return normalize_time(value.get('time', ''))
        # Quantity value
        if 'amount' in value:
            return normalize_quantity(value)
        # Monolingualtext
        if 'text' in value:
            return str(value.get('text', '')).strip()
        # Coordinate
        if 'lat' in value and 'lon' in value:
            try:
                return f"{float(value['lat']):.4f},{float(value['lon']):.4f}"
            except:
                return str(value)
    
    return str(value).strip()


def normalize_time(time_str: str) -> str:
    """
    Normalize Wikidata time values.
    
    Examples:
        +1990-01-15T00:00:00Z -> 1990-01-15
        +1990-01-00T00:00:00Z -> 1990-01
        +1990-00-00T00:00:00Z -> 1990
    """
    if not isinstance(time_str, str):
        return str(time_str)
    
    # Remove leading + or -
    time_str = time_str.lstrip('+')
    
    # Extract date part
    date_part = time_str.split('T')[0]
    
    # Handle -00 precision
    if date_part.endswith('-00-00'):
        return date_part[:-6]  # Year only
    elif date_part.endswith('-00'):
        return date_part[:-3]  # Year-month
    
    return date_part


def normalize_quantity(quantity_dict: Dict) -> str:
    """
    Normalize quantity values.
    
    Args:
        quantity_dict: Dict with 'amount' and optionally 'unit'
        
    Returns:
        Normalized string like "123@Q...(unit)" or just "123"
    """
    amount = quantity_dict.get('amount', '')
    unit = quantity_dict.get('unit', '1')
    
    # Clean amount (remove leading +)
    if isinstance(amount, str):
        amount = amount.lstrip('+')
    
    if unit and unit != '1':
        return f"{amount}@{unit}"
    return str(amount)


def extract_qid_from_value(value: Any) -> Optional[str]:
    """
    Extract QID if value represents an entity.
    
    Args:
        value: Value to check
        
    Returns:
        QID string if entity-valued, None otherwise
    """
    if isinstance(value, str):
        # Try JSON parse first
        try:
            parsed = json.loads(value)
            return extract_qid_from_value(parsed)
        except:
            pass
        
        # Check if it's a QID directly
        if value.startswith('Q') and len(value) > 1:
            rest = value[1:]
            if rest.isdigit() or (rest.startswith('-') and rest[1:].isdigit()):
                return value
    
    if isinstance(value, dict):
        return value.get('id')
    
    return None


# -------------------- Hashing Utilities --------------------

def sha256_short(s: str, length: int = 16) -> str:
    """Generate a short SHA256 hash."""
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:length]


def generate_instance_id(prefix: str, *args) -> str:
    """
    Generate a deterministic instance ID.
    
    Args:
        prefix: ID prefix (e.g., "kgc", "mkqa", "mfc")
        *args: Components to hash
        
    Returns:
        Unique instance ID
    """
    combined = "_".join(str(a) for a in args)
    hash_part = sha256_short(combined, 12)
    return f"{prefix}_{hash_part}"


# -------------------- Data Classes --------------------

@dataclass
class KGCTriple:
    """A KGC triple instance."""
    subject: str  # QID
    predicate: str  # PID
    object: str  # QID
    synset_id: str
    split: str
    
    def to_tsv_line(self) -> str:
        return f"{self.subject}\t{self.predicate}\t{self.object}"
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MKQAInstance:
    """An MKQA question instance."""
    id: str
    question: str
    logical_form: str
    answers: List[str]
    language: str
    synset_ids: List[str]
    split: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MFCInstance:
    """An MFC claim instance."""
    id: str
    claim: str
    label: str  # SUPPORTED, REFUTED, NEI
    language: str
    evidence: List[Dict]  # [{evidence_pointer, text, spans}]
    source_synset_id: str
    split: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


# -------------------- File I/O --------------------

def write_jsonl(items: List[Dict], path: str):
    """Write items to a JSONL file."""
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def read_jsonl(path: str) -> List[Dict]:
    """Read items from a JSONL file."""
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_tsv(lines: List[str], path: str, header: Optional[str] = None):
    """Write lines to a TSV file."""
    with open(path, 'w', encoding='utf-8') as f:
        if header:
            f.write(header + '\n')
        for line in lines:
            f.write(line + '\n')


def get_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.utcnow().isoformat() + 'Z'


# -------------------- Test Functions --------------------

def test_split_distribution(n_samples: int = 10000):
    """
    Test that split distribution is approximately 80/10/10.
    """
    from collections import Counter
    
    splits = Counter()
    for i in range(n_samples):
        fake_synset_id = f"synset_{sha256_short(str(i), 16)}"
        split = get_synset_split(fake_synset_id)
        splits[split] += 1
    
    print(f"Split distribution over {n_samples} samples:")
    for split in ['train', 'dev', 'test']:
        pct = 100 * splits[split] / n_samples
        print(f"  {split}: {splits[split]} ({pct:.1f}%)")
    
    return splits


if __name__ == "__main__":
    # Run tests
    print("Testing split distribution...")
    test_split_distribution(10000)
    
    print("\nTesting ES connection...")
    try:
        es = get_es_client()
        if es.ping():
            print("  ES connection: OK")
        else:
            print("  ES connection: FAILED (ping returned False)")
    except Exception as e:
        print(f"  ES connection: FAILED ({e})")
