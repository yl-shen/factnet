#!/usr/bin/env python3
"""
nohup python 05_build_factsense.py \
--file-list 01_factstatement/factstatements_part_0.parquet \
--outdir 05_factsense \
--job-id job-000 \
--workers 16 \
--batch-size 100 \
--log-dir 05_factsense/logs \
--resume \
> 05_factsense/runing.log 2>&1 &
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
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import ssl
import urllib3
import gc
import re

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
class FactSense:
    """FactSense 数据模型"""
    factsense_id: str
    belongs_to_statement_id: str
    subject_qid: str
    property_pid: str
    value_qid: Optional[str]
    language: str
    page_id: Optional[int]
    page_title: Optional[str]
    page_namespace: int
    match_type: str  # sitelink, label_based, sentence_cooccurrence, wikilink, redirect
    sentence: Optional[str]
    sentence_index: Optional[int]
    confidence: float
    subject_label: Optional[str]
    value_label: Optional[str]
    extraction_method: str
    extraction_ts: str
    
    def to_dict(self) -> Dict:
        """Convert to dict for storage"""
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
        # Test connection
        if not _es_client.ping():
            logging.warning("ES Ping failed in worker init")
    except Exception as e:
        logging.error(f"Worker init failed: {e}")


# -------------------- Logging --------------------

def setup_logging(log_dir: Path, name: str = "factsense_builder") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(ch)
    
    # File handler with rotation
    log_file = log_dir / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.handlers.RotatingFileHandler(
        str(log_file),
        maxBytes=50*1024*1024,  # 50MB
        backupCount=5,
        encoding='utf-8'
    )
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    
    return logger


# -------------------- Utilities --------------------

def sha256_short(s: str, length: int = 16) -> str:
    """Generate short hash"""
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:length]


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load checkpoint file"""
    if checkpoint_path.exists():
        try:
            with checkpoint_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load checkpoint: {e}")
    return {}


def save_checkpoint(checkpoint_path: Path, data: Dict[str, Any]):
    """Save checkpoint with atomic write"""
    tmp_path = checkpoint_path.with_suffix('.tmp')
    try:
        with tmp_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp_path.replace(checkpoint_path)
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")


def simple_sentence_split(text: str) -> List[str]:
    """Simple sentence splitter"""
    if not text:
        return []
    # Split by common sentence terminators
    sentences = re.split(r'[。.!?！？\n]+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


# -------------------- ES Index Creation --------------------

def create_factsense_index(es: Elasticsearch, index_name: str, logger: logging.Logger):
    """Create FactSense ES index with proper mapping"""
    if es.indices.exists(index=index_name):
        logger.info(f"Index {index_name} already exists")
        return
    
    mapping = {
        "settings": {
            "number_of_shards": 30,
            "number_of_replicas": 1,
            "refresh_interval": "30s",
            "analysis": {
                "analyzer": {
                    "lowercase_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "factsense_id": {"type": "keyword"},
                "belongs_to_statement_id": {"type": "keyword"},
                "subject_qid": {"type": "keyword"},
                "property_pid": {"type": "keyword"},
                "value_qid": {"type": "keyword"},
                "language": {"type": "keyword"},
                "page_id": {"type": "long"},
                "page_title": {
                    "type": "text",
                    "analyzer": "lowercase_analyzer",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "page_namespace": {"type": "integer"},
                "match_type": {"type": "keyword"},
                "sentence": {"type": "text", "analyzer": "lowercase_analyzer"},
                "sentence_index": {"type": "integer"},
                "confidence": {"type": "float"},
                "subject_label": {"type": "text", "analyzer": "lowercase_analyzer"},
                "value_label": {"type": "text", "analyzer": "lowercase_analyzer"},
                "extraction_method": {"type": "keyword"},
                "extraction_ts": {"type": "date"}
            }
        }
    }
    
    try:
        es.indices.create(index=index_name, body=mapping)
        logger.info(f"Successfully created index: {index_name}")
    except Exception as e:
        logger.error(f"Failed to create index {index_name}: {e}")
        raise


# -------------------- ES Query Functions --------------------

def query_labels_batch(es: Elasticsearch, qids: List[str], languages: List[str] = None) -> Dict[str, Dict[str, Dict]]:
    """
    Batch query labels for multiple QIDs
    Returns: {qid: {lang: {'label': str, 'aliases': [str]}}}
    """
    if not qids:
        return {}
    
    # Build query for all QIDs
    query = {
        "query": {
            "terms": {
                "subject_qid": qids
            }
        },
        "size": min(10000, len(qids) * 50)  # Estimate max results
    }
    
    if languages:
        query["query"] = {
            "bool": {
                "must": [
                    {"terms": {"subject_qid": qids}},
                    {"terms": {"language": languages}}
                ]
            }
        }
    
    results = {}
    try:
        resp = es.search(index="factnet_labels_v1", body=query)
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
        logging.warning(f"Failed to query labels: {e}")
    
    return results


def query_pages_by_title(es: Elasticsearch, titles: List[str], lang: str, limit: int = 100) -> List[Dict]:
    """
    Query Wikipedia pages by titles
    Returns: [{'page_id', 'title', 'namespace', 'text', 'sentences'}]
    """
    if not titles:
        return []
    
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"lang": lang}},
                    {"terms": {"title_text": titles}}
                ]
            }
        },
        "size": limit
    }
    
    pages = []
    try:
        resp = es.search(index="factnet_pages_v1", body=query)
        for hit in resp.get('hits', {}).get('hits', []):
            src = hit['_source']
            pages.append({
                'page_id': src.get('page_id'),
                'title': src.get('title'),
                'namespace': src.get('namespace', 0),
                'text': src.get('text', ''),
                'sentences': src.get('sentences', [])
            })
    except Exception as e:
        logging.warning(f"Failed to query pages by title: {e}")
    
    return pages


def query_page_by_id(es: Elasticsearch, page_id: int, lang: str) -> Optional[Dict]:
    """Query a single page by ID"""
    doc_id = f"{lang}_{page_id}"
    try:
        resp = es.get(index="factnet_pages_v1", id=doc_id)
        if resp.get('found'):
            src = resp['_source']
            return {
                'page_id': src.get('page_id'),
                'title': src.get('title'),
                'namespace': src.get('namespace', 0),
                'text': src.get('text', ''),
                'sentences': src.get('sentences', [])
            }
    except Exception:
        pass
    return None


def query_pages_by_redirect_target(es: Elasticsearch, target_title: str, lang: str, limit: int = 5) -> List[Dict]:
    """
    Query pages that have redirects pointing to the target title
    利用 redirects nested 字段查找重定向到目标标题的页面
    
    Returns: [{'page_id', 'title', 'namespace', 'text', 'sentences', 'pagelinks', 'redirects'}]
    """
    if not target_title:
        return []
    
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"lang": lang}},
                    {
                        "nested": {
                            "path": "redirects",
                            "query": {
                                "term": {
                                    "redirects.title": target_title
                                }
                            }
                        }
                    }
                ]
            }
        },
        "size": limit
    }
    
    pages = []
    try:
        resp = es.search(index="factnet_pages_v1", body=query)
        for hit in resp.get('hits', {}).get('hits', []):
            src = hit['_source']
            pages.append({
                'page_id': src.get('page_id'),
                'title': src.get('title'),
                'namespace': src.get('namespace', 0),
                'text': src.get('text', ''),
                'sentences': src.get('sentences', []),
                'pagelinks': src.get('pagelinks', []),
                'redirects': src.get('redirects', [])
            })
    except Exception as e:
        logging.warning(f"Failed to query pages by redirect target: {e}")
    
    return pages


def query_pages_batch_by_title(es: Elasticsearch, titles: List[str], lang: str) -> Dict[str, Dict]:
    """
    Batch query pages by titles.
    Returns: {title_lower: page_data_dict}
    """
    if not titles:
        return {}
    
    # Remove duplicates and empty
    titles = list(set([t for t in titles if t]))
    if not titles:
        return {}

    results = {}
    
    # Chunking to avoid too large query
    chunk_size = 100
    for i in range(0, len(titles), chunk_size):
        chunk = titles[i:i+chunk_size]
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"lang": lang}},
                        {"terms": {"title_text": chunk}}
                    ]
                }
            },
            "size": len(chunk) * 2  # slight buffer for case variations
        }
        
        try:
            resp = es.search(index="factnet_pages_v1", body=query)
            for hit in resp.get('hits', {}).get('hits', []):
                src = hit['_source']
                title = src.get('title')
                if title:
                    page_data = {
                        'page_id': src.get('page_id'),
                        'title': title,
                        'namespace': src.get('namespace', 0),
                        'text': src.get('text', ''),
                        'sentences': src.get('sentences', []),
                        'pagelinks': src.get('pagelinks', []),
                        'redirects': src.get('redirects', [])
                    }
                    results[title.lower()] = page_data
        except Exception as e:
            logging.warning(f"Failed to batch query pages by title: {e}")
            
    return results


def query_pages_batch_by_redirect(es: Elasticsearch, targets: List[str], lang: str) -> Dict[str, List[Dict]]:
    """
    Batch query pages by redirect targets.
    Returns: {target_title_lower: [page_data_list]}
    """
    if not targets:
        return {}
        
    targets = list(set([t for t in targets if t]))
    if not targets:
        return {}

    results = {}
    
    # Chunking
    chunk_size = 50
    for i in range(0, len(targets), chunk_size):
        chunk = targets[i:i+chunk_size]
        
        # We need to find pages where ANY of the redirects point to ANY of our targets
        # Best way is a terms query on the nested field
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"lang": lang}},
                        {
                            "nested": {
                                "path": "redirects",
                                "query": {
                                    "terms": {
                                        "redirects.title": chunk
                                    }
                                }
                            }
                        }
                    ]
                }
            },
            "size": len(chunk) * 5  # Assume avg 5 pages redirecting to one target
        }
        
        try:
            resp = es.search(index="factnet_pages_v1", body=query)
            for hit in resp.get('hits', {}).get('hits', []):
                src = hit['_source']
                page_data = {
                    'page_id': src.get('page_id'),
                    'title': src.get('title'),
                    'namespace': src.get('namespace', 0),
                    'text': src.get('text', ''),
                    'sentences': src.get('sentences', []),
                    'pagelinks': src.get('pagelinks', []),
                    'redirects': src.get('redirects', [])
                }
                
                # Determine which target(s) this page redirects TO
                # Iterate through redirects in the page source
                page_redirects = src.get('redirects', [])
                for r in page_redirects:
                    r_title = r.get('title', '')
                    if r_title and r_title in chunk: # Check if this redirect matches one of our targets
                        r_lower = r_title.lower()
                        if r_lower not in results:
                            results[r_lower] = []
                        results[r_lower].append(page_data)
                        
        except Exception as e:
            logging.warning(f"Failed to batch query pages by redirect: {e}")

    return results


# -------------------- FactSense Generation --------------------

def generate_factsense_from_statement_v2(
    statement: Dict,
    labels_data: Dict[str, Dict[str, Dict]],
    prefetched_pages: Dict[str, Dict[str, Dict]], # {lang: {title_lower: page_data}}
    prefetched_redirects: Dict[str, Dict[str, List[Dict]]], # {lang: {target_lower: [pages]}}
    target_languages: List[str] = None
) -> List[FactSense]:
    """
    Generate FactSense records for a single statement using PRE-FETCHED data.
    """
    factsenses = []
    
    statement_id = statement.get('core_id')
    subject_qid = statement.get('subject_qid')
    property_pid = statement.get('property_pid')
    value = statement.get('value')
    
    if not (statement_id and subject_qid and property_pid):
        return factsenses
    
    # Extract value_qid if value is a QID
    value_qid = None
    if isinstance(value, str) and value.startswith('Q') and value[1:].isdigit():
        value_qid = value
    
    # Parse sitelinks
    sitelinks = {}
    sitelinks_str = statement.get('sitelinks', '{}')
    if isinstance(sitelinks_str, str):
        try:
            sitelinks = json.loads(sitelinks_str)
        except:
            sitelinks = {}
    
    # Get subject labels
    subject_labels = labels_data.get(subject_qid, {})
    
    # Get value labels if value is QID
    value_labels = {}
    if value_qid:
        value_labels = labels_data.get(value_qid, {})
    
    # Determine languages to process
    if target_languages:
        languages = target_languages
    else:
        languages = set(subject_labels.keys())
        if value_labels:
            languages.update(value_labels.keys())
        languages = list(languages)
    
    if not languages:
        languages = ['en']
    
    extraction_ts = datetime.utcnow().isoformat() + 'Z'
    
    # Process each language
    for lang in languages:
        subject_label = subject_labels.get(lang, {}).get('label', subject_qid)
        value_label = None
        if value_qid and lang in value_labels:
            value_label = value_labels[lang].get('label', value_qid)
        
        # --- Strategy 1: Sitelink match ---
        sitelink_key = f"{lang}wiki"
        sitelink_title = sitelinks.get(sitelink_key)
        
        if sitelink_title:
            # Lookup in prefetched pages
            page_data = None
            if lang in prefetched_pages and sitelink_title.lower() in prefetched_pages[lang]:
                page_data = prefetched_pages[lang][sitelink_title.lower()]
            
            if page_data:
                # Strategy 1a: Pagelinks
                pagelinks = page_data.get('pagelinks', [])
                if value_label and pagelinks:
                    found_pagelink = False
                    for link in pagelinks:
                        link_title = link.get('title', '')
                        if link_title and value_label.lower() == link_title.lower():
                            found_pagelink = True
                            fs_id = sha256_short(f"{statement_id}_{lang}_{page_data.get('page_id')}_pagelink")
                            fs = FactSense(
                                factsense_id=fs_id,
                                belongs_to_statement_id=statement_id,
                                subject_qid=subject_qid,
                                property_pid=property_pid,
                                value_qid=value_qid,
                                language=lang,
                                page_id=page_data.get('page_id'),
                                page_title=page_data.get('title'),
                                page_namespace=page_data.get('namespace', 0),
                                match_type='sitelink_pagelink',
                                sentence=None,
                                sentence_index=None,
                                confidence=0.9,
                                subject_label=subject_label,
                                value_label=value_label,
                                extraction_method='sitelink_pagelink_match_v1',
                                extraction_ts=extraction_ts
                            )
                            factsenses.append(fs)
                            # Once found a pagelink match, we might still want sentence matches? 
                            # Let's keep going.
                
                # Strategy 1b: Sentences
                sentences = page_data.get('sentences', [])
                if isinstance(sentences, str):
                    sentences = simple_sentence_split(sentences)
                elif not isinstance(sentences, list):
                    sentences = []
                
                for idx, sent in enumerate(sentences[:50]):
                    if not sent or len(sent) < 20: continue
                    
                    subject_in_sent = subject_label.lower() in sent.lower()
                    value_in_sent = False
                    if value_label:
                        value_in_sent = value_label.lower() in sent.lower()
                    
                    if subject_in_sent or value_in_sent:
                        confidence = 0.8 if (subject_in_sent and value_in_sent) else 0.6
                        fs_id = sha256_short(f"{statement_id}_{lang}_{page_data.get('page_id')}_{idx}")
                        fs = FactSense(
                            factsense_id=fs_id,
                            belongs_to_statement_id=statement_id,
                            subject_qid=subject_qid,
                            property_pid=property_pid,
                            value_qid=value_qid,
                            language=lang,
                            page_id=page_data.get('page_id'),
                            page_title=page_data.get('title'),
                            page_namespace=page_data.get('namespace', 0),
                            match_type='sitelink',
                            sentence=sent[:1000],
                            sentence_index=idx,
                            confidence=confidence,
                            subject_label=subject_label,
                            value_label=value_label,
                            extraction_method='sitelink_sentence_match_v1',
                            extraction_ts=extraction_ts
                        )
                        factsenses.append(fs)
        
        # --- Strategy 2: Label-based (only if no sitelink matching produced results? or always?) --- 
        # Original logic: "elif lang in subject_labels" implies if sitelink match ran, we skip label match.
        # But wait, original code:
        # if sitelink_key in sitelinks: ...
        # elif lang in subject_labels: ...
        # So yes, it's exclusive. If sitelink exists, we trust it.
        
        elif lang in subject_labels and subject_label:
            # Lookup pages by label (title match)
            # In pre-fetching, we looked up 'subject_label'
            candidates = []
            if lang in prefetched_pages and subject_label.lower() in prefetched_pages[lang]:
                candidates.append(prefetched_pages[lang][subject_label.lower()])
            
            for page_data in candidates:
                # Strategy 2a: Pagelinks
                pagelinks = page_data.get('pagelinks', [])
                if value_label and pagelinks:
                    for link in pagelinks:
                        link_title = link.get('title', '')
                        if link_title and value_label.lower() == link_title.lower():
                            fs_id = sha256_short(f"{statement_id}_{lang}_{page_data.get('page_id')}_pagelink")
                            fs = FactSense(
                                factsense_id=fs_id,
                                belongs_to_statement_id=statement_id,
                                subject_qid=subject_qid,
                                property_pid=property_pid,
                                value_qid=value_qid,
                                language=lang,
                                page_id=page_data.get('page_id'),
                                page_title=page_data.get('title'),
                                page_namespace=page_data.get('namespace', 0),
                                match_type='label_pagelink',
                                sentence=None,
                                sentence_index=None,
                                confidence=0.85,
                                subject_label=subject_label,
                                value_label=value_label,
                                extraction_method='label_pagelink_match_v1',
                                extraction_ts=extraction_ts
                            )
                            factsenses.append(fs)

                # Strategy 2b: Sentences
                sentences = page_data.get('sentences', [])
                if isinstance(sentences, str):
                    sentences = simple_sentence_split(sentences)
                elif not isinstance(sentences, list):
                    sentences = []
                
                for idx, sent in enumerate(sentences[:30]):
                    if not sent or len(sent) < 20: continue
                    
                    subject_in_sent = subject_label.lower() in sent.lower()
                    value_in_sent = value_label.lower() in sent.lower() if value_label else False
                    
                    if subject_in_sent or value_in_sent:
                        confidence = 0.7 if (subject_in_sent and value_in_sent) else 0.5
                        fs_id = sha256_short(f"{statement_id}_{lang}_{page_data.get('page_id')}_{idx}")
                        fs = FactSense(
                            factsense_id=fs_id,
                            belongs_to_statement_id=statement_id,
                            subject_qid=subject_qid,
                            property_pid=property_pid,
                            value_qid=value_qid,
                            language=lang,
                            page_id=page_data.get('page_id'),
                            page_title=page_data.get('title'),
                            page_namespace=page_data.get('namespace', 0),
                            match_type='label_based',
                            sentence=sent[:1000],
                            sentence_index=idx,
                            confidence=confidence,
                            subject_label=subject_label,
                            value_label=value_label,
                            extraction_method='label_sentence_match_v1',
                            extraction_ts=extraction_ts
                        )
                        factsenses.append(fs)
        
        # --- Strategy 3: Redirect-based matching ---
        # Original: if lang in subject_labels and subject_label
        # Note: Original code ran Strategy 3 independent of Strategy 1/2?
        # Original: 
        # if sitelink: ...
        # elif label: ...
        # if lang in subject_labels: ... (This is OUTSIDE the elif, so it runs ALWAYS)
        # Wait, let me check the original indentation.
        # Lines 624 was at same level as `elif` (line 549)??
        # No, line 624 is `        if lang in subject_labels and subject_label:`
        # Line 549 was `        elif lang in subject_labels:`
        # Line 472 was `        if sitelink_key in sitelinks:`
        # So `if sitelink: ... elif label: ...` is one block.
        # `if lang in subject_labels: ...` for redirects was AFTER that block. So it runs in addition to sitelink/label match.
        
        if lang in subject_labels and subject_label:
            redirect_pages_list = []
            if lang in prefetched_redirects and subject_label.lower() in prefetched_redirects[lang]:
                 redirect_pages_list = prefetched_redirects[lang][subject_label.lower()]
            
            for page_data in redirect_pages_list:
                # Pagelinks check
                pagelinks = page_data.get('pagelinks', [])
                if value_label and pagelinks:
                    for link in pagelinks:
                        link_title = link.get('title', '')
                        if link_title and value_label.lower() == link_title.lower():
                            fs_id = sha256_short(f"{statement_id}_{lang}_{page_data.get('page_id')}_redirect_pagelink")
                            fs = FactSense(
                                factsense_id=fs_id,
                                belongs_to_statement_id=statement_id,
                                subject_qid=subject_qid,
                                property_pid=property_pid,
                                value_qid=value_qid,
                                language=lang,
                                page_id=page_data.get('page_id'),
                                page_title=page_data.get('title'),
                                page_namespace=page_data.get('namespace', 0),
                                match_type='redirect_pagelink',
                                sentence=None,
                                sentence_index=None,
                                confidence=0.75,
                                subject_label=subject_label,
                                value_label=value_label,
                                extraction_method='redirect_pagelink_match_v1',
                                extraction_ts=extraction_ts
                            )
                            factsenses.append(fs)

    return factsenses


# -------------------- Worker Function --------------------


def worker_process_batch(
    batch_statements: List[Dict],
    # es_settings: Dict[str, Any], # No longer used inside, but kept for signature if needed or removed? 
    # Actually, we need to change signature in the caller too.
    # Let's remove es_settings from here and rely on global _es_client
    target_languages: List[str]
) -> Tuple[List[Dict], int, int, Dict]:
    """
    Worker function to process a batch of statements with pre-fetching
    """
    global _es_client
    if _es_client is None:
        # Fallback if init failed or not called (shouldn't happen with pool init)
        logging.error("Worker Global ES client not initialized!")
        return [], 0, 0, {}

    factsense_records = []
    statements_processed = 0
    factsenses_generated = 0
    
    stats = {
        'total_qids': 0, 'labels_found': 0, 'labels_not_found': 0,
        'value_qid_count': 0, 'total_languages': 0,
        'statements_with_labels': 0, 'statements_no_labels': 0,
        'match_sitelink': 0, 'match_label': 0, 'match_redirect': 0,
        'prefetch_sitelink_count': 0, 'prefetch_label_count': 0, 'prefetch_redirect_count': 0
    }
    
    try:
        # --- Phase 1: Collect QIDs and Languages ---
        qids = set()
        for stmt in batch_statements:
            subject_qid = stmt.get('subject_qid')
            if subject_qid:
                qids.add(subject_qid)
                stats['total_qids'] += 1
            value = stmt.get('value')
            if isinstance(value, str) and value.startswith('Q') and value[1:].isdigit():
                qids.add(value)
                stats['value_qid_count'] += 1
        
        # Batch query labels
        labels_data = query_labels_batch(_es_client, list(qids), target_languages)
        
        # Update stats
        for qid in qids:
            if qid in labels_data and labels_data[qid]:
                stats['labels_found'] += 1
                stats['total_languages'] += sum(len(langs) for langs in labels_data[qid].values())
            else:
                stats['labels_not_found'] += 1

        # --- Phase 2: Identify Pages to Fetch ---
        pages_to_fetch = {} # {lang: set(titles)}
        redirects_to_fetch = {} # {lang: set(target_titles)}
        
        for stmt in batch_statements:
            subject_qid = stmt.get('subject_qid')
            value_qid = None
            val = stmt.get('value')
            if isinstance(val, str) and val.startswith('Q') and val.isdigit(): value_qid = val
            
            # Determine languages
            stmt_langs = set()
            if target_languages:
                stmt_langs = set(target_languages)
            else:
                # Use languages from subject/value labels
                if subject_qid in labels_data:
                    stmt_langs.update(labels_data[subject_qid].keys())
                if value_qid and value_qid in labels_data:
                    stmt_langs.update(labels_data[value_qid].keys())
            
            if not stmt_langs: 
                stmt_langs = {'en'}

            # Sitelinks
            sitelinks = {}
            try:
                sitelinks = json.loads(stmt.get('sitelinks', '{}'))
            except: pass
            
            for lang in stmt_langs:
                # 1. Sitelink Title
                sitelink_key = f"{lang}wiki"
                if sitelink_key in sitelinks:
                    title = sitelinks[sitelink_key]
                    if lang not in pages_to_fetch: pages_to_fetch[lang] = set()
                    pages_to_fetch[lang].add(title)
                
                subj_label = None
                if subject_qid in labels_data and lang in labels_data[subject_qid]:
                    subj_label = labels_data[subject_qid][lang].get('label')
                
                if subj_label:
                    if lang not in pages_to_fetch: pages_to_fetch[lang] = set()
                    pages_to_fetch[lang].add(subj_label)
                    
                    if lang not in redirects_to_fetch: redirects_to_fetch[lang] = set()
                    redirects_to_fetch[lang].add(subj_label)

        # --- Phase 3: Batch Fetch Pages ---
        prefetched_pages = {} # {lang: {title_lower: page_data}}
        for lang, titles in pages_to_fetch.items():
            found = query_pages_batch_by_title(_es_client, list(titles), lang)
            if found:
                prefetched_pages[lang] = found
                stats['prefetch_sitelink_count'] += len(found) # Approx

        prefetched_redirects = {} # {lang: {target_lower: [pages]}}
        for lang, targets in redirects_to_fetch.items():
            found = query_pages_batch_by_redirect(_es_client, list(targets), lang)
            if found:
                prefetched_redirects[lang] = found
                stats['prefetch_redirect_count'] += len(found)

        # --- Phase 4: Process Statements ---
        for stmt in batch_statements:
            try:
                # Stat update
                subject_qid = stmt.get('subject_qid')
                if subject_qid in labels_data and labels_data[subject_qid]:
                    stats['statements_with_labels'] += 1
                else:
                    stats['statements_no_labels'] += 1

                # Generate
                factsenses = generate_factsense_from_statement_v2(
                    stmt,
                    labels_data,
                    prefetched_pages,
                    prefetched_redirects,
                    target_languages
                )
                
                for fs in factsenses:
                    match_type = fs.match_type
                    if 'sitelink' in match_type: stats['match_sitelink'] += 1
                    elif 'label' in match_type: stats['match_label'] += 1
                    elif 'redirect' in match_type: stats['match_redirect'] += 1
                    
                    factsense_records.append(fs.to_dict())
                    factsenses_generated += 1
                
                statements_processed += 1
                
            except Exception as e:
                logging.debug(f"Error processing statement {stmt.get('core_id')}: {e}")
                continue

    except Exception as e:
        logging.error(f"Worker batch processing error: {e}", exc_info=True)
    
    return factsense_records, statements_processed, factsenses_generated, stats


# -------------------- Main Processing --------------------

def process_factstatements(
    factstatement_dir: Optional[Path],
    outdir: Path,
    es_index: str,
    batch_size: int,
    workers: int,
    max_factsense_per_file: int,
    target_languages: List[str],
    resume: bool,
    logger: logging.Logger,
    max_statements: Optional[int] = None,
    file_list: Optional[List[str]] = None,
    job_id: Optional[str] = None
):
    """
    Main processing loop
    
    Args:
        job_id: Unique identifier for this job (for parallel cluster execution)
               If provided, checkpoint and output files will be isolated per job
    """
    
    outdir.mkdir(parents=True, exist_ok=True)
    
    # 为每个job创建独立的checkpoint和输出文件
    if job_id:
        checkpoint_path = outdir / f'checkpoint_{job_id}.json'
        job_output_prefix = f"{job_id}_"
        logger.info(f"Running in cluster mode with job_id: {job_id}")
    else:
        checkpoint_path = outdir / 'checkpoint.json'
        job_output_prefix = ""
        logger.info(f"Running in single mode")
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path) if resume else {}
    start_file_idx = checkpoint.get('file_index', 0)
    start_row = checkpoint.get('row_offset', 0)
    total_statements_processed = checkpoint.get('total_statements', 0)
    total_factsenses_generated = checkpoint.get('total_factsenses', 0)
    output_file_index = checkpoint.get('output_file_index', 0)
    
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Starting from file_index={start_file_idx}, row={start_row}")
    
    # Create ES client for index management
    es = ElasticFactory(ES_IP_LIST, ES_PORT, ES_USER, ES_PASSWARD).create()
    create_factsense_index(es, es_index, logger)
    
    # Find factstatement files
    if file_list:
        # Use provided file list (for cluster execution)
        fs_files = [Path(f) for f in file_list]
        logger.info(f"Using provided file list: {len(fs_files)} files")
        for f in fs_files:
            if not f.exists():
                logger.error(f"File not found: {f}")
                sys.exit(1)
    else:
        # Discover files from directory
        fs_files = sorted(factstatement_dir.glob('factstatements_part_*.parquet'))
        logger.info(f"Found {len(fs_files)} factstatement files in directory")
    
    if not fs_files:
        logger.error("No factstatement files found!")
        return
    
    # ES settings for workers
    es_settings = {
        'hosts': ES_IP_LIST,
        'port': ES_PORT,
        'user': ES_USER,
        'password': ES_PASSWARD
    }
    
    # Output buffers
    factsense_buffer = []
    current_output_rows = 0
    
    # Parquet writer
    parquet_writer = None
    current_output_path = None
    
    # Process each file
    for file_idx, fs_file in enumerate(fs_files):
        if file_idx < start_file_idx:
            continue
        
        logger.info(f"Processing file {file_idx + 1}/{len(fs_files)}: {fs_file.name}")
        
        try:
            pf = pq.ParquetFile(fs_file)
            total_rows = pf.metadata.num_rows
            
            # Read in batches
            skip_rows = start_row if file_idx == start_file_idx else 0
            current_row = start_row if file_idx == start_file_idx else 0
            
            for batch in pf.iter_batches(batch_size=batch_size):
                batch_df = batch.to_pandas()
                batch_len = len(batch_df)

                # --- 精确跳过已处理行 ---
                if skip_rows > 0:
                    if batch_len <= skip_rows:
                        skip_rows -= batch_len
                        current_row += batch_len
                        continue
                    else:
                        batch_df = batch_df.iloc[skip_rows:]
                        batch = pa.Table.from_pandas(batch_df)
                        processed_rows = len(batch_df)
                        skip_rows = 0
                else:
                    processed_rows = batch_len
                
                statements = batch.to_pylist()
                
                # Process with worker pool
                # Use initializer to create persistent ES connection
                with mp.Pool(processes=workers, initializer=worker_init, initargs=(es_settings,)) as pool:
                    # Split into chunks for workers
                    chunk_size = max(1, len(statements) // workers)
                    chunks = [statements[i:i+chunk_size] for i in range(0, len(statements), chunk_size)]
                    
                    # Process chunks
                    results = pool.starmap(
                        worker_process_batch,
                        [(chunk, target_languages) for chunk in chunks]
                    )
                
                # Aggregate results
                batch_stats = {
                    'total_qids': 0, 'labels_found': 0, 'labels_not_found': 0,
                    'value_qid_count': 0, 'total_languages': 0,
                    'statements_with_labels': 0, 'statements_no_labels': 0,
                    'match_sitelink': 0, 'match_label': 0, 'match_redirect': 0,
                    'prefetch_sitelink_count': 0, 'prefetch_redirect_count': 0
                }
                
                for result in results:
                    factsense_records, stmts_proc, fs_gen, worker_stats = result
                    total_statements_processed += stmts_proc
                    total_factsenses_generated += fs_gen
                    factsense_buffer.extend(factsense_records)
                    
                    # 聚合统计
                    for key in batch_stats:
                        batch_stats[key] += worker_stats.get(key, 0)
                
                #记录详细统计
                if batch_stats['total_qids'] > 0:
                    avg_fs_per_stmt = fs_gen / stmts_proc if stmts_proc > 0 else 0
                    labels_found_pct = batch_stats['labels_found'] / batch_stats['total_qids'] * 100 if batch_stats['total_qids'] > 0 else 0
                    avg_langs = batch_stats['total_languages'] / max(batch_stats['labels_found'], 1)
                    stmts_with_labels_pct = batch_stats['statements_with_labels'] / stmts_proc * 100 if stmts_proc > 0 else 0
                    
                    logger.info(
                        f"Batch: stmts={stmts_proc}, factsenses={fs_gen} (avg {avg_fs_per_stmt:.1f}/stmt) | "
                        f"QIDs: {batch_stats['total_qids']}, labels: {batch_stats['labels_found']} ({labels_found_pct:.1f}%), "
                        f"Prefetch: pages={batch_stats['prefetch_sitelink_count']}, redirects={batch_stats['prefetch_redirect_count']} | "
                        f"match: sl={batch_stats['match_sitelink']}, lb={batch_stats['match_label']}, rd={batch_stats['match_redirect']}"
                    )
                
                ### update
                current_row += processed_rows

                persist_ok = True

                # Write to Parquet when buffer is large enough
                if len(factsense_buffer) >= max_factsense_per_file:
                    # Define consistent schema (use same schema for file creation and table conversion)
                    factsense_schema = pa.schema([
                        ('factsense_id', pa.string()),
                        ('belongs_to_statement_id', pa.string()),
                        ('subject_qid', pa.string()),
                        ('property_pid', pa.string()),
                        ('value_qid', pa.string()),  # nullable string
                        ('language', pa.string()),
                        ('page_id', pa.int64()),
                        ('page_title', pa.string()),
                        ('page_namespace', pa.int32()),
                        ('match_type', pa.string()),
                        ('sentence', pa.string()),  # nullable string
                        ('sentence_index', pa.int32()),
                        ('confidence', pa.float32()),
                        ('subject_label', pa.string()),
                        ('value_label', pa.string()),  # nullable string
                        ('extraction_method', pa.string()),
                        ('extraction_ts', pa.string())
                    ])
                    
                    if parquet_writer is None:
                        current_output_path = outdir / f'factsense_part_{job_output_prefix}{output_file_index}.parquet'
                        parquet_writer = pq.ParquetWriter(current_output_path, factsense_schema, compression='snappy')
                        logger.info(f"Created new parquet file: {current_output_path}")
                    
                    # Write batch - IMPORTANT: use schema to enforce types
                    try:
                        table = pa.Table.from_pylist(factsense_buffer[:max_factsense_per_file], schema=factsense_schema)
                        parquet_writer.write_table(table)
                        current_output_rows += len(factsense_buffer[:max_factsense_per_file])
                    except Exception as e:
                        logger.error(f"Failed to write parquet batch: {e}")
                        persist_ok = False
                        # Try to recover by writing with inferred schema
                        logger.warning("Attempting to write with inferred schema...")
                        table = pa.Table.from_pylist(factsense_buffer[:max_factsense_per_file])
                        # Cast to target schema
                        table = table.cast(factsense_schema)
                        parquet_writer.write_table(table)
                        current_output_rows += len(factsense_buffer[:max_factsense_per_file])
                    
                    
                    # Bulk index to ES
                    try:
                        actions = [
                            {
                                '_op_type': 'index',
                                '_index': es_index,
                                '_id': rec['factsense_id'],
                                '_source': rec
                            }
                            for rec in factsense_buffer[:max_factsense_per_file]
                        ]
                        helpers.bulk(es, actions, chunk_size=20000, request_timeout=120)
                        logger.info(f"Indexed {len(actions)} FactSense records to ES")
                    except Exception as e:
                        logger.error(f"ES indexing error: {e}")
                        persist_ok = False
                    
                    # Clear buffer
                    factsense_buffer = factsense_buffer[max_factsense_per_file:]
                    
                    # Rotate file if needed
                    if current_output_rows >= max_factsense_per_file:
                        if parquet_writer:
                            parquet_writer.close()
                            parquet_writer = None
                        output_file_index += 1
                        current_output_rows = 0
                    
                    if persist_ok:
                        logger.info(f"Persisted {len(actions)} records to Parquet")
                        # ✅ 仅在数据已成功持久化后，更新 checkpoint
                        checkpoint_data = {
                            'file_index': file_idx,
                            'row_offset': current_row,   # 关键：已“持久化”的输入行数
                            'total_statements': total_statements_processed,
                            'total_factsenses': total_factsenses_generated,
                            'output_file_index': output_file_index,
                            'last_update': time.time()
                        }
                        save_checkpoint(checkpoint_path, checkpoint_data)
                    else:
                        logger.warning("Persist failed, checkpoint NOT updated")
                
                logger.info(f"Progress: statements={total_statements_processed}, factsenses={total_factsenses_generated}")
                
                # Check max limit
                if max_statements and total_statements_processed >= max_statements:
                    logger.info(f"Reached max_statements limit: {max_statements}")
                    break
            
            # Reset row offset for next file
            start_row = 0
            
        except Exception as e:
            logger.error(f"Error processing file {fs_file.name}: {e}", exc_info=True)
            continue
    
    # Final flush
    if factsense_buffer:
        # Define consistent schema
        factsense_schema = pa.schema([
            ('factsense_id', pa.string()),
            ('belongs_to_statement_id', pa.string()),
            ('subject_qid', pa.string()),
            ('property_pid', pa.string()),
            ('value_qid', pa.string()),
            ('language', pa.string()),
            ('page_id', pa.int64()),
            ('page_title', pa.string()),
            ('page_namespace', pa.int32()),
            ('match_type', pa.string()),
            ('sentence', pa.string()),
            ('sentence_index', pa.int32()),
            ('confidence', pa.float32()),
            ('subject_label', pa.string()),
            ('value_label', pa.string()),
            ('extraction_method', pa.string()),
            ('extraction_ts', pa.string())
        ])
        
        if parquet_writer is None and factsense_buffer:
            current_output_path = outdir / f'factsense_part_{job_output_prefix}{output_file_index}.parquet'
            parquet_writer = pq.ParquetWriter(current_output_path, factsense_schema, compression='snappy')
        
        if parquet_writer:
            try:
                table = pa.Table.from_pylist(factsense_buffer, schema=factsense_schema)
                parquet_writer.write_table(table)
                logger.info(f"Wrote final {len(factsense_buffer)} records to parquet")
            except Exception as e:
                logger.error(f"Failed to write final batch: {e}")
                # Try with cast
                table = pa.Table.from_pylist(factsense_buffer)
                table = table.cast(factsense_schema)
                parquet_writer.write_table(table)
                logger.info(f"Wrote final {len(factsense_buffer)} records to parquet (with cast)")
        
        # Final ES indexing
        try:
            actions = [
                {
                    '_op_type': 'index',
                    '_index': es_index,
                    '_id': rec['factsense_id'],
                    '_source': rec
                }
                for rec in factsense_buffer
            ]
            helpers.bulk(es, actions, chunk_size=100, request_timeout=120)
            logger.info(f"Final ES indexing: {len(actions)} records")
        except Exception as e:
            logger.error(f"Final ES indexing error: {e}")
    
    if parquet_writer:
        parquet_writer.close()
    
    # ✅ Final flush 成功后，更新 checkpoint
    checkpoint_data = {
        'file_index': checkpoint.get('file_index', start_file_idx),
        'row_offset': checkpoint.get('row_offset', start_row),
        'total_statements': total_statements_processed,
        'total_factsenses': total_factsenses_generated,
        'output_file_index': output_file_index,
        'last_update': time.time()
    }
    save_checkpoint(checkpoint_path, checkpoint_data)

    logger.info(f"Processing complete! Statements: {total_statements_processed}, FactSenses: {total_factsenses_generated}")


# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser(description='Build FactSense from FactStatements')
    parser.add_argument('--factstatement-dir', type=Path, required=False,
                        help='Directory containing factstatement parquet files')
    parser.add_argument('--file-list', type=str, nargs='+', required=False,
                        help='List of specific parquet files to process (for cluster execution)')
    parser.add_argument('--outdir', type=Path, required=True,
                        help='Output directory for FactSense parquet files')
    parser.add_argument('--workers', type=int, default=32,
                        help='Number of worker processes')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Number of statements per batch')
    parser.add_argument('--max-factsense-per-file', type=int, default=5_000_000,
                        help='Max FactSense records per output file')
    parser.add_argument('--log-dir', type=Path, default=Path('./logs'),
                        help='Log directory')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--es-index', type=str, default='factnet_factsense_v1',
                        help='Elasticsearch index name')
    parser.add_argument('--target-languages', type=str, nargs='+',
                        help='Target languages (e.g., en zh de fr)')
    parser.add_argument('--max-statements', type=int, default=None,
                        help='Max statements to process (for testing)')
    parser.add_argument('--job-id', type=str, default=None,
                        help='Job ID for cluster execution (isolates checkpoint and output files)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info(f"Starting FactSense builder with args: {args}")
    
    # Validate inputs
    if not args.factstatement_dir and not args.file_list:
        logger.error("Either --factstatement-dir or --file-list must be specified")
        sys.exit(1)
    
    if args.factstatement_dir and not args.factstatement_dir.exists():
        logger.error(f"FactStatement directory not found: {args.factstatement_dir}")
        sys.exit(1)
    
    # Process
    try:
        process_factstatements(
            factstatement_dir=args.factstatement_dir,
            outdir=args.outdir,
            es_index=args.es_index,
            batch_size=args.batch_size,
            workers=args.workers,
            max_factsense_per_file=args.max_factsense_per_file,
            target_languages=args.target_languages,
            resume=args.resume,
            logger=logger,
            max_statements=args.max_statements,
            file_list=args.file_list,
            job_id=args.job_id
        )
        logger.info("FactSense construction completed successfully!")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(e)
    finally:
        try:
            with open('/obssidecar/terminate/0', "w") as fo:
                fo.write(" ")
        except Exception:
            pass