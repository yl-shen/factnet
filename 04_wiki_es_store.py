"""
nohup python 04_wiki_es_store.py \
--resume \
--langs en \
--wikiextractor-out 01_wikiextractor_out \
--parquet-out 02_sql_to_parquet \
--index factnet_pages_v1 \
--workers 32 \
--pages-batch 20000 --updates-batch 20000 \
--log-dir 03_wiki_es_store/logs \
> 03_wiki_es_store/runing_en.log &
"""

import os
import sys
import argparse
import json
import logging
import logging.handlers
import signal
import threading
import time
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, Any, List, Optional, Tuple
import re
import ssl
import urllib3
import random
import psutil
import multiprocessing
from queue import Empty

# third-party
import nltk
from elasticsearch import Elasticsearch, helpers
import pyarrow.parquet as pq
import pyarrow as pa
from es_config import ES_IP_LIST, ES_PORT, ES_USER, ES_PASSWARD

# 关闭 HTTPS 未验证证书警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -------------------- ElasticFactory for cluster access --------------------
class ElasticFactory(object):
    def __init__(self, host: list, port: str, username: str, password: str):
        self.port = port
        self.host = host
        self.username = username
        self.password = password

    def create(self) -> Elasticsearch:
        context = ssl._create_unverified_context()
        addrs = [{"host": h, "port": self.port} for h in self.host]

        if self.username and self.password:
            elasticsearch = Elasticsearch(
                addrs,
                http_auth=(self.username, self.password),
                scheme="https",
                ssl_context=context
            )
        else:
            elasticsearch = Elasticsearch(addrs)
        return elasticsearch

# -------------------- ES Connection Pool --------------------
class ESConnectionPool:
    _pools = {}
    _lock = threading.Lock()

    @classmethod
    def get_client(cls, es_ip_list: list, es_port: str, es_user: Optional[str], es_pass: Optional[str], pool_size: int = 2) -> Tuple[int, Elasticsearch]:
        """获取或创建连接池中的ES客户端（pool_size 对每个进程不宜太大）"""
        key = (tuple(es_ip_list), es_port, es_user, es_pass)
        with cls._lock:
            if key not in cls._pools:
                cls._pools[key] = {
                    'clients': [ElasticFactory(es_ip_list, es_port, es_user, es_pass).create() for _ in range(max(1, pool_size))],
                    'in_use': [False] * max(1, pool_size),
                    'last_used': [time.time()] * max(1, pool_size)
                }
            pool = cls._pools[key]
            for i, in_use in enumerate(pool['in_use']):
                if not in_use:
                    pool['in_use'][i] = True
                    pool['last_used'][i] = time.time()
                    return i, pool['clients'][i]
            # all busy -> create temporary new client
            new_client = ElasticFactory(es_ip_list, es_port, es_user, es_pass).create()
            pool['clients'].append(new_client)
            pool['in_use'].append(True)
            pool['last_used'].append(time.time())
            return len(pool['clients']) - 1, new_client

    @classmethod
    def release_client(cls, es_ip_list: list, es_port: str, es_user: Optional[str], es_pass: Optional[str], client_id: int):
        key = (tuple(es_ip_list), es_port, es_user, es_pass)
        with cls._lock:
            if key in cls._pools:
                pool = cls._pools[key]
                if 0 <= client_id < len(pool['in_use']):
                    pool['in_use'][client_id] = False
                    pool['last_used'][client_id] = time.time()

    @classmethod
    def cleanup_old_connections(cls, max_idle_time: int = 600):
        now = time.time()
        with cls._lock:
            for key, pool in cls._pools.items():
                for i, in_use in enumerate(pool['in_use']):
                    if not in_use and now - pool['last_used'][i] > max_idle_time:
                        try:
                            if hasattr(pool['clients'][i], 'close'):
                                pool['clients'][i].close()
                        except Exception:
                            pass
                        pool['clients'][i] = ElasticFactory(*key).create()
                        pool['last_used'][i] = now

# -------------------- 信号优雅中断 --------------------
stop_flag = False
def _install_sigint_handler():
    def _handler(signum, frame):
        global stop_flag
        logging.warning("收到中断信号，当前正在完成的任务后将停止提交新任务...")
        stop_flag = True
    import signal as _signal
    _signal.signal(_signal.SIGINT, _handler)
    _signal.signal(_signal.SIGTERM, _handler)

# -------------------- 日志 --------------------
def setup_global_logging(log_dir: Path, logfile_name="index_pages.log"):
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(ch)

    # rotating file
    fh = logging.handlers.RotatingFileHandler(str(log_dir / logfile_name), maxBytes=20*1024*1024, backupCount=5, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

def add_language_logfile(log_dir: Path, lang: str):
    fh = logging.handlers.RotatingFileHandler(str(log_dir / f"{lang}.log"), maxBytes=10*1024*1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(fh)
    return fh

# -------------------- Checkpoint helpers --------------------
def load_checkpoint(cp_path: Path) -> Dict[str, Any]:
    if cp_path.exists():
        try:
            with cp_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning("加载 checkpoint 失败，忽略并重建: %s (%s)", cp_path, e)
    return {}

def save_checkpoint_with_retry(cp_path: Path, data: Dict[str, Any], max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            tmp = cp_path.with_suffix(f'.tmp.{attempt}')
            with tmp.open('w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp.replace(cp_path)
            return True
        except Exception as e:
            logging.warning(f"保存checkpoint失败(尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(1)
    logging.error("保存checkpoint达到最大重试次数，放弃保存")
    return False

# -------------------- NLTK tokenizer helper --------------------
PUNKT_NAME_MAP = {
    'en': 'english','eng': 'english','zh': 'chinese','zh-cn': 'chinese','zh_tw': 'chinese',
    'de': 'german','fr': 'french','es': 'spanish','cs': 'czech','ces': 'czech',
    'da': 'danish','dan': 'danish','nl': 'dutch','nld': 'dutch','et': 'estonian','est': 'estonian',
    'fi': 'finnish','fin': 'finnish','el': 'greek','ell': 'greek','gre': 'greek',
    'it': 'italian','ita': 'italian','no': 'norwegian','nb': 'norwegian','nn': 'norwegian',
    'nob': 'norwegian','nno': 'norwegian','pl': 'polish','pol': 'polish','pt': 'portuguese','por': 'portuguese',
    'pt-br': 'portuguese','pt_pt': 'portuguese','sl': 'slovene','slv': 'slovene','sv': 'swedish','swe': 'swedish',
    'tr': 'turkish','tur': 'turkish',
}
_tokenizer_cache: Dict[str, Any] = {}
def get_sentence_tokenizer_for_lang(lang: str):
    if not lang:
        lang = 'en'
    key = lang.lower()
    if key in _tokenizer_cache:
        return _tokenizer_cache[key]
    name = PUNKT_NAME_MAP.get(key, key)
    try:
        nltk.data.find('tokenizers/punkt')
    except Exception:
        try:
            nltk.download('punkt', quiet=True)
        except Exception:
            pass
    try:
        tok = nltk.data.load(f'tokenizers/punkt/{name}.pickle')
        _tokenizer_cache[key] = tok
        logging.info("Loaded punkt tokenizer: %s -> %s", lang, name)
        return tok
    except Exception as e:
        logging.warning("无法加载 punkt tokenizer for lang=%s (尝试名=%s): %s. 将回退到简单规则分句。", lang, name, e)
        _tokenizer_cache[key] = None
        return None

def simple_sentence_split(text: str, lang: str = 'en'):
    tok = get_sentence_tokenizer_for_lang(lang)
    if tok:
        try:
            sents = tok.tokenize(text)
            return [s.strip() for s in sents if s.strip()]
        except Exception:
            pass
    sents = re.split(r'(?<=[。.!?！？])\s+', text)
    return [s.strip() for s in sents if s.strip()]

# -------------------- helpers：读取 wikiextractor 输出的文件列表 --------------------
LANG_DIR_RE = re.compile(r"^(.+?)wiki", re.IGNORECASE)
def language_of_dirname(dirname: str) -> str:
    base = Path(dirname).name
    m = LANG_DIR_RE.search(base)
    if m:
        return m.group(1).lower()
    return base.lower()

def find_wikiextractor_lang_dirs(root: Path) -> Dict[str, Path]:
    langs = {}
    if not root.exists():
        return langs
    for p in sorted(root.iterdir()):
        if not p.is_dir(): continue
        lang = language_of_dirname(p.name)
        langs[lang] = p
    return langs

def list_wikiextractor_files_for_lang(lang_dir: Path) -> List[Path]:
    if not lang_dir or not lang_dir.exists():
        return []
    files = [p for p in lang_dir.rglob('*') if p.is_file()]
    files = [p for p in files if p.name.startswith('wiki_') or p.suffix == '' or (p.suffix.lower() in ['.json', '.txt'] and not p.name.endswith('.checkpoint'))]
    files = sorted(files)
    return files

def find_parquet_parts(parquet_root: Path, lang: str, table: str) -> List[Path]:
    p = parquet_root / lang / table
    if not p.exists():
        return []
    files = sorted([f for f in p.glob('*.parquet') if not f.name.endswith('.checkpoint')])
    return files

# -------------------- ES helpers --------------------
def make_es_client(es_ip_list: list, es_port: str, es_user: Optional[str], es_passwd: Optional[str]) -> Elasticsearch:
    return ElasticFactory(es_ip_list, es_port, es_user, es_passwd).create()

def ensure_index_exists(es: Elasticsearch, index_name: str, num_shards: int = 6, num_replicas: int = 1):
    if not es.indices.exists(index=index_name):
        logging.info(f"创建索引 {index_name}，设置 {num_shards} 主分片和 {num_replicas} 副本")
        index_settings = {
            "settings": {
                "number_of_shards": num_shards,
                "number_of_replicas": num_replicas,
                "refresh_interval": "30s",
                "translog": {"durability": "async", "sync_interval": "30s"},
            },
            "mappings": {
                "properties": {
                    "page_id": {"type": "long"},
                    "title": {"type": "text"},
                    "title_text": {"type": "keyword"},
                    "lang": {"type": "keyword"},
                    "namespace": {"type": "integer"},
                    "text": {"type": "text"},
                    "sentences": {"type": "text"},
                    "pagelinks": {
                        "type": "nested",
                        "properties": {"title": {"type": "keyword"}, "namespace": {"type": "integer"}}
                    },
                    "redirects": {
                        "type": "nested",
                        "properties": {"title": {"type": "keyword"}, "namespace": {"type": "integer"}}
                    }
                }
            }
        }
        es.indices.create(index=index_name, body=index_settings)
        logging.info(f"成功创建索引 {index_name}")
    else:
        logging.info(f"索引 {index_name} 已存在，使用现有索引")

def calculate_optimal_batch_size(file_size: int, base_batch_size: int) -> int:
    if file_size > 1024 * 1024 * 1024:
        return max(5000, min(base_batch_size, 20000))
    elif file_size > 500 * 1024 * 1024:
        return max(5000, min(base_batch_size, 20000))
    elif file_size > 100 * 1024 * 1024:
        return max(2000, min(int(base_batch_size * 1.2), 20000))
    else:
        return min(max(1000, int(base_batch_size)), 20000)

# -------------------- 页面索引 worker（单文件） --------------------
def index_single_wiki_file_optimized(file_path: str, lang: str, es_ip_list: list, es_port: str, es_user: Optional[str], es_pass: Optional[str],
                               index_name: str, pages_batch: int = 500, workers: int = 4) -> Dict[str, Any]:
    logger = logging.getLogger()
    inpath = Path(file_path)
    try:
        file_size = inpath.stat().st_size
    except Exception:
        file_size = 0
    adaptive_batch_size = calculate_optimal_batch_size(file_size, pages_batch)

    client_id, es = ESConnectionPool.get_client(es_ip_list, es_port, es_user, es_pass, pool_size=max(1, min(4, workers//4)))
    actions = []
    wrote = 0
    processed = 0
    last_gc_check = 0

    if file_path.endswith(".checkpoint"):
        checkpoint_path = Path(file_path)
    else:
        checkpoint_path = Path(file_path + ".checkpoint")
    checkpoint_data = {}
    if checkpoint_path.exists():
        try:
            with checkpoint_path.open('r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                if 'processed' in checkpoint_data:
                    processed = checkpoint_data['processed']
                    wrote = checkpoint_data.get('wrote', 0)
                    logger.info(f"从检查点继续处理: 已处理={processed}, 已写入={wrote}")
        except Exception as e:
            logger.warning(f"读取checkpoint失败: {e}, 将从头开始处理文件")

    def update_checkpoint():
        try:
            checkpoint_data.update({'processed': processed, 'wrote': wrote, 'last_update': time.time()})
            with checkpoint_path.open('w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"更新checkpoint失败: {e}")

    def flush_actions():
        nonlocal actions, wrote
        if not actions:
            return
        try:
            # 使用 chunk_size 控制每次小批量发送到 ES
            success, errors = helpers.bulk(es, actions, request_timeout=120, raise_on_error=False, chunk_size=1000)
            wrote += len(actions)
            if errors:
                logger.warning(f"pages 批量更新部分失败，errors exist")
            update_checkpoint()
        except Exception as e:
            logger.error(f"pages 批量索引失败: {e}")
        actions.clear()

    try:
        line_counter = 0
        with inpath.open('rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.strip():
                    continue
                line_counter += 1
                if line_counter <= processed:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                page_id = obj.get('id') or obj.get('page_id') or None
                title = obj.get('title') or obj.get('heading') or ''
                text = obj.get('text') or obj.get('content') or ''
                namespace = obj.get('namespace', obj.get('ns', 0) or 0)
                lang_hint = lang or (obj.get('url','').split('.')[0] if obj.get('url') else 'und')
                sents = simple_sentence_split(text, lang=lang_hint)
                _id = f"{lang}_{int(page_id)}" if page_id else f"{lang}_title_{title}"
                doc = {
                    "page_id": int(page_id) if page_id else None,
                    "title": title,
                    "title_text": title,
                    "lang": lang_hint,
                    "namespace": int(namespace) if namespace is not None else 0,
                    "text": text,
                    "sentences": sents,
                }
                actions.append({"_op_type": "index", "_index": index_name, "_id": _id, "_source": doc})
                processed += 1

                if processed - last_gc_check >= 20000:
                    last_gc_check = processed
                    try:
                        memory_percent = psutil.Process().memory_percent()
                        if memory_percent > 95:
                            logger.warning(f"内存使用率高: {memory_percent:.1f}%, 提前刷新批处理并执行GC")
                            flush_actions()
                            gc.collect()
                    except Exception:
                        pass

                if processed % 20000 == 0:
                    update_checkpoint()

                if len(actions) >= adaptive_batch_size:
                    flush_actions()

            if actions:
                flush_actions()

        checkpoint_data.update({'status': 'completed', 'processed': processed, 'wrote': wrote, 'completed_time': time.time()})
        with checkpoint_path.open('w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False)

        ESConnectionPool.release_client(es_ip_list, es_port, es_user, es_pass, client_id)
        return {"file": str(inpath), "status": "done", "message": "", "processed": processed, "wrote": wrote}
    except Exception as e:
        update_checkpoint()
        try:
            if actions:
                flush_actions()
        except Exception:
            pass
        ESConnectionPool.release_client(es_ip_list, es_port, es_user, es_pass, client_id)
        return {"file": str(inpath), "status": "error", "message": str(e), "processed": processed, "wrote": wrote}

# -------------------- pagelinks/redirects update worker（单分片） --------------------
def process_parquet_part_update_optimized(part_path: str, lang: str, table: str, es_ip_list: list, es_port: str, es_user: Optional[str], es_pass: Optional[str],
                                index_name: str, updates_batch: int = 2000, workers: int = 4) -> Dict[str, Any]:
    """
    优化版本：使用 stream 读取 parquet，按 doc 聚合更新以显著减少对 ES 的 update 次数
    """
    logger = logging.getLogger()
    inpath = Path(part_path)
    try:
        file_size = inpath.stat().st_size
    except Exception:
        file_size = 0
    adaptive_batch_size = calculate_optimal_batch_size(file_size, updates_batch)

    client_id, es = ESConnectionPool.get_client(es_ip_list, es_port, es_user, es_pass, pool_size=max(1, min(2, workers//8)))

    wrote = 0
    processed = 0
    last_gc_check = 0

    if part_path.endswith(".checkpoint"):
        checkpoint_path = Path(part_path)
    else:
        checkpoint_path = Path(part_path + ".checkpoint")
    checkpoint_data = {}
    if checkpoint_path.exists():
        try:
            with checkpoint_path.open('r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                if 'processed' in checkpoint_data:
                    processed = checkpoint_data['processed']
                    wrote = checkpoint_data.get('wrote', 0)
                    logger.info(f"从检查点继续处理parquet: 已处理={processed}, 已写入={wrote}")
        except Exception as e:
            logging.warning(f"读取parquet checkpoint失败: {e}, 将从头开始处理文件")

    def update_checkpoint():
        try:
            checkpoint_data.update({'processed': processed, 'wrote': wrote, 'last_update': time.time()})
            with checkpoint_path.open('w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"更新parquet checkpoint失败: {e}")

    # 聚合结构：doc_id -> {'pagelinks': [..], 'redirects': [..]}
    updates_by_doc = {}

    def add_update(doc_id: str, item: dict, field: str):
        lst = updates_by_doc.get(doc_id)
        if not lst:
            updates_by_doc[doc_id] = {'pagelinks': [], 'redirects': []}
            lst = updates_by_doc[doc_id]
        if field == 'pagelinks':
            lst['pagelinks'].append(item)
        else:
            lst['redirects'].append(item)

    def flush_aggregated_updates():
        nonlocal updates_by_doc, wrote
        if not updates_by_doc:
            return
        actions = []
        # build one update per doc_id (with params.links)
        for doc_id, payload in updates_by_doc.items():
            if payload['pagelinks']:
                links = payload['pagelinks']
                # build script to append multiple links idempotently
                script = {
                    "source": """
                        if (!ctx._source.containsKey(params.field)) { ctx._source[params.field] = []; }
                        for (int i = 0; i < params.links.length; i++) {
                            def l = params.links[i];
                            boolean exists = false;
                            for (int j = 0; j < ctx._source[params.field].size(); j++) {
                                if (ctx._source[params.field][j].title == l.title && ctx._source[params.field][j].namespace == l.namespace) {
                                    exists = true;
                                    break;
                                }
                            }
                            if (!exists) {
                                ctx._source[params.field].add(l);
                            }
                        }
                    """,
                    "lang": "painless",
                    "params": {"links": links, "field": "pagelinks"}
                }
                actions.append({
                    "_op_type": "update",
                    "_index": index_name,
                    "_id": doc_id,
                    "retry_on_conflict": 3,
                    "script": script,
                    "upsert": {"pagelinks": links}
                })
            if payload['redirects']:
                links = payload['redirects']
                script = {
                    "source": """
                        if (!ctx._source.containsKey(params.field)) { ctx._source[params.field] = []; }
                        for (int i = 0; i < params.links.length; i++) {
                            def l = params.links[i];
                            boolean exists = false;
                            for (int j = 0; j < ctx._source[params.field].size(); j++) {
                                if (ctx._source[params.field][j].title == l.title && ctx._source[params.field][j].namespace == l.namespace) {
                                    exists = true;
                                    break;
                                }
                            }
                            if (!exists) {
                                ctx._source[params.field].add(l);
                            }
                        }
                    """,
                    "lang": "painless",
                    "params": {"links": links, "field": "redirects"}
                }
                actions.append({
                    "_op_type": "update",
                    "_index": index_name,
                    "_id": doc_id,
                    "retry_on_conflict": 3,
                    "script": script,
                    "upsert": {"redirects": links}
                })
        # send to ES
        try:
            # chunk_size 小一些以防内存/连接阻塞
            success, errors = helpers.bulk(es, actions, request_timeout=300, raise_on_error=False, chunk_size=1000)
            wrote += len(actions)
            if errors:
                logger.warning("部分更新失败（aggregated updates）")
        except Exception as e:
            logger.error(f"aggregated bulk 更新失败: {e}")
        updates_by_doc = {}

    # Choose columns to read dynamically to minimize memory
    def choose_cols_from_parquet(pf: pq.ParquetFile, table_name: str):
        names = [c for c in pf.schema.names]
        ln = [n.lower() for n in names]
        colmap = {}
        if table_name == 'pagelinks':
            for cand in ['pl_from', 'from', 'page_id', 'col0']:
                if cand in ln:
                    colmap['from'] = names[ln.index(cand)]
                    break
            for cand in ['pl_title', 'pl_target', 'title', 'col2', 'col1']:
                if cand in ln:
                    colmap['title'] = names[ln.index(cand)]
                    break
            for cand in ['pl_namespace', 'pl_from_namespace', 'namespace', 'col1']:
                if cand in ln:
                    colmap['namespace'] = names[ln.index(cand)]
                    break
        else:
            for cand in ['rd_from', 'from', 'page_id', 'col0']:
                if cand in ln:
                    colmap['from'] = names[ln.index(cand)]
                    break
            for cand in ['rd_title', 'rd_to', 'title', 'col2', 'col1']:
                if cand in ln:
                    colmap['title'] = names[ln.index(cand)]
                    break
            for cand in ['rd_namespace', 'namespace', 'col1']:
                if cand in ln:
                    colmap['namespace'] = names[ln.index(cand)]
                    break
        if 'from' not in colmap and len(names) >= 1:
            colmap['from'] = names[0]
        if 'title' not in colmap and len(names) >= 3:
            colmap['title'] = names[2]
        elif 'title' not in colmap and len(names) >= 2:
            colmap['title'] = names[1]
        return colmap

    try:
        pf = pq.ParquetFile(str(inpath))
        colmap = choose_cols_from_parquet(pf, table)
        # iterate by record batch to limit memory
        batch_size = max(1024, min(10000, adaptive_batch_size))  # reasonable default
        row_index = processed
        total_rows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
        # iterate over row groups then batches
        seen_rows = 0
        for rg in range(pf.num_row_groups):
            if stop_flag:
                break
            row_group_table = pf.read_row_group(rg, columns=list(set(v for v in colmap.values() if v in pf.schema.names)))
            # convert to record batches for streaming
            for rb in row_group_table.to_batches(max_chunksize=batch_size):
                if stop_flag:
                    break
                pylist = rb.to_pylist()  # list of dicts (one dict per row)
                for rec in pylist:
                    seen_rows += 1
                    if seen_rows <= row_index:
                        continue
                    try:
                        from_val = rec.get(colmap['from'])
                        target_title = rec.get(colmap.get('title'))
                        ns = rec.get(colmap.get('namespace')) if colmap.get('namespace') else None
                        if from_val is None:
                            continue
                        try:
                            pid = int(from_val)
                            doc_id = f"{lang}_{pid}"
                        except Exception:
                            doc_id = f"{lang}_{from_val}"
                        if table == 'pagelinks':
                            link_obj = {"title": target_title, "namespace": int(ns) if ns is not None else None}
                            add_update(doc_id, link_obj, 'pagelinks')
                        else:
                            rd_obj = {"title": target_title, "namespace": int(ns) if ns is not None else None}
                            add_update(doc_id, rd_obj, 'redirects')
                        processed += 1
                    except Exception as e:
                        logging.debug("处理 parquet row 失败: %s", e)

                    if processed % 20000 == 0:
                        update_checkpoint()

                    if processed - last_gc_check >= 20000:
                        last_gc_check = processed
                        try:
                            memory_percent = psutil.virtual_memory().percent
                            if memory_percent > 95:
                                logger.warning(f"内存使用率高: {memory_percent:.1f}%, 提前 flush 并 GC")
                                flush_aggregated_updates()
                                gc.collect()
                        except Exception:
                            pass

                    # flush aggregated updates when too many docs accumulated
                    if len(updates_by_doc) >= max(5000, adaptive_batch_size // 2):
                        flush_aggregated_updates()

                # end batch loop

        # flush remaining
        if updates_by_doc:
            flush_aggregated_updates()

        checkpoint_data.update({'status': 'completed', 'processed': processed, 'wrote': wrote, 'completed_time': time.time()})
        with checkpoint_path.open('w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False)

        ESConnectionPool.release_client(es_ip_list, es_port, es_user, es_pass, client_id)
        return {"part": str(inpath), "status": "done", "message": "", "processed": processed, "wrote": wrote}
    except Exception as e:
        update_checkpoint()
        try:
            if updates_by_doc:
                flush_aggregated_updates()
        except Exception:
            pass
        ESConnectionPool.release_client(es_ip_list, es_port, es_user, es_pass, client_id)
        return {"part": str(inpath), "status": "error", "message": str(e), "processed": processed, "wrote": wrote}

# -------------------- 任务管理 --------------------
class TaskManager:
    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.workers = []
        self.tasks_submitted = 0
        self.tasks_completed = 0
        self.stop_event = multiprocessing.Event()

    def add_task(self, task_type, task_func, task_args):
        self.task_queue.put((task_type, task_func, task_args))
        self.tasks_submitted += 1

    def worker_process(self, worker_id):
        while not self.stop_event.is_set():
            try:
                try:
                    task_type, task_func, task_args = self.task_queue.get(timeout=1)
                except Empty:
                    continue
                try:
                    result = task_func(*task_args)
                    self.result_queue.put((task_type, result, None))
                except Exception as e:
                    self.result_queue.put((task_type, None, str(e)))
                if random.random() < 0.05:
                    gc.collect()
            except Exception as e:
                logging.error(f"Worker {worker_id} 发生错误: {e}")
                time.sleep(1)

    def start(self):
        for i in range(self.max_workers):
            p = multiprocessing.Process(target=self.worker_process, args=(i,))
            p.daemon = True
            p.start()
            self.workers.append(p)
        logging.info(f"启动了 {self.max_workers} 个工作进程")

    def process_results(self, result_handler, timeout=0.1):
        try:
            task_type, result, error = self.result_queue.get(timeout=timeout)
            self.tasks_completed += 1
            if error:
                logging.error(f"任务类型 {task_type} 执行失败: {error}")
                return task_type, None, error
            return task_type, result, None
        except Empty:
            return None, None, None

    def shutdown(self):
        self.stop_event.set()
        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except Empty:
                break
        logging.info(f"任务管理器已关闭. 提交任务: {self.tasks_submitted}, 完成任务: {self.tasks_completed}")

# -------------------- 准备任务列表函数 --------------------
def prepare_page_tasks(wikidir: Path, checkpoint: Dict[str, Any], resume: bool) -> List[Path]:
    if not wikidir or not wikidir.exists():
        return []
    page_files = list_wikiextractor_files_for_lang(wikidir)
    logging.info("找到 %d 个 wikiextractor 文件用于索引 pages", len(page_files))
    to_process_pages = []
    for p in page_files:
        key = str(p.resolve())
        entry = checkpoint.get('pages', {}).get(key)
        local_checkpoint_path = Path(str(p) + ".checkpoint")
        if local_checkpoint_path.exists():
            try:
                with local_checkpoint_path.open('r', encoding='utf-8') as f:
                    local_checkpoint = json.load(f)
                    if local_checkpoint.get('status') == 'completed' and resume:
                        logging.info("跳过已完成 page 文件：%s (已处理 %d 行)", p.name, local_checkpoint.get('processed', 0))
                        continue
            except Exception:
                pass
        elif entry and entry.get('status') == 'done' and resume:
            logging.info("跳过已完成 page 文件：%s", p.name)
            continue
        to_process_pages.append(p)
    return to_process_pages

def prepare_parquet_tasks(parquet_root: Path, lang: str, table: str, checkpoint: Dict[str, Any], resume: bool) -> List[Path]:
    parts = find_parquet_parts(parquet_root, lang, table)
    logging.info("找到 %d 个 %s parquet 分片", len(parts), table)
    to_process = []
    for p in parts:
        key = str(p.resolve())
        entry = checkpoint.get(table, {}).get(key)
        local_checkpoint_path = Path(str(p) + ".checkpoint")
        if local_checkpoint_path.exists():
            try:
                with local_checkpoint_path.open('r', encoding='utf-8') as f:
                    local_checkpoint = json.load(f)
                    if local_checkpoint.get('status') == 'completed' and resume:
                        logging.info("跳过已完成 %s 分片：%s (已处理 %d 行)", table, p.name, local_checkpoint.get('processed', 0))
                        continue
            except Exception:
                pass
        elif entry and entry.get('status') == 'done' and resume:
            logging.info("跳过已完成 %s 分片：%s", table, p.name)
            continue
        to_process.append(p)
    return to_process

# -------------------- process_language_optimized (keeps overall flow) --------------------
def process_language_optimized(lang: str, wikidir: Path, parquet_root: Path,
                     es_ip_list: list, es_port: str, es_user: Optional[str], es_pass: Optional[str],
                     index_name: str, workers: int, pages_batch: int, updates_batch: int,
                     log_dir: Path, resume: bool):
    lang_log = add_language_logfile(log_dir, lang)
    logger = logging.getLogger()
    logger.info("=== 开始处理语言：%s ===", lang)
    checkpoint_dir = parquet_root / lang
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cp_path = checkpoint_dir / 'checkpoint_indexing.json'
    checkpoint = load_checkpoint(cp_path)
    if 'pages' not in checkpoint:
        checkpoint['pages'] = {}
    if 'pagelinks' not in checkpoint:
        checkpoint['pagelinks'] = {}
    if 'redirects' not in checkpoint:
        checkpoint['redirects'] = {}

    page_tasks = prepare_page_tasks(wikidir, checkpoint, resume)
    pagelink_tasks = prepare_parquet_tasks(parquet_root, lang, 'pagelinks', checkpoint, resume)
    redirect_tasks = prepare_parquet_tasks(parquet_root, lang, 'redirects', checkpoint, resume)

    if not page_tasks and not pagelink_tasks and not redirect_tasks:
        logger.info("没有需要处理的任务，跳过语言 %s", lang)
        logging.getLogger().removeHandler(lang_log)
        return

    task_manager = TaskManager(max_workers=max(1, min(workers, 8)))
    task_manager.start()

    for p in page_tasks:
        task_manager.add_task('pages', index_single_wiki_file_optimized, (str(p), lang, es_ip_list, es_port, es_user, es_pass, index_name, pages_batch, workers))
    for p in pagelink_tasks:
        task_manager.add_task('pagelinks', process_parquet_part_update_optimized, (str(p), lang, 'pagelinks', es_ip_list, es_port, es_user, es_pass, index_name, updates_batch, workers))
    for p in redirect_tasks:
        task_manager.add_task('redirects', process_parquet_part_update_optimized, (str(p), lang, 'redirects', es_ip_list, es_port, es_user, es_pass, index_name, updates_batch, workers))

    total_tasks = len(page_tasks) + len(pagelink_tasks) + len(redirect_tasks)
    tasks_completed = 0
    last_checkpoint_save = time.time()
    last_progress_log = time.time()

    try:
        while tasks_completed < total_tasks and not stop_flag:
            task_type, result, error = task_manager.process_results(None)
            if task_type:
                tasks_completed += 1
                if task_type == 'pages' and result:
                    file_path = result.get('file')
                    checkpoint['pages'][file_path] = result
                    if result.get('status') == 'done':
                        logger.info("[pages-%s] 完成: %s (processed=%s wrote=%s)", lang, Path(file_path).name, result.get('processed'), result.get('wrote'))
                    else:
                        logger.error("[pages-%s] 错误: %s -> %s", lang, Path(file_path).name, result.get('message'))
                elif task_type in ('pagelinks', 'redirects') and result:
                    part_path = result.get('part')
                    checkpoint[task_type][part_path] = result
                    if result.get('status') == 'done':
                        logger.info("[%s-%s] 完成: %s (processed=%s wrote=%s)", task_type, lang, Path(part_path).name, result.get('processed'), result.get('wrote'))
                    else:
                        logger.error("[%s-%s] 错误: %s -> %s", task_type, lang, Path(part_path).name, result.get('message'))
                now = time.time()
                if now - last_checkpoint_save > 60:
                    save_checkpoint_with_retry(cp_path, checkpoint)
                    last_checkpoint_save = now
                if now - last_progress_log > 30:
                    progress = tasks_completed / total_tasks * 100
                    logger.info(f"[{lang}] 进度: {progress:.1f}% ({tasks_completed}/{total_tasks})")
                    last_progress_log = now
            else:
                time.sleep(0.01)
        save_checkpoint_with_retry(cp_path, checkpoint)
    except KeyboardInterrupt:
        logger.warning("接收到中断信号，正在清理...")
    finally:
        task_manager.shutdown()
    ESConnectionPool.cleanup_old_connections()
    logging.getLogger().removeHandler(lang_log)
    logger.info("=== 语言 %s 处理完毕 (%d/%d 任务) ===", lang, tasks_completed, total_tasks)

# -------------------- 全局资源监控 --------------------
def start_resource_monitoring(interval=60):
    def monitor_resources():
        while not stop_flag:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                logging.info(f"系统资源监控: CPU={cpu_percent}%, 内存={memory_percent}%")
                if memory_percent > 85:
                    logging.warning(f"内存使用率过高 ({memory_percent}%)，执行垃圾回收")
                    gc.collect()
            except Exception as e:
                logging.error(f"资源监控异常: {e}")
            time.sleep(interval)
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()
    return monitor_thread

# -------------------- main --------------------
def main():
    parser = argparse.ArgumentParser(description="Step3: Index WikiExtractor output + attach pagelinks & redirects (optimized)")
    parser.add_argument("--wikiextractor-out", required=True, type=Path, help="WikiExtractor 输出的根目录")
    parser.add_argument("--parquet-out", required=True, type=Path, help="parquet 根目录")
    parser.add_argument("--index", default=os.environ.get("ES_INDEX_PAGES","factnet_pages_v1"), help="ES index name")
    parser.add_argument("--num-shards", type=int, default=6, help="ES索引主分片数")
    parser.add_argument("--num-replicas", type=int, default=1, help="ES索引副本数")
    parser.add_argument("--log-dir", default=Path("./logs"), type=Path, help="日志目录")
    parser.add_argument("--workers", type=int, default=8, help="并行处理的worker数量（建议 <= CPU cores/2）")
    parser.add_argument("--pages-batch", type=int, default=50000, help="pages bulk batch size")
    parser.add_argument("--updates-batch", type=int, default=20000, help="pagelinks/redirects bulk update batch size")
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 续跑")
    parser.add_argument("--langs", nargs="*", default=None, help="只处理这些语言（lang code）")
    args = parser.parse_args()

    setup_global_logging(args.log_dir)
    _install_sigint_handler()

    monitor_thread = start_resource_monitoring()

    wikidir = args.wikiextractor_out.resolve()
    parquet_root = args.parquet_out.resolve()

    es = make_es_client(ES_IP_LIST, ES_PORT, ES_USER, ES_PASSWARD)
    try:
        es.ping()
        logging.info("成功连接到ES集群")
        ensure_index_exists(es, args.index, args.num_shards, args.num_replicas)
    except Exception as e:
        logging.error("无法连接 ES: %s", e)

    langs_map = find_wikiextractor_lang_dirs(wikidir)
    for p in sorted(parquet_root.iterdir()) if parquet_root.exists() else []:
        if p.is_dir():
            lang = p.name.lower()
            if lang not in langs_map:
                langs_map[lang] = parquet_root / lang

    all_langs = sorted(langs_map.keys())
    if args.langs:
        want = set([l.lower() for l in args.langs])
        all_langs = [l for l in all_langs if l in want]
    logging.info("检测到语言（候选）：%s", all_langs)

    multiprocessing.log_to_stderr(logging.INFO)

    task_manager = TaskManager(max_workers=max(1, min(args.workers, 16)))
    task_manager.start()

    lang_loggers = {}
    for lang in all_langs:
        lang_loggers[lang] = add_language_logfile(args.log_dir, lang)

    all_tasks = []
    lang_tasks_map = {}

    for lang in all_langs:
        if stop_flag:
            logging.warning("检测到全局中断标志，停止后续语言处理。")
            break
        wdir = None
        if wikidir.exists():
            possible_dirs = [d for d in wikidir.iterdir() if language_of_dirname(d.name) == lang]
            if possible_dirs:
                wdir = possible_dirs[0]
        candidate = langs_map.get(lang)
        if candidate and candidate.exists() and any(candidate.iterdir()):
            if any(str(x).lower().find('wiki')!=-1 for x in [candidate.name]):
                wdir = candidate
        if not wdir or not wdir.exists():
            found = None
            if wikidir.exists():
                for d in wikidir.iterdir():
                    if d.is_dir() and language_of_dirname(d.name) == lang:
                        found = d
                        break
            if found:
                wdir = found
            else:
                wdir = None

        logging.info("准备处理语言 %s: wikidir=%s parquetdir=%s", lang, (str(wdir) if wdir else "None"), str(parquet_root))

        checkpoint_dir = parquet_root / lang
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cp_path = checkpoint_dir / 'checkpoint_indexing.json'
        checkpoint = load_checkpoint(cp_path)
        if 'pages' not in checkpoint:
            checkpoint['pages'] = {}
        if 'pagelinks' not in checkpoint:
            checkpoint['pagelinks'] = {}
        if 'redirects' not in checkpoint:
            checkpoint['redirects'] = {}

        page_tasks = []
        if wdir:
            page_tasks = prepare_page_tasks(wdir, checkpoint, args.resume)
        pagelink_tasks = prepare_parquet_tasks(parquet_root, lang, 'pagelinks', checkpoint, args.resume)
        redirect_tasks = prepare_parquet_tasks(parquet_root, lang, 'redirects', checkpoint, args.resume)

        lang_tasks = page_tasks + pagelink_tasks + redirect_tasks
        lang_tasks_map[lang] = {'total': len(lang_tasks), 'completed': 0, 'checkpoint': checkpoint, 'cp_path': cp_path, 'last_checkpoint_save': time.time(), 'last_progress_log': time.time()}

        if not lang_tasks:
            logging.info("没有需要处理的任务，跳过语言 %s", lang)
            continue

        for p in page_tasks:
            all_tasks.append({'lang': lang, 'type': 'pages', 'func': index_single_wiki_file_optimized, 'args': (str(p), lang, ES_IP_LIST, ES_PORT, ES_USER, ES_PASSWARD, args.index, args.pages_batch, args.workers)})
        for p in pagelink_tasks:
            all_tasks.append({'lang': lang, 'type': 'pagelinks', 'func': process_parquet_part_update_optimized, 'args': (str(p), lang, 'pagelinks', ES_IP_LIST, ES_PORT, ES_USER, ES_PASSWARD, args.index, args.updates_batch, args.workers)})
        for p in redirect_tasks:
            all_tasks.append({'lang': lang, 'type': 'redirects', 'func': process_parquet_part_update_optimized, 'args': (str(p), lang, 'redirects', ES_IP_LIST, ES_PORT, ES_USER, ES_PASSWARD, args.index, args.updates_batch, args.workers)})

    total_tasks = len(all_tasks)
    logging.info("总共收集了 %d 个任务，开始处理", total_tasks)

    for task in all_tasks:
        task_manager.add_task((task['lang'], task['type']), task['func'], task['args'])

    tasks_completed = 0
    try:
        while tasks_completed < total_tasks and not stop_flag:
            task_info, result, error = task_manager.process_results(None)
            if task_info:
                lang, task_type = task_info
                tasks_completed += 1
                lang_tasks_map[lang]['completed'] += 1
                if task_type == 'pages' and result:
                    file_path = result.get('file')
                    lang_tasks_map[lang]['checkpoint']['pages'][file_path] = result
                    if result.get('status') == 'done':
                        logging.info("[pages-%s] 完成: %s (processed=%s wrote=%s)", lang, Path(file_path).name, result.get('processed'), result.get('wrote'))
                    else:
                        logging.error("[pages-%s] 错误: %s -> %s", lang, Path(file_path).name, result.get('message'))
                elif task_type in ('pagelinks', 'redirects') and result:
                    part_path = result.get('part')
                    lang_tasks_map[lang]['checkpoint'][task_type][part_path] = result
                    if result.get('status') == 'done':
                        logging.info("[%s-%s] 完成: %s (processed=%s wrote=%s)", task_type, lang, Path(part_path).name, result.get('processed'), result.get('wrote'))
                    else:
                        logging.error("[%s-%s] 错误: %s -> %s", task_type, lang, Path(part_path).name, result.get('message'))
                now = time.time()
                if now - lang_tasks_map[lang]['last_checkpoint_save'] > 60:
                    save_checkpoint_with_retry(lang_tasks_map[lang]['cp_path'], lang_tasks_map[lang]['checkpoint'])
                    lang_tasks_map[lang]['last_checkpoint_save'] = now
                if now - lang_tasks_map[lang]['last_progress_log'] > 30:
                    lang_progress = lang_tasks_map[lang]['completed'] / lang_tasks_map[lang]['total'] * 100 if lang_tasks_map[lang]['total'] else 100.0
                    logging.info(f"[{lang}] 进度: {lang_progress:.1f}% ({lang_tasks_map[lang]['completed']}/{lang_tasks_map[lang]['total']})")
                    lang_tasks_map[lang]['last_progress_log'] = now
                if lang_tasks_map[lang]['completed'] == lang_tasks_map[lang]['total']:
                    logging.info("=== 语言 %s 处理完毕 (%d/%d 任务) ===", lang, lang_tasks_map[lang]['completed'], lang_tasks_map[lang]['total'])
                    save_checkpoint_with_retry(lang_tasks_map[lang]['cp_path'], lang_tasks_map[lang]['checkpoint'])
            else:
                time.sleep(0.01)
            if tasks_completed % 100 == 0 and total_tasks:
                overall_progress = tasks_completed / total_tasks * 100
                logging.info(f"总体进度: {overall_progress:.1f}% ({tasks_completed}/{total_tasks})")
        for lang, info in lang_tasks_map.items():
            save_checkpoint_with_retry(info['cp_path'], info['checkpoint'])
    except KeyboardInterrupt:
        logging.warning("接收到中断信号，正在清理...")
    finally:
        task_manager.shutdown()
    for lang, handler in lang_loggers.items():
        logging.getLogger().removeHandler(handler)
    ESConnectionPool.cleanup_old_connections()
    logging.info("全部语言处理完毕，总计完成 %d/%d 任务。", tasks_completed, total_tasks)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(e)
    finally:
        # 保持与你原脚本相同的退出信号（如有）
        try:
            os.system('echo > /obssidecar/terminate/0')
        except Exception:
            pass