#!/usr/bin/env python3

"""
python 01_parse_wikidata_factstatement.py \
  --input **/latest-all.json.bz2 \
  --outdir 01_factstatement \
  --format parquet \
  --batch-size 50000 \
  --workers 30 \
  --log-dir logs
"""

import argparse
import bz2
import gzip
import json
import os
import sys
import math
import time
import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, List, Tuple
from pathlib import Path
from functools import wraps

# optional nice progress bar
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# pandas/pyarrow for parquet
try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pd = None
    pa = None
    pq = None

# ---------------- logging & retry ----------------

def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'factnet_extraction_{time.strftime("%Y%m%d_%H%M%S")}.log')
    logger = logging.getLogger('factnet')
    logger.setLevel(logging.INFO)
    # avoid adding multiple handlers on repeated imports
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

def retry(max_retries=3, delay=1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger('factnet')
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    logger.warning(f"Attempt {retries}/{max_retries} failed in {func.__name__}: {str(e)}", exc_info=False)
                    if retries >= max_retries:
                        logger.error(f"Max retries reached for {func.__name__}", exc_info=True)
                        raise
                    time.sleep(delay * (2 ** (retries - 1)))  # exponential backoff
            return None
        return wrapper
    return decorator

# ---------------- data model & helpers ----------------

@dataclass
class FactStatement:
    core_id: str
    subject_qid: Optional[str]
    property_pid: Optional[str]
    value: Any
    rank: Optional[str]
    qualifiers: Optional[Dict]
    references: Optional[List[Dict]]
    last_edit: Optional[str]
    provenance: Optional[Dict]
    confidence: float
    sitelinks: Optional[Dict] = None
    labels_present: Optional[List[str]] = None
    claim_hash: Optional[str] = None
    claim_hash_prefix: Optional[str] = None
    subject_prefix: Optional[str] = None

    def to_flat_dict(self) -> Dict:
        return {
            'core_id': self.core_id,
            'subject_qid': self.subject_qid,
            'property_pid': self.property_pid,
            'value': json.dumps(self.value, ensure_ascii=False),
            'rank': self.rank,
            'qualifiers': json.dumps(self.qualifiers or {}, ensure_ascii=False),
            'references': json.dumps(self.references or [], ensure_ascii=False),
            'last_edit': self.last_edit,
            'provenance': json.dumps(self.provenance or {}, ensure_ascii=False),
            'confidence': self.confidence,
            'sitelinks': json.dumps(self.sitelinks or {}, ensure_ascii=False),
            'labels_present': json.dumps(self.labels_present or [], ensure_ascii=False),
            'claim_hash': self.claim_hash or '',
            'claim_hash_prefix': self.claim_hash_prefix or '',
            'subject_prefix': self.subject_prefix or ''
        }

def smart_open(path: str):
    if path.endswith('.bz2'):
        return bz2.open(path, 'rt', encoding='utf-8')
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8')
    return open(path, 'r', encoding='utf-8')

def sha256_hex(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def normalize_value_for_hash(value) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(value)

def build_claim_hash(subject_qid: str, property_pid: str, value_repr: Any, qualifiers: Any) -> str:
    norm = {
        's': subject_qid,
        'p': property_pid,
        'v': normalize_value_for_hash(value_repr),
        'q': normalize_value_for_hash(qualifiers or {})
    }
    s = json.dumps(norm, sort_keys=True, ensure_ascii=False)
    return sha256_hex(s)

def subject_prefix_from_qid(qid: str) -> str:
    if not qid or not qid.startswith('Q'):
        return "xx"
    digits = ''.join(ch for ch in qid[1:] if ch.isdigit())
    if not digits:
        return "00"
    digits = digits.zfill(2)
    return digits[-2:]

def claim_hash_prefix(ch: str) -> str:
    if not ch:
        return "00"
    return ch[:2]

# ---------------- serialization / extraction ----------------

def serialize_value(claim: Dict) -> Any:
    mainsnak = claim.get('mainsnak') or {}
    dv = mainsnak.get('datavalue')
    if not dv:
        return None
    dvv = dv.get('value')
    t = dv.get('type') or mainsnak.get('datatype')
    if t == 'wikibase-entityid':
        return dvv.get('id')
    if t == 'string':
        return dvv
    if t == 'time':
        return dvv.get('time')
    if t == 'monolingualtext':
        return {'text': dvv.get('text'), 'lang': dvv.get('language')}
    if t == 'globecoordinate':
        return {'lat': dvv.get('latitude'), 'lon': dvv.get('longitude'), 'alt': dvv.get('altitude')}
    if t == 'quantity':
        return {'amount': dvv.get('amount'), 'unit': dvv.get('unit')}
    return dvv

def compute_confidence(rank: str, n_refs: int) -> float:
    base_map = {'preferred': 0.9, 'normal': 0.7, 'deprecated': 0.3}
    base = base_map.get(rank, 0.6)
    add = 0.0
    if n_refs > 0:
        add = min(0.09, math.log1p(n_refs) * 0.02)
    return round(min(1.0, base + add), 4)

def extract_statements_from_entity(entity: Dict) -> Tuple[List[FactStatement], List[Dict]]:
    """
    labels_record: {'subject_qid','language','label','aliases'}
    """
    results = []
    labels_out = []
    subject_qid = entity.get('id')
    if not subject_qid:
        return results, labels_out

    # labels and aliases
    labels_map = entity.get('labels', {}) or {}
    aliases_map = entity.get('aliases', {}) or {}
    # sitelinks
    sitelinks = {}
    for lang, sl in (entity.get('sitelinks') or {}).items():
        if isinstance(sl, dict) and 'title' in sl:
            sitelinks[lang] = sl.get('title')

    # produce labels records per language
    for lang, lab_obj in labels_map.items():
        lab = lab_obj.get('value') if isinstance(lab_obj, dict) else None
        alias_list = []
        if lang in aliases_map:
            alias_list = [a.get('value') for a in aliases_map.get(lang, []) if isinstance(a, dict) and a.get('value')]
        if lab:
            labels_out.append({'subject_qid': subject_qid, 'language': lang, 'label': lab, 'aliases': alias_list})

    # if no labels exist, add fallback record
    if not labels_map:
        labels_out.append({'subject_qid': subject_qid, 'language': 'und', 'label': subject_qid, 'aliases': []})

    claims = entity.get('claims', {}) or {}
    for pid, claim_list in claims.items():
        for claim in claim_list:
            core_id = claim.get('id')
            if not core_id:
                continue
            value = serialize_value(claim)
            rank = claim.get('rank')
            qualifiers = claim.get('qualifiers')
            references = claim.get('references')
            last_edit = entity.get('modified')
            provenance = {'entity_id': subject_qid}
            n_refs = len(references) if references else 0
            confidence = compute_confidence(rank, n_refs)
            ch = build_claim_hash(subject_qid, pid, value, qualifiers)
            ch_pref = claim_hash_prefix(ch)
            subj_pref = subject_prefix_from_qid(subject_qid)
            fs = FactStatement(
                core_id=core_id,
                subject_qid=subject_qid,
                property_pid=pid,
                value=value,
                rank=rank,
                qualifiers=qualifiers,
                references=references,
                last_edit=last_edit,
                provenance=provenance,
                confidence=confidence,
                sitelinks=sitelinks,
                labels_present=list(labels_map.keys()),
                claim_hash=ch,
                claim_hash_prefix=ch_pref,
                subject_prefix=subj_pref
            )
            results.append(fs)
    return results, labels_out


@retry(max_retries=3, delay=1.0)
def write_checkpoint(checkpoint_path: str, line_number: int, file_index: int):
    logger = logging.getLogger('factnet')
    tmp = checkpoint_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(f"{line_number},{file_index}\n")
    os.replace(tmp, checkpoint_path)
    logger.info(f"Checkpoint written: line={line_number}, file_index={file_index}")

@retry(max_retries=3, delay=1.0)
def write_batch_parquet_rows_pyarrow(rows: List[Dict], out_path: str, schema: Optional[pa.Schema] = None):
    logger = logging.getLogger('factnet')
    if pa is None or pq is None:
        raise RuntimeError("pyarrow is required for parquet output")
    # rows: list of flat dicts (strings)
    logger.info(f"Writing {len(rows)} rows to parquet file {out_path}")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path, compression='snappy')
    logger.info(f"Successfully wrote parquet {out_path}")

@retry(max_retries=3, delay=1.0)
def append_parquet_with_writer(writer: pq.ParquetWriter, table: pa.Table):
    logger = logging.getLogger('factnet')
    writer.write_table(table)
    logger.info(f"Appended table with {table.num_rows} rows to writer")

@retry(max_retries=3, delay=1.0)
def write_csv_rows(rows: List[Dict], out_path: str):
    logger = logging.getLogger('factnet')
    logger.info(f"Appending {len(rows)} rows to CSV {out_path}")
    header = not os.path.exists(out_path)
    import csv
    with open(out_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.info(f"Successfully wrote CSV {out_path}")

# ---------------- worker / parallel orchestration ----------------

def worker_task(entities_chunk: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    每个 worker 从 entities_chunk 中提取 fact statements & labels，并返回为平展的 dict 列表
    """
    rows = []
    labels_rows = []
    for entity in entities_chunk:
        try:
            fs_list, labels_list = extract_statements_from_entity(entity)
            for fs in fs_list:
                rows.append(fs.to_flat_dict())
            for l in labels_list:
                labels_rows.append(l)
        except Exception as e:
            continue
    return rows, labels_rows

# ---------------- main processing loop ----------------

def iter_wikidata_entities(path: str, start_line: int = 0) -> Iterator[Dict]:
    logger = logging.getLogger('factnet')
    with smart_open(path) as f:
        for i, raw in enumerate(f):
            if i < start_line:
                continue
            raw = raw.strip()
            if not raw:
                continue
            if raw in ('[', ']'):
                continue
            if raw.endswith(','):
                raw = raw[:-1]
            try:
                obj = json.loads(raw)
                yield obj, start_line + i
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error at line {start_line + i}: {str(e)}")
                continue

def process_dump_stream(
    input_path: str,
    outdir: str,
    fmt: str = 'parquet',
    batch_size: int = 200,
    max_entities: Optional[int] = None,
    workers: int = 4,
    max_rows_per_file: int = 5_000_000
):
    logger = logging.getLogger('factnet')
    os.makedirs(outdir, exist_ok=True)
    checkpoint_path = os.path.join(outdir, 'checkpoint.txt')
    labels_out_path_base = os.path.join(outdir, 'labels')  # will produce files like labels_part_{i}.parquet or .csv

    start_line = 0
    file_index = 0

    # 恢复 checkpoint（line_number, file_index）
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                ln, fi = f.read().strip().split(',')
                start_line = int(ln)
                file_index = int(fi)
                logger.info(f"Resuming from checkpoint line={start_line}, file_index={file_index}")
        except Exception as e:
            logger.warning(f"Failed to read checkpoint: {str(e)}. Starting from 0.")

    # Prepare multiprocessing pool
    pool = mp.Pool(processes=workers)
    total_entities = 0
    total_statements = 0
    total_labels = 0

    entity_buffer: List[Dict] = []

    # Parquet writer management (single writer per output part)
    parquet_writer = None
    current_rows = 0
    out_path = None

    # labels file index
    labels_file_index = 0

    # iterate entities pair (entity, line_no)
    iterator = iter_wikidata_entities(input_path, start_line=start_line)

    # For progress / logging
    last_log_ts = time.time()

    try:
        for entity_obj, absolute_line_no in iterator:
            entity_buffer.append(entity_obj)
            # process when we have a batch
            if len(entity_buffer) >= batch_size:
                # split buffer into chunks for workers
                chunksize = max(1, len(entity_buffer) // workers)
                chunks = [entity_buffer[i:i+chunksize] for i in range(0, len(entity_buffer), chunksize)]

                logger.info(f"Dispatching batch: {len(entity_buffer)} entities -> {len(chunks)} worker chunks")
                # map to workers
                try:
                    results = pool.map(worker_task, chunks)
                except Exception as e:
                    logger.error(f"Multiprocessing map failed: {str(e)}", exc_info=True)
                    # attempt to process serially as fallback
                    results = [worker_task(c) for c in chunks]

                # flatten results
                rows = []
                labels_rows = []
                for r_rows, r_labels in results:
                    rows.extend(r_rows)
                    labels_rows.extend(r_labels)

                # write rows out
                if rows:
                    if fmt == 'parquet':
                        # create arrow table
                        try:
                            table = pa.Table.from_pylist(rows)
                        except Exception as e:
                            logger.error(f"Failed to convert rows to pyarrow table: {str(e)}", exc_info=True)
                            table = None
                        if table is not None:
                            # open new writer if needed
                            if parquet_writer is None:
                                out_path = os.path.join(outdir, f'factstatements_part_{file_index}.parquet')
                                parquet_writer = pq.ParquetWriter(out_path, table.schema, compression='snappy')
                                logger.info(f"Opened new parquet writer: {out_path}")
                                file_index += 1
                            # append
                            append_parquet_with_writer(parquet_writer, table)
                            current_rows += table.num_rows
                            total_statements += table.num_rows

                            # rotate file if exceed limit
                            if current_rows >= max_rows_per_file:
                                parquet_writer.close()
                                logger.info(f"Closed parquet writer after {current_rows} rows")
                                parquet_writer = None
                                current_rows = 0
                    else:
                        # csv append
                        write_csv_rows(rows, os.path.join(outdir, f'factstatements_part_{file_index}.csv'))
                        total_statements += len(rows)

                # write labels out as separate file (parquet or csv)
                if labels_rows:
                    # deduplicate labels_rows by (subject_qid, language, label) might be expensive; we keep as is for simplicity
                    if fmt == 'parquet':
                        labels_table = pa.Table.from_pylist(labels_rows)
                        labels_out_path = os.path.join(labels_out_path_base + f"_part_{labels_file_index}.parquet")
                        write_batch_parquet_rows_pyarrow(labels_rows, labels_out_path)
                        labels_file_index += 1
                        total_labels += len(labels_rows)
                    else:
                        write_csv_rows(labels_rows, os.path.join(outdir, f'labels_part_{labels_file_index}.csv'))
                        labels_file_index += 1
                        total_labels += len(labels_rows)

                total_entities += len(entity_buffer)
                # update checkpoint: absolute_line_no is last entity line number processed in this batch
                write_checkpoint(checkpoint_path, absolute_line_no + 1, file_index)  # next start line is last+1
                logger.info(f"Batch processed: total_entities={total_entities}, total_statements={total_statements}, total_labels={total_labels}")
                entity_buffer.clear()

            # periodic log to avoid silence
            if time.time() - last_log_ts > 30:
                logger.info(f"Progress: processed entities ~{total_entities}, buffer={len(entity_buffer)}")
                last_log_ts = time.time()

            # check max_entities limit
            if max_entities and total_entities >= max_entities:
                logger.info(f"Reached max_entities limit {max_entities}. Stopping.")
                break

        # final flush for remaining buffer
        if entity_buffer:
            logger.info(f"Processing final buffer of {len(entity_buffer)} entities")
            chunksize = max(1, len(entity_buffer) // workers)
            chunks = [entity_buffer[i:i+chunksize] for i in range(0, len(entity_buffer), chunksize)]
            try:
                results = pool.map(worker_task, chunks)
            except Exception as e:
                logger.error(f"Multiprocessing final map failed: {str(e)}", exc_info=True)
                results = [worker_task(c) for c in chunks]

            rows = []
            labels_rows = []
            for r_rows, r_labels in results:
                rows.extend(r_rows)
                labels_rows.extend(r_labels)

            if rows:
                if fmt == 'parquet':
                    table = pa.Table.from_pylist(rows)
                    if parquet_writer is None:
                        out_path = os.path.join(outdir, f'factstatements_part_{file_index}.parquet')
                        parquet_writer = pq.ParquetWriter(out_path, table.schema, compression='snappy')
                        logger.info(f"Opened new parquet writer (final): {out_path}")
                        file_index += 1
                    append_parquet_with_writer(parquet_writer, table)
                    current_rows += table.num_rows
                    total_statements += table.num_rows
                else:
                    write_csv_rows(rows, os.path.join(outdir, f'factstatements_part_{file_index}.csv'))
                    total_statements += len(rows)

            if labels_rows:
                if fmt == 'parquet':
                    labels_out_path = os.path.join(labels_out_path_base + f"_part_{labels_file_index}.parquet")
                    write_batch_parquet_rows_pyarrow(labels_rows, labels_out_path)
                    labels_file_index += 1
                    total_labels += len(labels_rows)
                else:
                    write_csv_rows(labels_rows, os.path.join(outdir, f'labels_part_{labels_file_index}.csv'))
                    labels_file_index += 1
                    total_labels += len(labels_rows)

            total_entities += len(entity_buffer)
            write_checkpoint(checkpoint_path, absolute_line_no + 1, file_index)
            entity_buffer.clear()

    except Exception as e:
        logger.error(f"Fatal error in processing loop: {str(e)}", exc_info=True)
        raise
    finally:
        try:
            pool.close()
            pool.join()
        except Exception:
            pass
        if parquet_writer:
            try:
                parquet_writer.close()
            except Exception:
                pass

    logger.info(f"Done. Processed {total_entities} entities and extracted {total_statements} FactStatements to {outdir}")
    print(f"Done. Processed {total_entities} entities and extracted {total_statements} FactStatements to {outdir}")

# ---------------- CLI ----------------

def build_argparser():
    p = argparse.ArgumentParser(description='FactNet Stage1 Parallel extractor with checkpoint & logging')
    p.add_argument('--input', required=True, help='Input Wikidata entities NDJSON file (support .bz2/.gz)')
    p.add_argument('--outdir', required=True, help='Output directory')
    p.add_argument('--format', choices=['parquet', 'csv'], default='parquet')
    p.add_argument('--batch-size', type=int, default=200, help='Number of entities batched before dispatching to workers')
    p.add_argument('--max-entities', type=int, default=None, help='Max entities to process (for testing)')
    p.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 1), help='Worker processes for parsing')
    p.add_argument('--log-dir', default='logs', help='Directory for logs')
    p.add_argument('--max-rows-per-file', type=int, default=5_000_000, help='Max rows per parquet output file')
    return p

def main(argv=None):
    args = build_argparser().parse_args(argv)
    logger = setup_logging(args.log_dir)
    logger.info(f"Starting parse_wikidata_stream_parallel.py with args: {args}")

    # sanity checks
    if args.format == 'parquet' and (pa is None or pq is None):
        logger.error("pyarrow/pyarrow.parquet required for parquet output but not available")
        sys.exit(1)
    if args.format == 'csv' and pd is None:
        logger.warning("pandas not found; csv output will still work using csv writer")

    try:
        process_dump_stream(
            input_path=args.input,
            outdir=args.outdir,
            fmt=args.format,
            batch_size=args.batch_size,
            max_entities=args.max_entities,
            workers=args.workers,
            max_rows_per_file=args.max_rows_per_file
        )
        logger.info("Process completed successfully")
    except Exception as e:
        logger.error(f"Process failed with exception: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()