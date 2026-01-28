#!/usr/bin/env python3
"""
nohup python 03_sql_to_parquet.py \
--pagelinks-dir wiki_pagelinks \
--redirects-dir wiki_redirect \
--out-root 02_sql_to_parquet \
--workers 16 \
--log 02_sql_to_parquet/logs \
> 02_sql_to_parquet/runing_log.txt &
"""

import argparse
import gzip
import json
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# --- parsing helpers ---

def split_tuple_fields(s: str):
    fields = []
    cur = ''
    in_str = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "'":
            if in_str and i + 1 < len(s) and s[i + 1] == "'":
                cur += "''"
                i += 2
                continue
            in_str = not in_str
            cur += ch
        elif ch == ',' and not in_str:
            fields.append(cur.strip())
            cur = ''
        else:
            cur += ch
        i += 1
    if cur:
        fields.append(cur.strip())
    return fields


def clean_sql_field(f: str):
    f = f.strip()
    if f == 'NULL':
        return None
    if f.startswith("'") and f.endswith("'"):
        inner = f[1:-1].replace("''", "'")
        return inner
    return f


def parse_insert_line(line: str):
    """
    Parse a single INSERT INTO ... VALUES (...),(...),...; line.
    Returns a list of rows (list of fields).
    """
    # Find start of VALUES
    idx = line.find("VALUES")
    if idx == -1:
        return []

    rest = line[idx + 6:].strip().rstrip(';')

    rows = []
    current_tuple = ''
    depth = 0
    in_string = False
    i = 0

    while i < len(rest):
        ch = rest[i]
        current_tuple += ch

        if ch == "'":
            # Handle escaped single quote ''
            if in_string and i + 1 < len(rest) and rest[i + 1] == "'":
                current_tuple += "'"
                i += 1
            else:
                in_string = not in_string

        elif ch == '(' and not in_string:
            depth += 1
        elif ch == ')' and not in_string:
            depth -= 1
            if depth == 0:
                # end of tuple
                inner = current_tuple.strip()
                if inner.startswith('(') and inner.endswith(')'):
                    inner = inner[1:-1]
                fields = split_tuple_fields(inner)
                fields = [clean_sql_field(f) for f in fields]
                rows.append(fields)
                current_tuple = ''
                # skip over commas and spaces between tuples
                while i + 1 < len(rest) and rest[i + 1] in ', \\n\\t\\r':
                    i += 1
        i += 1

    return rows


# --- parquet writer ---

def write_chunk(rows: List[List[str]], outdir: Path, tablename: str, idx: int, logger: logging.Logger):
    if not rows:
        return
    maxcols = max(len(r) for r in rows)
    cols = {f'col{i}': [r[i] if i < len(r) else None for r in rows] for i in range(maxcols)}
    table = pa.Table.from_pydict(cols)
    outpath = outdir / f"{tablename}_part_{idx}.parquet"
    pq.write_table(table, str(outpath), compression='snappy')
    logger.info(f"Wrote {len(rows)} rows to {outpath}")


# --- process a single input file (worker) ---

def process_file_worker(infile: str, outdir: str, tablename: str, chunk_size: int = 100000):
    try:
        inpath = Path(infile)
        outdir_p = Path(outdir)
        outdir_p.mkdir(parents=True, exist_ok=True)
        chunk = []
        out_index = 0
        parts_written = 0
        
        with gzip.open(inpath, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if not line.startswith('INSERT INTO'):
                    continue
                rows = parse_insert_line(line)
                for fields in rows:
                    chunk.append(fields)
                    if len(chunk) >= chunk_size:
                        write_chunk(chunk, outdir_p, tablename, out_index, logger=_dummy_logger)
                        out_index += 1
                        parts_written += 1
                        chunk = []
        if chunk:
            write_chunk(chunk, outdir_p, tablename, out_index, logger=_dummy_logger)
            parts_written += 1
        return {"infile": str(inpath), "status": "done", "message": "", "parts_written": parts_written}
    except Exception as e:
        return {"infile": infile, "status": "error", "message": str(e), "parts_written": 0}


class _DummyLogger:
    def info(self, *args, **kwargs):
        pass

_dummy_logger = _DummyLogger()


# --- checkpoint helpers ---

def load_checkpoint(cp_path: Path):
    if cp_path.exists():
        try:
            with cp_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_checkpoint(cp_path: Path, data):
    tmp = cp_path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(cp_path)


# --- language detection ---

LANG_RE = re.compile(r'^([a-z0-9_+-]+?)wiki', re.IGNORECASE)

def detect_language_from_filename(name: str) -> str:
    m = LANG_RE.match(name)
    if m:
        return m.group(1).lower()
    # fallback: try first token before '-'
    token = name.split('-')[0]
    if token:
        return token.lower()
    return 'unknown'


# --- orchestration per language and table ---

def process_language(lang: str, files: List[Path], out_root: Path, tablename: str, workers: int, chunk_size: int, logger: logging.Logger, force: bool = False):
    logger.info(f"Processing language {lang} for table {tablename} ({len(files)} files)")
    lang_out = out_root / lang / tablename
    lang_out.mkdir(parents=True, exist_ok=True)
    cp_path = lang_out / 'checkpoint.json'

    checkpoint = load_checkpoint(cp_path)
    if 'files' not in checkpoint:
        checkpoint['files'] = {}

    to_process = []
    for p in sorted(files):
        key = str(p.resolve())
        entry = checkpoint['files'].get(key)
        if entry and entry.get('status') == 'done' and not force:
            logger.info(f"Skipping already completed file: {p.name}")
            continue
        to_process.append(p)

    if not to_process:
        logger.info(f"All files completed for language {lang} table {tablename}")
        return

    logger.info(f"Starting parallel processing for {len(to_process)} files with {workers} workers")
    futures = []
    with ProcessPoolExecutor(max_workers=workers) as exe:
        for p in to_process:
            futures.append(exe.submit(process_file_worker, str(p), str(lang_out), tablename, chunk_size))
        for fut in as_completed(futures):
            res = fut.result()
            infile = res.get('infile')
            status = res.get('status')
            msg = res.get('message')
            parts = res.get('parts_written', 0)
            checkpoint['files'][infile] = {"status": status, "parts_written": parts, "message": msg}
            save_checkpoint(cp_path, checkpoint)
            if status == 'done':
                logger.info(f"Completed {infile} (parts={parts})")
            else:
                logger.error(f"Error processing {infile}: {msg}")


# --- logging setup ---

def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('parse_sql_to_parquet')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(log_path), encoding='utf-8')
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# --- main: accept two input folders and process both in one run ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pagelinks-dir', help='Directory containing pagelinks .sql.gz files')
    parser.add_argument('--redirects-dir', help='Directory containing redirect .sql.gz files')
    parser.add_argument('--out-root', required=True, help='Output root for parquet files and checkpoints')
    parser.add_argument('--workers', type=int, default=max(1, os.cpu_count() // 2), help='Number of worker processes per language')
    parser.add_argument('--chunk-size', type=int, default=100000, help='Rows per parquet part')
    parser.add_argument('--force', action='store_true', help='Force re-processing even if checkpoint says done')
    parser.add_argument('--log', default='logs/parse_sql_to_parquet.log', help='Log file path')
    args = parser.parse_args()

    log_path = Path(args.log)
    logger = setup_logging(log_path)

    if not args.pagelinks_dir and not args.redirects_dir:
        logger.error('Please specify at least one of --pagelinks-dir or --redirects-dir')
        sys.exit(1)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Prepare jobs: for each table type, group files by detected language
    table_jobs = []  # list of (tablename, dict(lang -> list(Path)))

    if args.pagelinks_dir:
        pl_dir = Path(args.pagelinks_dir)
        if not pl_dir.exists():
            logger.error(f'Pagelinks dir does not exist: {pl_dir}')
            sys.exit(1)
        files = [p for p in pl_dir.glob('*.sql.gz') if p.is_file()]
        groups = {}
        for p in files:
            lang = detect_language_from_filename(p.name)
            groups.setdefault(lang, []).append(p)
        table_jobs.append(('pagelinks', groups))

    if args.redirects_dir:
        rd_dir = Path(args.redirects_dir)
        if not rd_dir.exists():
            logger.error(f'Redirects dir does not exist: {rd_dir}')
            sys.exit(1)
        files = [p for p in rd_dir.glob('*.sql.gz') if p.is_file()]
        groups = {}
        for p in files:
            lang = detect_language_from_filename(p.name)
            groups.setdefault(lang, []).append(p)
        table_jobs.append(('redirects', groups))

    # Process each table, each language. We'll process tables sequentially; inside each language we'll parallelize.
    for tablename, groups in table_jobs:
        logger.info(f"Processing table {tablename} with {len(groups)} detected languages")
        for lang, files in sorted(groups.items()):
            try:
                process_language(lang, files, out_root, tablename, args.workers, args.chunk_size, logger, force=args.force)
            except Exception as e:
                logger.exception(f"Unhandled exception processing {tablename} language {lang}: {e}")

    logger.info('All processing finished')


if __name__ == '__main__':
    main()