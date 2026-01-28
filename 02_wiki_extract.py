#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nohup python 02_wiki_extract.py \
--orig wikipedia_page \
--out 01_wikiextractor_out \
--processes 16 \
--log-dir 01_wikiextractor_out/logs \
> 01_wikiextractor_out/runing_log.txt &
"""

#!/usr/bin/env python3
import argparse
import logging
import logging.handlers
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

stop_flag = False
def _install_sigint_handler():
    def _handler(signum, frame):
        global stop_flag
        logging.warning("收到中断信号，当前任务完成后将停止...")
        stop_flag = True
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

# ---- 日志 ----
def setup_logging(log_dir: Path, logfile_name="wikiextractor_simple.log"):
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 控制台输出
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(ch)

    # 主日志文件
    fh = logging.handlers.RotatingFileHandler(
        str(log_dir / logfile_name), maxBytes=10*1024*1024, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

def add_language_logfile(log_dir: Path, lang: str):
    fh = logging.handlers.RotatingFileHandler(
        str(log_dir / f"{lang}.log"), maxBytes=5*1024*1024, backupCount=2, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(fh)
    return fh

# ---- 辅助 ----
LANG_RE = re.compile(r"^(.+?)wiki", re.IGNORECASE)
def language_of_filename(fn: str) -> str:
    base = Path(fn).name
    if base.endswith(".bz2"):
        base = base[:-4]
    m = LANG_RE.search(base)
    if m:
        return m.group(1).lower()
    return base.lower()

# ---- 主逻辑 ----
def main():
    parser = argparse.ArgumentParser(description="简单的 WikiExtractor 重跑脚本（按语言串行）")
    parser.add_argument("--orig", required=True, type=Path, help="原始 .bz2 文件目录")
    parser.add_argument("--out", required=True, type=Path, help="输出根目录")
    parser.add_argument("--log-dir", default=Path("./logs"), type=Path, help="日志目录")
    parser.add_argument("--processes", type=int, default=30, help="WikiExtractor 的 --processes 参数")
    parser.add_argument("--python", default=sys.executable, help="Python 可执行文件路径（默认当前环境）")
    args = parser.parse_args()

    setup_logging(args.log_dir)
    _install_sigint_handler()

    orig = args.orig.resolve()
    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    bz2_files = sorted(orig.glob("*.bz2"))
    if not bz2_files:
        logging.info("在 %s 下找不到任何 .bz2 文件。", orig)
        return

    # 按语言分组
    groups = {}
    for f in bz2_files:
        lang = language_of_filename(f.name)
        groups.setdefault(lang, []).append(f)

    logging.info("检测到 %d 个语言：%s", len(groups), list(groups.keys()))

    for lang, files in groups.items():
        if stop_flag:
            logging.warning("检测到中断，停止后续处理。")
            break

        lang_log = add_language_logfile(args.log_dir, lang)
        logging.info("=== 开始处理语言: %s ===", lang)

        for bz2_path in files:
            bn = bz2_path.stem
            final_outdir = out_root / bn
            tmp_outdir = out_root / (bn + ".__inprogress__")

            if final_outdir.exists():
                logging.info("[%s] 已存在，跳过。", bn)
                continue

            if tmp_outdir.exists():
                logging.warning("[%s] 检测到残留 __inprogress__ 目录，删除后重新运行。", bn)
                shutil.rmtree(tmp_outdir)

            logging.info("[%s] 开始运行 WikiExtractor...", bn)
            tmp_outdir.mkdir(parents=True, exist_ok=True)

            cmd = [
                args.python, "-m", "wikiextractor.WikiExtractor",
                "--json",
                "--processes", str(args.processes),
                "-o", str(tmp_outdir),
                str(bz2_path)
            ]
            logging.info("[%s] 执行命令: %s", bn, " ".join(cmd))
            ret = subprocess.call(cmd)

            if ret == 0:
                shutil.move(str(tmp_outdir), str(final_outdir))
                logging.info("[%s] 完成。输出目录: %s", bn, final_outdir)
            else:
                logging.error("[%s] WikiExtractor 运行失败，退出码=%d，保留 __inprogress__ 目录。", bn, ret)

            if stop_flag:
                logging.warning("检测到中断信号，提前结束。")
                break

        logging.info("=== 语言 %s 处理完毕 ===", lang)
        logging.getLogger().removeHandler(lang_log)

    logging.info("全部任务完成。")

if __name__ == "__main__":
    main()