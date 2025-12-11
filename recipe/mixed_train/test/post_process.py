#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from pathlib import Path

def sanitize_filename(name: str, max_len: int = 128) -> str:
    """把任意字符串转为安全的文件名（仅保留字母数字和-_.，空白变为_）"""
    s = str(name)
    s = "".join(ch if (ch.isalnum() or ch in "-_. ") else "_" for ch in s)
    s = re.sub(r"\s+", "_", s).strip("_")
    if not s:
        s = "item"
    return s[:max_len]

def main():
    parser = argparse.ArgumentParser(
        description="将 JSONL 文件的每一行拆分为单独的 JSON 文件"
    )
    parser.add_argument("jsonl", help="输入的 .jsonl 文件路径")
    parser.add_argument(
        "-o", "--outdir",
        help="输出目录（默认：<输入文件名>_split）"
    )
    parser.add_argument(
        "--prefix",
        help="输出文件名前缀（默认：<输入文件名>_）"
    )
    parser.add_argument(
        "--field-as-name",
        help="用某个 JSON 字段作为文件名（缺失则回退到序号），例如：--field-as-name id"
    )
    parser.add_argument(
        "--start", type=int, default=1,
        help="起始序号（默认：1）"
    )
    parser.add_argument(
        "--pad", type=int, default=5,
        help="序号零填充宽度（默认：5，设为0则不填充）"
    )
    parser.add_argument(
        "--indent", type=int, default=2,
        help="JSON 输出缩进（默认：2）"
    )
    parser.add_argument(
        "--ensure-ascii", action="store_true",
        help="开启后非 ASCII 字符会被转义（默认关闭，直接写入 UTF-8）"
    )
    parser.add_argument(
        "--encoding", default="utf-8",
        help="输入/输出文件编码（默认：utf-8）"
    )
    parser.add_argument(
        "--skip-invalid", action="store_true",
        help="遇到无法解析的行时跳过（默认：报错并退出）"
    )

    args = parser.parse_args()

    in_path = Path(args.jsonl)
    if not in_path.is_file():
        raise SystemExit(f"找不到输入文件：{in_path}")

    base = in_path.stem  # 不含扩展名
    outdir = Path(args.outdir) if args.outdir else in_path.with_name(base + "_split")
    outdir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix if args.prefix is not None else (base + "_")

    total_written = 0
    with in_path.open("r", encoding=args.encoding) as f:
        idx = args.start
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                # 空白行直接跳过
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                msg = f"[行 {line_no}] 解析失败：{e}"
                if args.skip_invalid:
                    print("跳过：", msg)
                    continue
                else:
                    raise SystemExit(msg)

            # 生成文件名
            name_from_field = None
            if args.field_as_name:
                v = obj
                for key in args.field_as_name.split("."):
                    if isinstance(v, dict) and key in v:
                        v = v[key]
                    else:
                        v = None
                        break
                if v is not None:
                    name_from_field = sanitize_filename(v)

            if name_from_field:
                filename = f"{prefix}{name_from_field}.json"
            else:
                if args.pad > 0:
                    filename = f"{prefix}{idx:0{args.pad}d}.json"
                else:
                    filename = f"{prefix}{idx}.json"

            out_path = outdir / filename
            with out_path.open("w", encoding=args.encoding) as wf:
                json.dump(obj, wf, ensure_ascii=args.ensure_ascii, indent=args.indent)
                wf.write("\n")  # 习惯性在文件末尾加换行

            total_written += 1
            idx += 1

    print(f"完成：写出 {total_written} 个 JSON 文件至目录：{outdir}")

if __name__ == "__main__":
    main()
