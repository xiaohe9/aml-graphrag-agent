"""下载并解压 AMLSim 数据集。

    uv run python scripts/fetch_data.py

**为什么要写这个脚本，而不是手动下载：**

1. **可复现**——别人 clone 仓库后一条命令就能拿到同一份数据；
2. **可核对**——脚本里写死了行数期望值，数据换了会立刻发现（而不是等模型效果变差）；
3. **不需要账号**——Kaggle 的下载接口会 302 重定向到一个**带签名的** Google Storage 链接，
   跟着跳转就能下，无需 API token。

数据是 Apache 2.0，但体积大（解压 61.5 MB），**不入库**——`data/` 已在 .gitignore 里。

**数据集的定位**（官方原话）：生成合成银行交易数据，以及一组已知的洗钱模式，
**主要用于测试机器学习模型和图算法**。这正是我们需要的。
"""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

# Windows 控制台默认是 GBK，而本脚本的输出里有中文和制表符号。
# 显式把标准输出切到 UTF-8 并允许替换，是为了**不让一句日志把整个脚本搞崩**——
# 实测在 GBK 控制台上打印 ✓ 会抛 UnicodeEncodeError，中断在下载完成的最后一步。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# 数据集来源。写成常量而不是散在代码里，是为了让"数据从哪来"只有一个答案。
KAGGLE_SLUG = "anshankul/ibm-amlsim-example-dataset"
ARCHIVE_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_SLUG}"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARCHIVE_PATH = DATA_DIR / "amlsim.zip"
EXTRACT_DIR = DATA_DIR / "extracted"

# 期望值。**数据换了必须在这里发现**，而不是等模型效果下降才怀疑。
#
# 这些数字来自 M1 的数据画像，记录在 docs/08-数据规格与真值设计.md。
EXPECTED = {
    "accounts.csv": 10_000,
    "transactions.csv": 1_323_234,
    "alerts.csv": 1_719,
}

# Kaggle 对没有 User-Agent 的请求可能返回 403，所以显式带上。
HEADERS = {"user-agent": "aml-agent/0.0.0 (research; dataset fetch)"}


def _download(url: str, destination: Path) -> None:
    """下载到本地。urllib 默认会跟随 302——Kaggle 正是靠这个把请求转到签名链接。"""
    request = urllib.request.Request(url, headers=HEADERS)  # noqa: S310 - URL 是写死的常量
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        total = int(response.headers.get("content-length") or 0)
        received = 0
        chunks: list[bytes] = []
        while True:
            chunk = response.read(1 << 20)  # 1 MiB 一块，兼顾内存与进度粒度
            if not chunk:
                break
            chunks.append(chunk)
            received += len(chunk)
            if total:
                percent = received * 100 // total
                print(f"\r    下载中 {percent:3d}%  ({received / 1e6:.1f} MB)", end="", flush=True)
        print()
    destination.write_bytes(b"".join(chunks))


def _extract(archive: Path, destination: Path) -> list[Path]:
    """解压并把文件名扁平化到目标目录。

    扁平化是刻意的：Kaggle 的 zip 里有时带一层目录，有时不带。**不希望下游代码去猜路径。**
    """
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if member.endswith("/"):
                continue
            target = destination / Path(member).name
            with bundle.open(member) as source:
                target.write_bytes(source.read())
            written.append(target)
    return written


def _verify(directory: Path) -> bool:
    """核对行数与期望值。

    行数用逐行计数而不是把整个文件读进来——transactions.csv 有 58 MB，
    而这个脚本可能在内存受限的环境里跑。
    """
    ok = True
    for name, expected_rows in EXPECTED.items():
        path = directory / name
        if not path.is_file():
            print(f"    [缺失] {name}")
            ok = False
            continue
        # 减 1 是表头
        with path.open("rb") as handle:
            actual = sum(1 for _ in handle) - 1
        if actual == expected_rows:
            print(f"    [OK]   {name:<20} {actual:>10,} 行")
        else:
            print(f"    [变了] {name:<20} {actual:>10,} 行（期望 {expected_rows:,}）——数据变了！")
            ok = False
    return ok


def main() -> int:
    if EXTRACT_DIR.is_dir() and _verify(EXTRACT_DIR):
        print("数据已就绪，无需重新下载。")
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not ARCHIVE_PATH.is_file():
        print(f"从 Kaggle 下载：{KAGGLE_SLUG}")
        _download(ARCHIVE_URL, ARCHIVE_PATH)
        print(
            f"    已保存到 {ARCHIVE_PATH.relative_to(PROJECT_ROOT)}"
            f"（{ARCHIVE_PATH.stat().st_size / 1e6:.1f} MB）"
        )

    print("解压中…")
    files = _extract(ARCHIVE_PATH, EXTRACT_DIR)
    for path in files:
        print(f"    {path.name:<24} {path.stat().st_size / 1e6:>7.1f} MB")

    print("核对行数：")
    if not _verify(EXTRACT_DIR):
        print(
            "\n数据与预期不符。请核对 docs/08-数据规格与真值设计.md，"
            "并同步更新本脚本的 EXPECTED 常量。"
        )
        return 1

    print(f"\n完成。数据集根目录：{EXTRACT_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
