"""跑全部门禁。**CI 和本地用的是同一个脚本。**

    uv run python scripts/check.py              # 静态检查 + 单元测试（快，约 30 秒）
    uv run python scripts/check.py --integration # 再加上需要数据和图库的测试

## 为什么要有这个脚本

CI 里写一遍命令、本地靠口头传一遍——**两者迟早会不一致**。
而「CI 绿但本地红」或反过来，会让人**不再信任门禁**，
然后开始绕过它。**同一个脚本，是让两者不可能漂移的唯一办法。**

## 门禁清单

| 门禁 | 拦什么 |
|---|---|
| `ruff check` | 语法、导入顺序、可疑写法 |
| `ruff format --check` | 格式 |
| `mypy --strict` | 类型错误（**strict**，不是默认档） |
| `pytest` | 行为 |
| **标签泄漏审计** | 在 `pytest` 里（`test_label_guard.py`）——**它必须是门禁的一部分，不是可选项** |

最后一条是刻意的：**防未来改动的检查，不进 CI 就等于没做。**
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def _run(label: str, command: list[str]) -> bool:
    """跑一条门禁。返回是否通过。

    `# noqa: S603` 是刻意的：这些命令是**脚本里写死的常量列表**，
    不含任何外部输入。而且 `shell=False`（`subprocess.run` 的默认），
    所以不存在 shell 注入面。
    """
    print(f"\n{'=' * 72}\n  {label}\n  $ {' '.join(command)}\n{'=' * 72}")
    started = time.monotonic()
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)  # noqa: S603
    elapsed = time.monotonic() - started
    mark = "通过" if result.returncode == 0 else "**未通过**"
    print(f"  -> {mark}（{elapsed:.1f} 秒）")
    return result.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="跑全部门禁。")
    parser.add_argument(
        "--integration",
        action="store_true",
        help="同时跑需要数据集与图库的测试（默认跳过）",
    )
    args = parser.parse_args(argv)

    # pytest 按标记选择跑哪一档。
    # `-m` 后面跟的是**标记表达式**，不是"-m 还是别的选项"——
    # 第一版写成了 `"-m" if args.integration else "-m"`，两个分支一模一样，
    # 被 ruff 的 RUF034 抓了出来。**那是没想清楚留下的残渣。**
    marker = "integration" if args.integration else "not integration"

    gates: list[tuple[str, list[str]]] = [
        ("ruff check", [sys.executable, "-m", "ruff", "check", "."]),
        ("ruff format --check", [sys.executable, "-m", "ruff", "format", "--check", "."]),
        ("mypy --strict", [sys.executable, "-m", "mypy"]),
        (
            f"pytest（{marker}）",
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-m", marker],
        ),
    ]

    failures: list[str] = []
    overall = time.monotonic()
    for label, command in gates:
        if not _run(label, command):
            failures.append(label)

    print(f"\n{'=' * 72}")
    if failures:
        print(f"  {len(failures)} 项门禁未通过：" + "、".join(failures))
        print(f"  总耗时 {time.monotonic() - overall:.1f} 秒")
        return 1
    print(f"  全部门禁通过（总耗时 {time.monotonic() - overall:.1f} 秒）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
