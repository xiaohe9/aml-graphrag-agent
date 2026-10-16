"""建图并校验。

    docker compose up -d neo4j          # 先起图库
    uv run python scripts/build_graph.py

**为什么建图和校验放在同一个命令里：**

建图错了不会报错——它只会安静地产出一张不对的图，然后所有下游指标都错。
**把校验绑在建图上，是让"错了"立刻可见的最省事办法。**

重建（会清空数据）：
    uv run python scripts/build_graph.py --reset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许直接 `python scripts/build_graph.py` 而不必先安装成包
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from aml_agent.data import loader  # noqa: E402
from aml_agent.graph import (  # noqa: E402
    GraphNotEmptyError,
    build_graph,
    connect,
    verify_and_report,
)
from aml_agent.graph.build import DEFAULT_BATCH_SIZE, DEFAULT_URI  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把 AMLSim 数据建成 Neo4j 图并校验。")
    parser.add_argument("--reset", action="store_true", help="先清空图再重建")
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args(argv)

    dataset = loader.load_cached()
    print(
        f"源数据：账户 {len(dataset.accounts):,}  交易 {len(dataset.transactions):,}  "
        f"告警 {len(dataset.alerts):,}"
    )

    driver = connect(args.uri)
    try:
        driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001 - CLI：给出可执行的下一步比抛栈有用
        print(f"连不上图库（{args.uri}）：{type(exc).__name__}: {exc}")
        print("先起图库：  docker compose up -d neo4j")
        return 2

    try:
        stats = build_graph(driver, dataset, batch_size=args.batch_size, reset=args.reset)
        print(
            f"\n建图完成（{stats.seconds:.1f} 秒）："
            f"账户 {stats.accounts:,}  实体 {stats.entities:,}  "
            f"交易边 {stats.transfers:,}  控制关系 {stats.controls:,}"
        )

        print("\n校验：")
        return 0 if verify_and_report(driver, dataset) else 1
    except GraphNotEmptyError as exc:
        print(f"\n{exc}")
        return 3
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
