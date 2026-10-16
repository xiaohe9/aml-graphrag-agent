"""评测 FR-4 社区发现——**用代理指标，并明确说清这是代理**。

    uv run python scripts/eval_communities.py

## ⚠️ 这个脚本**不能**回答"团伙识别准不准"

因为这份数据**没有团伙真值**（`ALERT_TYPE` 只有 cycle / fan_in 两种交易级标注）。

它能回答的是四个**弱得多但真实**的问题：

1. **跑得动吗** —— 去重图 6.9 万条边上的耗时；
2. **社区非平凡吗** —— 如果 99% 的节点挤在一个社区里，那这个划分没有信息；
3. **稳定吗** —— 换随机种子重跑，两次划分的一致率有多高；
   **不稳定的划分不能被引用**，因为换个种子结论就变了；
4. **与已有标签有关系吗（代理）** —— 标注过的账户是否更集中在某些社区。
   **这只是代理，不能当成团伙识别达标。**

**这四条都不会给出"召回率"或"精确率"** —— 因为没有真值就没有这些。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from aml_agent.data import loader  # noqa: E402
from aml_agent.detect.communities import (  # noqa: E402
    find_communities,
    label_purity,
    rand_agreement,
)
from aml_agent.graph import INSTITUTION, load  # noqa: E402


def _as_float(value: object) -> float:
    """把 `summary()` 里的值收窄成 float。

    `summary()` 返回 `dict[str, object]`（因为它混了 int / float / str），
    所以取出来是 `object`。**显式检查而不是 `float(...)  # type: ignore`**——
    非数值会立刻报错，不会悄悄变成 0 参与判断。
    """
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"期望数值，收到 {type(value).__name__}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评测社区发现（代理指标）。")
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--out", default=None, help="把结果写到 JSON")
    args = parser.parse_args(argv)

    print("加载边表…", flush=True)
    edges = load.load_edges()
    print(f"  {len(edges):,} 条边")

    print("\n=== 1. 跑得动吗 ===")
    scan = find_communities(edges, resolution=args.resolution)
    summary = scan.summary()
    print(
        f"  节点 {summary['nodes']:,}　社区 {summary['communities']:,}"
        f"　模块度 {summary['modularity']}　耗时 {summary['seconds']} 秒"
    )
    print(
        f"  最大社区 {summary['largest']:,}（占 {_as_float(summary['largestShare']) * 100:.1f}%）"
    )

    print("\n=== 2. 社区非平凡吗 ===")
    sizes = scan.sizes()
    print(f"  规模前 8：{sizes[:8]}")
    share = _as_float(summary["largestShare"])
    if share > 0.5:
        print(f"  ⚠️ 最大社区占了 {share * 100:.0f}% —— **划分几乎没有信息量**")
    elif len(sizes) < 2:
        print("  ⚠️ 只有一个社区 —— 划分没有信息量")
    else:
        print("  ✅ 社区规模呈分布，不是一个巨型社区")

    print("\n=== 3. 稳定吗（换种子重跑）===")
    other = find_communities(edges, resolution=args.resolution, seed=7)
    agreement = rand_agreement(scan.assignments, other.assignments)
    print(f"  两次划分的成对一致率：{agreement:.3f}")
    if agreement >= 0.95:
        print("  ✅ 稳定 —— 结论可以被引用")
    elif agreement >= 0.8:
        print("  ⚠️ 基本稳定，但换种子会有可见变化")
    else:
        print("  ❌ **不稳定 —— 这个划分不能用来下结论**")

    print("\n=== 4. 与已有标签的关系（⚠️ 代理指标，不是准确率）===")
    dataset = loader.load()
    alerts = dataset.alerts
    # 标签从 CSV 读，不从图里读（ADR-0002）
    labelled = {str(account) for account in alerts["SENDER_ACCOUNT_ID"].tolist()} | {
        str(account) for account in alerts["RECEIVER_ACCOUNT_ID"].tolist()
    }
    labels = {
        node: node.split(":", 1)[-1] in labelled and node.startswith(f"{INSTITUTION}:")
        for node in scan.assignments
    }
    purity = label_purity(scan.assignments, labels)
    for key, value in purity.items():
        print(f"  {key}: {value}")
    if "spread" in purity:
        spread = _as_float(purity["spread"])
        if spread > 0.05:
            print(f"  → 社区间标注率相差 {spread * 100:.1f} 个百分点，**有弱关联**")
        else:
            print(f"  → 社区间标注率只差 {spread * 100:.1f} 个百分点，**看不出关联**")
        print("  ⚠️ 再说一次：这是代理指标。标签是交易级的 cycle/fan_in，不是团伙标注。")

    print("\n=== 结论 ===")
    print("  这条需求【能实现】，但在这份数据上【无法验证准确率】——")
    print("  因为 ALERT_TYPE 只有 cycle / fan_in 两种交易级标注，没有团伙标注。")
    print("  上面四条都是代理指标。**代理就是代理，不能冒充达标。**")

    if args.out:
        target = Path(args.out)
        target.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "topSizes": sizes[:20],
                    "seedAgreement": round(agreement, 4),
                    "labelProxy": purity,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n已写入 {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
