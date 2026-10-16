"""检测层评测：我们的结构信号 vs 规则基线。

    uv run python scripts/run_detection_eval.py

**这个脚本产出的就是 M3 的第一个真实数字。**

## 评测口径（为什么这么定）

系统在真实场景里的约束是「**分析师一天只能看 K 条**」。
所以核心口径是 **「报警数固定为 K」时的精确率与召回率**——
而不是先随便定一个分数阈值。

这样每个方法都在**同一个工作量**下比较，比"各自选最优阈值"公平得多。

## 真值

`alerts.csv` 的 1,719 条交易（`IS_FRAUD=True`），并带原生模式类型（`cycle` / `fan_in`）。
所以除了总体指标，还能**分模式看**——这能暴露"某一类根本没被检出"这种问题。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd  # noqa: E402

from aml_agent.data import loader, schema  # noqa: E402
from aml_agent.detect import baseline, cycles, structural  # noqa: E402
from aml_agent.eval import metrics  # noqa: E402
from aml_agent.graph import load as graph_load  # noqa: E402

#: 分析师看多少条。三个档位看趋势。
ALERT_BUDGETS = (100, 500, 1000)

#: 固定召回目标
TARGET_RECALL = 0.9


def _operating_point(
    ranking: metrics.Ranking, truth: frozenset[str], budget: int
) -> metrics.ConfusionMatrix:
    """把「报警数固定为 budget」这个运行点转成混淆矩阵。

    ⚠️ **用 `confusion_at_k`（按位置切），不能用阈值切。**
    实测踩过：全图 242,981 条边分数完全并列，按阈值切会把 24 万条全报出来，
    算出 `R@100 = 14.7%`（= 253 条）——**前 100 名里放不下 253 条**。
    详见 `aml_agent.eval.metrics.confusion_at_k` 的文档。
    """
    return metrics.confusion_at_k(ranking, truth, k=budget)


def _evaluate(
    name: str,
    scores: pd.Series,
    tx_ids: pd.Series,
    truth: frozenset[str],
    *,
    truth_by_type: dict[str, frozenset[str]],
) -> dict[str, object]:
    ranking = metrics.Ranking(scores=dict(zip(tx_ids, scores.astype(float), strict=True)))

    row: dict[str, object] = {"方法": name}
    for budget in ALERT_BUDGETS:
        matrix = _operating_point(ranking, truth, budget)
        row[f"P@{budget}"] = matrix.precision
        row[f"R@{budget}"] = matrix.recall

    point = metrics.precision_at_recall(ranking, truth, target_recall=TARGET_RECALL)
    row["召回90%时精确率"] = point.precision if point.reached else float("nan")
    row["召回90%需报警数"] = point.alerts if point.reached else float("inf")

    # 分模式召回：能看出"哪一类根本没被检出"
    for pattern, pattern_truth in truth_by_type.items():
        matrix = _operating_point(ranking, pattern_truth, budget=ALERT_BUDGETS[-1])
        row[f"召回·{pattern}"] = matrix.recall
    return row


def _as_float(value: object) -> float | None:
    """尽力转成 float。转不了就返回 None。

    这里必须容错，而不是直接 `float(value)`：
    pandas 3.0 会把**混合了 int 和 float('inf') 的列**推断成字符串列，
    于是 `f"{value:.0%}"` 会抛 `Unknown format code`。
    真正踩到过这个坑——格式化代码不该对上游的 dtype 推断做假设。
    """
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _format_percent(value: object) -> str:
    number = _as_float(value)
    if number is None or number != number:  # NaN
        return "   n/a"
    return f"{number:>6.1%}"


def _format_count(value: object) -> str:
    number = _as_float(value)
    if number is None:
        return str(value)
    if number == float("inf"):
        return "  不可达"
    return f"{number:>8,.0f}"


def _print_table(rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows).set_index("方法")

    # 逐格格式化成字符串，避免依赖 pandas 的 dtype 推断
    display = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        if column == "召回90%需报警数":
            display[column] = [_format_count(v) for v in frame[column]]
        else:
            display[column] = [_format_percent(v) for v in frame[column]]
    print(display.to_string())


def main() -> int:
    started = time.monotonic()

    print("加载图与边表…")
    graph, edges = graph_load.load_graph()
    dataset = loader.load_cached()
    alerts = dataset.alerts
    print(f"  边 {len(edges):,}   节点 {graph.number_of_nodes():,}")

    truth = frozenset(str(t) for t in alerts[schema.TX_ID])
    truth_by_type = {
        str(pattern): frozenset(str(t) for t in group[schema.TX_ID])
        for pattern, group in alerts.groupby(schema.ALERT_TYPE)
    }
    print(
        f"  真值 {len(truth):,} 条  "
        + "  ".join(f"{k}={len(v):,}" for k, v in truth_by_type.items())
    )

    print("\n计算结构特征（O(E)，全图）…")
    features = structural.compute_edge_features(graph, edges)
    print(f"  完成，用时 {time.monotonic() - started:.0f} 秒")

    tx_ids = edges["txId"].astype(str)
    #
    # ⚠️ 这一行是**必须**的：真值的 key 来自 alerts 的 TX_ID，拼成了字符串；
    # 而边表的 txId 是 int64。**两边 key 类型不一致 = 永远不相交 = 全是 0。**
    #
    # 这个 bug 的表现是「所有方法（包括随机）都恰好 0.0%」——
    # **随机方法也全 0，本身就是接线错误的诊断信号**，因为随机不该恰好撞不上。
    # 评测代码一定要有这种「一看就不可能」的自检。

    print("\n=== 特征单独看：互转到底覆盖了多少标注 ===")
    print(
        f"  全图互转边 {int(features['isReciprocal'].sum()):,} 条 "
        f"（占全部 {features['isReciprocal'].mean():.2%}）"
    )
    for pattern, pattern_truth in truth_by_type.items():
        subset = features[features["txId"].astype(str).isin(pattern_truth)]
        print(
            f"  {pattern:<8} 共 {len(subset):>5,} 条，其中互转（长度 2 的环）"
            f"{int(subset['isReciprocal'].sum()):>5,} 条 "
            f"({subset['isReciprocal'].mean():.1%})"
        )

    print("\n=== ★ 分模式特征画像：两类模式的可识别特征完全不同 ===")
    print("   （这决定了下一步该造什么特征——不是猜，是看）")
    marked = features["txId"].astype(str)
    profiles: list[tuple[str, pd.DataFrame]] = [("全部交易", features)]
    for pattern in sorted(truth_by_type):
        profiles.append((f"标注·{pattern}", features[marked.isin(truth_by_type[pattern])]))
    print(
        f"  {'分组':<14}{'条数':>10}{'金额中位':>11}{'收款方入度中位':>15}"
        f"{'付款方出度中位':>15}{'同对笔数中位':>13}{'互转率':>9}"
    )
    for label, frame in profiles:
        print(
            f"  {label:<14}{len(frame):>10,}{frame['amount'].median():>11.2f}"
            f"{frame['receiverInDegree'].median():>15.0f}"
            f"{frame['senderOutDegree'].median():>15.0f}"
            f"{frame['pairTxCount'].median():>13.0f}"
            f"{frame['isReciprocal'].mean():>9.1%}"
        )

    print("\n=== 第二层：真正的环检测 ===")
    cycle_started = time.monotonic()
    scan = cycles.find_short_cycles(graph)
    print(
        f"  扫描 {scan.scanned_edges:,} 条去重边，用时 {scan.seconds:.1f} 秒"
        f"（总 {time.monotonic() - cycle_started:.1f} 秒）"
    )
    histogram = scan.length_histogram()
    print("  最短环长分布：" + "  ".join(f"环长{k}={v:,}" for k, v in histogram.items()))
    real_cycle = cycles.cycle_feature(edges, scan, min_cycle_length=3)
    print(
        f"  其中环长 ≥3 的交易 {int(real_cycle.sum()):,} 条"
        f"（占 {real_cycle.mean():.2%}）—— 环长 2 被排除，因为它没有区分度"
    )

    print("\n=== 特征单独评测：两个方向都测，让数据说话 ===")
    feature_rows: list[dict[str, object]] = []
    for column in structural.FEATURE_COLUMNS:
        values = features[column].astype(float)
        for label, sign in (("小/少 → 可疑", -1.0), ("大/多 → 可疑", 1.0)):
            feature_rows.append(
                _evaluate(
                    f"  {column}（{label}）",
                    sign * values,
                    tx_ids,
                    truth,
                    truth_by_type=truth_by_type,
                )
            )
    feature_rows.append(
        _evaluate(
            "  ★真实环（环长≥3）",
            real_cycle,
            tx_ids,
            truth,
            truth_by_type=truth_by_type,
        )
    )
    _print_table(feature_rows)

    print("\n=== 最终对比：我们的组合信号 vs 规则基线 ===")
    cycle_score, fan_in_score = structural.two_detector_scores(features)
    rows: list[dict[str, object]] = []

    def add(label: str, scores: pd.Series) -> None:
        rows.append(_evaluate(label, scores, tx_ids, truth, truth_by_type=truth_by_type))

    add("★ 双检测器·求和（偏精确）", structural.fuse_sum(cycle_score, fan_in_score))
    add("★ 双检测器·取大（偏召回）", structural.fuse_max(cycle_score, fan_in_score))
    add("  单检测器·仅cycle特征", cycle_score)
    add("  单检测器·仅fan_in特征", fan_in_score)
    add("  加权组合（旧）", structural.combined_score(features))

    # 环长分布 —— "有没有环"太粗（8% 的交易都在环上），要精确到"环长是多少"
    cycle_length = cycles.shortest_cycle_feature(edges, scan)
    is_length_three = (cycle_length == 3).astype(float)
    is_length_three_or_four = cycle_length.isin([3, 4]).astype(float)
    amount_rank = 1.0 - features["amount"].rank(pct=True)
    pair_rank = 1.0 - features["pairTxCount"].rank(pct=True)
    print(
        f"  环长恰好为 3 的交易 {int(is_length_three.sum()):,} 条"
        f"（占 {is_length_three.mean():.2%}，对照全图 8.13% 的『有环』）"
    )

    add("★★ 仅环长=3", is_length_three)
    add("★★ 环长=3 + 小额", is_length_three * 2.0 + amount_rank)
    add("★★ 环长=3 + 小额 + 少笔数", is_length_three * 2.0 + amount_rank + pair_rank)
    add("★  环长3or4 + 小额", is_length_three_or_four + amount_rank)
    add("  仅真实环（环长≥3）", real_cycle)

    for name, scorer in baseline.BASELINES.items():
        add(name, scorer(edges))
    _print_table(rows)

    print(f"\n总耗时 {time.monotonic() - started:.0f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
