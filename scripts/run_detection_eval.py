"""检测层评测：我们的结构信号 vs 规则基线。

    uv run python scripts/run_detection_eval.py

**这个脚本产出的就是 M3 的第一个真实数字。**

## 评测口径（为什么这么定）

系统在真实场景里的约束是「**分析师一天只能看 K 条**」。
所以核心口径是 **「报警数固定为 K」时的精确率与召回率**——
而不是先随便定一个分数阈值。

这样每个方法都在**同一个工作量**下比较，比"各自选最优阈值"公平得多。

## ⚠️ 性能：这个脚本曾经要 28 分钟

实测的时间分布（14 个方法）：

```text
  建字典  101.53 秒     ← 把 130 万条分数装进 Python dict
  排序     49.77 秒
  算指标   63.38 秒
  合计    214.67 秒     ← 评测循环；加上加载图与算特征，整轮 372 秒
```

**根因不是算法，是数据结构**：指标层原本围绕 `Mapping[str, float]`
（130 万个字符串键的 Python 字典）设计，每个方法都要重建一次。

改成向量化快路径（`ranking_order` + `curve_from_order`）之后：
**一次排序 + 一次前缀和，覆盖全部指标。**
正确性由 `test_curve_matches_the_ranking_path` 逐项锁住
（含**全部并列**的极端情况）。

## 真值

`alerts.csv` 的 1,719 条交易（`IS_FRAUD=True`），并带原生模式类型（`cycle` / `fan_in`）。
所以除了总体指标，还能**分模式看**——这能暴露"某一类根本没被检出"这种问题。

⚠️ **标签从 CSV 读，不经过图**（见 `docs/04` 4.1.1：图是特征存储，不是答案存储）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aml_agent.data import loader, schema  # noqa: E402
from aml_agent.detect import baseline, cycles, structural  # noqa: E402
from aml_agent.eval import metrics  # noqa: E402
from aml_agent.graph import load as graph_load  # noqa: E402

#: 分析师看多少条。三个档位看趋势。
ALERT_BUDGETS = (100, 500, 1000)

#: 固定召回目标
TARGET_RECALL = 0.9


# --------------------------------------------------------------------------- #
#  评测
# --------------------------------------------------------------------------- #


def _evaluate(
    name: str,
    scores: pd.Series,
    tie_break: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    timings: list[tuple[str, float, float]] | None = None,
) -> dict[str, object]:
    """评测一个方法。

    **一次排序 + 每个标签集一次前缀和。** 整体和分模式共用同一个排序——
    它们只有标签不同，排序完全一样。
    """
    started = time.monotonic()

    order = metrics.ranking_order(scores.to_numpy(dtype=np.float64), tie_break=tie_break)
    sorted_at = time.monotonic()

    overall = metrics.curve_from_order(order, masks["总体"])
    row: dict[str, object] = {"方法": name}
    for budget in ALERT_BUDGETS:
        matrix = overall.confusion_at_k(budget)
        row[f"P@{budget}"] = matrix.precision
        row[f"R@{budget}"] = matrix.recall

    point = overall.precision_at_recall(TARGET_RECALL)
    row["召回90%时精确率"] = point.precision if point.reached else float("nan")
    row["召回90%需报警数"] = point.alerts if point.reached else float("inf")

    # 分模式召回：能看出"哪一类根本没被检出"
    for pattern, mask in masks.items():
        if pattern == "总体":
            continue
        row[f"召回·{pattern}"] = metrics.curve_from_order(order, mask).recall_at(ALERT_BUDGETS[-1])

    finished = time.monotonic()
    if timings is not None:
        timings.append((name, sorted_at - started, finished - sorted_at))
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


# --------------------------------------------------------------------------- #
#  主流程
# --------------------------------------------------------------------------- #


def main() -> int:
    started = time.monotonic()

    print("加载边表…")
    load_started = time.monotonic()
    # ⚠️ 用 `load_simple_graph`（去重图，6.9 万条边），**不是** `load_graph`（多重图，132 万条）。
    #
    # 实测：往 NetworkX 里逐条加 132 万条边要 30 多秒，而去重图只要约 0.3 秒。
    # 而环检测本来就在去重图上做（平行边不改变环的结构）。
    graph, edges = graph_load.load_simple_graph()
    dataset = loader.load_cached()
    alerts = dataset.alerts
    load_seconds = time.monotonic() - load_started
    print(
        f"  边 {len(edges):,}（去重后 {graph.number_of_edges():,}）"
        f"   节点 {graph.number_of_nodes():,}   用时 {load_seconds:.1f} 秒"
    )

    # ---- 真值：从 CSV 读，不经过图 ---------------------------------------- #
    #
    # ⚠️ 用**整数 txId** 作为并列时的排序键。
    # `Ranking` 那边用的是字典键（字符串），两者在同分时的顺序不同
    # （字符串 "999" > "12345"，而整数相反）。
    # 这里统一用整数，并保证**可复现**（同一输入永远同一顺序）。
    tie_break = edges["txId"].to_numpy(dtype=np.int64)

    fraud_ids = {int(t) for t in alerts[schema.TX_ID]}
    masks: dict[str, np.ndarray] = {"总体": edges["txId"].isin(fraud_ids).to_numpy()}
    for pattern, group in alerts.groupby(schema.ALERT_TYPE):
        pattern_ids = {int(t) for t in group[schema.TX_ID]}
        masks[str(pattern)] = edges["txId"].isin(pattern_ids).to_numpy()

    print(
        f"  真值 {int(masks['总体'].sum()):,} 条  "
        + "  ".join(f"{name}={int(mask.sum()):,}" for name, mask in masks.items() if name != "总体")
    )

    print("\n计算结构特征（O(E)，全图）…")
    features_started = time.monotonic()
    features = structural.compute_edge_features(edges)
    features_seconds = time.monotonic() - features_started
    print(f"  完成，用时 {features_seconds:.1f} 秒")

    # ---- 第二层：真正的环检测 ---------------------------------------------- #
    print("\n=== 第二层：真正的环检测 ===")
    scan = cycles.find_short_cycles(graph)
    histogram = scan.length_histogram()
    print(
        f"  扫描 {scan.scanned_edges:,} 条去重边，用时 {scan.seconds:.1f} 秒\n"
        f"  最短环长分布：" + "  ".join(f"环长{k}={v:,}" for k, v in histogram.items())
    )
    real_cycle = cycles.cycle_feature(edges, scan, min_cycle_length=3)
    print(
        f"  其中环长 ≥3 的交易 {int(real_cycle.sum()):,} 条"
        f"（占 {real_cycle.mean():.2%}）—— 环长 2 被排除，因为它没有区分度"
    )

    # ---- 分模式特征画像 ---------------------------------------------------- #
    print("\n=== ★ 分模式特征画像：两类模式的可识别特征完全不同 ===")
    print("   （这决定了下一步该造什么特征——不是猜，是看）")
    profiles: list[tuple[str, pd.DataFrame]] = [("全部交易", features)]
    for pattern, mask in masks.items():
        if pattern == "总体":
            continue
        profiles.append((f"标注·{pattern}", features[mask]))
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

    # ---- 逐特征评测（两个方向都测）---------------------------------------- #
    print("\n=== 特征单独评测：两个方向都测，让数据说话 ===")
    feature_loop_started = time.monotonic()
    feature_rows: list[dict[str, object]] = []
    for column in structural.FEATURE_COLUMNS:
        values = features[column].astype(float)
        for label, sign in (("小/少 → 可疑", -1.0), ("大/多 → 可疑", 1.0)):
            feature_rows.append(
                _evaluate(f"  {column}（{label}）", sign * values, tie_break, masks)
            )
    feature_rows.append(_evaluate("  ★真实环（环长≥3）", real_cycle, tie_break, masks))
    _print_table(feature_rows)
    feature_loop_seconds = time.monotonic() - feature_loop_started
    print(f"  （这 10 个特征方向评测共 {feature_loop_seconds:.1f} 秒）")

    # ---- 最终对比 ---------------------------------------------------------- #
    print("\n=== 最终对比：我们的组合信号 vs 规则基线 ===")
    scores_started = time.monotonic()
    cycle_score, fan_in_score = structural.two_detector_scores(features)
    cycle_length = cycles.shortest_cycle_feature(edges, scan)
    is_length_three = (cycle_length == 3).astype(float)
    is_length_three_or_four = cycle_length.isin([3, 4]).astype(float)
    amount_rank = 1.0 - features["amount"].rank(pct=True)
    pair_rank = 1.0 - features["pairTxCount"].rank(pct=True)
    scores_seconds = time.monotonic() - scores_started
    print(f"  构造各方法的分数，用时 {scores_seconds:.1f} 秒")

    rows: list[dict[str, object]] = []
    timings: list[tuple[str, float, float]] = []

    def add(label: str, scores: pd.Series) -> None:
        rows.append(_evaluate(label, scores, tie_break, masks, timings=timings))

    add("★ 双检测器·求和（偏精确）", structural.fuse_sum(cycle_score, fan_in_score))
    add("★ 双检测器·取大（偏召回）", structural.fuse_max(cycle_score, fan_in_score))
    add("  单检测器·仅cycle特征", cycle_score)
    add("  单检测器·仅fan_in特征", fan_in_score)
    add("  加权组合（旧）", structural.combined_score(features))
    add("★★ 仅环长=3", is_length_three)
    add("★★ 环长=3 + 小额", is_length_three * 2.0 + amount_rank)
    add("★★ 环长=3 + 小额 + 少笔数", is_length_three * 2.0 + amount_rank + pair_rank)
    add("★  环长3or4 + 小额", is_length_three_or_four + amount_rank)
    add("  仅真实环（环长≥3）", real_cycle)
    for name, scorer in baseline.BASELINES.items():
        add(name, scorer(edges))
    _print_table(rows)

    # ---- 分段计时：**先量，再优化** ---------------------------------------- #
    print("\n=== 各方法耗时分解（秒）===")
    print(f"  {'方法':<26}{'排序':>10}{'算指标':>10}{'合计':>10}")
    sorted_total = 0.0
    compute_total = 0.0
    for label, sort_seconds, compute_seconds in timings:
        sorted_total += sort_seconds
        compute_total += compute_seconds
        print(
            f"  {label:<26}{sort_seconds:>10.2f}{compute_seconds:>10.2f}"
            f"{sort_seconds + compute_seconds:>10.2f}"
        )
    loop_total = sorted_total + compute_total
    print(f"  {'合计':<26}{sorted_total:>10.2f}{compute_total:>10.2f}{loop_total:>10.2f}")

    elapsed = time.monotonic() - started
    print(
        f"\n=== 总耗时 {elapsed:.0f} 秒 ===\n"
        f"  加载边表与数据              {load_seconds:>6.1f} 秒\n"
        f"  算结构特征                 {features_seconds:>6.1f} 秒\n"
        f"  扫环                       {scan.seconds:>6.1f} 秒\n"
        f"  构造分数                   {scores_seconds:>6.1f} 秒\n"
        f"  特征方向评测（10 个）        {feature_loop_seconds:>6.1f} 秒\n"
        f"  最终对比评测（{len(timings)} 个）      {loop_total:>6.1f} 秒\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
