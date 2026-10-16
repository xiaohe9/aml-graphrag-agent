"""评测指标的测试。

**这些测试的意义不只是「代码没写错」，更是把几个容易被混淆的概念钉死：**

1. `1 − 精确率` 和 `FPR` 是两个不同的数（在类不平衡数据上能差一个数量级）；
2. 「未产出结论」不能折算成「无异常」，否则指标会失真；
3. 一个什么都不报的系统，FPR 是 0，看起来很完美——但它毫无用处。

前两条是产线项目和本项目的实测教训，第三条是「为什么必须同时报两个指标」的反证。
"""

from __future__ import annotations

import pytest

from aml_agent.eval.metrics import (
    Ranking,
    confusion_at,
    confusion_at_k,
    precision_at_k,
    precision_at_recall,
    summarize,
)


def _doc_example() -> tuple[Ranking, frozenset[str]]:
    """构造文档里的那个例子。

    1000 笔交易，10 笔真的洗钱，系统报了 100 条警，其中 8 条是真的。

    期望：TP=8  FP=92  FN=2  TN=898
         召回 80%   精确率 8%   1−精确率 92%   FPR 9.29%
    """
    scores: dict[str, float] = {}
    positives = [f"p{i}" for i in range(10)]  # p0..p7 被抓到，p8/p9 漏掉
    negatives = [f"n{i}" for i in range(990)]

    for item in positives[:8]:
        scores[item] = 0.9
    for item in negatives[:92]:
        scores[item] = 0.9
    for item in positives[8:]:
        scores[item] = 0.1
    for item in negatives[92:]:
        scores[item] = 0.1

    return Ranking(scores=scores), frozenset(positives)


# --------------------------------------------------------------------------- #
#  1. 混淆矩阵与三个不同的「率」
# --------------------------------------------------------------------------- #


def test_confusion_matches_the_documented_example() -> None:
    ranking, truth = _doc_example()
    matrix = confusion_at(ranking, truth, threshold=0.5)

    assert (matrix.tp, matrix.fp, matrix.fn, matrix.tn) == (8, 92, 2, 898)
    assert matrix.recall == pytest.approx(0.80)
    assert matrix.precision == pytest.approx(0.08)


def test_one_minus_precision_is_not_the_false_positive_rate() -> None:
    """这条是本项目最容易被问倒的地方。

    `1 − 精确率` = FP/(TP+FP) = 92/100 = 92%
    `FPR`       = FP/(FP+TN) = 92/990 = 9.29%

    **分子一样，分母不同。** 在正样本极少的场景下两者差一个数量级。
    """
    ranking, truth = _doc_example()
    matrix = confusion_at(ranking, truth, threshold=0.5)

    assert matrix.alert_false_rate == pytest.approx(0.92)
    assert matrix.false_positive_rate == pytest.approx(92 / 990)
    # 差距要大到能被看出来，否则这个测试就没有说服力
    assert matrix.alert_false_rate > matrix.false_positive_rate * 9


def test_threshold_is_inclusive() -> None:
    """恰好等于阈值算报警。用 `>` 会让边界样本被算成漏报，与直觉不符。"""
    ranking = Ranking(scores={"a": 0.5, "b": 0.4999})
    matrix = confusion_at(ranking, {"a"}, threshold=0.5)
    assert (matrix.tp, matrix.fp) == (1, 0)


def test_confusion_at_k_slices_by_position_not_by_score() -> None:
    """**这个测试锁住一个真实踩到的 bug。**

    实测场景：全图 242,981 条边的互转分数**完全相同**（都是 1.0）。
    用「分数 >= 第 k 名的分数」切，会把 24 万条**全部**报出来，
    算出 `R@100 = 14.7%`（= 253 条）——**而前 100 名里根本放不下 253 条**。

    这个数字自相矛盾，正是发现 bug 的线索。
    """
    # 200 条并列，其中 3 条是真的
    scores = {f"a{i:03d}": 1.0 for i in range(200)}
    truth = {"a000", "a001", "a002"}
    ranking = Ranking(scores=scores)

    # 按位置切：正好 100 条
    by_position = confusion_at_k(ranking, truth, k=100)
    assert by_position.tp + by_position.fp == 100
    assert by_position.tp == 3
    assert by_position.recall == pytest.approx(1.0)  # 3 个正例都在前 100

    # 按分数切：**并列会让它报出全部 200 条**，工作量完全不是 100
    by_threshold = confusion_at(ranking, truth, threshold=1.0)
    assert by_threshold.tp + by_threshold.fp == 200
    assert by_threshold.recall == pytest.approx(1.0)  # 召回一样，但代价完全不同

    # 两者召回相同、代价差一倍——这就是必须按位置切的原因
    assert by_position.precision == pytest.approx(0.03)
    assert by_threshold.precision == pytest.approx(0.015)


def test_confusion_at_k_accounts_for_silence() -> None:
    """沉默的正例仍然要单独计数，不能因为「按位置切」就漏掉。"""
    ranking = Ranking(
        scores={f"a{i:03d}": 1.0 for i in range(20)},
        no_answer=frozenset({"z1", "z2"}),
    )
    matrix = confusion_at_k(ranking, {"a000", "z1", "z2"}, k=10)
    assert matrix.positive_silent == 2
    assert matrix.tp == 1
    # 召回分母含沉默：1 / (1 + 0 + 2)
    assert matrix.recall == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
#  2. 「未产出结论」必须单独计数  ← 产线项目的教训
# --------------------------------------------------------------------------- #


def test_silence_is_counted_separately_and_hurts_recall() -> None:
    """沉默的正例**不能被当作**「正确地没报警」。

    构造：10 个正例，其中 4 个系统从未给出结论（沉默）。
    如果沉默被算进 TN，召回率会虚高；正确的做法是让它压低召回。
    """
    scores = {f"p{i}": 0.9 for i in range(6)}  # 6 个给了分数且都报警
    silent = frozenset({f"p{i}" for i in range(6, 10)})  # 4 个沉默
    truth = frozenset(f"p{i}" for i in range(10))
    ranking = Ranking(scores=scores, no_answer=silent)

    matrix = confusion_at(ranking, truth, threshold=0.5)

    assert matrix.tp == 6
    assert matrix.positive_silent == 4
    assert matrix.no_answer == 4
    # 分母 = 6(抓到) + 0(漏报) + 4(沉默) = 10，所以召回只有 60%，不是 100%
    assert matrix.recall == pytest.approx(0.60)


def test_silence_does_not_inflate_the_false_positive_rate() -> None:
    """反过来看：沉默会**虚假地压低**误报率。

    这正是「沉默是工程缺陷」的可怕之处——它同时让检出率难看、
    又让误报率好看，两个方向的失真叠在一起，**指标完全不可信**。
    """
    # 10 个负例，系统全部沉默 → 一条警都不报
    ranking = Ranking(scores={}, no_answer=frozenset(f"n{i}" for i in range(10)))
    matrix = confusion_at(ranking, frozenset(), threshold=0.5)

    assert matrix.fp == 0
    assert matrix.false_positive_rate == 0.0
    assert matrix.negative_silent == 10
    # 所以必须靠 no_answer 这个单独的数字把问题暴露出来
    assert matrix.no_answer == 10


def test_an_object_cannot_be_both_scored_and_silent() -> None:
    """口径必须唯一，否则统计结果无法解释。这是**失败要响亮**的例子。"""
    ranking = Ranking(scores={"x": 0.5}, no_answer=frozenset({"x"}))
    with pytest.raises(ValueError, match="既给分数又标记沉默"):
        confusion_at(ranking, frozenset(), threshold=0.5)


# --------------------------------------------------------------------------- #
#  3. 为什么必须同时报两个指标  ← 反证
# --------------------------------------------------------------------------- #


def test_a_system_that_never_alerts_is_useless_but_has_perfect_fpr() -> None:
    """**反证：只报一个指标一定会被骗。**

    什么都不报的系统：FPR = 0（完美！），召回 = 0（毫无用处）。
    所以任何单一指标都能被「钻空子策略」刷到满分。
    """
    ranking = Ranking(scores={f"p{i}": 0.05 for i in range(10)})
    truth = frozenset(f"p{i}" for i in range(10))

    matrix = confusion_at(ranking, truth, threshold=0.5)

    assert matrix.false_positive_rate == 0.0  # 完美
    assert matrix.recall == 0.0  # 但一无是处


def test_a_system_that_alerts_everything_has_perfect_recall() -> None:
    """另一个方向的反证：全部标红，召回率 100%。"""
    ranking = Ranking(scores={f"p{i}": 0.9 for i in range(10)} | {f"n{i}": 0.9 for i in range(990)})
    truth = frozenset(f"p{i}" for i in range(10))

    matrix = confusion_at(ranking, truth, threshold=0.5)

    assert matrix.recall == 1.0  # 完美
    assert matrix.precision == pytest.approx(10 / 1000)  # 但精确率 1%


# --------------------------------------------------------------------------- #
#  4. 两个切点：Precision@K 与「固定召回下的精确率」
# --------------------------------------------------------------------------- #


def test_precision_at_k() -> None:
    # 前 3 名里 2 个是真的
    ranking = Ranking(scores={"a": 1.0, "b": 0.9, "c": 0.8, "d": 0.7})
    assert precision_at_k(ranking, {"a", "c"}, k=3) == pytest.approx(2 / 3)


def test_precision_at_k_uses_actual_length_when_candidates_are_fewer() -> None:
    """候选不足 K 条时，分母取实际条数。

    否则「只找到 1 条且是对的」会被算成 1/K，而 K 越大分数越低——不合理。
    """
    ranking = Ranking(scores={"a": 1.0})
    assert precision_at_k(ranking, {"a"}, k=50) == 1.0


def test_precision_at_recall_finds_the_cheapest_point() -> None:
    """按可疑度降序走到召回达标的那一点，取那时的精确率。"""
    # 排序：p0 p1 p2 n0 n1 p3 ...  → 目标召回 0.75（4 个正例里 3 个）
    scores = {"p0": 1.0, "p1": 0.9, "p2": 0.8, "n0": 0.7, "n1": 0.6, "p3": 0.5}
    ranking = Ranking(scores=scores)
    truth = frozenset({"p0", "p1", "p2", "p3"})

    point = precision_at_recall(ranking, truth, target_recall=0.75)

    assert point.reached is True
    assert point.alerts == 3  # 前 3 条就够到 75%
    assert point.achieved_recall == pytest.approx(0.75)
    assert point.precision == pytest.approx(1.0)  # 前 3 条全是真的


def test_unreachable_recall_is_reported_not_silently_downgraded() -> None:
    """目标召回达不到时**必须能看出来**，不能悄悄变成「尽力而为」。"""
    # 10 个正例，其中 5 个系统从未给结论 → 最高召回只有 50%
    ranking = Ranking(
        scores={f"p{i}": 0.9 for i in range(5)},
        no_answer=frozenset({f"p{i}" for i in range(5, 10)}),
    )
    truth = frozenset(f"p{i}" for i in range(10))

    point = precision_at_recall(ranking, truth, target_recall=0.9)

    assert point.reached is False
    assert point.achieved_recall == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
#  5. 可复现性（NFR-5）
# --------------------------------------------------------------------------- #


def test_ties_break_deterministically() -> None:
    """同分时按 id 兜底排序。

    否则字典的迭代顺序会渗进结果，同一输入可能给出不同输出——
    而「可复现」是我们写进非功能需求里的硬要求。
    """
    first = Ranking(scores={"c": 1.0, "a": 1.0, "b": 1.0})
    second = Ranking(scores={"b": 1.0, "c": 1.0, "a": 1.0})

    assert first.ranked() == second.ranked() == ("a", "b", "c")


# --------------------------------------------------------------------------- #
#  6. 汇总
# --------------------------------------------------------------------------- #


def test_summarize_reports_both_metrics_and_the_silence_count() -> None:
    ranking, truth = _doc_example()
    report = summarize(ranking, truth, threshold=0.5, k=100, target_recall=0.9)

    assert report["precisionAtK"]["k"] == 100
    assert report["precisionAtK"]["precision"] == pytest.approx(0.08)
    # 召回目标 0.9 需要抓到 9/10，但排序里只有 8 个正例排在前面 → 达不到
    assert report["precisionAtRecall"]["reached"] is True
    assert report["precisionAtRecall"]["achievedRecall"] >= 0.9
    # 沉默计数单独暴露，方便 CI 断言「必须为 0」
    assert report["noAnswer"] == 0
    # 两个误报口径都在
    assert "falsePositiveRate" in report["confusion"]
    assert "alertFalseRate" in report["confusion"]


def test_zero_positives_does_not_divide_by_zero() -> None:
    """没有正例时不能崩——某些对照集就是全负例。"""
    ranking = Ranking(scores={"n0": 0.9, "n1": 0.1})
    assert confusion_at(ranking, frozenset(), threshold=0.5).recall == 0.0
    assert precision_at_recall(ranking, frozenset(), target_recall=0.9).reached is False


def test_invalid_arguments_are_rejected() -> None:
    ranking = Ranking(scores={"a": 1.0})
    with pytest.raises(ValueError, match="k 必须为正数"):
        precision_at_k(ranking, {"a"}, k=0)
    with pytest.raises(ValueError, match="target_recall"):
        precision_at_recall(ranking, {"a"}, target_recall=0.0)
