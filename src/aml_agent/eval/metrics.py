"""评测指标。

**这个模块先于任何检测算法存在，是刻意的。**

如果评测最后写，会出现两个后果：每个模块做完时你不知道它有没有用，
只能凭感觉往下做；最后发现指标不行时，已经不知道该改哪一层。

---

## 三条设计原则（每条都有来源）

### 1. 系统输出是「可排序分数」，不是布尔值

这是 PRD 2.4「一个排序，两个切点」的直接后果：

```text
分数高 ┃████████████  ← 取前 K 个算精确率      → Precision@K
      ┃████████
      ┃██████        ← 放宽到召回 90% 算精确率  → 固定召回下的精确率
      ┃████
分数低 ┃██
```

**两个指标来自同一个排序，只是切点不同。** 所以底层数据结构是排序，不是「是/否」。

如果系统只能输出布尔值，这两个指标都算不出来——
**这就是「指标反向约束设计」的一个具体例子。**

### 2. 「未产出结论」必须**单独计数**

**这条来自产线项目的实测教训**：那一轮跑出 0% 检出率，
看起来像「Agent 能力不行」，实际是 15/50 个案例**耗尽步数、从未给出答案**，
却被统计成了「无异常」。

**「什么都没找到」和「根本没给答案」是两件完全不同的事：**

- 前者是**能力问题** → 可以量化、可以改进；
- 后者是**工程缺陷** → 同时压低检出率、又虚假地压低误报率
  （不报警自然不会有误报），**会让所有指标失真**。

所以本模块把「沉默」拆成正例沉默和负例沉默，**并且召回率的分母包含正例沉默**——
沉默**不算**「正确地没报警」。

### 3. `1 − 精确率` 和 `FPR` 是两个不同的数，都要算

```text
1 − 精确率 = FP / (TP + FP)     分母是「所有报出来的警」
FPR       = FP / (FP + TN)     分母是「所有正常样本」
```

**分子一样，分母不同。** 在洗钱这种**极度不平衡**的数据上，两者能差一个数量级
（实测例子见 `08-数据规格`）。

**AML 行业口语说的「误报率」通常指前者**，而统计学教材说的是后者。
两个都算出来，就不会在面试或评审时被问住。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Ranking:
    """系统的输出：一个「可疑度排序」，外加「哪些没给结论」。

    - `scores`：**给了结论**的对象 → 可疑度分数（越大越可疑）
    - `no_answer`：**未产出结论**的对象

    ⚠️ `no_answer` 里的对象**不能**被当作「分数很低」。它们不是被判定为正常，
    而是**系统没能给出判定**。混为一谈会让指标失真。

    ## 为什么排序结果要缓存（`_order`）

    实测代价：**130 万条边**的项目里，一次全排序约 2 秒。
    而一个评测要按多个 K 值（100 / 500 / 1000）和多个模式分别算混淆矩阵，
    于是一个方法要排 7 次——**16 个方法就是 112 次全排序，评测跑十几分钟**。

    **排序结果与调用方无关**（同一个 Ranking 永远是同一个顺序），
    所以缓存它没有正确性代价，只有收益。

    为了能缓存，这里**没有用 `slots=True`**——
    冻结 dataclass 加 slots 就没法在首次访问后写回缓存字段了。
    """

    scores: Mapping[str, float]
    no_answer: frozenset[str] = frozenset()
    #: 缓存。`compare=False` 让它不参与相等性判断——它是派生结果，不是状态。
    _order: tuple[str, ...] | None = field(default=None, repr=False, compare=False)

    def ranked(self) -> tuple[str, ...]:
        """按可疑度降序，**同分按 id 升序**。

        同分时用 id 兜底排序是刻意的：否则字典/集合的迭代顺序会渗进结果，
        **违反「同一输入必须同一输出」的可复现要求**（NFR-5）。
        """
        # 用局部变量而不是直接返回 self._order：
        # mypy 看不到 object.__setattr__ 的效果，直接 return 会因为
        # 字段声明成 `tuple | None` 而报类型不匹配。
        order = self._order
        if order is None:
            order = tuple(sorted(self.scores, key=lambda item: (-self.scores[item], item)))
            # 冻结 dataclass 只能用 object.__setattr__ 写回
            object.__setattr__(self, "_order", order)
        return order


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """混淆矩阵。

    「沉默」被拆成两项，而不是揉进 `tn` 或 `fn`——
    这是本模块最重要的一个设计决定，理由见模块文档第 2 条。
    """

    tp: int
    fp: int
    tn: int
    #: 给了结论，但没报警，而它真的有问题 → 能力问题（漏报）
    fn: int
    #: 未产出结论，而它真的有问题 → **工程缺陷**
    positive_silent: int
    #: 未产出结论，而它本来没问题
    negative_silent: int

    @property
    def no_answer(self) -> int:
        return self.positive_silent + self.negative_silent

    @property
    def recall(self) -> float:
        """召回率（= 检出率）。

        ⚠️ **分母包含「沉默的正例」。** 沉默不算「正确地没报警」——
        否则「什么都不答」就能拿到一个不难看的召回率。
        """
        denominator = self.tp + self.fn + self.positive_silent
        return self.tp / denominator if denominator else 0.0

    @property
    def precision(self) -> float:
        """精确率。分母是「所有报出来的警」。"""
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else 0.0

    @property
    def false_positive_rate(self) -> float:
        """FPR = FP / (FP + TN)。分母是「所有正常样本」。"""
        denominator = self.fp + self.tn
        return self.fp / denominator if denominator else 0.0

    @property
    def alert_false_rate(self) -> float:
        """`1 − 精确率`：报出来的警里有多少是假的。

        **AML 行业口语说的「误报率」通常指这个**，不是 FPR。两个都要报。
        """
        return 1.0 - self.precision if (self.tp + self.fp) else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "positiveSilent": self.positive_silent,
            "negativeSilent": self.negative_silent,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "falsePositiveRate": round(self.false_positive_rate, 4),
            "alertFalseRate": round(self.alert_false_rate, 4),
        }


@dataclass(frozen=True, slots=True)
class PrecisionAtRecall:
    """「固定召回下的精确率」的结果点。"""

    target_recall: float
    #: 为达到（或最接近）目标召回，需要报出多少条
    alerts: int
    achieved_recall: float
    precision: float
    #: 目标召回是否真的达到了。**达不到时必须能看出来**，不能悄悄降级成「尽力而为」
    reached: bool


def confusion_at(ranking: Ranking, truth: Iterable[str], *, threshold: float) -> ConfusionMatrix:
    """在给定阈值上切一刀，得到混淆矩阵。

    分数 **>= threshold** 视为报警。用 `>=` 而不是 `>`：阈值本身应当算作报警，
    否则一个「恰好等于阈值」的样本会被算成漏报，和直觉不符。
    """
    positives = frozenset(truth)
    answered = set(ranking.scores)
    silent = set(ranking.no_answer)

    overlap = answered & silent
    if overlap:
        raise ValueError(f"对象不能既给分数又标记沉默：{sorted(overlap)[:5]}")

    tp = fp = tn = fn = 0
    for item, score in ranking.scores.items():
        positive = item in positives
        alerted = score >= threshold
        if alerted and positive:
            tp += 1
        elif alerted and not positive:
            fp += 1
        elif positive:
            fn += 1
        else:
            tn += 1

    positive_silent = sum(1 for item in silent if item in positives)
    negative_silent = len(silent) - positive_silent

    return ConfusionMatrix(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        positive_silent=positive_silent,
        negative_silent=negative_silent,
    )


def confusion_at_k(ranking: Ranking, truth: Iterable[str], *, k: int) -> ConfusionMatrix:
    """报警数**正好**为 k 时的混淆矩阵。

    ## 为什么不能用 `confusion_at(threshold=第k名的分数)` 代替

    两者在**没有并列**时等价，**有并列时结果完全不同**。

    本项目实测踩到了这个坑：全图有 **242,981 条边**的互转分数**完全相同**（都是 1.0）。
    用「分数 >= 第 100 名的分数」切，会把 24 万条**全部**报出来——
    于是「前 100 名的召回率」被算成了「24 万条的召回率」，
    算出 `R@100 = 14.7%`（= 253 条），**而前 100 名里根本放不下 253 条**。

    **这个数字自相矛盾，正是发现 bug 的线索。**

    ## 为什么按位置切才对

    分析师的时间是有限的：他只会有耐心看完前 k 条。
    所以「工作量」的准确含义是**看了几条**，而不是「分数超过某条线」。
    按位置切才对应真实约束。
    """
    if k <= 0:
        raise ValueError("k 必须为正数")

    positives = frozenset(truth)
    answered = set(ranking.scores)
    silent = set(ranking.no_answer)

    overlap = answered & silent
    if overlap:
        raise ValueError(f"对象不能既给分数又标记沉默：{sorted(overlap)[:5]}")

    top = set(ranking.ranked()[:k])

    tp = sum(1 for item in top if item in positives)
    fp = len(top) - tp
    # 没进前 k 的正例 = 漏报；沉默的正例单独算
    fn = sum(1 for item in positives if item in answered and item not in top)
    tn = sum(1 for item in answered if item not in positives and item not in top)
    positive_silent = sum(1 for item in silent if item in positives)

    return ConfusionMatrix(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        positive_silent=positive_silent,
        negative_silent=len(silent) - positive_silent,
    )


def precision_at_k(ranking: Ranking, truth: Iterable[str], *, k: int) -> float:
    """前 K 条里，有多少是真的。

    **这个指标回答的问题是**：「分析师一天只看 K 条，其中多少值得查？」

    为什么需要它：痛点是「误报淹没了分析师」，而**分析师的注意力是有限的**。
    整体精确率不体现这个约束，Precision@K 体现。

    若候选不足 K 条，分母取实际条数——否则候选少反而会拉低分数，不合理。
    """
    if k <= 0:
        raise ValueError("k 必须为正数")
    positives = frozenset(truth)
    top = ranking.ranked()[:k]
    if not top:
        return 0.0
    return sum(1 for item in top if item in positives) / len(top)


def precision_at_recall(
    ranking: Ranking, truth: Iterable[str], *, target_recall: float
) -> PrecisionAtRecall:
    """把阈值放宽到刚好达到目标召回，此时的精确率是多少。

    **这个指标回答的问题是**：「如果不许漏，误报会多到什么程度？」

    做法是沿着排序往下走，每一步算一次当前的召回与精确率，
    取**第一个召回达标的前缀**。因为排序是按可疑度降序，
    精确率通常随前缀变长而下降，所以这是「达到目标召回时最省告警量」的那个点。

    ⚠️ 如果**无论报出多少条都达不到目标召回**（因为有些正例沉默或排在最末），
    返回 `reached=False` 并给出实际达到的最高召回——**不静默降级**。
    """
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall 必须落在 (0, 1] 区间")

    positives = frozenset(truth)
    total_positive = len(positives)
    order = ranking.ranked()

    if total_positive == 0:
        return PrecisionAtRecall(
            target_recall=target_recall,
            alerts=0,
            achieved_recall=0.0,
            precision=0.0,
            reached=False,
        )

    hits = 0
    best_recall = 0.0
    for index, item in enumerate(order, start=1):
        if item in positives:
            hits += 1
        recall = hits / total_positive
        best_recall = recall
        if recall >= target_recall:
            return PrecisionAtRecall(
                target_recall=target_recall,
                alerts=index,
                achieved_recall=recall,
                precision=hits / index,
                reached=True,
            )

    # 全部报出仍未达标
    return PrecisionAtRecall(
        target_recall=target_recall,
        alerts=len(order),
        achieved_recall=best_recall,
        precision=(hits / len(order)) if order else 0.0,
        reached=False,
    )


def summarize(
    ranking: Ranking,
    truth: Iterable[str],
    *,
    threshold: float,
    k: int,
    target_recall: float = 0.9,
) -> dict[str, Any]:
    """把两个主指标和辅助指标汇总成一份可写进报告的字典。

    报告里**始终同时包含召回率和两个误报口径**——
    只报一个指标一定会被骗：报低误报，可以「什么都不报」拿满分；
    报高召回，可以「全部标红」拿满分。
    """
    positives = frozenset(truth)
    matrix = confusion_at(ranking, positives, threshold=threshold)
    at_k = precision_at_k(ranking, positives, k=k)
    at_recall = precision_at_recall(ranking, positives, target_recall=target_recall)

    return {
        "confusion": matrix.as_dict(),
        "precisionAtK": {"k": k, "precision": round(at_k, 4)},
        "precisionAtRecall": {
            "targetRecall": at_recall.target_recall,
            "alerts": at_recall.alerts,
            "achievedRecall": round(at_recall.achieved_recall, 4),
            "precision": round(at_recall.precision, 4),
            "reached": at_recall.reached,
        },
        # 单独暴露，方便 CI 里断言「必须为 0」
        "noAnswer": matrix.no_answer,
    }
