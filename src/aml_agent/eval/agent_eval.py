"""Agent 评测：在**同时含正例与负例**的样本上量化它的表现。

## ⚠️ 为什么必须有负例

只测洗钱案例的话，**一个"见谁都说可疑"的 Agent 会拿满分**。

这不是理论担忧——它是最容易犯的评测错误：
样本全是有问题的，于是"检出率 100%"，而实际上系统毫无分辨力。

**所以样本必须两类都有**，而且最好在活跃度上可比
（否则 Agent 可能靠"这个账户的数据多不多"就区分开了，
而不是靠资金结构）。

## ⚠️ 「未产出结论」必须单独计数（M3 的框架直接复用）

`ConfusionMatrix` 里本来就有 `positive_silent` / `negative_silent`——
它们表示**"我看了，但没给出结论"**。

| | 含义 |
|---|---|
| `fn` | 我看了，说没问题 —— **漏报** |
| `positive_silent` | **我压根没给出结论** |

**把后者折进前者，会同时压低检出率、又虚假地压低误报率。**
M3 里为什么这么设计，在这里得到了回报：**换了个 Agent，指标口径一个字都不用改。**
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from aml_agent.agent import AgentRun
from aml_agent.eval.metrics import ConfusionMatrix

#: 抽样随机种子。**固定**，否则每次评测的样本不同，数字没法比。
DEFAULT_SEED = 20260101


@dataclass(frozen=True, slots=True)
class AgentCase:
    """一个评测案例。"""

    #: 账户键（图里的形式，如 `amlsim:4122`）
    account_key: str
    #: 真值。**来自 CSV，不来自图**（ADR-0002）。
    is_fraud: bool
    #: 这个账户在原数据里的交易笔数——用来检查正负例是否可比
    activity: int = 0


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    """一个案例的结果。"""

    case: AgentCase
    run: AgentRun

    @property
    def verdict(self) -> bool | None:
        """`True` 可疑 / `False` 未发现可疑 / `None` **没给出结论**。"""
        if self.run.conclusion is None:
            return None
        return self.run.conclusion.suspicious

    @property
    def confidence(self) -> float:
        if self.run.conclusion is None:
            return 0.0
        return self.run.conclusion.confidence

    def as_dict(self) -> dict[str, Any]:
        return {
            "account": self.case.account_key,
            "is_fraud": self.case.is_fraud,
            "activity": self.case.activity,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 3),
            "steps": len(self.run.steps),
            "seconds": round(self.run.seconds, 1),
            "exhausted": self.run.exhausted,
            "forced": self.run.forced,
            "premature": self.run.premature,
        }


@dataclass(frozen=True, slots=True)
class AgentEvalResult:
    """一次 Agent 评测的全部结果。"""

    outcomes: tuple[CaseOutcome, ...]
    seconds: float
    temperature: float
    max_steps: int
    notes: tuple[str, ...] = field(default=())

    # ---- 混淆矩阵（**未产出结论单独成列**）------------------------------- #

    @property
    def confusion(self) -> ConfusionMatrix:
        tp = fp = tn = fn = 0
        positive_silent = negative_silent = 0
        for outcome in self.outcomes:
            verdict = outcome.verdict
            if verdict is None:
                # ⚠️ **没给结论** —— 不是"说没问题"，是"什么都没说"
                if outcome.case.is_fraud:
                    positive_silent += 1
                else:
                    negative_silent += 1
            elif verdict and outcome.case.is_fraud:
                tp += 1
            elif verdict and not outcome.case.is_fraud:
                fp += 1
            elif not verdict and outcome.case.is_fraud:
                fn += 1
            else:
                tn += 1
        return ConfusionMatrix(
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            positive_silent=positive_silent,
            negative_silent=negative_silent,
        )

    # ---- 运行健康度 ------------------------------------------------------ #

    @property
    def total(self) -> int:
        return len(self.outcomes)

    def rate(self, predicate: Any) -> float:
        if not self.outcomes:
            return 0.0
        return sum(1 for outcome in self.outcomes if predicate(outcome)) / len(self.outcomes)

    @property
    def produced_conclusion_rate(self) -> float:
        return self.rate(lambda outcome: not outcome.run.exhausted)

    @property
    def exhausted_rate(self) -> float:
        """**步数耗尽、从未给结论的比例。** 这是工程缺陷指标，越低越好。"""
        return self.rate(lambda outcome: outcome.run.exhausted)

    @property
    def premature_rate(self) -> float:
        """**没做任何侦查就给结论的比例。**"""
        return self.rate(lambda outcome: outcome.run.premature)

    @property
    def forced_rate(self) -> float:
        """结论来自强制产出步的比例——高说明自由步数不够用。"""
        return self.rate(lambda outcome: outcome.run.forced)

    @property
    def mean_steps(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(len(outcome.run.steps) for outcome in self.outcomes) / len(self.outcomes)

    @property
    def mean_seconds(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(outcome.run.seconds for outcome in self.outcomes) / len(self.outcomes)

    def summary(self) -> dict[str, Any]:
        confusion = self.confusion
        return {
            "total": self.total,
            "confusion": confusion.as_dict(),
            "recall": round(confusion.recall, 4),
            "precision": round(confusion.precision, 4),
            "exhaustedRate": round(self.exhausted_rate, 4),
            "prematureRate": round(self.premature_rate, 4),
            "forcedRate": round(self.forced_rate, 4),
            "meanSteps": round(self.mean_steps, 2),
            "meanSeconds": round(self.mean_seconds, 1),
            "temperature": self.temperature,
            "maxSteps": self.max_steps,
        }

    def render(self) -> str:
        """给人看的报告。"""
        confusion = self.confusion
        lines = [
            "=" * 74,
            "Agent 评测报告",
            "=" * 74,
            f"  样本        {self.total} 个"
            f"（洗钱 {sum(1 for o in self.outcomes if o.case.is_fraud)}"
            f" / 正常 {sum(1 for o in self.outcomes if not o.case.is_fraud)}）",
            f"  参数        temperature={self.temperature}  max_steps={self.max_steps}",
            f"  总耗时      {self.seconds / 60:.1f} 分钟（平均 {self.mean_seconds:.0f} 秒/案例）",
            "",
            "  ── 结论质量 ──",
            f"  真阳性 TP   {confusion.tp:>3}   假阳性 FP   {confusion.fp:>3}",
            f"  真阴性 TN   {confusion.tn:>3}   假阴性 FN   {confusion.fn:>3}",
            "",
            "  ── ⚠️ 未产出结论（**不计入上面任何一格**）──",
            f"  洗钱案例没给结论   {confusion.positive_silent:>3}   ← 既不算漏报，也不算检出",
            f"  正常案例没给结论   {confusion.negative_silent:>3}",
            "",
            f"  召回率      {confusion.recall:.1%}   （分母含「没给结论」的洗钱案例）",
            f"  精确率      {confusion.precision:.1%}",
            # 直接用它算好的——**"1 − 精确率"和 FPR 是两回事**，
            # 而 `alert_false_rate` 就是前者（AML 口语里的"误报率"）
            f"  误报率      {confusion.alert_false_rate:.1%}"
            f"   （= 1 − 精确率；FPR 是另一个数：{confusion.false_positive_rate:.1%}）",
            "",
            "  ── 运行健康度 ──",
            f"  产出结论率  {self.produced_conclusion_rate:.1%}",
            f"  步数耗尽率  {self.exhausted_rate:.1%}   ← **工程缺陷指标，应趋近 0**",
            f"  过早结论率  {self.premature_rate:.1%}",
            f"  强制产出率  {self.forced_rate:.1%}   ← 高说明自由步数不够",
            f"  平均步数    {self.mean_steps:.2f}",
        ]
        if self.notes:
            lines.append("")
            lines.append("  ── 说明 ──")
            lines.extend(f"  · {note}" for note in self.notes)
        lines.append("=" * 74)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  抽样
# --------------------------------------------------------------------------- #


def sample_cases(
    accounts: Sequence[int],
    fraud_accounts: Sequence[int],
    activity: dict[int, int],
    *,
    per_class: int,
    seed: int = DEFAULT_SEED,
    match_activity: bool = True,
) -> list[AgentCase]:
    """**从两类里各抽同样多的案例。**

    ## ⚠️ 为什么默认要按活跃度配对

    第一版是两类各随机抽。dry-run 一跑就发现了问题：

    ```text
    活跃度中位数：洗钱 204 笔 / 正常 62 笔   ← 相差 3.3 倍
    ```

    **原因**：洗钱账户的标注来自注入的模式，而**注入模式本身会产生额外交易**，
    所以被标注的账户天然更活跃。

    **后果**：Agent 可以靠"这个账户的数据多不多"就蒙对，
    而不是靠资金结构——**那评测出来的就不是我们要测的能力**。

    所以默认做**最近邻配对**：每个正例配一个活跃度最接近的未使用负例。

    ## 为什么要报告 `activity`

    配对之后仍要在报告里对比两类的活跃度分布——
    **配对是缓解，不是消除**。残余差异要让读者看得到。
    """
    from aml_agent.graph import INSTITUTION

    fraud_set = set(fraud_accounts)
    positives = [account for account in accounts if account in fraud_set]
    negatives = [account for account in accounts if account not in fraud_set]

    if len(positives) < per_class:
        raise ValueError(f"正例不够：需要 {per_class}，只有 {len(positives)}")
    if len(negatives) < per_class:
        raise ValueError(f"负例不够：需要 {per_class}，只有 {len(negatives)}")

    # `random` 而非 `secrets`：这是**抽样**，不是密码学用途。
    # ruff 的 S311 对任何 `random` 用法都报，这里明确标注意图。
    rng = random.Random(seed)  # noqa: S311
    picked_positives = rng.sample(positives, per_class)

    if not match_activity:
        picked_negatives = rng.sample(negatives, per_class)
    else:
        # 最近邻配对：对每个正例，挑活跃度最接近的**尚未被用掉**的负例。
        #
        # 用"逐个挑最近的"而不是"整体最优匹配"：前者是贪心，可能不是全局最优，
        # 但**足够好、且行为可解释**——评测的抽样策略应该是能讲清楚的。
        picked_negatives = []
        used: set[int] = set()
        for positive in sorted(picked_positives, key=lambda a: -activity.get(a, 0)):
            target = activity.get(positive, 0)
            best = min(
                (n for n in negatives if n not in used),
                key=lambda n: (abs(activity.get(n, 0) - target), n),
            )
            used.add(best)
            picked_negatives.append(best)

    picked = picked_positives + picked_negatives
    # 打乱顺序：**正例全排前面会让"跑到一半"的中间结果严重偏乐观**
    rng.shuffle(picked)

    return [
        AgentCase(
            account_key=f"{INSTITUTION}:{int(account)}",
            is_fraud=account in fraud_set,
            activity=activity.get(int(account), 0),
        )
        for account in picked
    ]


def describe_balance(cases: Sequence[AgentCase]) -> tuple[str, ...]:
    """检查正负例在活跃度上是否可比。**不可比要说出来。**"""
    positives = [case.activity for case in cases if case.is_fraud]
    negatives = [case.activity for case in cases if not case.is_fraud]
    if not positives or not negatives:
        return ("样本里缺一类，无法比较活跃度——**这样的评测没有说服力**",)

    def median(values: list[int]) -> float:
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[middle])
        return (ordered[middle - 1] + ordered[middle]) / 2

    positive_median = median(positives)
    negative_median = median(negatives)
    notes = [f"活跃度中位数：洗钱 {positive_median:.0f} 笔 / 正常 {negative_median:.0f} 笔"]
    if negative_median > 0:
        ratio = positive_median / negative_median
        if 0.5 <= ratio <= 2.0:
            notes.append("两类活跃度可比 ✅ —— Agent 不能靠'数据多少'蒙对")
        else:
            notes.append(
                f"⚠️ 两类活跃度相差 {ratio:.1f} 倍，**Agent 可能靠活跃度而非资金结构区分**，"
                "结论要打折看"
            )
    return tuple(notes)


def question_for(case: AgentCase) -> str:
    """给 Agent 的问题。

    **刻意统一措辞**——问题一变，结果就没法横向比。
    """
    return (
        f"帮我查一下账户 {case.account_key} 有没有可疑的资金往来。"
        "结论里请引用你实际查到的数据（笔数、金额、时间跨度）。"
    )
