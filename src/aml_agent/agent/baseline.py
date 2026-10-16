"""**基线**：一个账户相对全图的位置。

## 这是 M7 评测实测出来的缺口的修法

Agent 评测跑出 **FPR = 100%**——它把 5 个正常账户里的 4 个判成可疑。
**但根因不是模型不行。**

追下去发现：`amlsim:2454`（洗钱）和 `amlsim:400`（正常）的结构几乎完全一样
（都是"一个对手方、约 180 笔、金额全同、`t=0..199`"）。

> **AMLSim 的背景交易本身就高度规则**——每个账户都和固定的一组对手方反复转账。
> **所以"结构规则"在这份数据里不是信号。**
>
> **而 Agent 看到「180 笔、金额全同」时，它不知道这在全图里算多还是算少。**

**它不是判断错了，它是没有基线可依。**

## 基线从哪来

**从检测层来**（`detect/structural.py`）——那里已经有一条 `two_detector_score`，
而它的输出**本来就是全图排名**。所以：

> **Agent 不该从原始事实里"自己看出异常"。
> 异常是相对于基线定义的，而基线在检测层。**

这正是 `docs/14` 里「三类证据装配」的第 ② 类——
之前只用在渲染上，**没有做成工具给 Agent**。

## ⚠️ 分位高 ≠ 有问题

这一点**必须写进工具输出里**，否则 Agent 会把分位当成结论。

- 这是**相对排名**，不是绝对判断；
- 而且**某些信号我们实测过几乎没有区分度**（环长 2 约 1.0 倍）；
- 所以输出里同时带上 `MEASURED_SIGNALS` 的实测依据。

## 为什么构造一次就够了

算全图检测分数要遍历 132 万条边（实测约 2–4 秒）。
**但一个 Agent 循环会问很多次**——每次都算一遍是浪费。
所以 `BaselineIndex.build()` 一次算完、之后查询是字典查找。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aml_agent.detect.structural import compute_edge_features, two_detector_score
from aml_agent.retrieve.fusion import MEASURED_SIGNALS


@dataclass(frozen=True, slots=True)
class FeaturePosition:
    """一个特征的取值 + **它相对全图的分位**。"""

    name: str
    label: str
    value: float
    percentile: float

    def render(self) -> str:
        return f"{self.label}：{self.value:.2f}（全图第 {self.percentile * 100:.0f} 分位）"


@dataclass(frozen=True, slots=True)
class AccountBaseline:
    """一个账户相对全图的位置。"""

    account: str
    detector_score: float
    #: **这个账户的检测分数在全图中的分位（0–1）** —— Agent 缺的就是它
    detector_percentile: float
    features: tuple[FeaturePosition, ...]

    def as_payload(self) -> dict[str, Any]:
        return {
            "account": self.account,
            "detectorScore": round(self.detector_score, 4),
            "detectorPercentile": round(self.detector_percentile, 4),
            "features": [
                {
                    "name": position.name,
                    "label": position.label,
                    "value": round(position.value, 4),
                    "percentile": round(position.percentile, 4),
                }
                for position in self.features
            ],
            # ⚠️ 这段必须说清楚，否则 Agent 会把分位当成结论
            "howToRead": (
                "这些是**相对全图的排名**，不是判断。"
                "分位高不等于有问题——它只说明这个值在全图里偏大/偏小。"
                "请结合实测依据（measuredSignals）判断这个信号值不值得采信。"
            ),
            "measuredSignals": {
                key: {"grade": grade, "basis": basis}
                for key, (grade, basis) in MEASURED_SIGNALS.items()
            },
        }


# --------------------------------------------------------------------------- #
#  特征定义
# --------------------------------------------------------------------------- #

#: 每个账户要算哪些特征、以及它们的中文名。
#:
#: **只选 Agent 在工具结果里真能看到的那几个**——
#: 给一个它在事实里见不到的特征算分位，它没法把两者联系起来。
ACCOUNT_FEATURES: tuple[tuple[str, str], ...] = (
    ("outDegree", "转出去向数"),
    ("inDegree", "转入来源数"),
    ("medianAmount", "单笔金额中位数"),
    ("maxPairTxCount", "同一对手方的最大笔数"),
    ("reciprocalRatio", "互转笔数占比"),
)


def _percentiles(values: pd.Series) -> pd.Series:
    """把一组数值转成**分位**（0–1）。

    用 `rank(pct=True)` 而不是自己排序：它能正确处理并列
    （并列取平均名次），而**手写排序在并列时会给不同的分位**——
    这个坑在本项目里踩过（24 万条边分数完全相同）。
    """
    return values.rank(pct=True, method="average")


def _account_features(features: pd.DataFrame) -> pd.DataFrame:
    """把逐边特征聚合成逐账户特征。"""
    frame = features[["sender", "receiver", "amount", "receiverInDegree", "pairTxCount"]].copy()
    frame["isReciprocal"] = features["isReciprocal"].astype(float)

    sender_stats = frame.groupby("sender").agg(
        outDegree=("receiver", "size"),
        medianAmount=("amount", "median"),
        maxPairTxCount=("pairTxCount", "max"),
        reciprocalRatio=("isReciprocal", "mean"),
    )
    receiver_stats = frame.groupby("receiver").agg(
        inDegree=("sender", "size"),
        medianAmount=("amount", "median"),
        maxPairTxCount=("pairTxCount", "max"),
        reciprocalRatio=("isReciprocal", "mean"),
    )

    # 一个账户既发也收：**取两者里信息更多的那个**
    # （度数相加，中位数/比例取该账户自己发出的那部分）。
    combined = sender_stats.reindex(sender_stats.index.union(receiver_stats.index), fill_value=0.0)
    combined["outDegree"] = sender_stats["outDegree"].reindex(combined.index, fill_value=0)
    combined["inDegree"] = receiver_stats["inDegree"].reindex(combined.index, fill_value=0)
    return combined


def _as_float(value: object) -> float:
    """把 pandas 取出来的值收窄成 `float`。

    pandas-stubs 给 `.at[]` 的类型是 `str | bytes | date | datetime | ... | complex`，
    而 `float()` 的 stub 不接受 `date`/`complex` 这些。

    这里**显式收窄并检查**，而不是 `float(value)  # type: ignore`：
    非数值会立刻报错，不会悄悄变成一个 0 参与分位计算。
    """
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    raise TypeError(f"期望数值，收到 {type(value).__name__}")


class BaselineIndex:
    """全图各账户的检测分数与特征分位。**构造一次，查询 O(1)。**"""

    def __init__(
        self,
        scores: Mapping[str, float],
        percentiles: Mapping[str, float],
        features: pd.DataFrame,
        feature_percentiles: pd.DataFrame,
    ) -> None:
        self._scores = scores
        self._percentiles = percentiles
        self._features = features
        self._feature_percentiles = feature_percentiles

    @classmethod
    def build(cls, edges: pd.DataFrame) -> BaselineIndex:
        """从边表构造。**实测约 2–4 秒**（132 万条边）。"""
        features = compute_edge_features(edges)
        edge_scores = two_detector_score(features)

        # 账户分数 = 它相关边里**最高**的那一条。
        # 用 max 而不是 mean：一个账户只要有一笔高度可疑就该被看到，
        # 平均会把那一笔稀释掉（这正是"取大"和"求和"两种融合的分歧点）。
        #
        # 而 `two_detector_score` 本来就是 `cycle + fan_in` 两个排名分之和，
        # 所以它天然是"分数"而不是"概率"——**分位才是有意义的输出**。
        scoring = pd.DataFrame(
            {
                "sender": features["sender"].to_numpy(),
                "receiver": features["receiver"].to_numpy(),
                "score": edge_scores.to_numpy(),
            }
        )
        by_sender = scoring.groupby("sender")["score"].max()
        by_receiver = scoring.groupby("receiver")["score"].max()
        account_score = pd.concat([by_sender, by_receiver]).groupby(level=0).max()

        account_features = _account_features(features)
        feature_percentiles = pd.DataFrame(
            {
                name: _percentiles(account_features[name])
                for name, _ in ACCOUNT_FEATURES
                if name in account_features.columns
            }
        )

        return cls(
            scores={str(key): float(value) for key, value in account_score.items()},
            percentiles={
                str(key): float(value) for key, value in _percentiles(account_score).items()
            },
            features=account_features,
            feature_percentiles=feature_percentiles,
        )

    @property
    def account_count(self) -> int:
        return len(self._scores)

    def lookup(self, account: str) -> AccountBaseline | None:
        """查一个账户。**查不到返回 `None`**，而不是编一个默认值。

        ⚠️ "查不到"和"分位很低"是两回事——
        返回 0.0 会让 Agent 以为"这个账户排名最低"，而实际是**它不在图里**。
        """
        if account not in self._scores:
            return None

        positions: list[FeaturePosition] = []
        if account in self._features.index:
            for name, label in ACCOUNT_FEATURES:
                if name not in self._features.columns:
                    continue
                percentiles = self._feature_percentiles[name]
                positions.append(
                    FeaturePosition(
                        name=name,
                        label=label,
                        value=_as_float(self._features.at[account, name]),
                        percentile=(
                            _as_float(percentiles.at[account])
                            if account in percentiles.index
                            else 0.5
                        ),
                    )
                )

        return AccountBaseline(
            account=account,
            detector_score=self._scores[account],
            detector_percentile=self._percentiles[account],
            features=tuple(positions),
        )

    def render(self, baseline: AccountBaseline) -> str:
        """给人和给模型看的文本形式。"""
        lines = [
            f"{baseline.account} 在全图 {self.account_count} 个账户中的位置：",
            f"- 检测分数分位：第 {baseline.detector_percentile * 100:.0f} 分位",
        ]
        lines.extend(f"- {position.render()}" for position in baseline.features)
        return "\n".join(lines)


def numpy_percentile(sorted_values: np.ndarray, value: float) -> float:
    """在一个已排序数组里查分位。

    单独抽出来是为了**能被测试**——`BaselineIndex.lookup` 依赖分位算得对，
    而分位算错的症状是"数字看起来合理但排序不对"，很难从端到端发现。
    """
    if sorted_values.size == 0:
        return 0.0
    return float(np.searchsorted(sorted_values, value, side="right") / sorted_values.size)
