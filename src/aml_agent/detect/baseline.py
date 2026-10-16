"""规则基线。

**为什么一定要有基线：**

没有基线的指标是**没有意义的数字**。「召回 60%」听起来不错，
但如果一条 5 行代码的规则就能到 55%，那这一堆东西的价值就接近于零。

**基线回答一个问题：我做的这些，比"什么都不做聪明点"强多少？**

## 这里实现两个基线

| 基线 | 内容 | 为什么选它 |
|---|---|---|
| **金额阈值** | 单笔金额越大越可疑 | **真实反洗钱系统里最常见的规则**，也是 PRD 要挑战的对象 |
| **反向金额** | 单笔金额越**小**越可疑 | 诚实对照：如果调对方向就能赢，说明结构特征没有价值 |
| **随机** | 随机打分 | 下限。任何方法都不该输给随机 |

**金额基线的预期**：M1 实测涉诈金额中位数 10.60、正常 156.71。
所以「金额越大越可疑」这个方向在这份数据上**是反的**——
它的召回率应该接近 0。**这个 0 不是 bug，是结论。**
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

#: 基线就是一个函数：边表 → 每条边的可疑度分数。
BaselineFn = Callable[[pd.DataFrame], pd.Series]


def amount_threshold_score(edges: pd.DataFrame) -> pd.Series:
    """经典规则：金额越大越可疑。

    **这一条就是 PRD 里说要挑战的东西**——
    「规则只能判断单笔交易，而洗钱的痕迹在路径上」。
    """
    return edges["amount"].astype(float)


def small_amount_score(edges: pd.DataFrame) -> pd.Series:
    """反向的金额规则：金额越**小**越可疑。

    **这个基线很重要**，因为它是金额基线的"作弊版"——
    如果我们只是在数据上把方向调对就能赢，那说明**我们的结构特征没有价值**。
    用它来做一个诚实的对照。
    """
    return -edges["amount"].astype(float)


def random_score(edges: pd.DataFrame, *, seed: int = 42) -> pd.Series:
    """随机打分。用固定种子，保证**可复现**（NFR-5）。"""
    generator = np.random.default_rng(seed)
    return pd.Series(generator.random(len(edges)), index=edges.index)


#: 参与对比的基线。名字 → 打分函数。
BASELINES: dict[str, BaselineFn] = {
    "金额阈值规则（越大越可疑）": amount_threshold_score,
    "金额阈值规则（越小越可疑）": small_amount_score,
    "随机": random_score,
}
