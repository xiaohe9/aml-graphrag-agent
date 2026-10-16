"""第一层检测：可全图运行的便宜结构信号。

## 为什么是"便宜"的信号

M3 的测量结论（见 `docs/09-M3-算法与评测.md`）：

- 全局环枚举**不可行**——长度 3 都跑不完（平均出度 132，组合爆炸）；
- **「节点在环上」没有区分度**——98.5% 的节点属于同一个强连通分量。

所以第一层**不做环枚举**，只算**每条边都能 O(1) 得到**的结构量。
第二层（Agent）再对少数候选做有预算的深挖。

**这不是妥协，这是分层设计的必然结果**——而且它让第一层可以全图跑、可复现、可解释。

## 特征清单（⚠️ 方向是**实测**出来的，不是猜的）

M3 实测的分组画像：

```text
分组            条数       金额中位  收款方入度中位  付款方出度中位  同对笔数中位  互转率
全部交易      1,323,234   156.71      233          180          20    18.4%
标注·cycle         936    14.91      121          156           1    19.3%
标注·fan_in        783     3.65      145          151           2     9.2%
```

| 特征 | 可疑方向 | 依据（实测） |
|---|---|---|
| `amount` | **越小越可疑** | 14.91 / 3.65 vs 156.71 |
| `pairTxCount` | **越少越可疑** | **1–2 vs 20** ← 最强信号，且方向与直觉相反 |
| `receiverInDegree` | **越低越可疑** | 121 / 145 vs 233 |
| `senderOutDegree` | **越低越可疑** | 156 / 151 vs 180（弱） |
| `isReciprocal` | ⚠️ **无信号** | 19.3% vs 18.4%，几乎没差别 |

### ⚠️ 这些方向**只在本数据集上成立**

它们来自 AMLSim 的注入方式（一次性转账、小额、"用完就走"的账户）。
**在真实数据上必须重新测**——照搬会得到一条看起来合理、实际无效的规则。

### ⚠️ 互转（长度 2 的环）为什么没用

我原本以为"资金原路返回"是最强的洗钱痕迹。实测：

> **全图 18.4% 的边都是互转**（242,981 条），
> 而标注为 `cycle` 的边只有 19.3% 是互转——**比基线只高 0.9 个百分点**。
>
> **这个图太稠密了**（平均出度 132），互转是常态而不是异常。
> 所以「互转」这个信号**在有真实结构的稠密图上是失效的**。
"""

from __future__ import annotations

import pandas as pd

try:  # networkx 的类型标注是可选依赖
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None  # type: ignore[assignment]

#: 同一条边会不会被两套机制同时计入（用于排查重复计数）——留作断言的锚点
FEATURE_COLUMNS: tuple[str, ...] = (
    "isReciprocal",
    "receiverInDegree",
    "senderOutDegree",
    "pairTxCount",
    "amount",
)


def compute_edge_features(graph: nx.MultiDiGraph[str], edges: pd.DataFrame) -> pd.DataFrame:
    """给每条边算便宜的结构特征。

    复杂度 O(E)，**可以全图跑**——这正是第一层存在的意义。
    """
    # 有向去重后的边集：判断"反向边是否存在"要忽略重边
    simple_pairs = set(graph.edges())

    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())

    # 同一对账户之间的转账笔数。用 (sender, receiver) 分组计数。
    pair_counts = (
        edges.groupby(["sender", "receiver"], sort=False)["txId"].transform("size").to_numpy()
    )

    features = pd.DataFrame(
        {
            "txId": edges["txId"].to_numpy(),
            "sender": edges["sender"].to_numpy(),
            "receiver": edges["receiver"].to_numpy(),
            "amount": edges["amount"].to_numpy(),
            "receiverInDegree": edges["receiver"].map(in_degree).to_numpy(),
            "senderOutDegree": edges["sender"].map(out_degree).to_numpy(),
            "pairTxCount": pair_counts,
        }
    )

    # 互转：反向边存在。用集合查询，整体 O(E)。
    features["isReciprocal"] = [
        (receiver, sender) in simple_pairs
        for sender, receiver in zip(edges["sender"], edges["receiver"], strict=True)
    ]
    return features


def reciprocal_score(features: pd.DataFrame) -> pd.Series:
    """只靠「互转」打分。

    **这是一个诚实的上限探测**：实测标注为 `cycle` 的 936 条里，
    有 181 条是长度 2 的环。所以单靠这个信号，召回上限就是约 19%。
    先把它单独测出来，才知道后面加的特征到底有没有用。
    """
    return features["isReciprocal"].astype(float)


def small_amount_reference_score(features: pd.DataFrame) -> pd.Series:
    """「金额越小越可疑」——作为**参照**，不是我们的方法。

    ⚠️ 它和 `aml_agent.detect.baseline.small_amount_score` 是同一件事，
    这里保留一份是为了在评测表里与结构特征**并排对照**。
    **它不是我们的成果**——如果它比我们的结构特征还好，
    那说明我们的特征没有价值，必须诚实承认（M3 实测确实出现过这个情况）。
    """
    return -features["amount"]


def two_detector_scores(features: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """**分模式**打分：两个检测器，各用各的有效信号。

    ## 为什么要分开

    实测的特征画像说明两类模式的**可识别特征完全不同**：

    | 模式 | 有效信号 | 无效信号 |
    |---|---|---|
    | `fan_in` | **金额极小**（召回 90.7%） | 同对笔数（7.5%） |
    | `cycle` | **同对笔数少**（召回 12.5%） | **金额（0.0%）** |

    用一个分数同时打两类，是把两个不同的检测器搅在一起——**两边都做不好**。
    实测印证：合并打分的 P@100 升到 96%，但 fan_in 召回**从 90.7% 掉到 31.8%**。

    **这也正好对应架构里的第二层**：`match_patterns` 工具本来就是按模式匹配的。
    """
    rank = features.rank(pct=True)
    # 转成百分位，让两个检测器的分数**可比较**——否则融合时会被量纲决定
    cycle = (1.0 - rank["pairTxCount"]).rank(pct=True)
    fan_in = (1.0 - rank["amount"]).rank(pct=True)
    return cycle, fan_in


def fuse_max(cycle: pd.Series, fan_in: pd.Series) -> pd.Series:
    """融合策略一：**取较大值**。

    语义是「**任一检测器认为可疑，就报**」——偏召回。
    """
    return pd.concat([cycle, fan_in], axis=1).max(axis=1)


def fuse_sum(cycle: pd.Series, fan_in: pd.Series) -> pd.Series:
    """融合策略二：**求和**。

    语义是「**两个检测器都认为可疑，才更可疑**」——偏精确。
    它会自然地把同时具备两种特征的边排到前面。
    """
    return cycle + fan_in


def two_detector_score(features: pd.DataFrame) -> pd.Series:
    """默认融合：求和（偏精确，适合"分析师只看前 K 条"的场景）。"""
    return fuse_sum(*two_detector_scores(features))


def combined_score(features: pd.DataFrame) -> pd.Series:
    """把便宜信号按**秩**组合。

    ## 权重和方向都来自实测，不是先验

    见模块文档的分组画像表。四个特征里**有三个的方向与直觉相反**，
    而**互转这个我最有信心的信号其实没有信号**——所以把它权重设为 0。

    ## 为什么按秩（rank）而不是按原值加权

    这几个量量纲完全不同——金额是金额、度数是笔数。
    直接加权等于让数值范围最大的量主导一切。
    换成秩（都落到 0..1）之后，**权重才真的表达"我认为这个信号多重要"**。

    ## ⚠️ 这些权重**不能拿测试集调**

    本数据集只用于最终评测（`docs/08` 第 5.4 节）。
    权重是按画像表的中位数差异**手设**的，属于先验——不是拟合出来的。
    真要调参，得靠 5.2 节说的注入式训练集。
    """
    rank = features.rank(pct=True)

    # 取 1-rank 就是"越小越可疑"。方向依据见模块文档。
    return (
        1.0 * (1.0 - rank["pairTxCount"])  # 最强信号：1–2 vs 20
        + 1.0 * (1.0 - rank["amount"])  # 14.91/3.65 vs 156.71
        + 0.5 * (1.0 - rank["receiverInDegree"])  # 121/145 vs 233
        + 0.5 * (1.0 - rank["senderOutDegree"])  # 156/151 vs 180（弱）
        # isReciprocal 权重 0：实测 19.3% vs 18.4%，没有信号。
        # 保留这个特征是为了**留下"我们测过它并且否掉了它"的证据**，
        # 而不是因为它有用。
        + 0.0 * rank["isReciprocal"]
    )
