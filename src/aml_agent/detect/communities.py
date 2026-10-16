"""社区发现（FR-4）：识别**可疑团伙**。

## ⚠️ 这个需求的原验证方式不成立

`docs/02` 的追踪矩阵原本写「注入团伙识别」作为验证方式——
**但这份数据集根本没有团伙真值**：

```text
ALERT_TYPE 取值：cycle 936 / fan_in 783     ← 只有这两种
```

**当时的我假设了一个不存在的真值。**

所以这里实现的**只是一个算法能力**，它的验证只能用**代理指标**，
而且**代理就是代理，不能冒充达标**：

| 能验证的 | 不能验证的 |
|---|---|
| 算法能跑、耗时多少 | **它识别的社区是不是"真的团伙"** |
| 社区是否非平凡（不是一团一整个图） | 召回率、精确率（没有真值就没有这些） |
| 跨随机种子的稳定性 | |
| 与**已有标签**的相关性（这是代理） | |

**把"这份数据无法验证团伙识别"写进结论，
比拿一个代理指标冒充达标要诚实得多。**

## 为什么必须在**去重图**上跑

同 `detect/cycles.py`：132 万条边里 **94.8% 是平行边**，
去重后只有 68,947 条。Louvain 在多重图上会把平行边的权重叠加，
**让"转账次数多"变成"连接紧密"**——那不是社区结构，是交易频次。
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass

import networkx as nx
import pandas as pd

#: Louvain 的分辨率。越大 → 社区越多越小。
#:
#: 默认 1.0 是 Louvain 的原始定义。
#: **它没有一个"正确"值**——社区粒度本来就是业务问题
#: （"团伙"是 5 个人还是 50 个人？），所以它必须可调，且调参结果要记录。
DEFAULT_RESOLUTION = 1.0

#: 随机种子。Louvain 有随机性，**不固定种子就没有可复现性**。
DEFAULT_SEED = 20260101


@dataclass(frozen=True, slots=True)
class CommunityScan:
    """一次社区发现的结果。"""

    #: 节点 → 社区编号
    assignments: Mapping[str, int]
    #: 模块度。**注意它只是"划分质量"，不是"可疑程度"**
    modularity: float
    seconds: float
    resolution: float
    seed: int

    @property
    def community_count(self) -> int:
        return len(set(self.assignments.values()))

    @property
    def node_count(self) -> int:
        return len(self.assignments)

    def sizes(self) -> list[int]:
        """各社区的规模，降序。"""
        counts: dict[int, int] = {}
        for community in self.assignments.values():
            counts[community] = counts.get(community, 0) + 1
        return sorted(counts.values(), reverse=True)

    def community_of(self, node: str) -> int | None:
        return self.assignments.get(node)

    def members(self, community: int) -> tuple[str, ...]:
        return tuple(sorted(node for node, value in self.assignments.items() if value == community))

    def summary(self) -> dict[str, object]:
        sizes = self.sizes()
        return {
            "nodes": self.node_count,
            "communities": self.community_count,
            "modularity": round(self.modularity, 4),
            "largest": sizes[0] if sizes else 0,
            "largestShare": round(sizes[0] / self.node_count, 3) if sizes else 0.0,
            "seconds": round(self.seconds, 1),
            "resolution": self.resolution,
            "seed": self.seed,
        }


def to_undirected_simple(edges: pd.DataFrame) -> nx.Graph[str]:
    """把边表转成**无向去重图**。

    ## 为什么社区发现用无向图

    社区发现问的是「**谁和谁是一伙的**」，不是「钱往哪个方向流」。
    同一伙人之间的资金往来是双向的还是单向的，**不影响他们是不是一伙**。

    （这与环检测相反——**环必须是有向的**，否则 A→B 和 B→A 会被当成同一个环。
    同一个图，两个算法要不同的视图，这是正常的。）
    """
    graph: nx.Graph[str] = nx.Graph()
    senders = edges["sender"].astype(str).to_numpy()
    receivers = edges["receiver"].astype(str).to_numpy()
    graph.add_nodes_from(set(senders) | set(receivers))
    # 用 `from_edgelist` 而不是逐条 add_edge：与 `to_simple_digraph` 同理，
    # 后者在 13 万条边上会慢一个数量级
    graph.add_edges_from(zip(senders, receivers, strict=True))
    return graph


def find_communities(
    edges: pd.DataFrame,
    *,
    resolution: float = DEFAULT_RESOLUTION,
    seed: int = DEFAULT_SEED,
    max_nodes: int | None = None,
) -> CommunityScan:
    """在去重无向图上跑 Louvain。

    `max_nodes` 是**安全阀**：社区发现是 O(E log V) 但常数不小，
    在超大图上必须能设上限，而不是把内存打满。
    """
    if resolution <= 0:
        raise ValueError(f"resolution 必须为正，收到 {resolution}")

    graph = to_undirected_simple(edges)
    if max_nodes is not None and graph.number_of_nodes() > max_nodes:
        raise ValueError(
            f"图有 {graph.number_of_nodes()} 个节点，超过 max_nodes={max_nodes}。"
            "**这是刻意的保护**——社区发现在超大图上的内存与耗时都不是线性的。"
        )

    started = time.monotonic()
    communities = nx.community.louvain_communities(graph, resolution=resolution, seed=seed)
    seconds = time.monotonic() - started

    assignments = {node: index for index, members in enumerate(communities) for node in members}
    return CommunityScan(
        assignments=assignments,
        modularity=float(nx.community.modularity(graph, communities)),
        seconds=seconds,
        resolution=resolution,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
#  代理指标（**不是真值，必须标明**）
# --------------------------------------------------------------------------- #


def rand_agreement(first: Mapping[str, int], second: Mapping[str, int]) -> float:
    """两次划分的**成对一致率**（Rand index 的简化版）。

    ## 为什么自己写而不是用现成库

    ARI/NMI 的标准实现要 `sklearn`，而这里只需要一个"稳不稳定"的粗判。
    自己写 10 行、无新依赖、**而且定义完全透明**——
    对代理指标来说，**定义透明比指标高级更重要**。

    ## 它衡量什么

    在所有**节点对**里，两次划分给出相同判断（同社区 / 不同社区）的比例。
    1.0 = 完全一致。

    ⚠️ **它不衡量划分好不好，只衡量稳不稳。**
    两次都错得一模一样，它也会给 1.0。
    """
    nodes = sorted(set(first) & set(second))
    if len(nodes) < 2:
        return 0.0

    agree = 0
    total = 0
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            same_first = first[left] == first[right]
            same_second = second[left] == second[right]
            total += 1
            if same_first == same_second:
                agree += 1
    return agree / total if total else 0.0


def label_purity(assignments: Mapping[str, int], labels: Mapping[str, bool]) -> dict[str, object]:
    """**代理指标**：社区划分与已有标签的关系。

    ⚠️⚠️ **这不是团伙识别的准确率。**

    这份数据的标签只有 `cycle` / `fan_in` 两种**交易级**标注，
    **它们不是团伙标注**。所以这里只能回答一个弱得多的问题：

    > 「标注过的账户，是否更倾向于落在**同一些**社区里？」

    - 如果**是** → 社区结构可能与可疑活动有关（**弱证据**）；
    - 如果**否** → 说明社区结构捕捉的是别的东西（比如交易活跃度），
      **那它作为"团伙识别"就是失败的**。

    **两种结果都有信息量，但都不能说成"团伙识别达标"。**
    """
    known = {node: value for node, value in labels.items() if node in assignments}
    if not known:
        return {"note": "没有可对齐的标签"}

    # 每个社区里"标注过的账户"占比，与全图基准比
    baseline = sum(1 for value in known.values() if value) / len(known)
    by_community: dict[int, list[bool]] = {}
    for node, value in known.items():
        by_community.setdefault(assignments[node], []).append(value)

    # 只看规模够大的社区，小社区的比例没有统计意义
    large = {key: values for key, values in by_community.items() if len(values) >= 30}
    if not large:
        return {
            "baselineRate": round(baseline, 4),
            "note": "没有规模足够的社区可供比较",
        }

    rates = sorted((sum(values) / len(values) for values in large.values()), reverse=True)
    return {
        "baselineRate": round(baseline, 4),
        "communitiesCompared": len(rates),
        "topRate": round(rates[0], 4),
        "bottomRate": round(rates[-1], 4),
        "spread": round(rates[0] - rates[-1], 4),
        "note": (
            "⚠️ **代理指标，不是团伙识别准确率。**"
            "标签是交易级的 cycle/fan_in，不是团伙标注。"
            "这里只回答：标注账户是否更集中在某些社区。"
        ),
    }
