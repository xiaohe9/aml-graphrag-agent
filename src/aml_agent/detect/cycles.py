"""第二层：**真正去找环**。

## 为什么不用 `nx.simple_cycles`

实测对比（去重图，68,947 条边）：

| 方法 | 长度 ≤3 |
|---|---|
| `nx.simple_cycles(length_bound=3)` | **145 个环 / 120 秒触顶** ❌ |
| 本模块的集合求交 | 400 个样本 **0.0 秒** ✅ |

**69,000 条边的图，120 秒只找到 145 个三角形——这个数字本身就不合理。**
问题不在图，在 NetworkX 那个带 `length_bound` 的实现。

## 本模块的做法

对每条边 `u→v`，求**从 `v` 回到 `u` 的最短路径长度**。
回程长 + 1 就是环长。关键在于**最后一步用集合求交**：

| 回程 | 环长 | 判断 |
|---|---|---|
| 1 | 2 | `u ∈ 出邻居(v)` |
| 2 | 3 | `出邻居(v) ∩ 入邻居(u) ≠ ∅` |
| 3 | 4 | `∃w ∈ 出邻居(v)`：`出邻居(w) ∩ 入邻居(u) ≠ ∅` |

## ⚠️ 为什么必须在**去重图**上做

原始边表有 132 万条边，但**只有 68,947 个不同的账户对**（平行边占 94.8%）。

**平行边不改变环的结构**——`u→v→u` 存在与否，和中间有几条平行边无关。
在多重图上枚举会把同一个环重复算 19 次，白白慢 19 倍。

**但仍要记得**：最后要把结果**映射回具体交易**——
因为"这条交易在环上"才是我们要评的东西。

## ⚠️ 环长 2 不算数

M3 实测：**普通交易里 19% 的边本来就是互转**（回程 1 跳），
所以「环长 2」**没有区分度**（cycle 标注 75 vs 普通 77）。

**有区分度的是环长 3（40×）和环长 4（5×）。**
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import networkx as nx
import pandas as pd

#: 找环时允许的最大环长。**依据是实测**：
#: 标注的 `cycle` 里 99.6% 的环长落在 2–5，最长 7（见 `docs/09` 1.3）。
#: 取 6 是留一点余量，同时把成本压在可控范围。
DEFAULT_MAX_CYCLE_LENGTH = 6


@dataclass(frozen=True, slots=True)
class CycleScan:
    """一次环扫描的结果。"""

    #: `(sender, receiver)` → **最短环长**。只包含确实在短环上的边。
    shortest_by_pair: dict[tuple[str, str], int]
    #: 扫描了多少条去重边
    scanned_edges: int
    seconds: float

    def length_histogram(self) -> dict[int, int]:
        histogram: dict[int, int] = {}
        for length in self.shortest_by_pair.values():
            histogram[length] = histogram.get(length, 0) + 1
        return dict(sorted(histogram.items()))


def build_neighbor_index(
    graph: nx.MultiDiGraph[str] | nx.DiGraph[str],
) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    """建出/入邻居索引（**去重**）。

    返回冻结集合：它们会被反复求交，冻结集合的哈希在构造时就缓存好了。
    而且要保证**不会被下游代码改动**——索引一旦被改，结果就不可复现。
    """
    out_neighbors = {node: frozenset(graph.successors(node)) for node in graph.nodes()}
    in_neighbors = {node: frozenset(graph.predecessors(node)) for node in graph.nodes()}
    return out_neighbors, in_neighbors


def shortest_return_length(
    out_neighbors: dict[str, frozenset[str]],
    in_neighbors: dict[str, frozenset[str]],
    sender: str,
    receiver: str,
    max_return: int,
) -> int | None:
    """从 `receiver` 回到 `sender` 的最短跳数；找不到返回 `None`。

    **必须排除"原地打转"**：中间节点不能是 `sender` 或 `receiver` 本身，
    也不能重复——否则会算出 `u→v→u→v` 这种**不是简单环**的东西，
    把结果虚高。
    """
    outgoing = out_neighbors.get(receiver)
    incoming = in_neighbors.get(sender)
    if outgoing is None or incoming is None:
        return None

    if max_return < 0:
        return None

    # 自环：自己转自己。回程 0 跳 → **环长 1**
    if sender == receiver:
        return 0

    banned = {sender, receiver}

    # 每一档都按"它代表几跳"门控。
    # 不门控的话，`max_cycle_length=1` 仍然会返回环长 2 的结果——
    # 参数就形同虚设了。
    if max_return >= 1 and sender in outgoing:
        # 回程 1 跳 → 环长 2（互转）
        return 1

    if max_return >= 2:
        # 回程 2 跳 → 环长 3
        middle = outgoing & incoming
        if middle - banned:
            return 2

    if max_return >= 3:
        # 回程 3 跳 → 环长 4
        for first in outgoing:
            if first in banned:
                continue
            second = out_neighbors.get(first)
            if second is None:
                continue
            middle = second & incoming
            if middle - banned - {first}:
                return 3

    return None


def find_short_cycles(
    graph: nx.MultiDiGraph[str] | nx.DiGraph[str],
    *,
    max_cycle_length: int = DEFAULT_MAX_CYCLE_LENGTH,
) -> CycleScan:
    """扫描**所有去重边**，找出每条边所在的最短环。

    复杂度：每条边约 `|出邻居|^2` 次集合操作，而去重图平均出度约 7——
    所以全图是**秒级**的，不是 M3 里 `simple_cycles` 那种跑不完的量级。
    """
    started = time.monotonic()
    out_neighbors, in_neighbors = build_neighbor_index(graph)
    max_return = max_cycle_length - 1

    # ⚠️ **必须用 `is_multigraph()`，不能用 `isinstance(graph, nx.DiGraph)`。**
    #
    # `MultiDiGraph` **是 `DiGraph` 的子类**，所以 `isinstance` 对它也返回 True——
    # 于是去重被整个跳过，平行边被扫了 19 遍。
    # 实测平行边占 94.8%，这个 bug 让扫描量变成 19 倍。
    # **是 `test_parallel_edges_do_not_change_the_cycle` 把它抓出来的。**
    simple = nx.DiGraph(graph) if graph.is_multigraph() else graph
    pairs = list(simple.edges())

    shortest: dict[tuple[str, str], int] = {}
    for sender, receiver in pairs:
        hops = shortest_return_length(out_neighbors, in_neighbors, sender, receiver, max_return)
        if hops is not None:
            shortest[(sender, receiver)] = hops + 1

    return CycleScan(
        shortest_by_pair=shortest,
        scanned_edges=len(pairs),
        seconds=time.monotonic() - started,
    )


def cycle_feature(edges: pd.DataFrame, scan: CycleScan, *, min_cycle_length: int = 3) -> pd.Series:
    """把环扫描结果映射回**每一条交易**。

    `min_cycle_length=3` 是刻意的：**环长 2 不算数**。

    M3 实测（400 条抽样）：

    | 回程 | 环长 | cycle 标注 | 普通交易 | 倍数 |
    |---|---|---|---|---|
    | 1 跳 | 2 | 75 | **77** | **1.0×** ❌ |
    | 2 跳 | 3 | **122** | **3** | **40×** |
    | 3 跳 | 4 | **104** | **21** | **5×** |

    互转在这张图里太普遍（18.4% 的边），**把它算进来会淹没真正的信号**。
    ⚠️ **不在任何短环上的边必须给 0**。

    这里踩过一个坑：如果用 `scan.shortest_by_pair.get(pair, 哨兵大数) >= min_cycle_length`
    来写，**不在环上的边会因为哨兵值很大而被判成 1**——
    正好把"没有环"错判成"有环"，是**最糟的一种反向错误**。
    """
    pairs = zip(edges["sender"], edges["receiver"], strict=True)
    return pd.Series(
        [
            # 三元表达式的条件里必须先判断"有没有环"，再判断环长
            1.0
            if (length := scan.shortest_by_pair.get((s, r))) is not None
            and length >= min_cycle_length
            else 0.0
            for s, r in pairs
        ],
        index=edges.index,
    )


def shortest_cycle_feature(edges: pd.DataFrame, scan: CycleScan) -> pd.Series:
    """最短环长本身作为特征（**越小越可疑**）。

    找不到环的给一个大数（`max_cycle_length + 1`），表示"没落在任何短环上"。
    """
    fallback = DEFAULT_MAX_CYCLE_LENGTH + 1
    return pd.Series(
        [
            float(scan.shortest_by_pair.get((s, r), fallback))
            for s, r in zip(edges["sender"], edges["receiver"], strict=True)
        ],
        index=edges.index,
    )
