"""把图从 Neo4j 取出来，变成 NetworkX 图。

**这是架构里"取子图"那一层的实现**（见 `docs/03-概要设计` §5 的分工）：
Neo4j 负责存储与查询，算法在 NetworkX 里跑。

## 为什么要缓存

从 Neo4j 拉 132 万条边要 **63 秒**。而调算法参数、做对照实验会反复用同一张图——
每次都重拉会让所有实验变得不可忍受。

缓存的是**边表**（Parquet），不是 NetworkX 对象：

- Parquet 可压缩、跨版本稳定，而且**能直接丢给 pandas 做统计**；
- NetworkX 对象用 pickle 存的话，版本一升就可能读不出来。

## 缓存目录为什么可配置

默认写在 `<项目>/data/cache`。但本机沙箱不允许**子进程**写项目目录，
所以这里允许用环境变量 `AML_CACHE_DIR` 指到别处。
**这不是为了本机方便，而是"派生数据的位置不该写死在代码里"这条原则的体现。**
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import networkx as nx
import pandas as pd

from aml_agent.graph import connect as _connect

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_FILE = "edges.parquet"

#: 从图里拉出来的**边表的列名**。
#:
#: ⚠️ **刻意和 CSV 的列名不同，这是有意的：**
#:
#: | 来源 | 命名风格 | 例子 |
#: |---|---|---|
#: | CSV（数据集原始列名） | `UPPER_CASE` | `TX_ID`、`TX_AMOUNT` |
#: | 图（Cypher 别名） | `camelCase` | `txId`、`amount` |
#:
#: 两套名字各自贴合自己的来源——把 CSV 的名字硬搬到 Cypher 别名上，
#: 或者反过来，都会让某一侧变得别扭。
#:
#: **代价是 join 时要显式映射**（`txId` ↔ `TX_ID`）。
#: 所以这里把列名**声明成常量**，让"契约"只有一个出处，
#: 而不是散在代码里靠人记得。
EDGE_COLUMNS: tuple[str, ...] = ("sender", "receiver", "txId", "amount", "timestamp")

#: 拉边的查询。
#:
#: ⚠️ **这里曾经有 `t.isFraud AS isFraud`** —— 那是标签，而且它已经不在图上了
#: （见 `docs/04` 4.1.1：图是特征存储，不是答案存储）。
#:
#: **它是被自动审计抓出来的**：审计一开始只扫了 `cypher` 和 `queries` 两个模块，
#: 而**没扫到的模块就是漏洞**——这正是「手写清单会漏」的同一个错误，
#: 只是换了个层次。现在审计覆盖全部持有查询的模块。
#:
#: 评测需要的标签**从 CSV 读**（`data/loader.py` 的 `alerts.csv`），不经过图。
FETCH_EDGES = """
MATCH (s:Account)-[t:TRANSFER]->(r:Account)
RETURN s.key AS sender, r.key AS receiver, t.txId AS txId,
       t.amount AS amount, t.timestamp AS timestamp
"""


def cache_dir() -> Path:
    override = os.environ.get("AML_CACHE_DIR")
    return Path(override) if override else DEFAULT_CACHE_DIR


def fetch_edges() -> pd.DataFrame:
    """从 Neo4j 拉全部边（含标签）。约 60 秒。"""
    driver = _connect()
    try:
        with driver.session() as session:
            records = session.run(FETCH_EDGES).data()
    finally:
        driver.close()
    frame = pd.DataFrame(records)
    # txId 是评测的 join key，必须能唯一定位一条边
    return frame.astype({"txId": "int64", "amount": "float64", "timestamp": "int64"})


def load_edges(*, use_cache: bool = True, refresh: bool = False) -> pd.DataFrame:
    """取边表，优先用缓存。"""
    path = cache_dir() / CACHE_FILE
    if use_cache and not refresh and path.is_file():
        return pd.read_parquet(path)

    started = time.monotonic()
    frame = fetch_edges()
    elapsed = time.monotonic() - started

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        note = f"，已缓存到 {path}"
    except Exception as exc:  # noqa: BLE001
        # 缓存写不进去**不该让整个流程失败**——它只是优化。
        #
        # 这里刻意接宽泛异常：写缓存可能因为磁盘满、缺 pyarrow、
        # 权限不足、路径过长等**各种**原因失败，而它们都不该中断主流程。
        # 真实踩到的是 ImportError（没装 pyarrow）——当时只接了 OSError，没接住。
        note = f"，缓存未写入（{type(exc).__name__}: {exc}）"
    print(f"  从 Neo4j 拉取 {len(frame):,} 条边，用时 {elapsed:.1f} 秒{note}")
    return frame


def to_multidigraph(edges: pd.DataFrame) -> nx.MultiDiGraph[str]:
    """边表 → NetworkX 多重有向图。

    **用 MultiDiGraph 而不是 DiGraph**：同一对账户之间有多笔转账，
    **保留每一条**是我们的建模决策（`docs/04` 3.3）——压成一条会丢掉时间和笔数。

    节点是字符串（`amlsim:123`），所以显式写成 `MultiDiGraph[str]`——
    mypy `--strict` 要求泛型参数，而**显式写出来比让它推断更可靠**。
    """
    graph: nx.MultiDiGraph[str] = nx.MultiDiGraph()
    graph.add_edges_from(
        zip(
            edges["sender"].to_numpy(),
            edges["receiver"].to_numpy(),
            edges["txId"].to_numpy(),
            strict=True,
        )
    )
    return graph


def load_graph(
    *, use_cache: bool = True, refresh: bool = False
) -> tuple[nx.MultiDiGraph[str], pd.DataFrame]:
    """一次拿到图和边表。边表用于评测（它带标签），图用于跑算法。"""
    edges = load_edges(use_cache=use_cache, refresh=refresh)
    return to_multidigraph(edges), edges
