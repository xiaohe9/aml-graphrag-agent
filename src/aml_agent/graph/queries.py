"""**唯一**允许从图里读数据的地方。

## 为什么要有这样一个模块

图上的属性会变，开发者会写新查询。如果每个调用方自己拼 Cypher，
**迟早有人写出 `RETURN t`**——而 `t` 身上有什么，取决于当时图里存了什么。

**所以把读路径收敛到一个模块**：这里所有查询的 `RETURN` 子句都**从白名单生成**，
而不是手写。手写会写错，生成不会。

## 三层防御的位置

| 层 | 在本项目哪里 |
|---|---|
| ① 不存 | `cypher.IMPORT_TRANSFERS`——标签**不写进图** |
| **② 白名单** | **本模块**——`RETURN` 从 `ALLOWED_*_PROPERTIES` 生成 |
| ③ 自动检查 | `guard.assert_query_safe()`——本模块每条查询执行前都过一遍 |

## 一个 Cypher 细节

变长路径的上界 `*1..n` 在 Cypher 里**必须是字面量，不能是参数**。
所以这里把校验过的整数拼进字符串。

**这是安全的**，因为它是 `int` 且先经过范围校验——不是用户输入的文本。

> **参数化用于「值」，校验过的字面量用于「结构」**——这是查询构建的通用原则。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neo4j import Driver

from aml_agent.graph import guard

# --------------------------------------------------------------------------- #
#  白名单
# --------------------------------------------------------------------------- #

#: `Account` 节点上允许读出的属性。
ALLOWED_ACCOUNT_PROPERTIES: tuple[str, ...] = (
    "key",
    "accountId",
    "initialBalance",
    "institution",
)

#: `TRANSFER` 关系上允许读出的属性。
#:
#: ⚠️ **这个元组就是安全边界。** 往里加字段前先问一句：
#: 「这个字段能出现在 Agent 的上下文里吗？」
ALLOWED_TRANSFER_PROPERTIES: tuple[str, ...] = (
    "txId",
    "amount",
    "timestamp",
    "isSentinelAmount",
)

#: 变长路径的跳数上界。对应工具契约里的 `hops ≤ 3`（见 `docs/06`）。
MAX_HOPS = 3

#: 找路径的长度上界。对应工具契约里的 `max_length ≤ 8`。
#: 这个 8 现在**有实测依据**：标注的环最长是 7（见 `docs/09` 1.3）。
MAX_PATH_LENGTH = 8

#: 单次查询返回的边数上限。**上界不是保守，是必须**——邻域随跳数指数增长，
#: 没有上限的话一次查询就能把内存打满。
DEFAULT_EDGE_LIMIT = 500
MAX_EDGE_LIMIT = 2_000


# --------------------------------------------------------------------------- #
#  返回结构
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TransferEdge:
    """一条交易边。**字段清单与白名单一一对应**——多一个都不可能。"""

    sender: str
    receiver: str
    tx_id: int
    amount: float
    timestamp: int
    is_sentinel_amount: bool


@dataclass(frozen=True, slots=True)
class Neighborhood:
    """一个账户的邻域侦查结果。"""

    center: str
    hops: int
    edges: tuple[TransferEdge, ...]
    #: 是否因为触到上限而被截断。
    #:
    #: ⚠️ **这个字段是必须的**：不区分「确认没有」和「没查完」，
    #: Agent 会把"被截断"误读成"没有更多关联"，然后据此下结论。
    truncated: bool


# --------------------------------------------------------------------------- #
#  查询构建
# --------------------------------------------------------------------------- #


def _transfer_return_clause(alias: str = "t") -> str:
    """从白名单**生成** RETURN 子句。

    生成而不是手写——手写会写错，生成不会。
    这个函数是全模块的安全核心。
    """
    return ",\n    ".join(f"{alias}.{name} AS {name}" for name in ALLOWED_TRANSFER_PROPERTIES)


def _edge_query(hops: int) -> str:
    """邻域侦查：从中心账户出发，沿出边扩张 `hops` 跳。"""
    if not 1 <= hops <= MAX_HOPS:
        raise ValueError(f"hops 必须在 1..{MAX_HOPS} 之间，收到 {hops}")

    query = f"""
MATCH (origin:Account {{key: $key}})
MATCH path = (origin)-[:TRANSFER*1..{hops}]->(:Account)
WITH relationships(path) AS rels
LIMIT $limit
UNWIND rels AS t
RETURN DISTINCT
    startNode(t).key AS sender,
    endNode(t).key AS receiver,
    {_transfer_return_clause("t")}
"""
    guard.assert_query_safe(query)
    return query


def _between_query() -> str:
    query = f"""
MATCH (s:Account {{key: $sender}})-[t:TRANSFER]->(r:Account {{key: $receiver}})
RETURN
    s.key AS sender,
    r.key AS receiver,
    {_transfer_return_clause("t")}
ORDER BY t.timestamp
LIMIT $limit
"""
    guard.assert_query_safe(query)
    return query


def _path_query(max_length: int) -> str:
    """两点之间的最短路径。"""

    if not 1 <= max_length <= MAX_PATH_LENGTH:
        raise ValueError(f"max_length 必须在 1..{MAX_PATH_LENGTH} 之间，收到 {max_length}")

    query = f"""
MATCH (source:Account {{key: $source}}), (target:Account {{key: $target}})
MATCH path = shortestPath((source)-[:TRANSFER*1..{max_length}]->(target))
WITH relationships(path) AS rels
UNWIND rels AS t
RETURN
    startNode(t).key AS sender,
    endNode(t).key AS receiver,
    {_transfer_return_clause("t")}
"""
    guard.assert_query_safe(query)
    return query


# --------------------------------------------------------------------------- #
#  读接口
# --------------------------------------------------------------------------- #


def _rows_to_edges(rows: list[dict[str, Any]]) -> tuple[TransferEdge, ...]:
    """行 → 强类型边。

    这里按**显式字段**构造，而不是 `TransferEdge(**row)`——
    万一查询多返回了一个字段，`**row` 会静默地接受它（或者报一个难懂的错），
    而显式列出字段会让"多出来的东西"无处可去。
    """
    return tuple(
        TransferEdge(
            sender=str(row["sender"]),
            receiver=str(row["receiver"]),
            tx_id=int(row["txId"]),
            amount=float(row["amount"]),
            timestamp=int(row["timestamp"]),
            is_sentinel_amount=bool(row["isSentinelAmount"]),
        )
        for row in rows
    )


def neighborhood(
    driver: Driver,
    key: str,
    *,
    hops: int = 1,
    limit: int = DEFAULT_EDGE_LIMIT,
) -> Neighborhood:
    """侦查一个账户的邻域（沿出边）。

    `truncated` 会告诉我们结果是不是被截断了——**Agent 必须能区分
    「确认没有」和「没查完」**，否则会把后者当成前者下结论。
    """
    capped = max(1, min(limit, MAX_EDGE_LIMIT))
    query = _edge_query(hops)
    with driver.session() as session:
        # 多取一条用来判断是否截断
        rows = session.run(query, key=key, limit=capped + 1).data()

    truncated = len(rows) > capped
    return Neighborhood(
        center=key,
        hops=hops,
        edges=_rows_to_edges(rows[:capped]),
        truncated=truncated,
    )


def transactions_between(
    driver: Driver,
    sender: str,
    receiver: str,
    *,
    limit: int = DEFAULT_EDGE_LIMIT,
) -> tuple[TransferEdge, ...]:
    """两个账户之间的全部交易，**按时间排序**。

    按时间排序是必须的：这个项目**刻意不聚合多笔转账**（`docs/04` 3.3），
    就是为了保住时间信息——不排序等于把那个设计决定浪费掉。
    """
    capped = max(1, min(limit, MAX_EDGE_LIMIT))
    with driver.session() as session:
        rows = session.run(_between_query(), sender=sender, receiver=receiver, limit=capped).data()
    return _rows_to_edges(rows)


def shortest_path(
    driver: Driver,
    source: str,
    target: str,
    *,
    max_length: int = MAX_PATH_LENGTH,
) -> tuple[TransferEdge, ...]:
    """两点之间的最短资金路径。

    找不到时返回**空元组**——调用方要把它解释成「在给定长度内没有路径」，
    **不能**解释成「两者没有关联」。
    """
    with driver.session() as session:
        rows = session.run(_path_query(max_length), source=source, target=target).data()
    return _rows_to_edges(rows)


def property_names(driver: Driver) -> tuple[str, ...]:
    """图上实际存在的**所有属性名**（节点 + 关系）。

    用于第 ① 层的验证：建完图之后对着数据库问「你身上有哪些属性」，
    再对照禁用清单。**代码可以改对，但库里的旧数据还在**——
    这个检查能发现"图是用旧版本建的"。
    """
    from aml_agent.graph import cypher

    with driver.session() as session:
        node_names = [str(record["property"]) for record in session.run(cypher.NODE_PROPERTY_NAMES)]
        rel_names = [
            str(record["property"]) for record in session.run(cypher.RELATIONSHIP_PROPERTY_NAMES)
        ]
    return tuple(sorted(set(node_names) | set(rel_names)))
