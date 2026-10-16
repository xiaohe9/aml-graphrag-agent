"""把三张 CSV 建成 Neo4j 图。

**四个关键决定，每个都会影响性能或正确性：**

### 1. 批量导入，不是逐条 CREATE

132 万条边如果逐条发请求，就是 132 万次网络往返。
用 `UNWIND $rows` 每批 5000 行，往返降到约 265 次——**差三个数量级**。

### 2. 约束先于导入

唯一约束会顺带建索引，让后面每一条 `MERGE` 走索引查找而不是全图扫描。
所以 `SETUP_STATEMENTS` 必须在导入之前跑。

### 3. 账户用 MERGE，交易用 CREATE

- **账户是实体**——它在数据里出现多次时指的是同一个东西，导入必须幂等；
- **交易是事件**——两笔金额相同的转账是两件事，合并会丢掉时间信息和笔数。

但这也带来一个后果：**重复导入会让交易翻倍**。
所以 `build_graph()` 会**先检查图是不是空的，非空就报错**，而不是静默地导成两倍。
（静默翻倍会让后面所有指标失真，而且很难查。）

### 4. 哨兵金额保留原值，只加标记

实测 3,053 笔金额等于 `2**31 / 100`，明显是整数溢出。
但**不改原始值**——否则以后说不清"这个数字是数据里的还是我们造的"。
只加一个 `isSentinelAmount` 标记，由使用方决定要不要排除。
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from neo4j import Driver, GraphDatabase

from aml_agent.data import schema
from aml_agent.data.loader import Dataset
from aml_agent.graph import cypher

DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"
# 本地图库的开发凭据，与 docker-compose.yml 里的一致。
# 它不保护任何真实数据——图是公开数据集派生出来的、随时可以重建。
# 硬编码密码在这一处是**正确**的（本地开发约定，不是机密），所以显式豁免 S105。
# ⚠️ 注意：这句话不能以 "noqa" 开头——ruff 会把注释里的它当成真的 noqa 指令，
#    然后报「未使用的 noqa」。这个坑真的踩到了。
DEFAULT_PASSWORD = "aml-agent-dev"  # noqa: S105

#: 每批行数。5000 是个折中：太小则往返次数多，太大则单次事务内存压力大。
DEFAULT_BATCH_SIZE = 5_000

#: 数据里没有机构维度（见 `docs/08-数据规格` Q7）。
#: 这里仍然写入 `institution` 属性，是为了**让跨机构的数据将来能直接进来**——
#: 键的设计要能容纳未来的维度。
INSTITUTION = "amlsim"


class GraphNotEmptyError(RuntimeError):
    """图里已经有数据了。

    **这是一个刻意的拦截。** 交易边用 CREATE 导入，重复导入会直接翻倍——
    而翻倍后的图会让所有指标失真，且**很难从结果上看出来**。
    宁可报错让人显式决定，也不要静默地污染数据。
    """


@dataclass(frozen=True, slots=True)
class BuildStats:
    """一次建图的结果。数字要和源数据对得上，对不上就是建错了。"""

    accounts: int
    entities: int
    transfers: int
    controls: int
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "accounts": self.accounts,
            "entities": self.entities,
            "transfers": self.transfers,
            "controls": self.controls,
            "seconds": round(self.seconds, 1),
        }


def connect(
    uri: str = DEFAULT_URI,
    user: str = DEFAULT_USER,
    password: str = DEFAULT_PASSWORD,
) -> Driver:
    """连上图库。凭据与 `docker-compose.yml` 里的一致。

    **关掉服务端通知**（`notifications_min_severity="OFF"`）是刻意的：
    空图上查 `MATCH (a:Account)` 会返回「标签不存在」的警告——
    这在建图前的空库检查里**必然发生**，而它会刷屏几十行。
    第一次跑这个项目的人看到满屏警告，会以为出错了。
    """
    return GraphDatabase.driver(uri, auth=(user, password), notifications_min_severity="OFF")


def _batched[T](rows: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _run_batches(driver: Driver, statement: str, rows: Sequence[dict[str, Any]], size: int) -> None:
    with driver.session() as session:
        for batch in _batched(rows, size):
            session.run(statement, rows=list(batch))


def counts(driver: Driver) -> dict[str, int]:
    """图里现在有多少东西。校验脚本和建图脚本都用它。"""
    with driver.session() as session:
        return {
            "accounts": session.run(cypher.COUNT_ACCOUNTS).single(strict=True)["n"],
            "entities": session.run(cypher.COUNT_ENTITIES).single(strict=True)["n"],
            "transfers": session.run(cypher.COUNT_TRANSFERS).single(strict=True)["n"],
            "controls": session.run(cypher.COUNT_CONTROLS).single(strict=True)["n"],
        }


def clear(driver: Driver) -> None:
    """清空图。

    ⚠️ **不能用"循环 + LIMIT"来分批。** 第一版写的是
    `MATCH (n) WITH n LIMIT 10000 DETACH DELETE n`，实测报内存错误——
    因为**"1 万个节点"不等于"1 万次删除"**：平均每个节点挂 264 条边，
    删 1 万个节点实际是**在一笔事务里删 130 万条关系**。

    **分批的单位是「事务」，不是「循环」。**
    用 Neo4j 5 的 `CALL ... IN TRANSACTIONS` 让每批各自提交，
    内存占用才有真正的上界。见 `cypher.CLEAR_GRAPH`。
    """
    with driver.session() as session:
        session.run(cypher.CLEAR_GRAPH)


def setup(driver: Driver) -> None:
    """建约束与索引。**必须在导入之前**——约束顺带建的索引会让 MERGE 快很多。"""
    with driver.session() as session:
        for statement in cypher.SETUP_STATEMENTS:
            session.run(statement)


def _account_rows(dataset: Dataset) -> list[dict[str, Any]]:
    accounts = dataset.accounts
    return [
        {
            # 键是字符串，不是整数——见 docs/04 第 3.1 节：
            # 将来接入跨机构数据时，键要能扩成 `{机构}:{账号}`，
            # 而这个扩展不该改数据库 schema。
            "key": f"{INSTITUTION}:{row[schema.ACCOUNT_ID]}",
            "accountId": int(row[schema.ACCOUNT_ID]),
            "initialBalance": float(row[schema.INIT_BALANCE]),
            "institution": INSTITUTION,
        }
        for _, row in accounts.iterrows()
    ]


def _entity_rows(dataset: Dataset) -> list[dict[str, Any]]:
    accounts = dataset.accounts
    return [
        {
            "key": f"{INSTITUTION}:{row[schema.ACCOUNT_ID]}",
            "entity": str(row[schema.CUSTOMER_ID]),
        }
        for _, row in accounts.iterrows()
    ]


def _transfer_rows(dataset: Dataset) -> list[dict[str, Any]]:
    transactions = dataset.transactions
    return [
        {
            "txId": int(row[schema.TX_ID]),
            "sender": f"{INSTITUTION}:{row[schema.SENDER]}",
            "receiver": f"{INSTITUTION}:{row[schema.RECEIVER]}",
            "amount": float(row[schema.TX_AMOUNT]),
            "timestamp": int(row[schema.TIMESTAMP]),
            # 只标记，不改值——原始数据保持原样
            "isSentinelAmount": bool(row[schema.TX_AMOUNT] >= schema.SENTINEL_AMOUNT),
            # ⚠️ 标签（IS_FRAUD）**刻意不写进来**。
            #
            # 图是**特征存储**，不是**答案存储**。曾经把 isFraud 写在边上，
            # 结果任何 `RETURN t` 都会把答案交给 Agent——而 Agent 一旦"看见答案"，
            # 后面所有评测都失去意义，**而且分数会很漂亮、很难发现**。
            #
            # 评测需要标签时从 CSV 读（`data/loader.py`），不经过图。
        }
        for _, row in transactions.iterrows()
    ]


def build_graph(
    driver: Driver,
    dataset: Dataset,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    reset: bool = False,
    progress: bool = True,
) -> BuildStats:
    """建图。非空且未指定 `reset` 时**报错**，不静默重复导入。"""
    started = time.monotonic()

    if reset:
        if progress:
            print("清空已有图…")
        clear(driver)

    existing = counts(driver)
    total_existing = sum(existing.values())
    if total_existing:
        raise GraphNotEmptyError(
            f"图里已经有 {total_existing:,} 个对象（{existing}）。"
            "交易边用 CREATE 导入，重复导入会让它翻倍，而翻倍后的指标是失真的。\n"
            "请显式选择：build_graph(..., reset=True) 清空重建，或者换一个空库。"
        )

    if progress:
        print("建约束与索引…")
    setup(driver)

    accounts = _account_rows(dataset)
    entities = _entity_rows(dataset)
    transfers = _transfer_rows(dataset)

    steps: Iterable[tuple[str, str, Sequence[dict[str, Any]]]] = (
        ("账户节点", cypher.IMPORT_ACCOUNTS, accounts),
        ("实体与控制关系", cypher.IMPORT_ENTITIES, entities),
        ("交易边", cypher.IMPORT_TRANSFERS, transfers),
    )
    for label, statement, rows in steps:
        if progress:
            print(f"  导入{label}：{len(rows):,} 行…")
        _run_batches(driver, statement, rows, batch_size)

    final = counts(driver)
    return BuildStats(
        accounts=final["accounts"],
        entities=final["entities"],
        transfers=final["transfers"],
        controls=final["controls"],
        seconds=time.monotonic() - started,
    )
