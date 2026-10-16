"""图谱层的集成测试：**对着真实数据与真实图库跑**。

这些测试和单元测试的分工是刻意的：

| | 单元测试 | 集成测试（本文件） |
|---|---|---|
| 依赖 | 无 | 数据集 + Neo4j |
| 速度 | 毫秒 | 秒 |
| 拦什么 | 逻辑错误 | **接线错误、真实数据与代码不一致** |
| 何时跳过 | 从不 | 数据/图库不在时**明确跳过**，不假装通过 |

**为什么两者都要有**：单元测试全绿但整条链路跑不通，是极常见的情况——
比如列名对不上、图库里的数据和代码假设的不一样。
**只有对着真东西跑一次才能发现。**
"""

from __future__ import annotations

import pytest
from neo4j import GraphDatabase

from aml_agent.data import loader
from aml_agent.graph import build, guard, queries, verify

pytestmark = pytest.mark.integration


def _driver_or_none() -> object | None:
    """连得上就返回 driver，连不上返回 None。

    用**短超时**：图库不在时，这个检查必须很快失败，
    否则整个测试收集阶段就会被拖住。
    """
    try:
        driver = GraphDatabase.driver(
            build.DEFAULT_URI,
            auth=(build.DEFAULT_USER, build.DEFAULT_PASSWORD),
            connection_timeout=3,
        )
        driver.verify_connectivity()
    except Exception:  # noqa: BLE001 - 连不上就是跳过，原因不重要
        return None
    return driver


requires_graph = pytest.mark.skipif(
    _driver_or_none() is None,
    reason="图库不可达；先 `docker compose up -d neo4j` 并跑 `scripts/build_graph.py`",
)


@requires_graph
def test_graph_counts_match_the_source_data() -> None:
    """**图的规模必须和源数据对得上。**

    这是 FR-1 的验收标准。对不上说明建图错了——
    而**建图错了不会报错**，它只会安静地产出一张不对的图，
    然后所有下游指标都建立在这个错误上。
    """
    driver = _driver_or_none()
    assert driver is not None
    try:
        dataset = loader.load_cached()
        counts = build.counts(driver)
        assert counts["accounts"] == len(dataset.accounts)
        assert counts["transfers"] == len(dataset.transactions)
        assert counts["controls"] == len(dataset.accounts)
    finally:
        driver.close()


@requires_graph
def test_graph_contains_no_label_properties() -> None:
    """**第一层防御的现场验证：图上不能有任何标签属性。**

    这个测试查的是**数据库里实际存了什么**，不是代码里写了什么。
    代码可以改对，但**库里的旧数据还在**——
    所以必须有一个测试去问数据库本身。

    这也是"重建图"这个动作的验收标准：
    如果哪天有人把标签加回 `IMPORT_TRANSFERS`，这里会立刻红。
    """
    driver = _driver_or_none()
    assert driver is not None
    try:
        names = queries.property_names(driver)
        assert names, "查不到任何属性名——连接或权限有问题"
        guard.assert_property_names_are_clean(names)
    finally:
        driver.close()


@requires_graph
def test_verify_graph_reports_all_checks_passing() -> None:
    """完整的校验流程必须全过（自环是信息项，不算失败）。"""
    driver = _driver_or_none()
    assert driver is not None
    try:
        dataset = loader.load_cached()
        checks = verify.verify_graph(driver, dataset)
        failed = [check for check in checks if not check.passed]
        assert not failed, "校验未通过：\n" + "\n".join(check.line() for check in failed)
    finally:
        driver.close()


@requires_graph
def test_safe_read_path_returns_real_edges() -> None:
    """安全读路径能取到真实数据，且字段与白名单一一对应。"""
    driver = _driver_or_none()
    assert driver is not None
    try:
        with driver.session() as session:
            record = session.run(
                "MATCH (a:Account)-[t:TRANSFER]->(b:Account) "
                "RETURN a.key AS sender, b.key AS receiver LIMIT 1"
            ).single(strict=True)
        edges = queries.transactions_between(
            driver, str(record["sender"]), str(record["receiver"]), limit=5
        )
        assert edges, "两个真实账户之间应该至少有一笔交易"
        for edge in edges:
            assert edge.sender == record["sender"]
            assert edge.receiver == record["receiver"]
            assert edge.amount >= 0
    finally:
        driver.close()


@requires_graph
def test_neighborhood_reports_truncation() -> None:
    """**截断标记必须真的会触发。**

    这里用一个很小的 limit，逼出截断——
    如果这个字段永远是 False，那它就是个摆设，
    而 Agent 会把「被截断」误读成「没有更多关联」。
    """
    driver = _driver_or_none()
    assert driver is not None
    try:
        with driver.session() as session:
            busiest = session.run(
                "MATCH (a:Account)-[t:TRANSFER]->() "
                "RETURN a.key AS key, count(t) AS n ORDER BY n DESC LIMIT 1"
            ).single(strict=True)
        result = queries.neighborhood(driver, str(busiest["key"]), hops=1, limit=5)
        assert result.truncated is True, "最忙的账户只取 5 条边，应该被截断"
        assert len(result.edges) <= 5
    finally:
        driver.close()


@requires_graph
def test_transactions_between_is_ordered_by_time() -> None:
    """同一对账户之间的多笔交易**必须按时间排序**。

    这个项目**刻意不聚合多笔转账**（`docs/04` 3.3），就是为了保住时间信息——
    不排序等于把那个设计决定浪费掉。
    """
    driver = _driver_or_none()
    assert driver is not None
    try:
        with driver.session() as session:
            pair = session.run(
                "MATCH (a:Account)-[t:TRANSFER]->(b:Account) "
                "WITH a.key AS sender, b.key AS receiver, count(t) AS n "
                "WHERE n >= 3 RETURN sender, receiver LIMIT 1"
            ).single(strict=True)
        edges = queries.transactions_between(
            driver, str(pair["sender"]), str(pair["receiver"]), limit=50
        )
        timestamps = [edge.timestamp for edge in edges]
        assert timestamps == sorted(timestamps), f"时间未排序：{timestamps}"
    finally:
        driver.close()


@requires_graph
def test_loading_edges_from_the_graph_has_no_label_column() -> None:
    """从图里拉出来的边表**不能有标签列**。

    评测需要的标签从 CSV 读，不经过图。这里验证那条路径没有被改回去。

    ⚠️ 用 `use_cache=False` **是刻意的**：要验证的是**那条查询本身**能跑、
    且返回的列符合契约——用缓存就变成在验缓存了。
    代价是这一个测试要 40 多秒（要从图库拉 132 万行）。
    **这是集成档，正确性优先于速度。**
    """
    from aml_agent.graph import load as graph_load

    edges = graph_load.load_edges(use_cache=False)

    # 最重要的一条：标签不在图里，也就不可能出现在边表里
    assert "isFraud" not in edges.columns
    assert "IS_FRAUD" not in edges.columns

    # 列必须**正好**是契约里声明的那几个——多一个少一个都要发现
    assert set(edges.columns) == set(graph_load.EDGE_COLUMNS), (
        f"边表列与契约不符：{sorted(edges.columns)}"
    )
