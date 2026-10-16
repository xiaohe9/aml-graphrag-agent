"""标签泄漏防护的测试。

## 这些测试守的是什么

数据集给的答案（`IS_FRAUD`）**如果出现在 Agent 能看到的地方**，评测就失去意义。
而**最坏的地方在于分数会很漂亮，很难发现**——指标不会自己说"我作弊了"。

## 三层防御，每一层都有测试

| 层 | 测试 |
|---|---|
| ① 不存（图里没有标签） | `test_import_statement_does_not_write_the_label` |
| ② 白名单（RETURN 从白名单生成） | `test_generated_queries_only_reference_whitelisted` |
| ③ 自动审计（扫描所有查询） | **`test_every_query_in_the_codebase_passes_the_guard`** ⭐ |
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import aml_agent.graph as graph_package
from aml_agent.graph import cypher, guard, load, queries

#: **审计范围。**
#:
#: ⚠️ 这个列表就是安全边界——**没扫到的模块就是漏洞**。
#: 它必须覆盖**每一个定义了 Cypher 查询的模块**，
#: 而 `test_audit_covers_every_module_that_defines_queries` 会反过来验证这一点
#: （遍历包内所有模块，找出有查询却没被列进来的）。
#:
#: 覆盖范围**自动验证**，而不是靠人记得更新——这是这个设计的关键。
MODULES_WITH_QUERIES = (cypher, queries, load)


def _all_queries() -> dict[str, str]:
    merged: dict[str, str] = {}
    for module in MODULES_WITH_QUERIES:
        prefix = module.__name__.rsplit(".", 1)[-1]
        for name, text in guard.collect_queries(module).items():
            merged[f"{prefix}.{name}"] = text
    return merged


# --------------------------------------------------------------------------- #
#  1. 防护本身能不能抓住泄漏
# --------------------------------------------------------------------------- #


def test_catches_returning_a_whole_relationship() -> None:
    """**这正是当初的漏洞形态**：一条 `RETURN t` 就把整个实体带出去了。"""
    with pytest.raises(guard.LabelLeakError, match="结构性泄漏风险"):
        guard.assert_query_safe("MATCH (a)-[t:TRANSFER]->(b) RETURN t")


def test_catches_returning_a_whole_node() -> None:
    with pytest.raises(guard.LabelLeakError, match="结构性泄漏风险"):
        guard.assert_query_safe("MATCH (n:Account) RETURN n LIMIT 10")


def test_catches_return_star() -> None:
    with pytest.raises(guard.LabelLeakError, match="结构性泄漏风险"):
        guard.assert_query_safe("MATCH (a)-[t:TRANSFER]->(b) RETURN *")


def test_catches_the_label_by_name() -> None:
    """即使显式写出属性名，也要拦住。"""
    for name in ("isFraud", "IS_FRAUD", "is_laundering", "groundTruth", "label"):
        with pytest.raises(guard.LabelLeakError):
            guard.assert_query_safe(
                f"MATCH (a)-[t:TRANSFER]->(b) RETURN t.txId AS id, t.{name} AS x"
            )


def test_allows_a_properly_whitelisted_query() -> None:
    """正常查询不能被误伤——误伤会让人把检查关掉，那比没有检查更糟。"""
    guard.assert_query_safe(
        "MATCH (a)-[t:TRANSFER]->(b) "
        "RETURN a.key AS sender, b.key AS receiver, t.txId AS txId, t.amount AS amount"
    )


def test_structural_allowlist_is_explicit() -> None:
    """例外必须**显式列出并写明理由**，不能靠放宽检查。"""
    for name, reason in cypher.STRUCTURAL_ALLOWLIST.items():
        assert reason.strip(), f"{name} 的例外理由不能为空"
        assert hasattr(cypher, name), f"{name} 不在 cypher 模块里"


# --------------------------------------------------------------------------- #
#  2. ★ 自动审计：代码库里所有查询都要过检查
# --------------------------------------------------------------------------- #


def test_every_query_in_the_codebase_passes_the_guard() -> None:
    """**这是第三层防御的核心测试。**

    它自动发现**每个持有查询的模块**里的全部查询常量
    （而不是靠手写清单——**手写清单会漏掉新增的查询，而漏掉的那条就是漏洞**），
    逐条过检查。

    ⚠️ **覆盖范围本身就是安全边界。**
    这个测试第一次写的时候只扫了 `cypher` 和 `queries`，
    结果漏掉了 `load.py` 里那条**还在读 `t.isFraud`** 的查询——
    **没扫到的模块就是漏洞**，这和"手写清单会漏"是同一个错误，只是换了个层次。
    """
    all_queries = _all_queries()
    assert all_queries, "没扫到任何查询——扫描逻辑坏了，这个测试就形同虚设"

    problems = guard.audit_queries(
        all_queries, allow_structural=frozenset(cypher.STRUCTURAL_ALLOWLIST)
    )
    assert not problems, "有查询未通过标签泄漏检查：\n" + "\n".join(problems)


def test_audit_covers_every_module_that_defines_queries() -> None:
    """**防止审计范围悄悄变窄。**

    做法是反过来查：遍历 `aml_agent.graph` 包下的**每一个模块**，
    找出其中定义了查询的，再确认它们**都在** `MODULES_WITH_QUERIES` 里。

    没有这个测试，将来有人新建一个模块写查询、却忘了加进审计范围，
    **上面那个测试仍然会绿**——而漏洞就在那个新模块里。
    """
    audited = {module.__name__ for module in MODULES_WITH_QUERIES}
    missed: list[str] = []
    for info in pkgutil.iter_modules(graph_package.__path__):
        module_name = f"{graph_package.__name__}.{info.name}"
        module = importlib.import_module(module_name)
        if guard.collect_queries(module) and module_name not in audited:
            missed.append(module_name)

    assert not missed, (
        f"这些模块定义了查询但没被审计：{missed}。\n"
        "把它们加进 MODULES_WITH_QUERIES——**没扫到的模块就是漏洞**。"
    )


def test_the_audit_actually_scans_a_meaningful_number_of_queries() -> None:
    """防止"扫描逻辑坏了但测试还绿"这种假安全。"""
    all_queries = _all_queries()
    assert len(all_queries) >= 10, f"只扫到 {len(all_queries)} 条查询，像是漏扫了"


def test_audit_reports_problems_instead_of_silently_passing() -> None:
    """审计函数本身要能报出问题——否则它只是个永远返回空的摆设。"""
    problems = guard.audit_queries({"BAD": "MATCH (a)-[t]->(b) RETURN t"})
    assert problems and "BAD" in problems[0]


# --------------------------------------------------------------------------- #
#  3. 第一层：图上根本不该有标签
# --------------------------------------------------------------------------- #


def test_import_statement_does_not_write_the_label() -> None:
    """建图语句里不能出现标签——**最强的防御是不把秘密放进去**。"""
    assert "isFraud" not in cypher.IMPORT_TRANSFERS
    assert "IS_FRAUD" not in cypher.IMPORT_TRANSFERS
    # 但要保留非标签的派生标记
    assert "isSentinelAmount" in cypher.IMPORT_TRANSFERS


def test_assert_property_names_are_clean_raises_for_labels() -> None:
    for dirty in (["isFraud", "amount"], ["txId", "IS_FRAUD"], ["ground_truth"]):
        with pytest.raises(guard.LabelLeakError, match="图上存在标签属性"):
            guard.assert_property_names_are_clean(dirty)


def test_assert_property_names_are_clean_accepts_our_real_schema() -> None:
    """我们实际会写进图的属性名必须全部通过。"""
    guard.assert_property_names_are_clean(
        [
            "key",
            "accountId",
            "initialBalance",
            "institution",
            "txId",
            "amount",
            "timestamp",
            "isSentinelAmount",
            "name",
        ]
    )


# --------------------------------------------------------------------------- #
#  4. 第二层：白名单
# --------------------------------------------------------------------------- #


def test_whitelist_contains_no_forbidden_names() -> None:
    """白名单自己是干净的——它是安全边界，边界被污染就全完了。"""
    guard.assert_property_names_are_clean(queries.ALLOWED_ACCOUNT_PROPERTIES)
    guard.assert_property_names_are_clean(queries.ALLOWED_TRANSFER_PROPERTIES)


def test_generated_queries_only_reference_whitelisted() -> None:
    """生成的 `RETURN` 子句里，除了白名单字段和节点标识，不该有别的东西。"""
    for hops in (1, 2, 3):
        query = queries._edge_query(hops)
        for name in queries.ALLOWED_TRANSFER_PROPERTIES:
            assert f"t.{name}" in query
        assert "isFraud" not in query
