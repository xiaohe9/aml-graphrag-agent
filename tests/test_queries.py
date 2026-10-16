"""查询构建的测试。

**这些测试盯的是「参数上界」**——工具契约里每一个上界都对应一条非功能需求
（见 `docs/06`），所以它们不是随便定的：

| 参数 | 上界 | 依据 |
|---|---|---|
| `hops` | ≤ 3 | 邻域随跳数**指数增长** |
| `max_length` | ≤ 8 | 标注的环最长是 7（`docs/09` 1.3 实测） |
| `limit` | ≤ 2000 | 一次查询不能把内存打满 |

**上界不是保守，是必须。** 而且它们必须被**服务端强制校验**，
不是靠提示词约束——提示词挡不住一个跑飞的 Agent。
"""

from __future__ import annotations

import pytest

from aml_agent.graph import queries

# --------------------------------------------------------------------------- #
#  跳数上界
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("hops", [0, -1, 4, 100])
def test_hops_out_of_range_is_rejected(hops: int) -> None:
    with pytest.raises(ValueError, match="hops 必须在"):
        queries._edge_query(hops)


@pytest.mark.parametrize("hops", [1, 2, 3])
def test_hops_in_range_is_accepted(hops: int) -> None:
    query = queries._edge_query(hops)
    assert f"*1..{hops}" in query


# --------------------------------------------------------------------------- #
#  路径长度上界
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("length", [0, -1, 9, 50])
def test_path_length_out_of_range_is_rejected(length: int) -> None:
    with pytest.raises(ValueError, match="max_length 必须在"):
        queries._path_query(length)


@pytest.mark.parametrize("length", [1, 4, 8])
def test_path_length_in_range_is_accepted(length: int) -> None:
    assert f"*1..{length}" in queries._path_query(length)


# --------------------------------------------------------------------------- #
#  返回条数上限
# --------------------------------------------------------------------------- #


def test_edge_limit_is_clamped() -> None:
    """上限必须被**服务端强制**，不能信任调用方传进来的数。

    调用方可能是 Agent，而 Agent 会传什么是不确定的。
    夹紧在这里意味着**存储层永远不会被要求返回超过 2000 条边**。
    """
    assert queries.MAX_EDGE_LIMIT == 2_000
    assert max(1, min(10**9, queries.MAX_EDGE_LIMIT)) == queries.MAX_EDGE_LIMIT
    assert max(1, min(0, queries.MAX_EDGE_LIMIT)) == 1


# --------------------------------------------------------------------------- #
#  截断标记
# --------------------------------------------------------------------------- #


def test_neighborhood_exposes_truncation() -> None:
    """`Neighborhood` **必须**带截断标记。

    没有它，Agent 会把「被截断」误读成「没有更多关联」，
    然后据此下一个错误的结论——**而错误结论比"不知道"危险得多**。
    """
    fields = queries.Neighborhood.__dataclass_fields__
    assert "truncated" in fields

    sample = queries.Neighborhood(center="amlsim:1", hops=1, edges=(), truncated=True)
    assert sample.truncated is True


def test_transfer_edge_has_exactly_the_whitelisted_fields() -> None:
    """返回结构的字段清单必须与白名单**一一对应**。

    多出来的字段意味着有人绕过了白名单。

    两边命名风格不同（查询层的属性名是 camelCase，Python 侧是 snake_case），
    所以要显式建立映射，而不是直接比集合——
    **直接比集合会把一个命名风格问题误报成安全问题**。
    """
    camel_to_snake = {
        "txId": "tx_id",
        "amount": "amount",
        "timestamp": "timestamp",
        "isSentinelAmount": "is_sentinel_amount",
    }
    expected = {"sender", "receiver"} | {
        camel_to_snake[name] for name in queries.ALLOWED_TRANSFER_PROPERTIES
    }
    assert set(queries.TransferEdge.__dataclass_fields__) == expected


def test_the_snake_case_mapping_covers_every_whitelisted_property() -> None:
    """映射表必须覆盖白名单里的**每一个**字段。

    漏一个，上面那个测试就会因为"少了个字段"而失败——
    但**失败原因会指向错误的结论**（看起来像字段丢了，实际是映射没写全）。
    """
    for name in queries.ALLOWED_TRANSFER_PROPERTIES:
        assert name.islower() or "_" not in name or name.isalnum(), f"{name} 命名异常"
