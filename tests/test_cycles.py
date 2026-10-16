"""环检测的测试。

**这些测试锁住三件容易搞错的事：**

1. **"原地打转"不算环**——`u→v→u→v` 不是简单环，把它算进去会让结果虚高；
2. **不在环上的边必须给 0**——这里踩过一个坑：用哨兵大数比较，
   会把"没有环"错判成"有环"，是**最糟的一种反向错误**；
3. **环长 2 要能单独排除**——实测证明互转没有区分度（cycle 标注 75 vs 普通 77）。
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest

from aml_agent.detect import cycles


def _graph(edges: list[tuple[str, str]]) -> nx.MultiDiGraph[str]:
    graph: nx.MultiDiGraph[str] = nx.MultiDiGraph()
    graph.add_edges_from(edges)
    return graph


def _scan(edges: list[tuple[str, str]], *, max_cycle_length: int = 6) -> cycles.CycleScan:
    return cycles.find_short_cycles(_graph(edges), max_cycle_length=max_cycle_length)


# --------------------------------------------------------------------------- #
#  1. 基本环长识别
# --------------------------------------------------------------------------- #


def test_two_cycle() -> None:
    """互转。回程 1 跳 → 环长 2。"""
    scan = _scan([("a", "b"), ("b", "a")])
    assert scan.shortest_by_pair[("a", "b")] == 2
    assert scan.shortest_by_pair[("b", "a")] == 2


def test_three_cycle() -> None:
    """a→b→c→a。每条边的最短环长都是 3。"""
    scan = _scan([("a", "b"), ("b", "c"), ("c", "a")])
    assert set(scan.shortest_by_pair.values()) == {3}
    assert scan.shortest_by_pair[("a", "b")] == 3


def test_four_cycle() -> None:
    scan = _scan([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])
    assert set(scan.shortest_by_pair.values()) == {4}


def test_no_cycle() -> None:
    """链状图，没有任何环。"""
    scan = _scan([("a", "b"), ("b", "c"), ("c", "d")])
    assert scan.shortest_by_pair == {}


def test_shortest_is_reported_when_several_cycles_exist() -> None:
    """一个环长 2 和一个环长 3 同时存在时，取**最短**的那个。"""
    scan = _scan([("a", "b"), ("b", "a"), ("b", "c"), ("c", "a")])
    assert scan.shortest_by_pair[("a", "b")] == 2
    # b→c→a→b 也是环，但 b→a 直接回去更短
    assert scan.shortest_by_pair[("b", "c")] == 3


def test_max_cycle_length_is_respected() -> None:
    """限制环长 3 时，环长 4 的环不该被找到。"""
    edges = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]
    assert _scan(edges, max_cycle_length=3).shortest_by_pair == {}
    assert _scan(edges, max_cycle_length=4).shortest_by_pair != {}


# --------------------------------------------------------------------------- #
#  2. 「原地打转」不算环
# --------------------------------------------------------------------------- #


def test_back_and_forth_is_not_a_long_cycle() -> None:
    """`a→b→a→b` **不是**简单环。

    如果实现里不排除重复节点，会把它当成一个更长的环，
    于是**每条互转边都会得到一个虚假的"环长 3"**。
    """
    scan = _scan([("a", "b"), ("b", "a")])
    # 只有环长 2，不该出现 3
    assert set(scan.shortest_by_pair.values()) == {2}


def test_self_loop_is_length_one() -> None:
    """自环（自己转自己）。要能被识别，但**不能**被当成正常环。"""
    scan = _scan([("a", "a")])
    # 回程 0 跳 —— 实现里 `sender in outgoing` 命中，返回 1
    assert scan.shortest_by_pair[("a", "a")] == 1


# --------------------------------------------------------------------------- #
#  3. 映射回交易：不在环上必须给 0
# --------------------------------------------------------------------------- #


def _edges_frame(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sender": [s for s, _ in pairs],
            "receiver": [r for _, r in pairs],
            "txId": list(range(len(pairs))),
        }
    )


def test_cycle_feature_is_zero_for_edges_not_on_a_cycle() -> None:
    """**这个测试锁住一个真实踩到的 bug。**

    如果用 `shortest_by_pair.get(pair, 哨兵大数) >= min_cycle_length` 来写，
    **不在环上的边会因为哨兵值很大而被判成 1**——
    正好把"没有环"错判成"有环"，是最糟的一种反向错误。
    """
    pairs = [("a", "b"), ("b", "a"), ("x", "y")]
    frame = _edges_frame(pairs)
    scan = _scan(pairs)

    feature = cycles.cycle_feature(frame, scan, min_cycle_length=3)
    assert list(feature) == [0.0, 0.0, 0.0]  # 只有环长 2 和一条无环边


def test_cycle_feature_keeps_three_cycles_but_drops_two_cycles() -> None:
    """实测依据：环长 2 没有区分度（cycle 标注 75 vs 普通 77），必须排除。"""
    pairs = [("a", "b"), ("b", "a"), ("c", "d"), ("d", "e"), ("e", "c")]
    frame = _edges_frame(pairs)
    scan = _scan(pairs)

    # 环长 ≥3：只有后面的三角形
    assert list(cycles.cycle_feature(frame, scan, min_cycle_length=3)) == [
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
    ]
    # 不设下限时，互转也算
    assert list(cycles.cycle_feature(frame, scan, min_cycle_length=2)) == [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]


def test_shortest_cycle_feature_uses_fallback_for_no_cycle() -> None:
    frame = _edges_frame([("a", "b"), ("b", "c"), ("c", "a"), ("x", "y")])
    scan = _scan([("a", "b"), ("b", "c"), ("c", "a"), ("x", "y")])
    feature = cycles.shortest_cycle_feature(frame, scan)
    assert list(feature) == [3.0, 3.0, 3.0, float(cycles.DEFAULT_MAX_CYCLE_LENGTH + 1)]


# --------------------------------------------------------------------------- #
#  4. 平行边不影响环结构
# --------------------------------------------------------------------------- #


def test_parallel_edges_do_not_change_the_cycle() -> None:
    """同一对账户之间转 19 次，环长还是 3——**平行边不改变环**。

    这是"必须在去重图上扫描"的依据：实测平行边占 94.8%，
    在多重图上扫会白慢 19 倍。
    """
    few = _scan([("a", "b"), ("b", "c"), ("c", "a")])
    many = _scan([("a", "b")] * 19 + [("b", "c")] * 7 + [("c", "a")] * 3)
    assert few.shortest_by_pair == many.shortest_by_pair
    assert few.scanned_edges == many.scanned_edges == 3


def test_scanned_edges_counts_distinct_pairs() -> None:
    scan = _scan([("a", "b")] * 5 + [("b", "a")])
    assert scan.scanned_edges == 2


def test_length_histogram() -> None:
    scan = _scan([("a", "b"), ("b", "a"), ("c", "d"), ("d", "e"), ("e", "c")])
    assert scan.length_histogram() == {2: 2, 3: 3}


@pytest.mark.parametrize("bad_max", [1, 0])
def test_max_cycle_length_below_two_finds_nothing(bad_max: int) -> None:
    """环长至少要 2（互转）。参数不合法时**安静返回空**，而不是抛异常——
    调用方可能用它来表示"不做环检测"。
    """
    scan = _scan([("a", "b"), ("b", "a")], max_cycle_length=bad_max)
    assert scan.shortest_by_pair == {}
