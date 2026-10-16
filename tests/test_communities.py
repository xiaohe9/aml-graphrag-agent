"""社区发现的测试。

## 这个文件守的核心：**代理指标不能冒充达标**

`docs/02` 原本给 FR-4 写的验证方式是「注入团伙识别」——
**而这份数据没有团伙真值**（`ALERT_TYPE` 只有 cycle / fan_in）。

所以测试里必须反复确认：
**输出里带着"这是代理"的说明**，而不是一个看起来像准确率的数字。
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest

from aml_agent.detect.communities import (
    CommunityScan,
    find_communities,
    label_purity,
    rand_agreement,
    to_undirected_simple,
)


def _edges(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sender": sender, "receiver": receiver, "amount": 1.0, "txId": index}
            for index, (sender, receiver) in enumerate(rows)
        ]
    )


def _two_cliques() -> pd.DataFrame:
    """两个明显分离的团 —— 社区发现应该正好找到 2 个。"""
    rows: list[tuple[str, str]] = []
    for group in ("a", "b"):
        members = [f"{group}{index}" for index in range(6)]
        rows.extend(
            (left, right) for index, left in enumerate(members) for right in members[index + 1 :]
        )
    # 团之间只连一条边
    rows.append(("a0", "b0"))
    return _edges(rows)


# --------------------------------------------------------------------------- #
#  1. 图视图
# --------------------------------------------------------------------------- #


def test_undirected_view_collapses_parallel_and_direction() -> None:
    """社区发现用**无向去重图**。

    A→B 和 B→A 是同一伙人之间的往来，**方向不影响他们是不是一伙**。

    （这与环检测相反——**环必须是有向的**，否则 A→B 和 B→A 会被当成同一个环。
    同一个图，两个算法要不同的视图，这是正常的。）
    """
    graph = to_undirected_simple(_edges([("a", "b"), ("b", "a"), ("a", "b")]))
    assert graph.number_of_edges() == 1
    assert set(graph.nodes) == {"a", "b"}


# --------------------------------------------------------------------------- #
#  2. 算法本身
# --------------------------------------------------------------------------- #


def test_finds_the_two_obvious_cliques() -> None:
    scan = find_communities(_two_cliques(), seed=1)
    assert scan.community_count == 2
    assert scan.assignments["a1"] == scan.assignments["a2"]
    assert scan.assignments["b1"] == scan.assignments["b2"]
    assert scan.assignments["a1"] != scan.assignments["b1"]


def test_is_reproducible_for_a_fixed_seed() -> None:
    """**Louvain 有随机性，不固定种子就没有可复现性。**

    而"同一输入两次结果不同"会让任何基于它的结论都失效。
    """
    edges = _two_cliques()
    first = find_communities(edges, seed=42)
    second = find_communities(edges, seed=42)
    assert first.assignments == second.assignments


def test_rejects_a_non_positive_resolution() -> None:
    with pytest.raises(ValueError, match="resolution"):
        find_communities(_two_cliques(), resolution=0)


def test_max_nodes_is_a_real_guard_not_a_suggestion() -> None:
    """`max_nodes` 是**安全阀**，不是提示。

    社区发现在超大图上的内存与耗时都不是线性的——
    设了上限就必须真的拦住，否则它只是一句注释。
    """
    with pytest.raises(ValueError, match="max_nodes"):
        find_communities(_two_cliques(), max_nodes=3)


def test_summary_reports_sizes_and_share() -> None:
    scan = find_communities(_two_cliques(), seed=1)
    summary = scan.summary()
    assert summary["communities"] == 2
    assert summary["nodes"] == 12
    assert 0.0 < float(summary["largestShare"]) <= 1.0


def test_members_and_lookup_are_consistent() -> None:
    scan = find_communities(_two_cliques(), seed=1)
    community = scan.community_of("a1")
    assert community is not None
    assert "a1" in scan.members(community)


# --------------------------------------------------------------------------- #
#  3. ★ 稳定性（代理指标之一）
# --------------------------------------------------------------------------- #


def test_rand_agreement_is_one_for_identical_partitions() -> None:
    first = {"a": 0, "b": 0, "c": 1}
    assert rand_agreement(first, first) == pytest.approx(1.0)


def test_rand_agreement_drops_when_partitions_differ() -> None:
    first = {"a": 0, "b": 0, "c": 1}
    second = {"a": 0, "b": 1, "c": 0}
    assert rand_agreement(first, second) < 1.0


def test_rand_agreement_handles_tiny_and_disjoint_inputs() -> None:
    """节点太少或没有交集时返回 0，**不能崩**。"""
    assert rand_agreement({"a": 0}, {"a": 0}) == 0.0
    assert rand_agreement({"a": 0}, {"b": 0}) == 0.0


def test_rand_agreement_documentation_says_what_it_does_not_measure() -> None:
    """它的文档必须写明：**它不衡量划分好不好，只衡量稳不稳**。

    两次都错得一模一样，它也会给 1.0——
    不写清这一点，读的人会把它当成质量指标。
    """
    doc = rand_agreement.__doc__ or ""
    assert "不衡量" in doc
    assert "稳不稳" in doc


# --------------------------------------------------------------------------- #
#  4. ★★ 代理指标不能冒充达标
# --------------------------------------------------------------------------- #


def test_label_purity_always_carries_the_proxy_warning() -> None:
    """**这是这个模块最重要的约束。**

    输出里必须带着「这是代理，不是准确率」——
    否则那个 `spread` 数字会被当成"团伙识别效果"。
    """
    assignments = {f"a{index}": 0 for index in range(40)}
    assignments.update({f"b{index}": 1 for index in range(40)})
    labels = {f"a{index}": index < 20 for index in range(40)}
    labels.update({f"b{index}": False for index in range(40)})

    result = label_purity(assignments, labels)
    assert "代理指标" in str(result["note"])
    assert "不是团伙识别准确率" in str(result["note"])


def test_label_purity_ignores_communities_that_are_too_small() -> None:
    """小社区的比例没有统计意义——**不能被算进"最好/最差社区"**。"""
    assignments = {"big0": 0, "big1": 0, "big2": 0}
    assignments.update({f"small{index}": 1 for index in range(40)})
    labels = {"big0": True, "big1": False, "big2": False}
    labels.update({f"small{index}": False for index in range(40)})

    result = label_purity(assignments, labels)
    assert result["communitiesCompared"] == 1, "只有 1 个社区规模 ≥ 30"


def test_label_purity_handles_no_matching_labels() -> None:
    result = label_purity({"a": 0}, {"zzz": True})
    assert "没有可对齐的标签" in str(result["note"])


def test_label_purity_says_so_when_no_community_is_large_enough() -> None:
    result = label_purity({"a": 0, "b": 0}, {"a": True, "b": False})
    assert "规模足够" in str(result["note"])


# --------------------------------------------------------------------------- #
#  5. 扫描对象
# --------------------------------------------------------------------------- #


def test_scan_summary_is_json_serialisable() -> None:
    import json

    scan = find_communities(_two_cliques(), seed=1)
    json.dumps(scan.summary(), ensure_ascii=False)


def test_scan_is_frozen() -> None:
    """扫描结果不可变——**证据一旦采集完就不该被下游改动**。"""
    import dataclasses

    scan = find_communities(_two_cliques(), seed=1)
    assert dataclasses.is_dataclass(CommunityScan)
    with pytest.raises(dataclasses.FrozenInstanceError):
        scan.seed = 999  # type: ignore[misc]


def test_single_node_graph_does_not_crash() -> None:
    """只有自环的图——Louvain 应该给出 1 个社区而不是报错。"""
    scan = find_communities(_edges([("a", "a")]))
    assert scan.node_count == 1


def test_modularity_is_reported_but_not_dressed_up_as_quality_of_detection() -> None:
    """模块度是**划分质量**，不是"可疑程度"，也不是识别准确率。

    这里只确认它被如实报出来（0–1 之间），
    **不做"模块度高就是好"的断言**——那是两回事。
    """
    scan = find_communities(_two_cliques(), seed=1)
    assert 0.0 <= scan.modularity <= 1.0
    assert isinstance(nx.Graph(), nx.Graph)  # networkx 确实被用到了
