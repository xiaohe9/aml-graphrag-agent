"""基线模块的测试。

## 这个文件守的是 M7 那个实测缺口的修法

Agent 评测跑出 **FPR = 100%**，根因不是模型不行——
是它看到「180 笔、金额全同」时**不知道这在全图里算多还是算少**。

所以这里测的是「**相对位置算得对不对**」，
而不是「分数绝对值是多少」（那个值本身没有意义，分位才有）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aml_agent.agent import (
    ACCOUNT_FEATURES,
    AccountBaseline,
    BaselineIndex,
    FeaturePosition,
    numpy_percentile,
)
from aml_agent.agent.baseline import _as_float


def _edges(rows: list[tuple[str, str, float, int]]) -> pd.DataFrame:
    """`(sender, receiver, amount, txId)` → 边表。"""
    return pd.DataFrame(
        [
            {"sender": sender, "receiver": receiver, "amount": amount, "txId": tx_id}
            for sender, receiver, amount, tx_id in rows
        ]
    )


def _star_graph() -> pd.DataFrame:
    """一个中心账户向 5 个账户转账——保证有结构可算。"""
    return _edges([("hub", f"leaf{index}", 10.0, index) for index in range(5)])


# --------------------------------------------------------------------------- #
#  1. 分位数本身
# --------------------------------------------------------------------------- #


def test_percentile_matches_rank_semantics() -> None:
    """分位的定义：**小于等于它的占比**。

    用 `side="right"`：如果有并列，并列的每一个都应该拿到同一个（较高的）分位。
    """
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert numpy_percentile(values, 5.0) == pytest.approx(1.0)
    assert numpy_percentile(values, 1.0) == pytest.approx(0.2)
    assert numpy_percentile(values, 0.0) == pytest.approx(0.0)


def test_percentile_handles_ties_consistently() -> None:
    """并列值必须拿到**相同的**分位。

    ⚠️ 这个坑本项目踩过：全图 24 万条边分数完全相同，
    手写排序在并列时给出了不同的次序。
    """
    values = np.array([1.0, 2.0, 2.0, 2.0, 5.0])
    assert numpy_percentile(values, 2.0) == pytest.approx(0.8)
    # 再来一次结果必须一样
    assert numpy_percentile(values, 2.0) == pytest.approx(0.8)


def test_percentile_of_empty_array_is_zero() -> None:
    """空数组不能崩——**返回 0 而不是抛异常**，因为"没有可比对象"是合法状态。"""
    assert numpy_percentile(np.array([]), 1.0) == 0.0


# --------------------------------------------------------------------------- #
#  2. 数值收窄
# --------------------------------------------------------------------------- #


def test_as_float_accepts_python_and_numpy_numbers() -> None:
    assert _as_float(3) == 3.0
    assert _as_float(3.5) == 3.5
    assert _as_float(np.int64(7)) == 7.0
    assert _as_float(np.float64(2.5)) == 2.5


def test_as_float_rejects_non_numbers_loudly() -> None:
    """**非数值要立刻报错**，不能悄悄变成 0 参与分位计算。

    pandas-stubs 给 `.at[]` 的类型宽到包含 `date`/`complex`，
    要是图省事写 `float(value)  # type: ignore`，
    一个日期字段就会变成 0.0 一路混进分位里。
    """
    with pytest.raises(TypeError, match="期望数值"):
        _as_float("不是数字")
    with pytest.raises(TypeError, match="期望数值"):
        _as_float(None)


# --------------------------------------------------------------------------- #
#  3. 索引构造与查询
# --------------------------------------------------------------------------- #


def test_index_covers_every_account_in_the_edges() -> None:
    edges = _star_graph()
    baseline = BaselineIndex.build(edges)
    assert baseline.account_count == 6  # hub + 5 leaves


def test_lookup_returns_percentiles_in_unit_interval() -> None:
    baseline = BaselineIndex.build(_star_graph())
    found = baseline.lookup("hub")
    assert found is not None
    assert 0.0 <= found.detector_percentile <= 1.0
    for position in found.features:
        assert 0.0 <= position.percentile <= 1.0


def test_lookup_of_an_unknown_account_returns_none_not_zero() -> None:
    """**"查不到"和"分位很低"是两回事。**

    返回一个 0 分位会让 Agent 以为"这个账户排名最低"——
    而实际是**它根本不在图里**。
    """
    baseline = BaselineIndex.build(_star_graph())
    assert baseline.lookup("不存在的账户") is None


def test_every_declared_feature_is_computed() -> None:
    """`ACCOUNT_FEATURES` 里声明的每个特征都要真能算出来。

    **声明和实现漂移的症状是"某个特征永远是 0.5 分位"**，很难发现。
    """
    baseline = BaselineIndex.build(_star_graph())
    found = baseline.lookup("hub")
    assert found is not None
    names = {position.name for position in found.features}
    assert names == {name for name, _ in ACCOUNT_FEATURES}


def test_index_is_built_once_and_queried_many_times() -> None:
    """构造是一次性的重活，查询必须便宜——**Agent 循环会问很多次**。"""
    import time

    baseline = BaselineIndex.build(_edges([("a", f"b{i}", 1.0, i) for i in range(200)]))
    started = time.monotonic()
    for _ in range(200):
        baseline.lookup("a")
    assert time.monotonic() - started < 1.0, "查询应该远快于构造"


# --------------------------------------------------------------------------- #
#  4. ★ 输出里必须写明"分位不等于结论"
# --------------------------------------------------------------------------- #


def _sample() -> AccountBaseline:
    return AccountBaseline(
        account="amlsim:1",
        detector_score=1.5,
        detector_percentile=0.9,
        features=(
            FeaturePosition(name="outDegree", label="转出去向数", value=180, percentile=0.8),
        ),
    )


def test_payload_warns_that_percentile_is_not_a_verdict() -> None:
    """**这一条是模块存在的意义所在。**

    如果 Agent 把「第 90 分位」读成「90% 是洗钱」，那这个工具就是在制造假阳性——
    而它本来是为了减少假阳性才加的。
    """
    payload = _sample().as_payload()
    how_to_read = str(payload["howToRead"])
    assert "相对全图" in how_to_read
    assert "不是判断" in how_to_read
    assert "不等于有问题" in how_to_read


def test_payload_carries_the_measured_signal_basis() -> None:
    """光给分数不够——**还要给"这个信号值不值得信"的实测依据**。

    实测里「环长 2」几乎没有区分度（约 1.0 倍），
    而「环长 3」是约 40 倍。**不分青红皂白地把分位甩给模型，是在误导它。**
    """
    payload = _sample().as_payload()
    signals = payload["measuredSignals"]
    assert isinstance(signals, dict)
    assert "cycle_length_2" in signals
    assert "cycle_length_3" in signals
    assert "40" in signals["cycle_length_3"]["basis"]


def test_payload_is_json_serialisable() -> None:
    import json

    json.dumps(_sample().as_payload(), ensure_ascii=False)


def test_render_is_human_readable() -> None:
    baseline = BaselineIndex.build(_star_graph())
    found = baseline.lookup("hub")
    assert found is not None
    text = baseline.render(found)
    assert "hub" in text
    assert "分位" in text
    assert "转出去向数" in text
