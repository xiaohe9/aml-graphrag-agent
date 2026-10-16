"""数据层的测试。

**两组测试目的不同：**

- **单元测试**（不依赖数据文件）：验证标签隔离、错误信息、列名过滤逻辑。
  这些必须永远能跑——**CI 里不该为了跑一个单元测试去下 61 MB 数据**。
- **集成测试**（依赖真实数据）：验证画像数字与文档一致。
  数据不在时**明确跳过**，而不是假装通过。
"""

from __future__ import annotations

import pytest

from aml_agent.data import loader, schema

# --------------------------------------------------------------------------- #
#  单元测试：标签隔离（NFR-6）
# --------------------------------------------------------------------------- #


def test_feature_columns_never_include_labels() -> None:
    """这是 NFR-6 的执行点：标签列绝不能出现在特征里。"""
    for table in loader.FILES:
        features = loader.feature_columns(table)
        assert "IS_FRAUD" not in features, f"{table} 的特征列里混进了标签"


def test_assert_no_labels_raises_loudly() -> None:
    """失败要响亮——宁可构建时报错，也不要让标签悄悄流入模型。"""
    with pytest.raises(ValueError, match="标签泄漏"):
        schema.assert_no_labels({"TX_AMOUNT", "IS_FRAUD"})

    # 干净的一组不应该报错
    schema.assert_no_labels({"TX_AMOUNT", "TIMESTAMP"})


def test_unknown_table_is_rejected() -> None:
    with pytest.raises(KeyError, match="未知的表"):
        loader.feature_columns("customers")


def test_column_drift_is_detected(tmp_path) -> None:
    """列名变了必须立刻发现，而不是等下游莫名其妙地出错。"""
    bad = tmp_path / "accounts.csv"
    bad.write_text("WRONG_COLUMN\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="缺少预期列"):
        loader._read(bad, ("ACCOUNT_ID", "CUSTOMER_ID"), "accounts")


def test_missing_data_gives_the_command_to_fix_it(tmp_path) -> None:
    """数据不在时，错误信息必须直接告诉人该怎么办。"""
    with pytest.raises(loader.DataNotReadyError, match=r"fetch_data\.py"):
        loader.load(tmp_path)


# --------------------------------------------------------------------------- #
#  集成测试：真实数据（不在就跳过）
# --------------------------------------------------------------------------- #

requires_data = pytest.mark.skipif(
    not loader.DEFAULT_DATA_DIR.is_dir(),
    reason="数据未下载；运行 `uv run python scripts/fetch_data.py` 后再跑",
)

# ⚠️ 这里**不能**写 `integration = [requires_data, pytest.mark.integration]`
# 然后 `@integration` —— **list 不可调用**，pytest 收集时会抛
# `TypeError: 'list' object is not callable`。
# 多个标记要**逐个叠上去**（下面的 `@requires_data` + `@pytest.mark.integration`）。
integration = pytest.mark.integration


@requires_data
@integration
def test_dataset_shape_matches_the_specification() -> None:
    """行数与 docs/08-数据规格与真值设计.md 一致。

    这些数字是**契约**：数据换了、或者文档写错了，这里会立刻发现。
    """
    dataset = loader.load_cached()
    stats = dataset.stats()

    assert stats["accounts"] == 10_000
    assert stats["transactions"] == 1_323_234
    assert stats["alerts"] == 1_719


@requires_data
@integration
def test_native_patterns_are_cycle_and_fan_in() -> None:
    """原生只有两种模式。这是 ADR-0004 修正需求范围的依据。"""
    dataset = loader.load_cached()
    assert set(dataset.alerts[schema.ALERT_TYPE].unique()) == {"cycle", "fan_in"}


@requires_data
@integration
def test_fraud_transactions_are_much_smaller_than_normal_ones() -> None:
    """**这是「单笔金额规则检出率为 0%」的证据。**

    涉诈交易的金额中位数远低于正常交易——因为洗钱的人**必须**把钱拆小，
    否则就触发上报。所以任何基于单笔金额的规则都抓不到它们。
    """
    stats = loader.load_cached().stats()

    normal = stats["medianAmountNormal"]
    fraud = stats["medianAmountFraud"]
    assert isinstance(normal, float) and isinstance(fraud, float)
    assert fraud < normal / 5, f"涉诈金额中位数 {fraud} 应该远低于正常 {normal}"
