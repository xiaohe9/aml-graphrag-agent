"""加载 AMLSim 数据集。

**三条原则：**

### 1. 标签隔离是加载时就要挡住的（NFR-6）

数据集里混着特征和标签（`IS_FRAUD`）。**标签泄漏会让评测分数异常漂亮，而且很难发现。**
所以本模块提供 `feature_columns()`，它调用 `schema.assert_no_labels()`——
任何试图把标签当特征用的代码**会在加载阶段就炸**，而不是在评测时给出一个假分数。

### 2. 不改原始数据

实测有 3,053 笔交易的金额等于 `2**31 / 100`，明显是整数溢出的哨兵值。
**我们标记它，但不修改它**——原始数据保持原样，清洗只发生在使用它的地方。
理由：一旦开始改原始数据，你就再也说不清"这个数字是数据里的还是我们造的"。

### 3. 只读一次

transactions.csv 有 132 万行、58 MB。反复读它会让所有下游实验变慢，
所以 `load()` 用 `lru_cache` 缓存——同一进程内只读一次。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from aml_agent.data import schema

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "extracted"

#: 各表的文件与期望列。列名核对是刻意的——
#: 数据集换了、列名变了，应该在这里立刻发现，而不是等某个下游环节莫名其妙地出错。
FILES: dict[str, tuple[str, tuple[str, ...]]] = {
    "accounts": ("accounts.csv", schema.ACCOUNT_COLUMNS),
    "transactions": ("transactions.csv", schema.TRANSACTION_COLUMNS),
    "alerts": (
        "alerts.csv",
        (
            "ALERT_ID",
            schema.ALERT_TYPE,
            "IS_FRAUD",
            schema.TX_ID,
            schema.SENDER,
            schema.RECEIVER,
            schema.TX_TYPE,
            schema.TX_AMOUNT,
            schema.TIMESTAMP,
        ),
    ),
}


class DataNotReadyError(FileNotFoundError):
    """数据没准备好。错误信息里直接给出该跑哪条命令。"""


@dataclass(frozen=True, slots=True)
class Dataset:
    """数据集的三张表。"""

    accounts: pd.DataFrame
    transactions: pd.DataFrame
    alerts: pd.DataFrame

    def stats(self) -> dict[str, object]:
        """画像摘要。写进评测报告，也用于和 `docs/08-数据规格` 里的数字对照。"""
        transactions = self.transactions
        fraud = transactions[transactions["IS_FRAUD"].astype(bool)]
        return {
            "accounts": int(len(self.accounts)),
            "transactions": int(len(transactions)),
            "alerts": int(len(self.alerts)),
            "alertTypes": {
                str(k): int(v) for k, v in self.alerts[schema.ALERT_TYPE].value_counts().items()
            },
            "fraudTransactions": int(len(fraud)),
            "fraudRatio": round(len(fraud) / len(transactions), 6) if len(transactions) else 0.0,
            # 涉诈金额中位数远低于正常交易——这是「单笔金额规则检出率为 0」的原因
            "medianAmountNormal": float(transactions.loc[~transactions["IS_FRAUD"].astype(bool), schema.TX_AMOUNT].median()),
            "medianAmountFraud": float(fraud[schema.TX_AMOUNT].median()) if len(fraud) else None,
            "sentinelAmountRows": int((transactions[schema.TX_AMOUNT] >= schema.SENTINEL_AMOUNT).sum()),
        }


def _read(path: Path, expected_columns: tuple[str, ...], name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in expected_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{name} 缺少预期列 {missing}。实际列：{list(frame.columns)}。"
            "数据集可能换了版本——请核对 docs/08-数据规格与真值设计.md。"
        )
    return frame


def load(data_dir: Path | None = None) -> Dataset:
    """读取三张表。默认目录见 `DEFAULT_DATA_DIR`。"""
    directory = DEFAULT_DATA_DIR if data_dir is None else data_dir
    missing = [filename for filename, _ in FILES.values() if not (directory / filename).is_file()]
    if missing:
        raise DataNotReadyError(
            f"{directory} 里缺少 {missing}。请先运行：\n"
            "    uv run python scripts/fetch_data.py"
        )

    frames = {
        name: _read(directory / filename, columns, name)
        for name, (filename, columns) in FILES.items()
    }
    return Dataset(
        accounts=frames["accounts"],
        transactions=frames["transactions"],
        alerts=frames["alerts"],
    )


@lru_cache(maxsize=1)
def load_cached() -> Dataset:
    """带缓存的加载。

    132 万行的表反复读会拖慢所有实验，而它是只读的——缓存没有副作用。
    """
    return load()


# --------------------------------------------------------------------------- #
#  特征列：标签隔离在这里兜底
# --------------------------------------------------------------------------- #


def feature_columns(table: str) -> tuple[str, ...]:
    """返回某张表可以进模型的列——**自动剔除标签**。

    这是 NFR-6「标签隔离」的执行点。所有构造特征的地方都应该走这里，
    而不是自己写列名列表。
    """
    if table not in FILES:
        raise KeyError(f"未知的表 {table!r}，可选：{sorted(FILES)}")
    _, columns = FILES[table]
    features = tuple(column for column in columns if column not in schema.LABEL_COLUMNS)

    # 再走一道断言：万一上面的过滤逻辑被改错，这里会立刻发现。
    schema.assert_no_labels(features)
    return features
