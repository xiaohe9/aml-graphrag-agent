"""数据集字段契约。

**为什么把字段名集中在一个文件里：**

字段名是**外部契约**——它由数据集决定，不由我们决定。一旦上游改名，
我们希望只有一个地方需要改，而不是在十几个文件里搜索字符串。

**更重要的是：这里同时标注了每个字段的「可用性」。**

数据集里混着**特征**和**标签**。标签是答案，一旦泄漏进模型输入，
评测分数会变得非常漂亮——**而且很难发现**。所以「哪些字段绝对不能进模型」
必须是**一处声明、全局遵守**的，而不是靠每个开发者记得。
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------- #
#  accounts.csv —— 账户（图里的 Account 节点）
# --------------------------------------------------------------------------- #
ACCOUNT_ID: Final = "ACCOUNT_ID"
CUSTOMER_ID: Final = "CUSTOMER_ID"
INIT_BALANCE: Final = "INIT_BALANCE"
COUNTRY: Final = "COUNTRY"
ACCOUNT_TYPE: Final = "ACCOUNT_TYPE"
TX_BEHAVIOR_ID: Final = "TX_BEHAVIOR_ID"

ACCOUNT_COLUMNS: Final = (
    ACCOUNT_ID,
    CUSTOMER_ID,
    INIT_BALANCE,
    COUNTRY,
    ACCOUNT_TYPE,
    "IS_FRAUD",
    TX_BEHAVIOR_ID,
)

# --------------------------------------------------------------------------- #
#  transactions.csv —— 交易（图里的 TRANSFER 边）
# --------------------------------------------------------------------------- #
TX_ID: Final = "TX_ID"
SENDER: Final = "SENDER_ACCOUNT_ID"
RECEIVER: Final = "RECEIVER_ACCOUNT_ID"
TX_TYPE: Final = "TX_TYPE"
TX_AMOUNT: Final = "TX_AMOUNT"
TIMESTAMP: Final = "TIMESTAMP"
ALERT_ID: Final = "ALERT_ID"

TRANSACTION_COLUMNS: Final = (
    TX_ID,
    SENDER,
    RECEIVER,
    TX_TYPE,
    TX_AMOUNT,
    TIMESTAMP,
    "IS_FRAUD",
    ALERT_ID,
)

# --------------------------------------------------------------------------- #
#  alerts.csv —— 原生标注的模式
# --------------------------------------------------------------------------- #
ALERT_TYPE: Final = "ALERT_TYPE"

# --------------------------------------------------------------------------- #
#  标签隔离（对应非功能需求 NFR-6）
# --------------------------------------------------------------------------- #

#: ⚠️ **绝对不允许进入模型输入**的字段。
#:
#: `IS_FRAUD` 是数据集给的**答案**。用它做特征 = 看着答案答题，
#: 评测出来的数字全是假的，而且分数会异常漂亮、不易察觉。
#:
#: 代码里凡是构造特征的地方，都应该用 `assert_no_labels()` 过一道。
LABEL_COLUMNS: Final = frozenset({"IS_FRAUD"})

#: 无告警的哨兵值。数据集用 `-1` 表示「这笔交易没有触发告警」，
#: 而不是「告警 ID 是 -1」。区分这两者会影响聚合统计。
NO_ALERT: Final = -1

#: 疑似哨兵/溢出的金额。实测有 3,053 笔等于 `2**31 / 100`，
#: 明显是 32 位整数上限除以 100，不是真实金额。
#: 统计金额分布时应排除，但**不改原始数据**——原始数据保持原样。
SENTINEL_AMOUNT: Final = 2_147_483_6.47


def assert_no_labels(columns: frozenset[str] | set[str] | tuple[str, ...]) -> None:
    """确认一组字段里没有标签列。有就抛异常。

    **这是一个「失败要响亮」的检查**：宁可构建时报错，
    也不要让标签悄悄流入模型、然后在评测时给出一个漂亮但虚假的分数。
    """
    leaked = LABEL_COLUMNS.intersection(columns)
    if leaked:
        raise ValueError(
            f"标签泄漏：{sorted(leaked)} 是数据集给的答案，不能作为模型输入。"
            "如果你确实需要它，那一定是在评测代码里，不是在特征代码里。"
        )
