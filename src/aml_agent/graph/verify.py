"""图的一致性校验。

**这是 SRS 里 FR-1 的验收标准，不是"顺手写的检查"。**

验收标准要求四条：

1. 节点数、边数与源数据**对得上**
2. `Account.key` **唯一**
3. 跨行同名账号**未被合并**（本数据集不体现，但原则保留，见 `docs/04` 3.1）
4. 索引**建立且可用**

**为什么校验脚本要单独存在，而不是"看一眼计数"：**

> 建图错了**不会报错**——它只会安静地产出一张不对的图，
> 然后所有下游指标都建立在这个错误上。
> **计数对不上是唯一能早期发现这件事的信号。**

所以校验必须能被**自动化调用**（CI、或者建图后立刻跑），而不是靠人记得。
"""

from __future__ import annotations

from dataclasses import dataclass

from neo4j import Driver

from aml_agent.data import schema
from aml_agent.data.loader import Dataset
from aml_agent.graph import cypher, guard, queries
from aml_agent.graph.build import connect


@dataclass(frozen=True, slots=True)
class Check:
    """一条校验结果。"""

    name: str
    passed: bool
    detail: str

    def line(self) -> str:
        mark = "[OK]  " if self.passed else "[失败]"
        return f"  {mark} {self.name:<28} {self.detail}"


def _count(driver: Driver, statement: str, key: str = "n") -> int:
    with driver.session() as session:
        record = session.run(statement).single(strict=True)
        return int(record[key])


def verify_graph(driver: Driver, dataset: Dataset) -> list[Check]:
    """对照源数据校验图。返回全部检查项——**失败的不隐藏**。"""
    checks: list[Check] = []

    # ---- 1. 规模对得上 ---------------------------------------------------- #
    expected_accounts = len(dataset.accounts)
    actual_accounts = _count(driver, cypher.COUNT_ACCOUNTS)
    checks.append(
        Check(
            "账户节点数",
            actual_accounts == expected_accounts,
            f"图中 {actual_accounts:,} / 源数据 {expected_accounts:,}",
        )
    )

    expected_entities = dataset.accounts[schema.CUSTOMER_ID].nunique()
    actual_entities = _count(driver, cypher.COUNT_ENTITIES)
    checks.append(
        Check(
            "实体节点数",
            actual_entities == expected_entities,
            f"图中 {actual_entities:,} / 源数据 {expected_entities:,}",
        )
    )

    expected_transfers = len(dataset.transactions)
    actual_transfers = _count(driver, cypher.COUNT_TRANSFERS)
    checks.append(
        Check(
            "交易边数",
            actual_transfers == expected_transfers,
            f"图中 {actual_transfers:,} / 源数据 {expected_transfers:,}",
        )
    )

    actual_controls = _count(driver, cypher.COUNT_CONTROLS)
    checks.append(
        Check(
            "控制关系数",
            actual_controls == expected_accounts,
            f"图中 {actual_controls:,} / 每账户一条应有 {expected_accounts:,}",
        )
    )

    # ---- 2. 键唯一 -------------------------------------------------------- #
    with driver.session() as session:
        duplicates = session.run(cypher.DUPLICATE_ACCOUNT_KEYS).data()
    checks.append(
        Check(
            "账户键唯一",
            not duplicates,
            "无重复" if not duplicates else f"发现 {len(duplicates)} 组重复：{duplicates[:3]}",
        )
    )

    # ---- 3. 索引可用 ------------------------------------------------------ #
    with driver.session() as session:
        indexes = session.run(cypher.SHOW_INDEXES).data()
    ours = [
        i
        for i in indexes
        if str(i.get("name", "")).startswith(("account_", "entity_", "transfer_"))
    ]
    not_online = [i["name"] for i in ours if i.get("state") != "ONLINE"]
    checks.append(
        Check(
            "索引在线",
            bool(ours) and not not_online,
            f"{len(ours)} 个索引" + (f"，未在线：{not_online}" if not_online else "，全部 ONLINE"),
        )
    )

    # ---- 4. 标签隔离：图上不能有答案（第 ① 层防御的验证）------------------ #
    #
    # 这一条**必须查数据库**，不能只查代码：
    # 代码可以改对，但**库里的旧数据还在**。它防的是"图是用旧版本建的"。
    names = queries.property_names(driver)
    try:
        guard.assert_property_names_are_clean(names)
        checks.append(Check("图上无标签属性", True, f"属性：{', '.join(names) or '（无）'}"))
    except guard.LabelLeakError as exc:
        checks.append(Check("图上无标签属性", False, str(exc).splitlines()[0]))

    # ---- 5. 自环：信息项，不是失败 ---------------------------------------- #    #
    # 源数据本身就有 181 笔「自己转给自己」。这是**数据的特征，不是建图的错误**。
    # 把它算成失败会让人去"修"一个本来没坏的东西。
    # 它真正影响的是**环路检测**——检测时必须显式排除自环，
    # 否则它们会被当成长度为 1 的环，制造大量假阳性。
    self_loops = _count(driver, cypher.COUNT_SELF_LOOPS)
    checks.append(
        Check(
            "自环（信息项，非失败）",
            True,
            f"{self_loops} 笔 —— 环路检测必须显式排除，否则会当成长度 1 的环",
        )
    )

    return checks


def verify_and_report(driver: Driver, dataset: Dataset) -> bool:
    """跑校验并打印。返回是否**全部通过**（信息项不算失败）。"""
    checks = verify_graph(driver, dataset)
    for check in checks:
        print(check.line())
    failed = [c for c in checks if not c.passed]
    if failed:
        print(f"\n{len(failed)} 项未通过。图不可信，先修再往下走。")
        return False
    print("\n全部通过。")
    return True


def main() -> int:
    """命令行入口：`python -m aml_agent.graph.verify`"""
    from aml_agent.data import loader

    driver = connect()
    try:
        driver.verify_connectivity()
        dataset = loader.load_cached()
        print("校验图与源数据的一致性：")
        return 0 if verify_and_report(driver, dataset) else 1
    except Exception as exc:  # noqa: BLE001 - CLI 入口，把原因打清楚比抛栈有用
        print(f"校验失败：{type(exc).__name__}: {exc}")
        print("图库起来了吗？  docker compose up -d neo4j")
        return 2
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
