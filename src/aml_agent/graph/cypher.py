"""图谱层的 Cypher 语句。

**为什么把 Cypher 单独放一个文件，而不是散在 Python 里：**

Cypher 是**另一种语言**。把它和 Python 混在一起，两边的语法高亮、格式、审查都做不好。
单独放还有个好处：**这些查询可以直接贴进 Neo4j Browser 里跑**，
调试时不用改 Python 代码。
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
#  约束与索引
# --------------------------------------------------------------------------- #

#: 唯一约束。**必须在导入数据之前建**——它会顺带创建索引，
#: 让后面每一条 MERGE 走索引查找，而不是全图扫描。
#:
#: 132 万条边如果每条都全图扫描，导入会从几分钟变成几小时。
CREATE_ACCOUNT_CONSTRAINT = """
CREATE CONSTRAINT account_key IF NOT EXISTS
FOR (a:Account) REQUIRE a.key IS UNIQUE
"""

CREATE_ENTITY_CONSTRAINT = """
CREATE CONSTRAINT entity_name IF NOT EXISTS
FOR (e:Entity) REQUIRE e.name IS UNIQUE
"""

#: 时间索引。流水线里的时序筛选（分层、快进快出）都是**范围查询**，
#: 没有它就退化成全表扫描——而"30 秒响应"是写进非功能需求的硬约束。
#:
#: Neo4j 5 支持关系属性索引，这是它相对早期版本的一个重要能力。
CREATE_TRANSFER_TIMESTAMP_INDEX = """
CREATE INDEX transfer_timestamp IF NOT EXISTS
FOR ()-[t:TRANSFER]-() ON (t.timestamp)
"""

#: 金额索引。金额区间筛选（拆分、大额）走它。
CREATE_TRANSFER_AMOUNT_INDEX = """
CREATE INDEX transfer_amount IF NOT EXISTS
FOR ()-[t:TRANSFER]-() ON (t.amount)
"""

SETUP_STATEMENTS: tuple[str, ...] = (
    CREATE_ACCOUNT_CONSTRAINT,
    CREATE_ENTITY_CONSTRAINT,
    CREATE_TRANSFER_TIMESTAMP_INDEX,
    CREATE_TRANSFER_AMOUNT_INDEX,
)

# --------------------------------------------------------------------------- #
#  导入
# --------------------------------------------------------------------------- #

#: 导入账户。
#:
#: **用 MERGE 而不是 CREATE**：账户是**实体**——它在数据里出现多次时指的是同一个东西，
#: 导入必须幂等。重跑一次不该产生两倍节点。
IMPORT_ACCOUNTS = """
UNWIND $rows AS row
MERGE (a:Account {key: row.key})
SET a.accountId = row.accountId,
    a.initialBalance = row.initialBalance,
    a.institution = row.institution
"""

#: 导入实体（客户）并连上账户。
#:
#: 关系方向是 `(Entity)-[:CONTROLS]->(Account)`——
#: **实体控制账户**，而不是反过来。方向搞反了，路径查询的语义就错了。
IMPORT_ENTITIES = """
UNWIND $rows AS row
MERGE (e:Entity {name: row.entity})
WITH e, row
MATCH (a:Account {key: row.key})
MERGE (e)-[:CONTROLS]->(a)
"""

#: 导入交易。
#:
#: **用 CREATE 而不是 MERGE**：交易是**事件**——两笔金额相同的转账是两件事，
#: 合并它们会丢掉时间信息和笔数。
#:
#: ## ⚠️ 标签（`isFraud`）**刻意不导入**
#:
#: 图是**特征存储**，不是**答案存储**。
#:
#: 曾经把 `isFraud` 写在边上，结果是：**任何 `RETURN t` 都会把答案交给 Agent**。
#: 而 Agent 一旦"看见答案"，后面所有评测都失去意义——
#: 更糟的是**分数会很漂亮，很难发现**。
#:
#: 评测需要标签，但它**从 CSV 读**（`data/loader.py`），不需要经过图。
#:
#: **最强的访问控制，是不把秘密放在那里。**
IMPORT_TRANSFERS = """
UNWIND $rows AS row
MATCH (s:Account {key: row.sender})
MATCH (r:Account {key: row.receiver})
CREATE (s)-[:TRANSFER {
    txId: row.txId,
    amount: row.amount,
    timestamp: row.timestamp,
    isSentinelAmount: row.isSentinelAmount
}]->(r)
"""

# --------------------------------------------------------------------------- #
#  校验
# --------------------------------------------------------------------------- #

COUNT_ACCOUNTS = "MATCH (a:Account) RETURN count(a) AS n"
COUNT_ENTITIES = "MATCH (e:Entity) RETURN count(e) AS n"
COUNT_TRANSFERS = "MATCH ()-[t:TRANSFER]->() RETURN count(t) AS n"
COUNT_CONTROLS = "MATCH ()-[c:CONTROLS]->() RETURN count(c) AS n"

#: 检查有没有账户节点被合并（key 重复）。
#:
#: 唯一约束理论上已经保证了，但**这是验收标准里明确列的一条**，
#: 所以要用独立查询再证一次——约束是可以被删掉的，校验不能只依赖它。
#:
#: ⚠️ 这里写成 `a.key AS key` 而不是 `RETURN key`，是**刻意的**：
#: `RETURN key` 里的 `key` 是个裸标识符，`guard` 无法区分
#: 「标量别名」和「整个实体」——**规范写法让检查能保持严格而不误伤**。
DUPLICATE_ACCOUNT_KEYS = """
MATCH (a:Account)
WITH a.key AS key, count(*) AS n
WHERE n > 1
RETURN key, n LIMIT 10
"""

#: 图中自环数量。实测源数据有 181 笔自己转自己，
#: 环路检测要显式排除它们（否则会把它们当成长度为 1 的环）。
COUNT_SELF_LOOPS = "MATCH (a:Account)-[t:TRANSFER]->(a) RETURN count(t) AS n"

#: 确认索引真的建好了，并且处于可用状态。
SHOW_INDEXES = "SHOW INDEXES YIELD name, type, state, labelsOrTypes, properties"

# --------------------------------------------------------------------------- #
#  标签泄漏检查
# --------------------------------------------------------------------------- #

#: 列出节点上出现过的**所有属性名**。
#:
#: 用 `keys()` 而不是写死属性名，是为了**发现意料之外的属性**——
#: 如果哪天有人往图上加了新字段，这个查询会把它的名字暴露出来，
#: 然后由 Python 侧对照禁用清单判断。
#:
#: 写死检查项只能防住你已经想到的泄漏；列全部属性才能防住没想到的。
NODE_PROPERTY_NAMES = """
MATCH (n)
WITH DISTINCT keys(n) AS ks
UNWIND ks AS property
RETURN DISTINCT property
"""

RELATIONSHIP_PROPERTY_NAMES = """
MATCH ()-[r]->()
WITH DISTINCT keys(r) AS ks
UNWIND ks AS property
RETURN DISTINCT property
"""

# --------------------------------------------------------------------------- #
#  清空
# --------------------------------------------------------------------------- #

#: 分批清空全图。
#:
#: ## 为什么要用 `CALL ... IN TRANSACTIONS`
#:
#: 第一版是 `MATCH (n) WITH n LIMIT 10000 DETACH DELETE n`，实测**报内存错误**：
#:
#: ```text
#: Neo.TransientError.General.MemoryPoolOutOfMemoryError
#: The allocation of an extra 24.0 MiB would use more than the limit 1.4 GiB
#: ```
#:
#: 原因：**"1 万个节点"不等于"1 万次删除"**。这里平均每个节点挂着 264 条边，
#: 于是"删 1 万个节点"实际是**在一笔事务里删 130 万条关系**——
#: 而 Neo4j 的事务内存上限是固定 1.4 GiB，跟堆大小无关。
#:
#: **正确修法不是调大上限，而是把工作拆成多笔事务。**
#: `IN TRANSACTIONS OF 500 ROWS` 让每 500 个节点各自提交，
#: 内存占用就有了上界——**这才是"分批"的真正含义：分批的单位是事务，不是循环。**
CLEAR_GRAPH = """
MATCH (n)
CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 500 ROWS
"""

#: **允许结构性返回的查询，必须写明理由。**
#:
#: 这两个查询返回的是属性**名字**，不是属性**值**——`keys()` 的结果本身不含数据，
#: 所以它们不构成泄漏。但它们**长得像** `RETURN 裸标识符`，
#: 会被 `guard` 的结构性检查拦下。
#:
#: 用显式白名单而不是放宽检查：**放宽检查会让所有查询都失去保护**，
#: 而白名单把例外限制在这两个、并且**在代码审查里看得见**。
STRUCTURAL_ALLOWLIST: dict[str, str] = {
    "NODE_PROPERTY_NAMES": "返回属性名清单，用于检测泄漏；不含任何属性值",
    "RELATIONSHIP_PROPERTY_NAMES": "同上",
}
