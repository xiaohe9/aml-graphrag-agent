"""标签泄漏防护。

## 威胁是什么

数据集给的答案（`IS_FRAUD`）**如果出现在 Agent 能看到的地方**，
评测就失去意义——而且**最坏的地方在于分数会很漂亮，很难发现**：

- Agent「看见答案」→ 检出率虚高 → 看起来效果很好；
- 你没有任何信号提示"这是作弊"，因为**指标不会自己说"我作弊了"**。

## 三层防御

| 层 | 做法 | 强度 |
|---|---|---|
| **① 不存** | 标签**不写进图**（图是特征存储，不是答案存储） | **最强** |
| **② 白名单** | 查询的 `RETURN` 子句**从允许的属性列表生成** | 强（不可能写错） |
| **③ 自动检查** | 扫描所有查询文本，禁用模式一律报错 | 兜底（防未来改动） |

**第 ① 层是最重要的**：**最强的访问控制，是不把秘密放在那里。**
第 ②③ 层是纵深防御——万一将来有人又把标签加回图上，还能挡住。

## 为什么第 ③ 层不能只查已知的字段名

写死 `if "isFraud" in query` 只能防住**你已经想到的**泄漏。
所以这里同时禁用**结构性模式**（`RETURN t`、`RETURN *`），
它们**与字段名无关**，能防住没想到的情况。
"""

from __future__ import annotations

import re
from collections.abc import Iterable

#: 禁止出现在查询里的属性名（大小写不敏感）。
#:
#: 命名上做了宽匹配：凡是名字里带这些片段的都算标签。
#: 宁可误伤（要求改名），也不要漏掉一个真标签。
FORBIDDEN_PROPERTY_PATTERNS: tuple[str, ...] = (
    "isfraud",
    # ⚠️ 下划线版本必须**单独列**：`IS_FRAUD` 里有个下划线，
    # 不带下划线的 `isfraud` 匹配不上它——**第一版就漏了这个**，
    # 而 `IS_FRAUD` 恰恰是数据集里的原始列名。
    "is_fraud",
    "is_laundering",
    "islaundering",
    "label",
    "groundtruth",
    "ground_truth",
    "target",
    "answer",
)


def _property_position_regexes(name: str) -> tuple[re.Pattern[str], ...]:
    """把禁用名编译成**只在属性位置**才匹配的正则。

    ## 为什么不能直接做子串匹配

    第一版是 `if "label" in query.lower()`。实测**误伤了自己**：

    - `SHOW INDEXES YIELD ... labelsOrTypes ...` —— 含 `label`；
    - `MATCH (target:Account {key: $target})` —— 含 `target`。

    **误伤会让人把检查关掉，那比没有检查更糟。**
    一个总在报错的检查，最后一定会被加 `# noqa` 或被删掉。

    所以改成只匹配**属性出现的位置**：

    | 形式 | 例子 |
    |---|---|
    | 属性访问 | `t.isFraud`、`t['isFraud']` |
    | map 的键 | `{isFraud: row.x}` |
    | 输出别名 | `... AS isFraud` |

    **变量名、过程名、配置项名都不会命中**，
    因为它们不带前导点、也不出现在 `AS` 后面。
    """
    escaped = re.escape(name)
    return (
        # t.isFraud / t['isFraud'] / t["isFraud"]
        re.compile(rf"[.\[]\s*['\"]?{escaped}\b", re.IGNORECASE),
        # {isFraud: ...} / , isFraud: ...
        #
        # ⚠️ 必须是"`{` 或 `,` 紧跟属性名"——**不能只写 `name:`**。
        # 因为 Cypher 的**节点标签语法**也是 `标识符:标签`：
        #   (target:Account {key: $target})
        # 只用 `\btarget\s*:` 会把它误判成泄漏。
        # **误伤会让人把检查关掉，那比没有检查更糟。**
        re.compile(rf"[\{{,]\s*{escaped}\s*:", re.IGNORECASE),
        # ... AS isFraud
        re.compile(rf"\bAS\s+{escaped}\b", re.IGNORECASE),
    )


_FORBIDDEN_IN_PROPERTY_POSITION: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, regex)
    for name in FORBIDDEN_PROPERTY_PATTERNS
    for regex in _property_position_regexes(name)
)

#: 结构性禁用：**与字段名无关**，防的是"整体返回实体"。
#:
#: ## 为什么不能只看 `RETURN 裸标识符`
#:
#: 第一版就是那么写的，结果**误伤了我们自己的查询**：
#: `RETURN key, n` 里的 `key` 和 `n` 是**标量别名**（来自 `WITH ... AS`），
#: 不是实体。但正则分不出来——**它只看得到"RETURN 后面跟了个词"**。
#:
#: 而误伤会让人把检查关掉，**那比没有检查更糟**。
#:
#: ## 改成看「这个标识符有没有被声明成节点或关系」
#:
#: Cypher 里实体变量一定在模式里出现过：`(a:Account)`、`[t:TRANSFER]`、`(b)`。
#: 所以规则变成：**RETURN 里的裸标识符，如果它是本查询声明过的实体变量，就拦。**
#:
#: | 写法 | 判定 |
#: |---|---|
#: | `MATCH (a)-[t]->(b) RETURN t` | `t` 是声明过的关系变量 → **拦下** ✅ |
#: | `WITH a.key AS key RETURN key, n` | `key` 只是别名，不是实体 → 放行 ✅ |
#:
#: 这仍然是启发式（真正的判断要解析 Cypher），但它**对准了真正的威胁**，
#: 而且不再误伤。
_ENTITY_VARIABLE = re.compile(r"[\(\[]\s*([a-zA-Z_]\w*)\s*(?:[\)\]:\[{]|\s)", re.IGNORECASE)

_BARE_RETURN_ITEM = re.compile(
    r"(?:\bRETURN\b|,)\s+(?:DISTINCT\s+)?([a-zA-Z_]\w*)\s*(?=,|$|LIMIT|ORDER|UNION)",
    re.IGNORECASE,
)

FORBIDDEN_STRUCTURAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    # RETURN * —— 无条件危险
    re.compile(r"\bRETURN\s+\*", re.IGNORECASE),
    # WITH *
    re.compile(r"\bWITH\s+\*", re.IGNORECASE),
)


def _declared_entity_variables(cypher: str) -> set[str]:
    """找出查询里被声明成**节点或关系变量**的标识符。"""
    return {match.group(1).lower() for match in _ENTITY_VARIABLE.finditer(cypher)}


def _returns_whole_entity(cypher: str) -> list[str]:
    """返回**被整体返回出去的实体变量名**（空列表表示安全）。"""
    declared = _declared_entity_variables(cypher)
    if not declared:
        return []
    return sorted(
        {
            match.group(1)
            for match in _BARE_RETURN_ITEM.finditer(cypher)
            if match.group(1).lower() in declared
        }
    )


class LabelLeakError(ValueError):
    """查询里出现了标签或结构性泄漏风险。

    **这是一个"失败要响亮"的检查**：宁可查询构建时报错，
    也不要让标签悄悄流进 Agent 的上下文、然后在评测时给出一个漂亮但虚假的分数。
    """


def check_query(cypher: str, *, allow_structural: bool = False) -> list[str]:
    """检查一条 Cypher，返回**问题描述列表**（空列表表示安全）。

    `allow_structural=True` 用于**确实需要返回整实体**的极少数地方，
    比如属性名枚举查询本身——它返回的是属性**名字**，不是值。
    用这个开关时**必须写明理由**，它会让这个查询出现在代码审查里。
    """
    problems: list[str] = []

    for name, regex in _FORBIDDEN_IN_PROPERTY_POSITION:
        if regex.search(cypher):
            problems.append(f"查询在属性位置引用了禁用名 {name!r}——图是特征存储，不是答案存储。")

    if not allow_structural:
        for regex in FORBIDDEN_STRUCTURAL_PATTERNS:
            match = regex.search(cypher)
            if match:
                problems.append(
                    f"结构性泄漏风险：{match.group(0)!r} 会把整个实体返回出去，"
                    "实体上的任何属性都会跟着泄漏。请改成显式列出属性。"
                )

        leaked = _returns_whole_entity(cypher)
        if leaked:
            problems.append(
                f"结构性泄漏风险：{leaked} 是本查询里声明过的节点/关系变量，"
                "整体返回它们会把**身上的所有属性**一起带出去（包括将来新加的字段）。"
                "请改成显式列出属性，例如 `t.txId AS txId`。"
            )
    return problems


def assert_query_safe(cypher: str, *, allow_structural: bool = False) -> None:
    """不安全就抛 `LabelLeakError`。"""
    problems = check_query(cypher, allow_structural=allow_structural)
    if problems:
        raise LabelLeakError(
            "查询未通过标签泄漏检查：\n  - "
            + "\n  - ".join(problems)
            + f"\n\n完整查询：\n{cypher.strip()}"
        )


def assert_property_names_are_clean(names: Iterable[str]) -> None:
    """检查**图上实际存在的属性名**里有没有标签。

    这是第 ① 层的验证：建完图之后，对着数据库问「你身上有哪些属性」，
    再拿结果对照禁用清单。

    **为什么要查数据库而不是查代码**：代码可以改对，但**库里的旧数据还在**。
    这个检查能发现"图是用旧版本建的"这种情况。

    ⚠️ 这里用**子串匹配**（比 `check_query` 更严），是刻意的：

    这是在核对**数据库里实际存了什么**，不是在检查查询写法。
    属性名是标识符，一个叫 `is_fraud_flag` 或 `fraud_label` 的属性
    **确实需要被拦下**——而在这里误伤的代价很小（改个属性名），
    漏掉的代价很大（答案在图上）。
    """
    dirty = [
        name
        for name in names
        if any(pattern in name.lower() for pattern in FORBIDDEN_PROPERTY_PATTERNS)
    ]
    if dirty:
        raise LabelLeakError(
            f"图上存在标签属性：{dirty}。\n"
            "图是**特征存储**，不是答案存储。请重建图（build_graph(..., reset=True)），"
            "并确认 `cypher.IMPORT_TRANSFERS` 里没有把标签写进去。"
        )


#: Cypher 语句的粗略识别：出现这些关键字之一就当它是查询。
_CYPHER_KEYWORDS = ("MATCH", "CREATE", "MERGE", "RETURN", "SHOW", "UNWIND", "DELETE")


def looks_like_cypher(text: str) -> bool:
    """判断一段文本是不是 Cypher 查询。

    ⚠️ **必须要求有空白**（即"不只一个词"）。

    第一版只查关键字，结果把 `guard._CYPHER_KEYWORDS` 那个
    `("MATCH", "CREATE", ...)` **元组本身当成了查询**——
    于是覆盖测试报「`guard.py` 定义了查询但没被审计」。

    **那个报警其实是对的**：扫描器的判据太松。
    真实的查询一定是多词的（`MATCH (n) RETURN n`），
    单个关键字只是一个词，不是查询。
    """
    if not any(keyword in text.upper() for keyword in _CYPHER_KEYWORDS):
        return False
    # 必须有空白 —— 排除"关键字列表"这类误判
    return any(character.isspace() for character in text)


def collect_queries(module: object) -> dict[str, str]:
    """从一个模块里挑出**全部查询常量**（大写的字符串常量，含元组里的）。

    用扫描而不是维护一份手写清单：**手写清单会漏掉新增的查询**，
    而漏掉的那条就是漏洞。自动发现才防得住"以后加的新查询"。

    元组也要扫——`SETUP_STATEMENTS` 就是一组查询，
    漏掉它就等于给约束语句开了后门。
    """
    found: dict[str, str] = {}
    for name, value in vars(module).items():
        if not name.isupper():
            continue
        if isinstance(value, str) and looks_like_cypher(value):
            found[name] = value
        elif isinstance(value, tuple):
            for index, item in enumerate(value):
                if isinstance(item, str) and looks_like_cypher(item):
                    found[f"{name}[{index}]"] = item
    return found


def audit_queries(
    queries: dict[str, str], *, allow_structural: frozenset[str] = frozenset()
) -> list[str]:
    """审计一组具名查询，返回问题列表（空表示全部通过）。

    `allow_structural` 里的查询名会跳过**结构性**检查（但仍会查禁用属性名）。
    这个开关必须配合**写明理由**使用——见 `cypher.STRUCTURAL_ALLOWLIST`。
    """
    problems: list[str] = []
    for name, query in sorted(queries.items()):
        for problem in check_query(query, allow_structural=name in allow_structural):
            problems.append(f"{name}: {problem}")
    return problems
