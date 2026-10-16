"""Agent 的工具：**4 个侦查工具 + 1 个产出结论工具**。

## ⚠️ 参数上界由**服务端**强制，不是靠提示词约定

工具契约里写了 `hops ≤ 3`、`top_k ≤ 5`、`limit ≤ 500`、`max_length ≤ 8`
（见 `docs/06`）。这些**不是"建议"**——

> **提示词挡不住一个跑飞的 Agent。** 邻域随跳数指数增长，
> 一个 `hops=10` 的请求就能把内存打满。
>
> 所以每个上界都在 `Toolbox.call()` 里**真实校验**。

## ⚠️ 参数错误要**返回给模型**，而不是抛异常中断循环

模型调错参数是**正常现象**，不是程序错误。正确做法是把错误信息作为
工具结果返回给它，**让它自己纠正**。

抛异常会把"模型犯了个小错"升级成"整个案例失败"——
**而产线项目的教训正是"工程缺陷会让所有指标失真"。**

## ⚠️ 工具粒度是**刻意的**（这是产线踩过的坑）

上一个项目里，诊断工具一次只能查**一个传感器**，
Agent 把 4 步预算全花在逐个查询上，**压根没走到给结论**，跑出 0% 检出率。

**看起来像"Agent 能力不行"，实际是工具设计的锅。**

所以这里的 `investigate_account` **一次返回整个邻域**（已按账户对归并），
而不是"一次查一条边"。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from neo4j import Driver

from aml_agent.embed import Embedder
from aml_agent.graph import queries
from aml_agent.knowledge import Pattern
from aml_agent.retrieve import PatternMatch, search
from aml_agent.retrieve.fusion import group_facts

#: 参数上界。**与 `docs/06` 的工具契约一一对应。**
MAX_HOPS = 3
MAX_TOP_K = 5
MAX_LIMIT = 500
MAX_PATH_LENGTH = 8

#: 结论工具的名字。**它是"是否产出结论"的唯一判据**——
#: 实测验证过：模型可能"说"它调用了这个名字，但只有真的 `tool_calls` 才算，
#: 所以这个判据是可数的、不会误判。
CONCLUDE_TOOL_NAME = "submit_conclusion"


@dataclass(frozen=True, slots=True)
class ToolResult:
    """一次工具调用的结果。"""

    name: str
    #: 塞回对话历史的内容（JSON 字符串）
    content: str
    ok: bool = True
    #: 供程序化使用（评测、测试断言）
    payload: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  工具的对外描述（**描述本身就是 Agent 设计的一部分**）
# --------------------------------------------------------------------------- #

INVESTIGATION_TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "investigate_account",
            "description": (
                "侦查一个账户的邻域，一次返回它相关的全部交易边（按对方账户归并，"
                "含笔数、金额区间、时间跨度）。**这是你应该最先调用的工具**——"
                "一次调用就能看到全貌。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "account_key": {
                        "type": "string",
                        "description": "账户键，形如 amlsim:4122",
                    },
                    "hops": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_HOPS,
                        "description": f"向外扩展几跳，1–{MAX_HOPS}。默认 1。",
                    },
                },
                "required": ["account_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_transactions",
            "description": (
                "查两个账户之间的全部交易，按时间排序。"
                "当你想知道'同一对账户之间转了多少次、时间上怎么分布'时用它。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sender": {"type": "string"},
                    "receiver": {"type": "string"},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_LIMIT,
                        "description": f"最多返回几条，默认 50，上限 {MAX_LIMIT}。",
                    },
                },
                "required": ["sender", "receiver"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trace_path",
            "description": (
                "求两个账户之间的**最短资金路径**。"
                "当你想验证'资金是不是绕回来了'或'这两点之间有没有通路'时用它。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "max_length": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_PATH_LENGTH,
                        "description": f"路径最长几跳，默认 {MAX_PATH_LENGTH}。",
                    },
                },
                "required": ["source", "target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "match_patterns",
            "description": (
                "把你观察到的结构用自然语言描述出来，"
                "去匹配已知的洗钱手法（知识库有图结构模式与监管类型学两类）。"
                "**返回的是'像什么'，不是'就是什么'**，请据此谨慎表述。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "structure_description": {
                        "type": "string",
                        "description": "用自然语言描述你看到的资金结构",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_TOP_K,
                        "description": f"返回几条，默认 3，上限 {MAX_TOP_K}。",
                    },
                },
                "required": ["structure_description"],
            },
        },
    },
)

#: 结论工具。**它的描述要写明"必须调用它才算完成"**——
#: 工具描述是 Agent 设计的一部分，不是文档。
CONCLUDE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": CONCLUDE_TOOL_NAME,
        "description": (
            "给出最终结论。**必须调用它才算完成任务**——只输出文字不算。"
            "如果信息不足，就如实说'信息不足'，并说明还缺什么。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "suspicious": {
                    "type": "boolean",
                    "description": "是否存在可疑迹象",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "置信度 0–1",
                },
                "reason": {
                    "type": "string",
                    "description": "判断理由。**要引用你实际查到的数据**，不要空泛。",
                },
                "missing": {
                    "type": "string",
                    "description": "还缺什么信息/哪里没查完。没有就留空。",
                },
            },
            "required": ["suspicious", "confidence", "reason"],
        },
    },
}


# --------------------------------------------------------------------------- #
#  工具箱
# --------------------------------------------------------------------------- #


class ToolArgumentError(ValueError):
    """参数不合法。**这个错误会作为工具结果返回给模型**，让它自己纠正。"""


@dataclass
class Toolbox:
    """把工具接到真实的图库与知识库上。

    ⚠️ 它持有一个 Neo4j `driver` 和检索用的索引——
    **这些是外部资源，由调用方负责生命周期**（创建与关闭），
    工具自己不管理它们。
    """

    driver: Driver
    patterns: Sequence[Pattern]
    index: Any
    embedder: Embedder
    min_similarity: float = 0.35
    #: 记录每个工具被调用了多少次——**用于诊断"Agent 是不是在瞎调"**
    call_counts: dict[str, int] = field(default_factory=dict)

    def schemas(self, *, include_conclude: bool = True) -> list[dict[str, Any]]:
        """本轮要提供给模型的工具清单。"""
        schemas = list(INVESTIGATION_TOOL_SCHEMAS)
        if include_conclude:
            schemas.append(CONCLUDE_TOOL_SCHEMA)
        return schemas

    def call(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        """执行一次工具调用。**任何参数问题都返回错误结果，不抛异常。**"""
        self.call_counts[name] = self.call_counts.get(name, 0) + 1
        try:
            if name == "investigate_account":
                return self._investigate(arguments)
            if name == "get_transactions":
                return self._transactions(arguments)
            if name == "trace_path":
                return self._trace(arguments)
            if name == "match_patterns":
                return self._match(arguments)
        except ToolArgumentError as exc:
            # 参数错 → **返回给模型让它自己改**，而不是中断整个循环
            return ToolResult(
                name=name,
                content=f"参数错误：{exc}",
                ok=False,
                payload={"error": str(exc)},
            )
        except Exception as exc:  # noqa: BLE001 - 工具失败要让模型知道，而不是让流程崩
            return ToolResult(
                name=name,
                content=f"工具执行失败（{type(exc).__name__}）：{str(exc)[:200]}",
                ok=False,
                payload={"error": type(exc).__name__},
            )
        return ToolResult(
            name=name,
            content=f"未知工具：{name}",
            ok=False,
            payload={"error": "unknown_tool"},
        )

    # ---- 各工具的实现 ---------------------------------------------------- #

    @staticmethod
    def _as_int(arguments: Mapping[str, Any], key: str, default: int, low: int, high: int) -> int:
        """取一个整数参数并**强制上下界**。

        这是"上界由服务端强制"的执行点。
        """
        raw = arguments.get(key, default)
        if isinstance(raw, str):
            try:
                raw = int(raw)
            except ValueError as exc:
                raise ToolArgumentError(f"{key} 应该是整数，收到 {raw!r}") from exc
        if not isinstance(raw, int) or isinstance(raw, bool):
            raise ToolArgumentError(f"{key} 应该是整数，收到 {type(raw).__name__}")
        if not low <= raw <= high:
            raise ToolArgumentError(f"{key} 必须在 {low}–{high} 之间，收到 {raw}")
        return raw

    @staticmethod
    def _as_key(arguments: Mapping[str, Any], key: str) -> str:
        raw = arguments.get(key)
        if not isinstance(raw, str) or not raw.strip():
            raise ToolArgumentError(f"{key} 必须是非空字符串，收到 {raw!r}")
        return raw.strip()

    def _investigate(self, arguments: Mapping[str, Any]) -> ToolResult:
        key = self._as_key(arguments, "account_key")
        hops = self._as_int(arguments, "hops", 1, 1, MAX_HOPS)

        neighborhood = queries.neighborhood(self.driver, key, hops=hops, limit=MAX_LIMIT)
        groups = group_facts(neighborhood.edges)

        # ⚠️ 截断必须**明确写进工具结果**——
        # 否则模型会把"没查完"当成"没有更多关联"
        payload: dict[str, Any] = {
            "account": neighborhood.center,
            "hops": neighborhood.hops,
            "edge_count": len(neighborhood.edges),
            "relation_count": len(groups),
            "relations": [
                {
                    "counterparty": (
                        group.receiver if group.sender == neighborhood.center else group.sender
                    ),
                    "direction": "out" if group.sender == neighborhood.center else "in",
                    "count": group.count,
                    "amount": group.describe_amount(),
                    "time_span": group.describe_time(),
                }
                for group in groups
            ],
        }
        if neighborhood.truncated:
            payload["truncated"] = (
                f"结果被截断：只取了前 {len(neighborhood.edges)} 条边，可能还有更多关联未返回"
            )
        return ToolResult(name="investigate_account", content=_to_json(payload), payload=payload)

    def _transactions(self, arguments: Mapping[str, Any]) -> ToolResult:
        sender = self._as_key(arguments, "sender")
        receiver = self._as_key(arguments, "receiver")
        limit = self._as_int(arguments, "limit", 50, 1, MAX_LIMIT)

        edges = queries.transactions_between(self.driver, sender, receiver, limit=limit)
        payload = {
            "sender": sender,
            "receiver": receiver,
            "count": len(edges),
            "transactions": [
                {"txId": edge.tx_id, "amount": edge.amount, "timestamp": edge.timestamp}
                for edge in edges
            ],
        }
        return ToolResult(name="get_transactions", content=_to_json(payload), payload=payload)

    def _trace(self, arguments: Mapping[str, Any]) -> ToolResult:
        source = self._as_key(arguments, "source")
        target = self._as_key(arguments, "target")
        max_length = self._as_int(arguments, "max_length", MAX_PATH_LENGTH, 1, MAX_PATH_LENGTH)

        edges = queries.shortest_path(self.driver, source, target, max_length=max_length)
        payload: dict[str, Any] = {
            "source": source,
            "target": target,
            "max_length": max_length,
            "found": bool(edges),
            "path": [
                {
                    "from": edge.sender,
                    "to": edge.receiver,
                    "amount": edge.amount,
                    "timestamp": edge.timestamp,
                }
                for edge in edges
            ],
        }
        if not edges:
            # ⚠️ **必须说清"没找到"的确切含义**——
            # 否则模型会把"这个长度内没有路径"读成"两者没有关联"
            payload["note"] = (
                f"在 {max_length} 跳以内**没有找到**路径。"
                "这不等于两者没有关联——只是在这个长度限制内没找到。"
            )
        return ToolResult(name="trace_path", content=_to_json(payload), payload=payload)

    def _match(self, arguments: Mapping[str, Any]) -> ToolResult:
        description = self._as_key(arguments, "structure_description")
        top_k = self._as_int(arguments, "top_k", 3, 1, MAX_TOP_K)

        matches: list[PatternMatch] = search(
            description,
            self.index,
            self.patterns,
            self.embedder,
            top_k=top_k,
            min_similarity=self.min_similarity,
        )
        payload: dict[str, Any] = {
            "query": description,
            "matches": [
                {
                    "name": match.pattern.name,
                    "category": match.pattern.category,
                    "similarity": round(match.similarity, 3),
                    "definition": match.pattern.definition,
                    "source": match.pattern.source,
                }
                for match in matches
            ],
        }
        if not matches:
            # **空结果是有价值的结论**，不是"没查到"
            payload["note"] = (
                f"没有相似度超过 {self.min_similarity} 的已知手法。"
                "这本身是个结论：**它不像任何已知手法**，而不是'查不到'。"
            )
        return ToolResult(name="match_patterns", content=_to_json(payload), payload=payload)


def _to_json(payload: Mapping[str, Any]) -> str:
    """转成工具结果字符串。

    `ensure_ascii=False` 是刻意的——中文股名、手法名要保持可读，
    转义成 `\\uXXXX` 会让模型多一层解析负担。
    """
    import json

    return json.dumps(payload, ensure_ascii=False)
