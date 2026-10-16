"""Agent 循环：工具调用 + **预留强制产出步**。

## 这个循环要解决的核心问题

**「什么都没找到」和「根本没给答案」是两件完全不同的事。**

- 前者是**能力问题**——可以量化、可以改进；
- 后者是**工程缺陷**——它同时**压低检出率**、又**虚假地压低误报率**，
  **会让所有指标失真**。

**而产线项目正是被后者坑过**：15/50 个案例耗尽步数从未给出答案，
却被统计成了"无异常"。

## 三个机制

### ① 用**工具调用**作为"是否给出结论"的判据

不是解析模型输出的文本，而是看它**有没有真的调用 `submit_conclusion`**。

**实测验证过这个判据的可靠性**：模型会"说"它调用了这个工具，
但只有真的 `tool_calls` 才算——

```text
没调工具，直接说话：submit_conclusion("信息不足，无法进行反洗钱分析。")
```

**所以这个判据是可数的、不会误判的。**

### ② 最后一步**只给结论工具**

这就是"预留强制产出步"：

```text
第 1..N-1 步：4 个侦查工具 + 结论工具
第 N 步：    只有结论工具          ← 它必须给出结论，没有别的选择
```

**这是"一定给结论"的机制保证，不是靠提示词求它。**

### ③ 三种异常**分别计数**，绝不合并

| 字段 | 含义 |
|---|---|
| `exhausted` | 步数耗尽，**从未产出结论** ← 工程缺陷 |
| `forced` | 结论来自强制产出步（说明自由步数不够用） |
| `premature` | **没做任何侦查就给结论** ← 实测发现模型会这样 |

`premature` 是实测发现的：问"介绍你自己"，它直接调了 `submit_conclusion`。
**不拦截它，但要计数**——因为"没查就下结论"和"查了才下结论"可信度完全不同。

## ⚠️ 步数预算

实测单次调用 **10–17 秒**（冷启动 60 秒）。所以步数必须省着用：

> **默认 `max_steps=6`，按每步 15 秒算约 1.5 分钟/案例。**
>
> 这直接来自产线教训：**工具粒度太细 + 步数少 = Agent 跑不完**。
> 所以我们的 `investigate_account` 一次给够，让 2–3 步就能完成。
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from aml_agent.agent.llm import ChatModel, ToolCall
from aml_agent.agent.tools import (
    CONCLUDE_TOOL_NAME,
    CONCLUDE_TOOL_SCHEMA,
    Toolbox,
    ToolResult,
)

#: 默认步数预算。
#:
#: **不是越大越好**：一步约 15 秒，10 步就是 2.5 分钟一个案例，
#: 30 个案例的评测要一个多小时——**慢到不会有人跑**。
#: 所以宁可把工具做粗，也不要靠加步数。
DEFAULT_MAX_STEPS = 6

#: 强制产出步给几次机会。
#:
#: **实测：在干净对话里 1 次就够（2/2 成功），但真跑时对话很脏，
#: 模型返回了空响应。** 所以给 2 次——第 2 次会用不容含糊的措辞再要求一遍。
#:
#: 不给更多，是因为**每一步都花 10–17 秒**，而"它就是不配合"也是有效结论
#: （会被如实记成 `exhausted`，而不是掩盖过去）。
FORCED_STEP_ATTEMPTS = 2

#: 强制产出步最后一次机会时用的措辞。
#:
#: 刻意**不留余地**：不写"请考虑给出结论"，而写"必须调用，文字不算"。
#: 因为温和的措辞正是第一次失败的原因之一。
FORCED_NUDGE = (
    "你还没有给出结论。**文字回复不算完成**——"
    "必须调用 submit_conclusion 工具，把 suspicious / confidence / reason 填好。"
    "如果信息不足，就把 suspicious 填 false、confidence 填低，并在 reason 里说明缺什么。"
)

#: 系统提示词。
#:
#: ⚠️ **提示词里刻意不写"大多数案例是正常的"这类先验**——
#: 上个项目正是这么写的（"Most windows are healthy"），
#: 而实际有一半案例是故障的，**结果把 Agent 带偏了**。
#:
#: **不告诉它先验，让它自己查。**
SYSTEM_PROMPT = """你是一名反洗钱分析师助手。

工作方式：
1. 先用 investigate_account 看账户全貌——
   **它同时返回该账户在全图的分位（baseline）**，所以通常一次调用就够了；
2. 只有当你想查**别的账户**（比如某个对手方）的位置时，才用 get_detector_scores；
3. 确实需要更多细节时再用其他工具（查两账户间明细、追路径、匹配已知手法）；
4. 最后**必须调用 submit_conclusion 给出结论**——只输出文字不算完成。

原则：
- **只依据工具返回的事实说话**，不要臆测。
- **分位是相对排名，不是判断。** 分位高不等于有问题；
  返回里会带 `measuredSignals`（各信号的实测等级），**用它决定该不该采信某个信号**。
- **不要为了"多查一点"而浪费步数**——步数预算很紧（每步十几秒）。
  信息够下判断就给结论，"信息不足"也是一个合格的结论。
- 工具结果里如果有 `truncated`、`note` 字段，**必须读它**——
  它说明"没查完"或"没找到"，这两者都不等于"没有异常"。
- 判断理由里**要引用你实际查到的数据**（笔数、金额、时间跨度、分位），不要空泛。
"""


@dataclass(frozen=True, slots=True)
class Conclusion:
    """模型的结论。"""

    suspicious: bool
    confidence: float
    reason: str
    missing: str = ""

    @classmethod
    def from_arguments(cls, arguments: Mapping[str, Any]) -> Conclusion:
        """从工具参数构造。

        ⚠️ **缺字段要能容错**——模型不保证每次都把参数给全。
        但 `confidence` 要夹到 [0,1]：模型给出 1.5 或 -0.2 都是出现过的。
        """
        raw_confidence = arguments.get("confidence", 0.0)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            # 模型给过 "很高" 这种值——**容错，但不掩盖**：
            # 置信度归零，让下游看得出"这条结论没有置信度"
            confidence = 0.0
        return cls(
            suspicious=bool(arguments.get("suspicious", False)),
            confidence=min(1.0, max(0.0, confidence)),
            reason=str(arguments.get("reason") or "").strip(),
            missing=str(arguments.get("missing") or "").strip(),
        )


@dataclass(frozen=True, slots=True)
class Step:
    """循环里的一步。"""

    index: int
    tool_calls: tuple[ToolCall, ...]
    results: tuple[ToolResult, ...]
    content: str
    seconds: float
    #: 这一步是不是**强制产出步**（只给了结论工具）
    forced: bool

    @property
    def used_tools(self) -> bool:
        return bool(self.tool_calls)


@dataclass(frozen=True, slots=True)
class AgentRun:
    """一次完整的 Agent 运行。"""

    question: str
    #: `None` 表示**从未产出结论**——与"结论是无异常"是两回事
    conclusion: Conclusion | None
    steps: tuple[Step, ...]
    #: 步数耗尽，从未给出结论 ← **工程缺陷，必须单独统计**
    exhausted: bool
    #: 结论来自强制产出步（说明自由步数不够用）
    forced: bool
    #: 没做任何侦查就给结论
    premature: bool
    seconds: float
    tool_counts: Mapping[str, int] = field(default_factory=dict)
    #: 未产出结构化结论时，模型最后说的话。
    #:
    #: ⚠️ **它不是结论。** 放在这里仅供人看（写报告、排查为什么没结论），
    #: **不参与任何指标**——`exhausted` 仍然是 `True`。
    #:
    #: 一段自由文本没有 `suspicious`、没有 `confidence`，
    #: 硬把它当成"结论：未发现异常"正是本项目一直在反对的那种混淆。
    final_text: str = ""

    @property
    def produced_conclusion(self) -> bool:
        return self.conclusion is not None

    def summary(self) -> dict[str, Any]:
        """给评测与日志用的摘要。**单独暴露 `exhausted`，方便 CI 断言它为 0。**"""
        return {
            "question": self.question,
            "conclusion": self.conclusion.reason[:80] if self.conclusion else None,
            "suspicious": self.conclusion.suspicious if self.conclusion else None,
            "confidence": self.conclusion.confidence if self.conclusion else None,
            "steps": len(self.steps),
            "forced": self.forced,
            "premature": self.premature,
            "exhausted": self.exhausted,
            "seconds": round(self.seconds, 1),
            "tools": dict(self.tool_counts),
            # 只报长度，不报内容——**它不是结论，不该被当成结论读**
            "final_text_chars": len(self.final_text),
        }


def run(
    question: str,
    *,
    chat: ChatModel,
    toolbox: Toolbox,
    max_steps: int = DEFAULT_MAX_STEPS,
    system_prompt: str = SYSTEM_PROMPT,
) -> AgentRun:
    """跑一次 Agent。

    步骤分配：
    - 第 1 .. `max_steps-1` 步：**自由侦查**（全部工具）
    - 第 `max_steps` 步：**强制产出**（只给结论工具）
    """
    if max_steps < 2:
        raise ValueError(
            "max_steps 至少为 2——**必须给「强制产出」留一步**，否则和「没有终止机制」是一回事"
        )

    started = time.monotonic()
    # 类型是 `Mapping` 而不是 `dict`：`response.raw` 是 Mapping，
    # 而它必须能原样塞回对话历史（Ollama 要求）。写成 dict 会逼出一个无谓的拷贝。
    messages: list[Mapping[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    steps: list[Step] = []
    investigated = False
    last_text = ""

    for index in range(max_steps):
        is_forced_step = index == max_steps - 1
        # ⚠️ 强制产出步**只给结论工具**——它没有别的选择
        schemas = (
            [CONCLUDE_TOOL_SCHEMA] if is_forced_step else toolbox.schemas(include_conclude=True)
        )
        # ⚠️ 但"只给一个工具"**不等于"它一定会用"**。
        #
        # 实测：在干净对话里 2/2 成功；但真跑时对话很脏
        # （长长一段"只说话"、一条失败的工具结果），它就返回了空响应。
        #
        # 所以强制步**给有限次机会**（每一轮都记为一个 Step，因为每轮都花时间）。
        attempts = FORCED_STEP_ATTEMPTS if is_forced_step else 1

        for attempt in range(attempts):
            step_started = time.monotonic()
            response = chat.chat(messages, tools=schemas)

            if not response.tool_calls:
                steps.append(
                    Step(
                        index=index,
                        tool_calls=(),
                        results=(),
                        content=response.content,
                        seconds=time.monotonic() - step_started,
                        forced=is_forced_step,
                    )
                )
                if response.content.strip():
                    last_text = response.content.strip()
                messages.append(response.raw or {"role": "assistant", "content": response.content})
                if attempt < attempts - 1:
                    # 强制步最后一搏：把要求说得**不容含糊**
                    messages.append({"role": "user", "content": FORCED_NUDGE})
                elif not is_forced_step:
                    messages.append({"role": "user", "content": "请调用工具继续。"})
                continue

            # ---- 有工具调用 ------------------------------------------------ #
            conclusion: Conclusion | None = None
            results: list[ToolResult] = []
            for call in response.tool_calls:
                if call.name == CONCLUDE_TOOL_NAME:
                    conclusion = Conclusion.from_arguments(call.arguments)
                    break
                investigated = True
                results.append(toolbox.call(call.name, call.arguments))

            steps.append(
                Step(
                    index=index,
                    tool_calls=response.tool_calls,
                    results=tuple(results),
                    content=response.content,
                    seconds=time.monotonic() - step_started,
                    forced=is_forced_step,
                )
            )

            if conclusion is not None:
                return AgentRun(
                    question=question,
                    conclusion=conclusion,
                    steps=tuple(steps),
                    exhausted=False,
                    forced=is_forced_step,
                    # **没做过任何一次侦查就给结论** —— 实测发现模型会这样
                    premature=not investigated,
                    seconds=time.monotonic() - started,
                    tool_counts=dict(toolbox.call_counts),
                )

            # 把 assistant 消息和工具结果塞回历史（顺序不能反）
            messages.append(response.raw or {"role": "assistant", "content": ""})
            for result in results:
                messages.append({"role": "tool", "content": result.content})
            break  # 这一步调过工具了，进下一步

    # ---- 步数耗尽，从未产出结论 ------------------------------------------- #
    #
    # ⚠️ 注意这里**没有**把 `last_text` 包装成一个 Conclusion。
    #
    # 一段自由文本**不是结论**——它没有 `suspicious`、没有 `confidence`。
    # 硬把它记成"结论：未发现异常"就是本项目一直在反对的那种混淆：
    # **"我看了，没问题"和"我没给出结论"会被算成同一件事。**
    #
    # 所以文本单独放在 `final_text` 里供人看，而 `exhausted=True` 保持不变，
    # **指标里它永远属于"未产出结论"这一类。**
    return AgentRun(
        question=question,
        conclusion=None,
        steps=tuple(steps),
        exhausted=True,
        forced=False,
        premature=False,
        seconds=time.monotonic() - started,
        tool_counts=dict(toolbox.call_counts),
        final_text=last_text,
    )


def investigate_tool_names() -> Sequence[str]:
    """侦查工具的名字（不含结论工具）。

    ⚠️ 从 `tools.py` 的 schema 里**推**出来，而不是手写一份——
    **手写清单会在加了新工具之后忘记更新**（这个教训在本项目里出现过三次）。
    """
    from aml_agent.agent.tools import INVESTIGATION_TOOL_SCHEMAS

    return tuple(schema["function"]["name"] for schema in INVESTIGATION_TOOL_SCHEMAS)
