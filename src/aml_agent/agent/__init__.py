"""Agent 编排层：工具调用 + 分层侦查 + **预留强制产出步**。

## 三个模块

| 模块 | 职责 |
|---|---|
| `llm` | 对话模型接口（`OllamaChat` 真实 / `ScriptedChat` 测试） |
| `tools` | 4 个侦查工具 + 1 个结论工具；**参数上界由服务端强制** |
| `loop` | 循环 + **强制产出步** + 三种异常分别计数 |

## 为什么接口化是必需的（不是"为了好看"）

真实实现走本地 Ollama，**但 CI 里跑不了它**——单次调用 10–17 秒，
一个循环好几步。**依赖它的测试会一直跳过，而"一直跳过"等于没有测试。**

所以 `ChatModel` 有两个实现，测试用 `ScriptedChat` 按剧本走，
**整个循环逻辑都能在毫秒级被验证**。

## 这个包守的那条原则

> **「什么都没找到」和「根本没给答案」是两件完全不同的事。**
>
> 前者是能力问题，后者是**工程缺陷**——它同时压低检出率、
> 又虚假地压低误报率，**会让所有指标失真**。
>
> 所以 `AgentRun.exhausted` 是**独立字段**，永远不会被折算成"无异常"。
"""

from aml_agent.agent.baseline import (
    ACCOUNT_FEATURES,
    AccountBaseline,
    BaselineIndex,
    FeaturePosition,
    numpy_percentile,
)
from aml_agent.agent.llm import (
    DEFAULT_MODEL,
    ChatModel,
    ChatResponse,
    LLMError,
    OllamaChat,
    ScriptedChat,
    ToolCall,
    says,
    tool_call,
)
from aml_agent.agent.loop import (
    DEFAULT_MAX_STEPS,
    FORCED_NUDGE,
    FORCED_STEP_ATTEMPTS,
    SYSTEM_PROMPT,
    AgentRun,
    Conclusion,
    Step,
    investigate_tool_names,
    run,
)
from aml_agent.agent.tools import (
    CONCLUDE_TOOL_NAME,
    CONCLUDE_TOOL_SCHEMA,
    INVESTIGATION_TOOL_SCHEMAS,
    MAX_HOPS,
    MAX_LIMIT,
    MAX_PATH_LENGTH,
    MAX_TOP_K,
    ToolArgumentError,
    Toolbox,
    ToolResult,
)

__all__ = [
    "ACCOUNT_FEATURES",
    "CONCLUDE_TOOL_NAME",
    "CONCLUDE_TOOL_SCHEMA",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MODEL",
    "FORCED_NUDGE",
    "FORCED_STEP_ATTEMPTS",
    "INVESTIGATION_TOOL_SCHEMAS",
    "MAX_HOPS",
    "MAX_LIMIT",
    "MAX_PATH_LENGTH",
    "MAX_TOP_K",
    "SYSTEM_PROMPT",
    "AccountBaseline",
    "AgentRun",
    "BaselineIndex",
    "ChatModel",
    "ChatResponse",
    "Conclusion",
    "FeaturePosition",
    "LLMError",
    "OllamaChat",
    "ScriptedChat",
    "Step",
    "ToolArgumentError",
    "ToolCall",
    "ToolResult",
    "Toolbox",
    "investigate_tool_names",
    "numpy_percentile",
    "run",
    "says",
    "tool_call",
]
