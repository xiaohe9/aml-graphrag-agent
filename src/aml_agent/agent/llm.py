"""Agent 的 LLM 客户端：**接口 + 两个实现**。

## 为什么又做接口（和嵌入层同样的理由）

真实实现走本地 Ollama（`qwen3:4b`，零 API 成本），但 **CI 里跑不了它**：
实测单次调用 **10–17 秒**（冷启动时到 60 秒），而一个 Agent 循环要好几步。

**依赖它的测试在 CI 里会一直跳过——而"一直跳过"等于没有测试。**

所以：

| 实现 | 用途 |
|---|---|
| `OllamaChat` | 真实跑 |
| `ScriptedChat` | **测试用**：按剧本返回预设的响应 |

## 实测数据（决定了步数预算）

```text
冷启动            54–66 秒      ← 第一次，模型载入
热调用            10–17 秒      ← 波动很大，按 15 秒做预算
think=False       17.2 秒       ← 反而比默认慢
/no_think（提示）   9.8 秒       ← 只快 1.3 倍
工具调用正确率     4/4 变体全部正确
```

**结论**：thinking 开关收益很小，**真正的变量是模型有没有热**。
按 **15 秒/步** 做容量规划——所以 **步数必须省着用**。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:4b"


class LLMError(RuntimeError):
    """LLM 服务不可用或返回了意料之外的形状。

    **失败要响亮**——如果这里静默退化，Agent 会"什么都没查到"，
    而调用方分不清那是"确实没有"还是"模型没响应"。
    """


# --------------------------------------------------------------------------- #
#  消息与响应
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ToolCall:
    """模型请求调用一个工具。"""

    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """一轮对话的结果。"""

    #: 模型请求调用的工具（**可能为空**——那就是它在"说话"而不是"做事"）
    tool_calls: tuple[ToolCall, ...]
    #: 纯文本内容
    content: str
    #: 原始消息体，要原样塞回对话历史（Ollama 要求）
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_tool_call(self) -> bool:
        return bool(self.tool_calls)


@runtime_checkable
class ChatModel(Protocol):
    """对话模型的接口。"""

    def chat(
        self, messages: Sequence[Mapping[str, Any]], *, tools: Sequence[Mapping[str, Any]]
    ) -> ChatResponse:
        """跑一轮对话。

        `tools` 为空序列时表示"这一轮不给任何工具"——
        调用方会用它来实现**强制产出步**。
        """
        ...


def _parse_tool_calls(message: Mapping[str, Any]) -> tuple[ToolCall, ...]:
    """从 Ollama 的响应里解析工具调用。

    ⚠️ `arguments` 有时是**字符串**（模型输出的 JSON 文本）而不是对象。
    实测踩到过：`{'account_key': 'amlsim:4122', 'hops': 1}` 有时是 dict，
    有时是它的字符串形式。**不做归一化的话，下游会拿到一个字符串当参数用。**
    """
    calls: list[ToolCall] = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function", {})
        name = function.get("name")
        if not isinstance(name, str):
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                # 解析不了就保留原样——让调用方看到真实情况，
                # 而不是悄悄替换成一个空 dict
                arguments = {"_raw": arguments}
        if not isinstance(arguments, Mapping):
            arguments = {"_raw": arguments}
        calls.append(ToolCall(name=name, arguments=arguments))
    return tuple(calls)


# --------------------------------------------------------------------------- #
#  真实实现
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class OllamaChat:
    """走本地 Ollama 的对话模型。**零 API 成本。**"""

    model: str = DEFAULT_MODEL
    host: str = DEFAULT_OLLAMA_HOST
    timeout: float = 300.0
    #: 关闭思考模式。实测收益很小（9.8 vs 12.6 秒），默认不开。
    think: bool | None = None
    #: 采样温度。
    #:
    #: ⚠️ **Agent 必须用低温**，这不只是"更好"的问题——
    #: 实测同一个案例跑两次：一次 2 步给出正确结论，另一次 5 步还耗尽步数。
    #: **一个行为不可复现的 Agent 根本没法评测**（同一输入两次结果不同，
    #: 测出来的任何数字都没有意义）。
    #:
    #: 默认 0.1：保留极少量随机性以避免完全退化的重复，但足够稳定。
    temperature: float = 0.1

    def chat(
        self, messages: Sequence[Mapping[str, Any]], *, tools: Sequence[Mapping[str, Any]]
    ) -> ChatResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            # Ollama 把采样参数放在 `options` 里，不是顶层
            "options": {"temperature": self.temperature},
        }
        if tools:
            body["tools"] = list(tools)
        if self.think is not None:
            body["think"] = self.think

        request = urllib.request.Request(  # noqa: S310 - host 是本机常量
            f"{self.host}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(
                f"连不上本地 Ollama（{self.host}）：{exc}\n"
                "先确认服务在跑：ollama serve\n"
                f"再确认模型已拉取：ollama pull {self.model}"
            ) from exc

        message = payload.get("message")
        if not isinstance(message, Mapping):
            raise LLMError(f"Ollama 返回里没有 message 字段：{str(payload)[:200]}")

        return ChatResponse(
            tool_calls=_parse_tool_calls(message),
            content=str(message.get("content") or ""),
            raw=message,
        )


# --------------------------------------------------------------------------- #
#  测试实现
# --------------------------------------------------------------------------- #


@dataclass
class ScriptedChat:
    """按剧本返回响应的假模型。**测试用。**

    ## 为什么它不能只是个"永远返回固定值"的桩

    Agent 的行为**取决于上一步返回了什么**（这轮调工具 → 下轮基于结果决定）。
    一个固定返回值的桩会让所有测试都只覆盖一条路径。

    所以这里按**轮次**消费剧本：第 N 次调用返回剧本里的第 N 项。
    剧本用完后再调用会抛错——**因为"Agent 比预期多跑了步"本身就是要发现的问题**。

    ## 它记下了什么

    `calls` 记录每次收到的 `messages` 和 `tools`——
    测试可以据此断言"**强制产出步确实只给了 conclude 工具**"。
    """

    responses: list[ChatResponse]
    calls: list[dict[str, Any]] = field(default_factory=list)
    _index: int = 0

    def chat(
        self, messages: Sequence[Mapping[str, Any]], *, tools: Sequence[Mapping[str, Any]]
    ) -> ChatResponse:
        self.calls.append({"messages": list(messages), "tools": list(tools)})
        if self._index >= len(self.responses):
            raise LLMError(
                f"剧本用完了：第 {self._index + 1} 次调用没有预设响应。\n"
                "**Agent 比预期多跑了步**——这通常意味着终止条件没有生效。"
            )
        response = self.responses[self._index]
        self._index += 1
        return response


def tool_call(name: str, **arguments: Any) -> ChatResponse:
    """构造一个"调用工具"的剧本项。测试里用它拼剧本。"""
    return ChatResponse(
        tool_calls=(ToolCall(name=name, arguments=arguments),),
        content="",
        raw={
            "role": "assistant",
            "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
        },
    )


def says(text: str) -> ChatResponse:
    """构造一个"只说话、不调工具"的剧本项。"""
    return ChatResponse(tool_calls=(), content=text, raw={"role": "assistant", "content": text})


def timed(fn: Any) -> Any:
    """给一次调用计时。测试与真实运行都用得上。"""
    started = time.monotonic()
    result = fn()
    return result, time.monotonic() - started
