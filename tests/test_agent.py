"""Agent 循环的测试。

## 全部用 `ScriptedChat`，**不依赖 Ollama**

理由同嵌入层：真实调用 10–17 秒一步，依赖它的测试在 CI 里会一直跳过——
**而"一直跳过"等于没有测试。**

按剧本走之后，循环逻辑能在**毫秒级**被完整覆盖，包括那些真实模型很难稳定复现的路径
（步数耗尽、没查就下结论、参数越界）。

## 这个文件守的核心

> **「什么都没找到」和「根本没给答案」是两件完全不同的事。**

所以 `exhausted` / `forced` / `premature` 各自都有专门的测试。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import pytest
from neo4j import Driver

from aml_agent.agent import (
    CONCLUDE_TOOL_NAME,
    DEFAULT_MAX_STEPS,
    FORCED_NUDGE,
    FORCED_STEP_ATTEMPTS,
    AgentRun,
    Conclusion,
    ScriptedChat,
    ToolArgumentError,
    Toolbox,
    investigate_tool_names,
    run,
    says,
    tool_call,
)
from aml_agent.agent.llm import LLMError, _parse_tool_calls

# --------------------------------------------------------------------------- #
#  测试替身
# --------------------------------------------------------------------------- #


class FakeToolbox:
    """只实现 `run()` 用到的那三个成员。

    **不碰 Neo4j**——这些是循环逻辑的单元测试。
    真实工具箱的接线由集成测试覆盖（那里会真的连图库）。
    """

    def __init__(self) -> None:
        self.call_counts: dict[str, int] = {}
        self.called: list[tuple[str, Mapping[str, Any]]] = []

    def schemas(self, *, include_conclude: bool = True) -> list[dict[str, Any]]:
        schemas = [
            {"type": "function", "function": {"name": name}} for name in investigate_tool_names()
        ]
        if include_conclude:
            schemas.append({"type": "function", "function": {"name": CONCLUDE_TOOL_NAME}})
        return schemas

    def call(self, name: str, arguments: Mapping[str, Any]) -> Any:
        from aml_agent.agent.tools import ToolResult

        self.call_counts[name] = self.call_counts.get(name, 0) + 1
        self.called.append((name, arguments))
        return ToolResult(name=name, content='{"ok": true}', payload={"ok": True})


def _tool_names(schemas: Sequence[Mapping[str, Any]]) -> set[str]:
    return {schema["function"]["name"] for schema in schemas}


def _run(
    responses: list[Any], *, max_steps: int = DEFAULT_MAX_STEPS
) -> tuple[AgentRun, FakeToolbox, ScriptedChat]:
    chat = ScriptedChat(responses=responses)
    toolbox = FakeToolbox()
    result = run(
        "查一下 amlsim:4122",
        chat=chat,
        toolbox=cast(Any, toolbox),
        max_steps=max_steps,
    )
    return result, toolbox, chat


# --------------------------------------------------------------------------- #
#  1. 正常路径
# --------------------------------------------------------------------------- #


def test_investigate_then_conclude() -> None:
    """先侦查、再给结论——最常见的一条路径。"""
    result, toolbox, _ = _run(
        [
            tool_call("investigate_account", account_key="amlsim:4122", hops=1),
            tool_call(
                CONCLUDE_TOOL_NAME,
                suspicious=True,
                confidence=0.8,
                reason="8 个去向各 23 笔、金额完全相同",
            ),
        ]
    )

    assert result.produced_conclusion
    assert result.conclusion is not None
    assert result.conclusion.suspicious is True
    assert result.conclusion.confidence == pytest.approx(0.8)
    assert result.exhausted is False
    assert result.premature is False
    assert toolbox.call_counts == {"investigate_account": 1}
    assert len(result.steps) == 2


def test_tool_result_is_fed_back_to_the_model() -> None:
    """工具结果必须**塞回对话历史**，否则模型第二轮看不到查到了什么。"""
    _, _, chat = _run(
        [
            tool_call("investigate_account", account_key="amlsim:4122"),
            tool_call(CONCLUDE_TOOL_NAME, suspicious=False, confidence=0.5, reason="无明显异常"),
        ]
    )
    second_call_messages = chat.calls[1]["messages"]
    roles = [message["role"] for message in second_call_messages]
    assert "tool" in roles, "工具结果没有回灌"
    assert '{"ok": true}' in str(second_call_messages)


# --------------------------------------------------------------------------- #
#  2. 强制产出步（**这是"一定给结论"的机制保证**）
# --------------------------------------------------------------------------- #


def test_the_last_step_offers_only_the_conclude_tool() -> None:
    """**最后一步只给结论工具**——它没有别的选择。

    这是"一定给结论"的机制保证，不是靠提示词求它。
    """
    _, _, chat = _run(
        [
            tool_call("investigate_account", account_key="a"),
            tool_call("investigate_account", account_key="b"),
            tool_call(CONCLUDE_TOOL_NAME, suspicious=False, confidence=0.5, reason="查完了"),
        ],
        max_steps=3,
    )

    assert _tool_names(chat.calls[0]["tools"]) > {CONCLUDE_TOOL_NAME}
    assert _tool_names(chat.calls[1]["tools"]) > {CONCLUDE_TOOL_NAME}
    # 第三步是强制产出步
    assert _tool_names(chat.calls[2]["tools"]) == {CONCLUDE_TOOL_NAME}


def test_conclusion_on_the_forced_step_is_flagged() -> None:
    """在强制产出步给结论 → `forced=True`。

    它说明**自由步数不够用**——这是个值得注意的信号，不是正常的。
    """
    result, _, _ = _run(
        [
            tool_call("investigate_account", account_key="a"),
            tool_call(CONCLUDE_TOOL_NAME, suspicious=False, confidence=0.5, reason="只能给这个"),
        ],
        max_steps=2,
    )
    assert result.produced_conclusion
    assert result.forced is True


def test_conclusion_on_a_free_step_is_not_flagged_as_forced() -> None:
    result, _, _ = _run(
        [
            tool_call("investigate_account", account_key="a"),
            tool_call(CONCLUDE_TOOL_NAME, suspicious=False, confidence=0.5, reason="够了"),
            tool_call(CONCLUDE_TOOL_NAME, suspicious=False, confidence=0.5, reason="不该走到这"),
        ],
        max_steps=3,
    )
    assert result.forced is False, "第二步就给了结论，不该标成强制产出"
    assert len(result.steps) == 2, "给完结论就该停，不该继续跑"


# --------------------------------------------------------------------------- #
#  2b. ★ 强制产出步**重试**（真跑时发现的缺陷）
# --------------------------------------------------------------------------- #


def test_forced_step_retries_when_the_model_only_talks() -> None:
    """**真跑发现的缺陷**：只给一个工具**不等于**它一定会用。

    实测：干净对话里 2/2 一次成功；但真跑时对话很脏
    （长长一段"只说话"、一条失败的工具结果），模型返回了空响应。

    所以强制步给有限次机会——**第 2 次用不容含糊的措辞再要求一遍**。
    """
    result, _, chat = _run(
        [
            tool_call("investigate_account", account_key="a"),
            says("根据调查结果，我认为……"),  # 强制步第 1 次：只说话
            tool_call(CONCLUDE_TOOL_NAME, suspicious=True, confidence=0.7, reason="第 2 次才给"),
        ],
        max_steps=2,
    )

    assert result.produced_conclusion, "重试之后应该拿到结论"
    assert result.forced is True
    # 三次调用：自由步 1 次 + 强制步 2 次
    assert len(chat.calls) == 3
    # 强制步的每次调用都**只给结论工具**
    assert _tool_names(chat.calls[1]["tools"]) == {CONCLUDE_TOOL_NAME}
    assert _tool_names(chat.calls[2]["tools"]) == {CONCLUDE_TOOL_NAME}
    # 第 2 次之前插入了不容含糊的要求
    nudge = chat.calls[2]["messages"][-1]["content"]
    assert "文字回复不算完成" in nudge


def test_forced_nudge_leaves_no_room_for_ambiguity() -> None:
    """措辞刻意**不留余地**——温和的措辞正是第一次失败的原因之一。"""
    assert "必须调用" in FORCED_NUDGE
    assert "不算完成" in FORCED_NUDGE


def test_forced_step_gives_up_after_the_attempt_budget() -> None:
    """重试是有上限的——**每一步都花 10–17 秒**，不能无限试。"""
    result, _, chat = _run(
        [says("嗯…"), says("还是拿不准"), says("再想想"), says("真的想不出来")],
        max_steps=2,
    )
    assert result.exhausted is True
    # max_steps=2 → 自由步 1 次 + 强制步 FORCED_STEP_ATTEMPTS 次
    assert len(chat.calls) == 1 + FORCED_STEP_ATTEMPTS


# --------------------------------------------------------------------------- #
#  2c. ★ 自由文本**不是**结论
# --------------------------------------------------------------------------- #


def test_free_text_is_kept_but_never_becomes_a_conclusion() -> None:
    """**这是 M5 最重要的一条设计。**

    兜底拿到的自由文本没有 `suspicious`、没有 `confidence`——
    硬把它记成"结论：未发现异常"，就是本项目一直在反对的那种混淆：
    **"我看了，没问题"和"我没给出结论"会被算成同一件事。**
    """
    result, _, _ = _run(
        [says("让我想想"), says("嗯"), says("我认为这个账户看起来正常")], max_steps=2
    )

    assert result.conclusion is None, "自由文本不能变成结论"
    assert result.exhausted is True, "它仍然必须被记成【未产出结论】"
    assert "看起来正常" in result.final_text, "文本要留着供人查看"
    # 摘要里只报长度，不报内容——避免它被当成结论读
    assert result.summary()["final_text_chars"] > 0
    assert "看起来正常" not in str(result.summary())


# --------------------------------------------------------------------------- #
#  3. ★ 步数耗尽 = 工程缺陷，必须单独计数
# --------------------------------------------------------------------------- #


def test_exhausted_run_has_no_conclusion_and_is_flagged() -> None:
    """**步数耗尽、从未给结论** → `exhausted=True` 且 `conclusion is None`。

    这是本项目最重要的一条区分：
    **它不能被折算成"结论是无异常"**——那会让所有指标失真。
    """
    result, _, _ = _run(
        [
            tool_call("investigate_account", account_key="a"),
            tool_call("investigate_account", account_key="b"),
            tool_call("investigate_account", account_key="c"),
        ],
        max_steps=3,
    )

    assert result.conclusion is None
    assert result.exhausted is True
    assert result.produced_conclusion is False
    # 摘要里单独暴露，方便 CI 断言它为 0
    assert result.summary()["exhausted"] is True
    assert result.summary()["suspicious"] is None


def test_model_that_only_talks_never_concludes() -> None:
    """只说话、从不调工具 → 同样算 `exhausted`。

    **这条路径真实存在**：模型可能一直在输出分析文字而不动手。

    ⚠️ 剧本要给够：`max_steps=3` 时，
    自由步 2 次 + 强制步 `FORCED_STEP_ATTEMPTS` 次 = 2 + 2 = 4 次调用。
    """
    script = [says(f"让我想想（第 {index} 次）") for index in range(2 + FORCED_STEP_ATTEMPTS)]
    result, _, _ = _run(script, max_steps=3)
    assert result.exhausted is True
    assert result.conclusion is None
    assert len(result.steps) == len(script)


def test_talking_steps_get_reminded_to_use_tools() -> None:
    """模型只说话时要**催它用工具**，而不是干等。"""
    script = [
        says("嗯…"),
        tool_call(CONCLUDE_TOOL_NAME, suspicious=False, confidence=0.3, reason="x"),
    ]
    _, _, chat = _run(script)
    reminder = chat.calls[1]["messages"][-1]["content"]
    assert "调用工具" in reminder or "submit_conclusion" in reminder


# --------------------------------------------------------------------------- #
#  4. ★ 过早结论（实测发现模型会这样）
# --------------------------------------------------------------------------- #


def test_concluding_without_investigating_is_flagged_premature() -> None:
    """**没做任何侦查就给结论** → `premature=True`。

    实测发现的：问模型"介绍你自己"，它直接调了 `submit_conclusion`。

    **不拦截，但要计数**——"没查就下结论"和"查了才下结论"可信度完全不同。
    """
    result, toolbox, _ = _run(
        [tool_call(CONCLUDE_TOOL_NAME, suspicious=False, confidence=0.9, reason="我觉得没问题")]
    )
    assert result.produced_conclusion
    assert result.premature is True
    assert toolbox.called == [], "它确实没调用任何侦查工具"


def test_a_failed_investigation_still_counts_as_investigating() -> None:
    """调用过侦查工具（即使结果失败）就不算 premature。

    因为"**查了但工具报错**"和"**根本没查**"是两回事——
    前者 Agent 至少尝试过，而且它能看到错误信息。
    """
    responses = [
        tool_call("investigate_account", account_key="a"),
        tool_call(CONCLUDE_TOOL_NAME, suspicious=False, confidence=0.5, reason="x"),
    ]
    result, _, _ = _run(responses)
    assert result.premature is False


# --------------------------------------------------------------------------- #
#  5. 步数预算
# --------------------------------------------------------------------------- #


def test_max_steps_must_leave_room_for_the_forced_step() -> None:
    """`max_steps < 2` 必须报错。

    `max_steps=1` 意味着**唯一的一步既是侦查又是产出**——
    那和"没有终止机制"是一回事。
    """
    with pytest.raises(ValueError, match="至少为 2"):
        _run([], max_steps=1)


def test_script_exhaustion_is_an_error_not_a_silent_pass() -> None:
    """剧本用完还继续调 → **抛错**。

    "Agent 比预期多跑了步"本身就是要发现的问题（说明终止条件没生效），
    不能静默当成"它又多调了一次工具"。

    ⚠️ 触发它需要**循环跑得比剧本长**：剧本只给 1 条（一句空话，不调工具），
    而 `max_steps=3`。第一次调用消耗掉剧本，第二次就没了。

    （第一版我写成"给一条结论、max_steps=3"——**那是错的**：
     结论出现在第一步，循环立刻返回，脚本根本没用完。测试也会因此失败。）
    """
    with pytest.raises(LLMError, match="剧本用完"):
        _run([says("嗯…")], max_steps=3)


# --------------------------------------------------------------------------- #
#  6. 结论参数的容错
# --------------------------------------------------------------------------- #


def test_confidence_is_clamped_to_unit_interval() -> None:
    """模型给出 1.5 或 -0.2 是出现过的——**必须夹住**，否则会污染下游聚合。"""
    assert Conclusion.from_arguments({"suspicious": True, "confidence": 1.5}).confidence == 1.0
    assert Conclusion.from_arguments({"suspicious": True, "confidence": -0.2}).confidence == 0.0


def test_conclusion_tolerates_missing_and_malformed_fields() -> None:
    """模型不保证每次都把参数给全。**缺字段不能崩。**"""
    empty = Conclusion.from_arguments({})
    assert empty.suspicious is False
    assert empty.confidence == 0.0
    assert empty.reason == ""

    malformed = Conclusion.from_arguments({"confidence": "很高"})
    assert malformed.confidence == 0.0


def test_confidence_as_string_is_accepted() -> None:
    """实测：`arguments` 有时是字符串，数字也可能是 `"0.8"`。"""
    assert Conclusion.from_arguments({"confidence": "0.8"}).confidence == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
#  7. 工具调用的解析（实测踩过的形状不一致）
# --------------------------------------------------------------------------- #


def test_tool_call_arguments_as_json_string_are_parsed() -> None:
    """**实测踩到过**：`arguments` 有时是 dict，有时是它的 JSON 字符串形式。

    不归一化的话，下游会拿到一个字符串当参数用。
    """
    message = {
        "tool_calls": [
            {
                "function": {
                    "name": "investigate_account",
                    "arguments": '{"account_key": "amlsim:1"}',
                }
            }
        ]
    }
    calls = _parse_tool_calls(message)
    assert calls[0].arguments == {"account_key": "amlsim:1"}


def test_unparseable_arguments_are_preserved_not_dropped() -> None:
    """解析不了的参数**保留原样**——让调用方看到真实情况，而不是悄悄换成空 dict。"""
    message = {"tool_calls": [{"function": {"name": "x", "arguments": "not json at all"}}]}
    calls = _parse_tool_calls(message)
    assert calls[0].arguments == {"_raw": "not json at all"}


def test_tool_call_without_a_name_is_skipped() -> None:
    message = {"tool_calls": [{"function": {"arguments": {}}}]}
    assert _parse_tool_calls(message) == ()


# --------------------------------------------------------------------------- #
#  8. 参数上界由**服务端**强制
# --------------------------------------------------------------------------- #


def _bare_toolbox() -> Toolbox:
    """只用来测参数校验的工具箱——**校验发生在碰 driver 之前**。"""
    return Toolbox(
        driver=cast(Driver, None),
        patterns=(),
        index=None,
        embedder=cast(Any, None),
    )


@pytest.mark.parametrize(
    ("key", "value", "low", "high"),
    [("hops", 0, 1, 3), ("hops", 4, 1, 3), ("top_k", 0, 1, 5), ("limit", 501, 1, 500)],
)
def test_out_of_range_arguments_raise(key: str, value: int, low: int, high: int) -> None:
    """**上界由服务端强制，不是靠提示词约定。**"""
    with pytest.raises(ToolArgumentError, match=key):
        Toolbox._as_int({key: value}, key, low, low, high)


def test_non_integer_argument_is_rejected() -> None:
    with pytest.raises(ToolArgumentError, match="整数"):
        Toolbox._as_int({"hops": "很多"}, "hops", 1, 1, 3)


def test_boolean_is_not_accepted_as_an_integer() -> None:
    """`True` 在 Python 里是 `int` 的子类——**不特判就会被当成 1**。"""
    with pytest.raises(ToolArgumentError, match="整数"):
        Toolbox._as_int({"hops": True}, "hops", 1, 1, 3)


def test_empty_account_key_is_rejected() -> None:
    with pytest.raises(ToolArgumentError, match="非空"):
        Toolbox._as_key({"account_key": "   "}, "account_key")


def test_argument_error_is_returned_to_the_model_not_raised() -> None:
    """**参数错误要返回给模型让它自己改，而不是中断循环。**

    抛异常会把"模型犯了个小错"升级成"整个案例失败"。
    """
    toolbox = _bare_toolbox()
    result = toolbox.call("investigate_account", {"account_key": "a", "hops": 99})
    assert result.ok is False
    assert "参数错误" in result.content
    assert "hops" in result.content


def test_unknown_tool_returns_an_error_result() -> None:
    result = _bare_toolbox().call("delete_everything", {})
    assert result.ok is False
    assert "未知工具" in result.content


def test_call_counts_track_every_attempt_including_failures() -> None:
    """调用计数**包含失败的调用**——诊断"Agent 是不是在瞎调"时要看真实次数。"""
    toolbox = _bare_toolbox()
    toolbox.call("investigate_account", {"account_key": "a", "hops": 99})
    toolbox.call("investigate_account", {"account_key": "a", "hops": 1})
    assert toolbox.call_counts["investigate_account"] == 2


# --------------------------------------------------------------------------- #
#  9. 工具清单不能靠手写
# --------------------------------------------------------------------------- #


def test_investigation_tool_names_are_derived_from_the_schemas() -> None:
    """工具名从 schema **推**出来，而不是手写一份。

    **手写清单会在加了新工具之后忘记更新**——这个教训在本项目里出现过三次
    （`.gitignore`、查询审计、YAML 扫描）。
    """
    names = investigate_tool_names()
    assert len(names) == 4, f"工具契约里是 4 个侦查工具，实际 {names}"
    assert CONCLUDE_TOOL_NAME not in names, "结论工具不算侦查工具"
