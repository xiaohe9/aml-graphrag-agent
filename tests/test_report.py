"""证据展示（HTML 报告）的测试。

**不跑模型、不连图库**——全部用构造好的 `AgentRun` 渲染。

## 这个文件守的两件事

### 1. **三类证据必须视觉可分**

如果"结构化事实"和"语义匹配"在报告里长得一样，
读的人就会把它们当成同等可信——
**那这份报告就在制造误解，而不是消除它。**

### 2. **「未获取到的信息」必须显眼**

它**不能**被排成一个小灰字脚注。
> **"我看了，没问题" 和 "我没看" 是两件完全不同的事。**

### 3. **未产出结论不能被渲染成"未发现异常"**

这是整个项目的主线。报告如果让这两者看起来一样，就是在骗读的人。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aml_agent.agent import AgentRun, Conclusion, Step, ToolCall, ToolResult
from aml_agent.agent.baseline import AccountBaseline, FeaturePosition
from aml_agent.report import ReportInput, render_html


def _result(name: str, payload: Mapping[str, Any], *, ok: bool = True) -> ToolResult:
    return ToolResult(name=name, content="{}", ok=ok, payload=payload)


def _run(
    *,
    suspicious: bool | None,
    results: list[ToolResult] | None = None,
    exhausted: bool = False,
    forced: bool = False,
    premature: bool = False,
    content: str = "",
) -> AgentRun:
    steps: list[Step] = []
    if results is not None:
        steps.append(
            Step(
                index=0,
                tool_calls=(ToolCall(name=results[0].name, arguments={"account_key": "amlsim:1"}),),
                results=tuple(results),
                content="",
                seconds=40.0,
                forced=False,
            )
        )
    if suspicious is not None or exhausted:
        steps.append(
            Step(
                index=len(steps),
                tool_calls=(
                    (ToolCall(name="submit_conclusion", arguments={}),)
                    if suspicious is not None
                    else ()
                ),
                results=(),
                content=content,
                seconds=20.0,
                forced=forced,
            )
        )
    conclusion = (
        None
        if suspicious is None
        else Conclusion(suspicious=suspicious, confidence=0.75, reason="因为查到了 X 和 Y")
    )
    return AgentRun(
        question="q",
        conclusion=conclusion,
        steps=tuple(steps),
        exhausted=exhausted or conclusion is None,
        forced=forced,
        premature=premature,
        seconds=60.0,
        tool_counts={},
        final_text=content,
    )


def _baseline() -> AccountBaseline:
    return AccountBaseline(
        account="amlsim:1",
        detector_score=1.2,
        detector_percentile=0.68,
        features=(
            FeaturePosition(name="outDegree", label="转出去向数", value=181, percentile=0.81),
        ),
    )


def _html(**kwargs: Any) -> str:
    return render_html(ReportInput(run=_run(**kwargs), baseline=_baseline()))


# --------------------------------------------------------------------------- #
#  1. 自包含
# --------------------------------------------------------------------------- #


def test_report_is_self_contained() -> None:
    """**不引用任何外部资源**——没有 CDN、没有外链脚本、没有图片。

    这是"一个文件发过去就能看"的前提。
    """
    html = _html(suspicious=True)
    assert "<style>" in html
    assert "http://" not in html.replace("http://www.w3.org", ""), "不该有外链"
    assert "<script" not in html, "不该有脚本——报告不需要执行任何代码"
    assert "<img" not in html, "不该引用图片"


def test_report_renders_without_a_baseline() -> None:
    """没有基线时也要能渲染（只是少一节），不能崩。"""
    html = render_html(ReportInput(run=_run(suspicious=False)))
    assert "未发现可疑" in html


# --------------------------------------------------------------------------- #
#  2. ★ 三类证据必须视觉可分
# --------------------------------------------------------------------------- #


def test_all_three_evidence_kinds_have_their_own_tag() -> None:
    """三类证据各用**不同的标签颜色**，而且标题里写明它们的可信度性质。"""
    html = _html(
        suspicious=True,
        results=[
            _result(
                "investigate_account",
                {
                    "relations": [
                        {
                            "counterparty": "amlsim:2",
                            "direction": "out",
                            "count": 23,
                            "amount": "13.63（每笔相同）",
                            "time_span": "t=27..184",
                        }
                    ]
                },
            ),
            _result(
                "match_patterns",
                {
                    "matches": [
                        {
                            "name": "回流环路",
                            "category": "structural",
                            "similarity": 0.66,
                            "definition": "资金绕一圈回到起点",
                            "source": "arXiv:2306.16424",
                        }
                    ]
                },
            ),
        ],
    )
    # 三个标签类各出现一次
    assert 'class="tag fact"' in html
    assert 'class="tag judge"' in html
    assert 'class="tag knowledge"' in html
    # 可信度性质写在标题区
    assert "未经推断" in html
    assert "不代表认定" in html


def test_facts_block_carries_amount_and_time_span() -> None:
    html = _html(
        suspicious=True,
        results=[
            _result(
                "investigate_account",
                {
                    "relations": [
                        {
                            "counterparty": "amlsim:2",
                            "direction": "out",
                            "count": 23,
                            "amount": "13.63（每笔相同）",
                            "time_span": "t=27..184",
                        }
                    ]
                },
            )
        ],
    )
    assert "amlsim:2" in html
    assert "13.63" in html
    assert "t=27..184" in html


def test_knowledge_block_shows_similarity_and_source() -> None:
    """相似手法要带**相似度和来源**——来源可追溯是知识库的硬要求。"""
    html = _html(
        suspicious=True,
        results=[
            _result(
                "match_patterns",
                {
                    "matches": [
                        {
                            "name": "回流环路",
                            "category": "structural",
                            "similarity": 0.6612,
                            "definition": "资金绕一圈回到起点",
                            "source": "arXiv:2306.16424",
                        }
                    ]
                },
            )
        ],
    )
    assert "0.661" in html
    assert "arXiv:2306.16424" in html


def test_measured_signal_grades_are_shown_and_weak_ones_flagged() -> None:
    """**弱信号必须显眼。**

    如果「环长 2」和「环长 3」以同样格式呈现，读的人没有依据区别对待——
    **而不标注就等于在误导。**
    """
    signals = {
        "cycle_length_2": "很弱，不建议作为依据",
        "cycle_length_3": {"grade": "强", "basis": "实测：约为普通交易的 40 倍"},
    }
    html = _html(
        suspicious=True,
        results=[_result("get_detector_scores", {"measuredSignals": signals})],
    )
    assert "很弱，不建议作为依据" in html
    assert "40 倍" in html
    assert "不标注就等于在误导" in html


# --------------------------------------------------------------------------- #
#  3. ★ 「未获取到的信息」必须显眼
# --------------------------------------------------------------------------- #


def test_missing_information_is_prominent_not_a_footnote() -> None:
    """**不能排成小灰字。** 它必须有独立的区块和警示类。"""
    html = _html(
        suspicious=True,
        results=[
            _result(
                "investigate_account",
                {
                    "relations": [],
                    "truncated": "结果被截断：只取了前 500 条边",
                },
            )
        ],
    )
    assert 'class="missing"' in html
    assert "请勿当作" in html
    assert "结果被截断" in html
    # 要说清「我看了，没问题」和「我没看」的区别
    assert "我没看" in html


def test_path_not_found_note_is_carried_into_missing() -> None:
    """`trace_path` 的「没找到」不等于「没有关联」——那句话必须传进报告。"""
    html = _html(
        suspicious=True,
        results=[
            _result(
                "trace_path",
                {"found": False, "note": "在 8 跳以内没有找到路径，这不等于两者没有关联"},
            )
        ],
    )
    assert "不等于两者没有关联" in html


def test_no_missing_section_when_nothing_is_missing() -> None:
    """该拿的都拿到了就不出现——**否则警告会变成噪音**。"""
    html = _html(suspicious=True, results=[_result("investigate_account", {"relations": []})])
    assert 'class="missing"' not in html


# --------------------------------------------------------------------------- #
#  4. ★ 未产出结论不能被渲染成「未发现异常」
# --------------------------------------------------------------------------- #


def test_no_conclusion_renders_as_a_warning_not_as_normal() -> None:
    """**这是整个项目的主线，报告里也不能破例。**"""
    html = _html(suspicious=None, exhausted=True, content="我觉得这个账户看起来正常")
    assert "未产出结论" in html
    assert "这不是「未发现异常」" in html or "这不是" in html
    assert "既不算漏报，也不算检出" in html
    # 而且那份自由文本**不该**被当成结论呈现
    assert "🟢 未发现可疑" not in html


def test_forced_and_premature_conclusions_are_flagged_in_the_report() -> None:
    """强制产出 / 过早结论都要在报告里说出来——**它们是可信度信息**。"""
    forced = _html(suspicious=True, forced=True)
    assert "强制产出步" in forced
    premature = _html(suspicious=True, premature=True)
    assert "没有做过任何侦查" in premature


def test_verdict_shows_confidence_and_cost() -> None:
    html = _html(suspicious=True)
    assert "0.75" in html
    assert "步" in html and "秒" in html


# --------------------------------------------------------------------------- #
#  5. 分位必须带"不是判断"的说明
# --------------------------------------------------------------------------- #


def test_baseline_block_says_percentile_is_not_a_verdict() -> None:
    html = _html(suspicious=True)
    assert "分位" in html
    assert "相对排名，不是判断" in html
    assert "分位高不等于有问题" in html


def test_execution_trace_is_included() -> None:
    """结论能不能被核对，取决于轨迹这一节。"""
    html = _html(suspicious=True, results=[_result("investigate_account", {"relations": []})])
    assert "执行轨迹" in html
    assert "investigate_account" in html


def test_html_escapes_model_output() -> None:
    """模型输出必须转义——**否则一次工具返回就能注入 HTML**。"""
    html = _html(
        suspicious=True,
        results=[
            _result(
                "investigate_account",
                {
                    "relations": [
                        {
                            "counterparty": "<script>alert(1)</script>",
                            "direction": "out",
                            "count": 1,
                            "amount": "1",
                            "time_span": "t=1",
                        }
                    ]
                },
            )
        ],
    )
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
