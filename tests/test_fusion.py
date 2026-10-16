"""融合层的测试。

## 这里守的是两件**很容易做错**的事

### 1. 弱信号必须被显式标注

实测告诉我们「环长 2」几乎没有区分度（正常交易里 18.4% 的边本来就是互转）。
如果它和「环长 3」以同样格式呈现，**模型没有任何依据去区别对待**——
**不标注，就等于在误导模型。**

### 2. 「没查到」不能被当成「没有」

这与评测里「未产出结论必须单独计数」是**同一条原则**。
子图被截断了、某个工具失败了——这些必须说出来，
否则模型会默认"没有更多信息"，**而它无法分辨「确实没有」和「根本没查」**。
"""

from __future__ import annotations

from aml_agent.graph.queries import TransferEdge
from aml_agent.knowledge import Pattern
from aml_agent.retrieve import (
    Finding,
    InvestigationContext,
    PatternMatch,
    derive_findings,
    group_facts,
    render,
    signal_reliability,
)


def _edge(sender: str, receiver: str, *, amount: float = 10.0, timestamp: int = 1) -> TransferEdge:
    return TransferEdge(
        sender=sender,
        receiver=receiver,
        tx_id=abs(hash((sender, receiver, timestamp))) % 10**6,
        amount=amount,
        timestamp=timestamp,
        is_sentinel_amount=False,
    )


def _pattern(name: str = "回流环路", category: str = "structural") -> Pattern:
    return Pattern(
        id="cycle",
        name=name,
        aliases=("cycle",),
        definition="资金绕一圈回到起点。",
        source="arXiv:2306.16424",
        category=category,
        extras={},
    )


# --------------------------------------------------------------------------- #
#  1. 弱信号必须被标注
# --------------------------------------------------------------------------- #


def test_the_two_cycle_is_explicitly_marked_as_weak() -> None:
    """**这是融合层最重要的一条设计。**

    环长 2 与环长 3 在实测上差了约 40 倍。如果渲染出来长得一样，
    模型就会把它们当成同等强度的证据。
    """
    grade, basis = signal_reliability("cycle_length_2")
    assert "弱" in grade, f"环长 2 的等级里必须出现「弱」，实际是 {grade!r}"
    assert "1.0" in basis or "同样普遍" in basis, "依据里要写出实测的对比"

    strong_grade, strong_basis = signal_reliability("cycle_length_3")
    assert "强" in strong_grade
    assert "40" in strong_basis, "依据里要有具体倍数，不能只给形容词"


def test_unknown_signal_is_marked_as_unverified() -> None:
    """**"我没测过这个信号"本身就是模型需要知道的信息。**"""
    grade, basis = signal_reliability("some_signal_we_never_measured")
    assert "未验证" in grade
    assert "谨慎" in basis


def test_reciprocal_edge_is_marked_ineffective() -> None:
    """互转是实测**完全无效**的信号（18.4% vs 19.3%），必须标出来。"""
    grade, _ = signal_reliability("reciprocal_edge")
    assert "无效" in grade


def test_every_measured_signal_has_a_concrete_basis() -> None:
    """每条依据都要有**具体内容**，不能是"实测表明"这种空话。"""
    from aml_agent.retrieve import MEASURED_SIGNALS

    for signal, (grade, basis) in MEASURED_SIGNALS.items():
        assert grade.strip(), f"{signal} 没有等级"
        assert len(basis) > 10, f"{signal} 的依据太短，像是占位符：{basis!r}"


# --------------------------------------------------------------------------- #
#  2. 从事实推导判断
# --------------------------------------------------------------------------- #


def test_cycle_lengths_become_separate_findings() -> None:
    """**环长 2 和环长 3 必须分成两条**——它们的可靠度差了 40 倍。

    合成一条"存在环"，就丢掉了这个区别。
    """
    findings = derive_findings([], cycle_lengths=[2, 3])
    kinds = {finding.kind for finding in findings}
    assert kinds == {"cycle_length_2", "cycle_length_3"}
    for finding in findings:
        assert finding.basis.strip(), "每条判断都必须带依据"


def test_fan_out_and_fan_in_are_derived_from_distinct_nodes() -> None:
    """扇入扇出要按**不同节点数**算，不是按边数。

    同一对账户之间转 19 次不是"扇出"——**边多不等于去向多**。
    """
    facts = [_edge("a", "b"), _edge("a", "b"), _edge("a", "b")]
    kinds = {finding.kind for finding in derive_findings(facts)}
    assert "fan_out_candidate" not in kinds, "三个同去向不算扇出"

    spread = [_edge("a", "b"), _edge("a", "c"), _edge("a", "d")]
    kinds = {finding.kind for finding in derive_findings(spread)}
    assert "fan_out_candidate" in kinds


def test_high_degree_triggers_an_insufficient_subgraph_warning() -> None:
    """度数很高时**不是**说"更可疑"，而是说"子图可能不够"。

    这个区分很重要：把"信息不足"说成"可疑度更高"是在制造假阳性。
    """
    facts = [_edge("a", f"target{index}") for index in range(50)]
    findings = derive_findings(facts, max_rendered_degree=30)
    kinds = {finding.kind for finding in findings}
    assert "subgraph_may_be_insufficient" in kinds
    warning = next(f for f in findings if f.kind == "subgraph_may_be_insufficient")
    assert "子图" in warning.basis


def test_no_facts_gives_no_degree_findings() -> None:
    assert derive_findings([]) == ()
    assert derive_findings([], cycle_lengths=[]) == ()


# --------------------------------------------------------------------------- #
#  3. 渲染：三类证据必须可区分
# --------------------------------------------------------------------------- #


def _context(**overrides: object) -> InvestigationContext:
    defaults: dict[str, object] = {
        "subject": "amlsim:4122",
        "hops": 1,
        "facts": (_edge("amlsim:4122", "amlsim:3", amount=13.63, timestamp=27),),
        "findings": (),
        "patterns": (),
        "missing": (),
    }
    defaults.update(overrides)
    return InvestigationContext(**defaults)  # type: ignore[arg-type]


def test_render_has_three_clearly_separated_evidence_sections() -> None:
    """三类证据各自成节，**并且标题里写明它们的可信度性质**。

    这是"融合"与"把结果拼起来"的区别所在。
    """
    text = render(_context())
    assert "一、结构化事实" in text
    assert "未经推断" in text, "事实这一节的标题要说明它是硬数据"
    assert "二、算法判断" in text
    assert "实测依据" in text, "判断这一节的标题要说明它带实测依据"
    assert "三、相似手法" in text
    assert "不代表认定" in text, "语义这一节要说明它只是相似"


def test_render_marks_weak_cycle_explicitly() -> None:
    """渲染出来的文本里必须能看出**哪个信号弱**。"""
    findings = derive_findings([], cycle_lengths=[2, 3])
    text = render(_context(findings=findings))
    weak_line = next(line for line in text.splitlines() if "长度为 2" in line)
    strong_line = next(line for line in text.splitlines() if "长度为 3" in line)
    assert "弱" in weak_line
    assert "强" in strong_line


def test_render_shows_grouped_facts_with_amount_and_time_span() -> None:
    """事实按账户对归并渲染——**而且时间跨度要一眼可见**。

    时间跨度是真正的信号（"固定间隔的连续转账"是分层手法的典型特征）。
    逐条渲染会把它淹没在重复里。
    """
    facts = tuple(_edge("a", "b", amount=13.63, timestamp=stamp) for stamp in (27, 38, 64, 184))
    text = render(_context(facts=facts, subject="a"))
    assert "a → b：4 笔" in text
    assert "13.63（每笔相同）" in text
    assert "t=27..184" in text, "时间跨度必须一眼可见"


def test_render_says_grouping_is_not_data_aggregation() -> None:
    """**必须说明这是表示层归并，不是丢信息的聚合。**

    否则读的人（或模型）会以为图里只有一组边——
    而 ADR-0001 明确要求保留每一条边的时间戳。
    """
    facts = tuple(_edge("a", "b", timestamp=stamp) for stamp in (1, 2, 3))
    text = render(_context(facts=facts, subject="a"))
    assert "按账户对归并" in text
    assert "未做丢失信息的聚合" in text or "每条边都在" in text


def test_render_truncates_long_group_lists_but_says_so() -> None:
    """组太多时会截断——**但必须说明截断了**，否则模型以为看到的就是全部。"""
    from aml_agent.retrieve import MAX_FACTS_RENDERED

    facts = tuple(_edge("a", f"b{index}") for index in range(MAX_FACTS_RENDERED + 15))
    text = render(_context(facts=facts, subject="a"))
    assert "另有 15 组未列出" in text


# --------------------------------------------------------------------------- #
#  3b. 事实分组（表示层压缩）
# --------------------------------------------------------------------------- #


def test_group_facts_collapses_repeated_pairs() -> None:
    """同一对账户的多笔转账**归并成一组**——这是渲染层的事，不是存储层的。"""
    facts = tuple(_edge("a", "b", timestamp=stamp) for stamp in (5, 1, 9))
    groups = group_facts(facts)
    assert len(groups) == 1
    group = groups[0]
    assert group.count == 3
    assert group.first_timestamp == 1
    assert group.last_timestamp == 9


def test_group_facts_keeps_distinct_pairs_separate() -> None:
    groups = group_facts([_edge("a", "b"), _edge("a", "c")])
    assert len(groups) == 2


def test_group_facts_sorts_by_count_descending() -> None:
    """笔数多的排前面——**它本身就是信号**（长期往来 vs 一次性转账）。"""
    facts = [_edge("a", "b")] * 5 + [_edge("a", "c")] * 1 + [_edge("a", "d")] * 3
    counts = [group.count for group in group_facts(facts)]
    assert counts == sorted(counts, reverse=True)


def test_group_facts_is_reproducible_for_equal_counts() -> None:
    """笔数相同时按账户对排序——**保证同一输入同一输出**（NFR-5）。"""
    facts = [_edge("a", "z"), _edge("a", "b")]
    first = [group.receiver for group in group_facts(facts)]
    second = [group.receiver for group in group_facts(list(reversed(facts)))]
    assert first == second == ["b", "z"]


def test_group_facts_reports_amount_range_when_it_varies() -> None:
    groups = group_facts([_edge("a", "b", amount=10.0), _edge("a", "b", amount=99.5)])
    assert "10.00–99.50" in groups[0].describe_amount()


def test_group_facts_counts_sentinel_amounts() -> None:
    from aml_agent.graph.queries import TransferEdge

    sentinel = TransferEdge(
        sender="a", receiver="b", tx_id=1, amount=2.1e7, timestamp=1, is_sentinel_amount=True
    )
    groups = group_facts([sentinel, _edge("a", "b")])
    assert groups[0].sentinel_count == 1
    assert "疑似哨兵金额" in groups[0].render_line()


# --------------------------------------------------------------------------- #
#  4. ★「没查到」不能被当成「没有」
# --------------------------------------------------------------------------- #


def test_missing_section_warns_against_reading_it_as_no_anomaly() -> None:
    """**这条测试守的是整个设计里最容易被忽略的地方。**

    如果不显式警告，模型很容易把"未获取到信息"读成"没发现问题"——
    而这两件事的含义完全相反。
    """
    text = render(_context(missing=("邻域被截断，只取了前 500 条边",)))
    assert "未获取到的信息" in text
    assert "请勿当作" in text, "必须明确警告不要把未获取当成没有异常"
    assert "邻域被截断" in text


def test_missing_is_rendered_last() -> None:
    """放在最后是刻意的——**模型对末尾内容记得更牢**。"""
    text = render(
        _context(
            findings=derive_findings([], cycle_lengths=[3]),
            patterns=(PatternMatch(pattern=_pattern(), similarity=0.66),),
            missing=("某个工具超时",),
        )
    )
    positions = [
        text.index("一、结构化事实"),
        text.index("二、算法判断"),
        text.index("三、相似手法"),
        text.index("未获取到的信息"),
    ]
    assert positions == sorted(positions), "缺失项那一节必须在最后"


def test_empty_facts_says_so_explicitly() -> None:
    """「该范围内没有交易边」和「这一节不渲染」是两回事。

    前者是**有信息的**（范围里确实没有），后者会让模型以为漏了什么。
    """
    text = render(_context(facts=()))
    assert "没有交易边" in text


def test_no_findings_says_so_explicitly() -> None:
    text = render(_context(findings=()))
    assert "没有触发任何结构判据" in text


def test_no_pattern_matches_is_described_as_a_valuable_conclusion() -> None:
    """**检索不到也是结论。** 空列表说的是"这不像任何已知手法"，不是"没查到"。"""
    text = render(_context(patterns=()))
    assert "有价值的结论" in text


def test_empty_missing_renders_no_warning_section() -> None:
    """该拿的都拿到了，就不该出现警告——**否则警告会变成噪音**。"""
    text = render(_context())
    assert "未获取到的信息" not in text


# --------------------------------------------------------------------------- #
#  5. 上下文对象本身
# --------------------------------------------------------------------------- #


def test_summary_is_compact_and_has_no_conclusion() -> None:
    """摘要只报**数量**，不报任何推断结论——

    因为它会被写进日志和报告，而**结论必须由 Agent 显式给出**，
    不能从日志里"顺带"冒出来。
    """
    context = _context(
        findings=derive_findings([], cycle_lengths=[2, 3]),
        patterns=(PatternMatch(pattern=_pattern(), similarity=0.66),),
        missing=("x",),
    )
    summary = context.summary()
    assert summary == {
        "subject": "amlsim:4122",
        "hops": 1,
        "facts": 1,
        "findings": 2,
        "patterns": 1,
        "missing": ["x"],
    }
    flat = str(summary)
    assert "可疑" not in flat and "洗钱" not in flat, "摘要里不能出现结论性词汇"


def test_context_is_immutable() -> None:
    """不可变——**证据一旦采集完就不该被下游改动**。"""
    import dataclasses

    assert dataclasses.is_dataclass(InvestigationContext)
    context = _context()
    try:
        context.subject = "改掉了"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:  # pragma: no cover
        raise AssertionError("InvestigationContext 应该是冻结的")


def test_finding_carries_detail_for_programmatic_use() -> None:
    """`Finding` 既有给人看的句子，也有给程序用的结构化字段。"""
    findings = derive_findings([], cycle_lengths=[3])
    assert findings[0].detail == {"cycle_length": 3}
    assert isinstance(findings[0], Finding)
