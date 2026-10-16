"""融合层：把三类**异构证据**拼成 LLM 能用的上下文。

## 融合不是"把结果拼成一个字符串"

HLD §3.3 的论断是：**融合是把三种异构证据装配起来**，不是两份结果列表加权排序。
这三种证据的**可信度完全不同**：

| 证据 | 来自 | 可信度性质 |
|---|---|---|
| **结构化事实** | 图数据库 | **硬数据**——谁转给谁、多少、何时，没有推断成分 |
| **算法判断** | 检测层 | **有阈值依赖**——同一个结构，阈值不同结论不同 |
| **语义知识** | 向量检索 | **只是"像"**——相似度高不等于就是那种手法 |

**把三者混在一个字符串列表里，等于把它们的可信度抹平了**——
模型会像对待事实一样对待一句"这看起来像分层"。

## 本模块的核心设计：**给每条判断标注实测可靠度**

`Finding` 带一个 `basis`，它**不是拍的，来自 `docs/09` 的实验**：

```text
环长 3     实测提升约 40 倍        ← 强
环长 4     实测约 5 倍             ← 中
环长 2     实测约 1.0 倍（无用）    ← 必须标出来，否则模型会拿它当依据
小额       召回 90.7%
同对笔数少   cycle 召回 12.5%
```

**这一条是本项目最独特的地方**：一般系统只报告"发现了什么"，
而这里**同时报告"这个发现有多可信、以及为什么"——依据是我们自己跑出来的数字。**

## ⚠️ 为什么必须显式标注"弱的信号"

如果"存在长度 2 的环"和"存在长度 3 的环"以同样格式呈现，
**模型没有任何依据去区别对待**——而实测告诉我们前者几乎等于噪声
（全图 18.4% 的边本来就是互转）。

**不标注，就等于在误导模型。**

## ⚠️ "没查到什么"必须显式传递

与评测里「未产出结论必须单独计数」是**同一条原则**：

- 子图被**截断**（只取了前 N 条边）→ 必须说
- 某个工具**失败或超时** → 必须说

什么都不说的时候，模型会默认"没有更多信息"，
**而它无法分辨「确实没有」和「根本没查」**。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from aml_agent.graph.queries import TransferEdge
from aml_agent.retrieve.matches import PatternMatch

# --------------------------------------------------------------------------- #
#  实测可靠度表
# --------------------------------------------------------------------------- #

#: 各结构信号的**实测**可靠度。
#:
#: ⚠️ **这里的每一个数字都来自 `docs/09` 的实验，不是拍的。**
#: 改这张表必须同时改文档，否则两边会漂移。
#:
#: 格式：信号名 → (等级, 依据)
MEASURED_SIGNALS: Mapping[str, tuple[str, str]] = {
    "cycle_length_3": (
        "强",
        "实测：环长 3 在标注交易中的占比约为普通交易的 40 倍",
    ),
    "cycle_length_4": (
        "中",
        "实测：环长 4 约为 5 倍",
    ),
    "cycle_length_2": (
        "很弱，不建议作为依据",
        "实测：环长 2（互转）在正常交易里同样普遍，约 1.0 倍——全图 18.4% 的边本来就是互转",
    ),
    "small_amount": (
        "强",
        "实测：涉诈金额中位数只有正常交易的十五分之一；该信号单独召回 90.7%",
    ),
    "pair_tx_count_low": (
        "中",
        "实测：标注交易的同对笔数中位数 1–2，而全图为 20；该信号 cycle 召回 12.5%",
    ),
    "high_degree": (
        "弱",
        "实测：标注交易的收款方入度中位数（121 / 145）反而低于全图（233）——方向与直觉相反",
    ),
    "reciprocal_edge": (
        "无效",
        "实测：标注 19.3% vs 全图 18.4%，几乎没有差别",
    ),
}


def signal_reliability(signal: str) -> tuple[str, str]:
    """查一个信号的实测可靠度。

    **未知信号明确标为"未验证"**，而不是返回空值——
    因为"我没测过这个信号"本身就是模型需要知道的信息。
    """
    return MEASURED_SIGNALS.get(signal, ("未验证", "该信号没有在本项目的数据上测过，请谨慎采信"))


# --------------------------------------------------------------------------- #
#  证据类型
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Finding:
    """一条**算法判断**。

    与 `TransferEdge`（事实）的区别在于：**它是对事实的推断**，依赖阈值与算法选择。
    所以它必须带实测依据。
    """

    kind: str
    summary: str
    #: 实测依据。**直接给数字，不只给形容词**——
    #: 形容词容易被当成修辞，具体数字不会。
    basis: str
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InvestigationContext:
    """一次调查的**全部证据**，按可信度分层。"""

    subject: str
    hops: int
    #: ① 结构化事实——硬数据
    facts: tuple[TransferEdge, ...]
    #: ② 算法判断——带实测依据
    findings: tuple[Finding, ...]
    #: ③ 语义知识——只是"像"
    patterns: tuple[PatternMatch, ...]
    #: ⚠️ 未获取到的信息。
    #:
    #: **空元组表示"该拿的都拿到了"，不是"没有异常"。**
    #: 这两件事在文本里看起来一样，但含义完全不同。
    missing: tuple[str, ...] = ()

    def summary(self) -> dict[str, Any]:
        """给日志与报告用的紧凑摘要（**不含标签，也不含任何推断结论**）。"""
        return {
            "subject": self.subject,
            "hops": self.hops,
            "facts": len(self.facts),
            "findings": len(self.findings),
            "patterns": len(self.patterns),
            "missing": list(self.missing),
        }


# --------------------------------------------------------------------------- #
#  渲染成 LLM 上下文
# --------------------------------------------------------------------------- #

#: 事实最多渲染多少**组**。**不是越多越好**——
#: 上下文塞得越满，模型对每一条的注意力就越弱。
MAX_FACTS_RENDERED = 40


# --------------------------------------------------------------------------- #
#  渲染前的事实分组（⚠️ 这是**表示层**压缩，不是数据聚合）
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class EdgeGroup:
    """同一对账户之间的转账，**压缩成一行可读的表示**。

    ## ⚠️ 这**不是** ADR-0001 拒绝的那种聚合

    ADR-0001 说"同一对账户的多笔转账**不聚合**"，指的是**存储层**——
    要保住每一条的时间戳，否则时序模式（快进快出、固定间隔分层）就检测不了。

    **这里的对象完全不同**：图里那些边一条都没少，`facts` 元组仍是完整列表。
    分组**只发生在渲染给 LLM 看的这一段文本里**。

    ## 为什么必须分组（看真实输出发现的）

    实测渲染一个账户的 1 跳邻域，40 行里几乎全是同一对账户的重复：

    ```text
    - amlsim:4122 → amlsim:3　金额 13.63　时刻 t=64
    - amlsim:4122 → amlsim:3　金额 13.63　时刻 t=38
    - amlsim:4122 → amlsim:3　金额 13.63　时刻 t=37
    …（再重复十几遍）
    ```

    **这是在浪费 LLM 的注意力预算**，而且**丢掉了真正有信息量的东西**——
    那十几笔的时间跨度本身才是信号（"固定间隔的连续转账"是分层手法的典型特征）。

    | | 逐条渲染 | 分组渲染 |
    |---|---|---|
    | 行数 | 19 行 | **1 行** |
    | 时间信息 | 散在 19 行里，看不出跨度 | **`t=27..184` 一眼可见** |
    | 金额变化 | 要自己比对 | **`13.63（不变）` 直接说出** |

    **压缩之后信息量反而更大**——因为它把"重复"换成了"统计特征"。
    """

    sender: str
    receiver: str
    count: int
    min_amount: float
    max_amount: float
    first_timestamp: int
    last_timestamp: int
    sentinel_count: int = 0

    def describe_amount(self) -> str:
        if self.min_amount == self.max_amount:
            return f"{self.min_amount:.2f}（每笔相同）"
        return f"{self.min_amount:.2f}–{self.max_amount:.2f}"

    def describe_time(self) -> str:
        if self.first_timestamp == self.last_timestamp:
            return f"t={self.first_timestamp}"
        return f"t={self.first_timestamp}..{self.last_timestamp}"

    def render_line(self) -> str:
        sentinel = f"（其中 {self.sentinel_count} 笔疑似哨兵金额）" if self.sentinel_count else ""
        return (
            f"- {self.sender} → {self.receiver}：{self.count} 笔，"
            f"金额 {self.describe_amount()}，{self.describe_time()}{sentinel}"
        )


def group_facts(facts: Sequence[TransferEdge]) -> tuple[EdgeGroup, ...]:
    """把事实按 `(转出方, 转入方)` 分组。

    分组结果**按笔数降序**——重复最多的关系最值得先看，
    而且它本身就是"长期往来 vs 一次性转账"的信号
    （实测：标注交易的同对笔数中位数 1–2，全图是 20）。
    """
    buckets: dict[tuple[str, str], list[TransferEdge]] = {}
    for edge in facts:
        buckets.setdefault((edge.sender, edge.receiver), []).append(edge)

    groups = [
        EdgeGroup(
            sender=key[0],
            receiver=key[1],
            count=len(items),
            min_amount=min(item.amount for item in items),
            max_amount=max(item.amount for item in items),
            first_timestamp=min(item.timestamp for item in items),
            last_timestamp=max(item.timestamp for item in items),
            sentinel_count=sum(1 for item in items if item.is_sentinel_amount),
        )
        for key, items in buckets.items()
    ]
    # 笔数降序；同笔数按账户对排序，保证**可复现**
    return tuple(sorted(groups, key=lambda group: (-group.count, group.sender, group.receiver)))


def render(context: InvestigationContext) -> str:
    """把上下文渲染成**分层的**文本。

    ## 三处分层是刻意的

    1. **三类证据各自成节**，标题里写明它们的**可信度性质**——
       来自图数据库的硬数据 / 算法判断并附实测依据 / 相似度匹配仅供参考；
    2. **弱信号显式标注**——见模块文档，不标注等于误导；
    3. **未获取到的信息单独成节并放在最后**——
       模型对末尾内容记得更牢，而且用 ⚠️ 开头让它在长文本里也不会被略过。

    ## 为什么用 Markdown 而不是 JSON

    这里的消费者是 LLM。**Markdown 的层级（标题、列表、加粗）本身就是信号**，
    模型在预训练里见过大量这样的结构。

    但**结构化字段仍然保留在 `InvestigationContext` 里**——
    需要程序化处理时（比如做评测）直接用对象，不用解析这段文本。
    """
    lines: list[str] = []

    lines.append(f"# 调查对象：{context.subject}")
    lines.append("")

    # ---- ① 结构化事实 ---------------------------------------------------- #
    #
    # ⚠️ 这里渲染的是**分组后的表示**，不是逐条边。
    # 分组只发生在这一段文本里——`context.facts` 仍是完整的边列表。
    # 理由见 `EdgeGroup` 的文档：逐条渲染会浪费注意力预算，
    # 而且**把时间跨度这个真正的信号淹没在重复里**。
    lines.append("## 一、结构化事实（来自图数据库，**未经推断**）")
    if context.facts:
        groups = group_facts(context.facts)
        lines.append(
            f"共 {len(context.facts)} 条交易边（子图范围 {context.hops} 跳），"
            f"按账户对归并后为 {len(groups)} 组："
        )
        for group in groups[:MAX_FACTS_RENDERED]:
            lines.append(group.render_line())
        hidden = len(groups) - MAX_FACTS_RENDERED
        if hidden > 0:
            lines.append(f"- …（另有 {hidden} 组未列出）")
        lines.append("")
        lines.append(
            "注：上面是**逐条边的完整列表按账户对归并后的表示**"
            "（图里每条边都在，未做丢失信息的聚合）。"
        )
    else:
        lines.append("（该范围内没有交易边）")
    lines.append("")

    # ---- ② 算法判断 ------------------------------------------------------ #
    lines.append("## 二、算法判断（**依赖阈值与算法选择，已标注实测依据**）")
    if context.findings:
        for finding in context.findings:
            lines.append(f"- **{finding.summary}**")
            lines.append(f"  - 依据：{finding.basis}")
    else:
        lines.append("（没有触发任何结构判据）")
    lines.append("")

    # ---- ③ 语义知识 ------------------------------------------------------ #
    lines.append("## 三、相似手法（**语义匹配，仅供参考，不代表认定**）")
    if context.patterns:
        for rank, match in enumerate(context.patterns, start=1):
            pattern = match.pattern
            lines.append(
                f"{rank}. **{pattern.name}**（相似度 {match.similarity:.3f}，"
                f"类别：{pattern.category}）"
            )
            lines.append(f"   - 定义：{pattern.definition}")
            lines.append(f"   - 来源：{pattern.source}")
    else:
        lines.append("（没有达到相似度下限的已知手法——**这本身是有价值的结论**）")
    lines.append("")

    # ---- ⚠️ 未获取到的信息（放最后，模型对末尾记得更牢）------------------ #
    #
    # 标题里刻意写明"请勿当作没有异常"——因为这句话不加，
    # 模型很可能把"未获取到"读成"没发现问题"。
    if context.missing:
        lines.append("## ⚠️ 未获取到的信息（**请勿当作「没有异常」**）")
        for item in context.missing:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  从事实推导判断
# --------------------------------------------------------------------------- #


def derive_findings(
    facts: Sequence[TransferEdge],
    *,
    cycle_lengths: Sequence[int] = (),
    max_rendered_degree: int = 30,
) -> tuple[Finding, ...]:
    """从结构化事实推导出**带实测依据的判断**。

    ## 为什么把"环长 2"也单独报出来

    因为它**很常见**——全图 18.4% 的边本来就是互转。

    如果只报"存在环"，模型会以为这是个发现；
    **而把它单独列出来、并注明"约 1.0 倍、不建议作为依据"，才是诚实的做法。**

    ## 为什么这里**不做**"综合可疑度打分"

    **打分是检测层的事**（`detect` 包），融合层只负责**如实呈现证据**。
    在这里偷偷合成一个分数，会让"这个分数怎么来的"变得无法追溯——
    而**可解释性是这个项目的硬约束**。
    """
    findings: list[Finding] = []

    for length in sorted(set(cycle_lengths)):
        key = f"cycle_length_{length}"
        grade, basis = signal_reliability(key)
        findings.append(
            Finding(
                kind=key,
                summary=f"该笔交易位于长度为 {length} 的环上（可靠度：{grade}）",
                basis=basis,
                detail={"cycle_length": length},
            )
        )

    if not facts:
        return tuple(findings)

    distinct_receivers = len({edge.receiver for edge in facts})
    distinct_senders = len({edge.sender for edge in facts})

    # 只在真的"多对一/一对多"时才报，避免把单条边也说成扇入扇出
    if distinct_receivers >= 2:
        _, basis = signal_reliability("high_degree")
        findings.append(
            Finding(
                kind="fan_out_candidate",
                summary=f"该账户在子图内指向 {distinct_receivers} 个不同去向（疑似扇出）",
                basis=basis,
                detail={"distinct_receivers": distinct_receivers},
            )
        )
    if distinct_senders >= 2:
        _, basis = signal_reliability("high_degree")
        findings.append(
            Finding(
                kind="fan_in_candidate",
                summary=f"该账户在子图内来自 {distinct_senders} 个不同来源（疑似扇入）",
                basis=basis,
                detail={"distinct_senders": distinct_senders},
            )
        )

    # 度数很高时额外提示一句——**这是"子图可能不足以判断"的信号**，
    # 而不是"这个人更可疑"。这个区分很重要。
    if max(distinct_receivers, distinct_senders) > max_rendered_degree:
        findings.append(
            Finding(
                kind="subgraph_may_be_insufficient",
                summary=(
                    f"该账户的连接数（{max(distinct_receivers, distinct_senders)}）"
                    "超出当前子图能完整呈现的范围"
                ),
                basis="子图取了固定跳数，高度数节点的邻域可能不完整——请以缺失项为准",
                detail={
                    "distinct_receivers": distinct_receivers,
                    "distinct_senders": distinct_senders,
                },
            )
        )

    return tuple(findings)
