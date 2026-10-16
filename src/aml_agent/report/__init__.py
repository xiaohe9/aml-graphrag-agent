"""证据展示：把一次调查渲染成**自包含的 HTML 报告**。

## ⚠️ 为什么是"自包含 HTML"而不是一个 Web 应用

| | 自包含 HTML | SPA + 服务 |
|---|---|---|
| 部署 | **一个文件，双击就开** | 要起服务、要配环境 |
| 发给别人 | 直接发 | 得给地址和账号 |
| 离线 | ✅ | ❌ |
| 交互 | 有限 | 丰富 |

**而这个项目的证据展示，本质上是"一份可核对的报告"，不是"一个工具"。**

- 分析师要的是**能读、能引用、能放进上报材料**的东西；
- 面试官要的是**打开就能看**，不是"你先 clone 再起服务"。

**所以交互性不是这里的瓶颈，可移植性才是。**

## ⚠️ 渲染的核心约束：**三类证据必须视觉可分**

这是整个项目的主线（见 `retrieve/fusion.py`）：

| 证据 | 可信度性质 |
|---|---|
| 结构化事实 | **硬数据**，未经推断 |
| 算法判断 | **有阈值依赖**，必须附实测依据 |
| 语义知识 | **只是"像"**，不代表认定 |

**如果三类证据在报告里长得一样，读的人就会把它们当成同等可信**——
那这份报告就在制造误解，而不是消除它。

所以每一类有**独立的区块、独立的颜色、以及写明可信度性质的标题**。

## ⚠️ 第二条约束：**「未获取到的信息」必须显眼**

它**不能**被排成一个小灰字脚注。

> **"我看了，没问题" 和 "我没看" 是两件完全不同的事。**
> 报告如果让这两者看起来一样，就是在骗读的人。

所以它单独成块、用警示色、并写明「**请勿当作没有异常**」。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from html import escape
from typing import Any

from aml_agent.agent import AgentRun
from aml_agent.agent.baseline import AccountBaseline

#: 报告样式。**内联**——自包含是这份报告的核心价值，不能依赖外部 CSS。
STYLE = """
:root {
  --ink: #16181d; --muted: #6b7280; --line: #e5e7eb; --bg: #f7f8fa;
  --fact: #1d4ed8; --judge: #b45309; --knowledge: #6d28d9; --missing: #b91c1c;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 20px 80px; background: var(--bg); color: var(--ink);
  font: 15px/1.75 -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
}
main { max-width: 960px; margin: 0 auto; }
h1 { font-size: 25px; margin: 0 0 4px; }
.sub { color: var(--muted); font-size: 13px; margin-bottom: 26px; }
.verdict {
  border-radius: 12px; padding: 20px 24px; margin-bottom: 26px;
  background: #fff; border: 1px solid var(--line); border-left-width: 5px;
}
.verdict.suspicious { border-left-color: #dc2626; }
.verdict.normal { border-left-color: #16a34a; }
.verdict.none { border-left-color var(--missing); border-left-color: var(--missing); }
.verdict h2 { margin: 0 0 8px; font-size: 20px; }
.reason { margin: 10px 0 0; }
section { background: #fff; border: 1px solid var(--line); border-radius: 12px;
          padding: 20px 24px; margin-bottom: 18px; }
section > h3 { margin: 0 0 4px; font-size: 16px; }
.nature { font-size: 12.5px; color: var(--muted); margin: 0 0 14px; }
.tag { display: inline-block; font-size: 11.5px; padding: 1px 8px; border-radius: 999px;
       margin-right: 6px; color: #fff; vertical-align: 2px; }
.tag.fact { background: var(--fact); }
.tag.judge { background: var(--judge); }
.tag.knowledge { background: var(--knowledge); }
.tag.missing { background: var(--missing); }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; font-size: 12.5px; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.bar { position: relative; height: 7px; background: #eef0f4; border-radius: 4px; width: 100%; }
.bar > i { position: absolute; inset: 0 auto 0 0; border-radius: 4px; background: var(--fact); }
.findings li { margin-bottom: 10px; }
.basis { color: var(--muted); font-size: 12.5px; display: block; }
.missing { border-left: 5px solid var(--missing); background: #fef2f2; }
.missing > h3 { color: var(--missing); }
.steps { font-size: 13px; }
.steps li { margin-bottom: 12px; }
.cmd { font-family: ui-monospace, Consolas, monospace; font-size: 12.5px;
       background: #f3f4f6; padding: 2px 7px; border-radius: 5px; }
.tool-ok { color: #16a34a; font-weight: 600; }
.tool-fail { color: var(--missing); font-weight: 600; }
footer { color: var(--muted); font-size: 12.5px; text-align: center; margin-top: 34px; }
"""


@dataclass(frozen=True, slots=True)
class ReportInput:
    """渲染一份报告需要的全部东西。"""

    run: AgentRun
    baseline: AccountBaseline | None = None
    #: 这次运行用的参数（写进报告，便于复现）
    parameters: Mapping[str, Any] | None = None


def _tag(kind: str, text: str) -> str:
    return f'<span class="tag {kind}">{escape(text)}</span>'


def _percentile_bar(percentile: float) -> str:
    """分位条。**方向必须是"从左往右越长越高"**——反过来读会误解。"""
    width = max(0.0, min(1.0, percentile)) * 100
    return f'<div class="bar"><i style="width:{width:.1f}%"></i></div>'


def _verdict_block(item: ReportInput) -> str:
    run = item.run
    if run.conclusion is None:
        # ⚠️ 未产出结论**不能**渲染成"未发现异常"——那正是本项目一直在防的混淆
        return (
            '<div class="verdict none">'
            "<h2>⚠️ 未产出结论</h2>"
            '<p class="reason">Agent <strong>耗尽步数也没给出结论</strong>。'
            "这不是「未发现异常」——它什么都没说。"
            "评测里它被单独计数，<strong>既不算漏报，也不算检出</strong>。</p>"
            f'<p class="sub">共 {len(run.steps)} 步，{run.seconds:.0f} 秒</p>'
            "</div>"
        )

    conclusion = run.conclusion
    css = "suspicious" if conclusion.suspicious else "normal"
    title = "🔴 存在可疑迹象" if conclusion.suspicious else "🟢 未发现可疑"
    extra = ""
    if run.premature:
        extra += '<p class="sub">⚠️ <strong>没有做过任何侦查就给了结论</strong>，请谨慎采信。</p>'
    if run.forced:
        extra += '<p class="sub">⚠️ 结论来自<strong>强制产出步</strong>（自由步数不够用）。</p>'
    missing = (
        f'<p class="reason">还缺：{escape(conclusion.missing)}</p>' if conclusion.missing else ""
    )
    return (
        f'<div class="verdict {css}">'
        f"<h2>{title}</h2>"
        f'<p class="sub">置信度 {conclusion.confidence:.2f}'
        f"　·　{len(run.steps)} 步　·　{run.seconds:.0f} 秒</p>"
        f'<p class="reason">{escape(conclusion.reason)}</p>'
        f"{missing}{extra}</div>"
    )


def _baseline_block(item: ReportInput) -> str:
    baseline = item.baseline
    if baseline is None:
        return ""
    rows = "".join(
        f"<tr><td>{escape(position.label)}</td>"
        f'<td class="num">{position.value:,.1f}</td>'
        f'<td class="num">{position.percentile * 100:.0f}%</td>'
        f"<td>{_percentile_bar(position.percentile)}</td></tr>"
        for position in baseline.features
    )
    return (
        "<section>"
        f"<h3>{_tag('judge', '相对位置')}该账户在全图中的位置</h3>"
        '<p class="nature">检测分数与各特征的<strong>分位</strong>——'
        "<strong>这是相对排名，不是判断</strong>：分位高不等于有问题。</p>"
        f"<p>检测分数分位：<strong>{baseline.detector_percentile * 100:.0f}%</strong></p>"
        '<table><thead><tr><th>特征</th><th class="num">取值</th>'
        '<th class="num">分位</th><th>分布</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></section>"
    )


def _facts_block(relations: list[Mapping[str, Any]]) -> str:
    """渲染结构化事实。

    ## 为什么从**工具结果**渲染，而不是重新去查一遍

    这份报告要展示的是「**Agent 当时看到了什么**」——
    因为它就是基于那些东西下的结论。

    重新查一遍得到的可能是**另一个时刻、另一组参数**下的数据，
    那样报告就不再是"这次调查的证据"，而是一份"关于这个账户的资料"——
    **两者看起来很像，但后者无法用来核对结论。**
    """
    if not relations:
        return (
            "<section>"
            f"<h3>{_tag('fact', '硬数据')}一、结构化事实</h3>"
            '<p class="nature">来自图数据库，<strong>未经推断</strong>。</p>'
            "<p>该范围内没有交易边。</p></section>"
        )
    rows = "".join(
        f"<tr><td>{escape(str(rel.get('counterparty', '?')))}</td>"
        f"<td>{escape(str(rel.get('direction', '?')))}</td>"
        f'<td class="num">{escape(str(rel.get("count", "?")))}</td>'
        f"<td>{escape(str(rel.get('amount', '?')))}</td>"
        f"<td>{escape(str(rel.get('time_span', '?')))}</td></tr>"
        for rel in relations[:40]
    )
    hidden = len(relations) - 40
    more = f'<p class="sub">…另有 {hidden} 组未列出</p>' if hidden > 0 else ""
    return (
        "<section>"
        f"<h3>{_tag('fact', '硬数据')}一、结构化事实</h3>"
        '<p class="nature">来自图数据库，<strong>未经推断</strong>。'
        "按账户对归并——<strong>图里每条边都在</strong>，未做丢失信息的聚合。</p>"
        '<table><thead><tr><th>对手方</th><th>方向</th><th class="num">笔数</th>'
        "<th>金额</th><th>时间跨度</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>{more}</section>"
    )


def _findings_block(signals: Mapping[str, Any]) -> str:
    """渲染算法判断——**本项目实测出来的各信号可靠度**。

    这一节的存在理由，就是整个项目对"可解释"的核心主张：
    **不只报「发现了什么」，还要报「这个发现有多可信、以及为什么」。**

    而依据不是形容词，是**我们自己在这份数据上跑出来的数字**。
    """
    if not signals:
        return ""
    rows = "".join(
        f"<tr><td><code>{escape(str(name))}</code></td><td>{escape(str(value))}</td></tr>"
        for name, value in signals.items()
    )
    return (
        "<section>"
        f"<h3>{_tag('judge', '算法判断')}二、各信号的实测可靠度</h3>"
        '<p class="nature">依赖阈值与算法选择，<strong>依据来自本项目在'
        "这份数据上的实测</strong>——不是形容词，是数字。</p>"
        '<p class="nature">⚠️ <strong>弱信号被明确标注</strong>：'
        "如果「环长 2」和「环长 3」以同样格式呈现，读的人没有依据区别对待它们"
        "——<strong>而不标注就等于在误导。</strong></p>"
        f"<table><thead><tr><th>信号</th><th>实测等级与依据</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></section>"
    )


def _knowledge_block(patterns: list[Mapping[str, Any]]) -> str:
    if not patterns:
        return (
            "<section>"
            f"<h3>{_tag('knowledge', '语义知识')}三、相似手法</h3>"
            '<p class="nature">语义匹配，<strong>不代表认定</strong>。</p>'
            "<p>没有达到相似度下限的已知手法——"
            "<strong>这本身是有价值的结论</strong>：它不像任何已知手法。</p></section>"
        )
    rows = "".join(
        f'<tr><td class="num">{float(match.get("similarity", 0)):.3f}</td>'
        f"<td>{escape(str(match.get('name', '?')))}"
        f'<span class="basis">{escape(str(match.get("definition", "")))}</span>'
        f'<span class="basis">来源：{escape(str(match.get("source", "")))}</span></td>'
        f"<td>{escape(str(match.get('category', '')))}</td></tr>"
        for match in patterns
    )
    return (
        "<section>"
        f"<h3>{_tag('knowledge', '语义知识')}三、相似手法</h3>"
        '<p class="nature">向量检索的语义匹配，<strong>只是"像"，不代表认定</strong>。'
        "每条都附来源——<strong>知识库里的条目必须可追溯</strong>。</p>"
        "<table><thead><tr><th>相似度</th><th>手法</th><th>类别</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></section>"
    )


def _missing_block(missing: Iterable[str]) -> str:
    items = list(missing)
    if not items:
        return ""
    rendered = "".join(f"<li>{escape(item)}</li>" for item in items)
    return (
        '<section class="missing">'
        f"<h3>{_tag('missing', '注意')}⚠️ 未获取到的信息</h3>"
        '<p class="nature"><strong>请勿当作「没有异常」</strong>——'
        "「我看了，没问题」和「我没看」是两件完全不同的事。</p>"
        f"<ul>{rendered}</ul></section>"
    )


def _steps_block(run: AgentRun) -> str:
    if not run.steps:
        return ""
    items = []
    for step in run.steps:
        marker = ' <span class="sub">[强制产出步]</span>' if step.forced else ""
        head = f"<strong>第 {step.index + 1} 步</strong>{marker}　{step.seconds:.0f} 秒"
        body = []
        if not step.tool_calls:
            said = escape(step.content[:220]) or "（空响应）"
            body.append(f'<span class="sub">只说话，没调工具：{said}</span>')
        for call in step.tool_calls:
            arguments = escape(str(dict(call.arguments))[:150])
            body.append(f'<span class="cmd">{escape(call.name)}({arguments})</span>')
        for result in step.results:
            status = "tool-ok" if result.ok else "tool-fail"
            label = "成功" if result.ok else "失败"
            body.append(
                f'<span class="{status}">← {label}</span> '
                f'<span class="sub">{escape(result.content[:220])}</span>'
            )
        items.append(f"<li>{head}<br>" + "<br>".join(body) + "</li>")
    return (
        "<section><h3>附：Agent 的执行轨迹</h3>"
        '<p class="nature">每一步调了什么工具、拿到了什么——'
        "<strong>结论能不能被核对，取决于这一节</strong>。</p>"
        f'<ul class="steps">{"".join(items)}</ul></section>'
    )


def render_html(item: ReportInput, *, title: str = "反洗钱可疑交易调查报告") -> str:
    """渲染成一份**自包含**的 HTML。

    **不引用任何外部资源**——没有 CDN、没有字体、没有图片。
    这样它可以被邮件发出去、离线打开、永久存档。
    """
    run = item.run
    subject = (
        run.steps[0].tool_calls[0].arguments.get("account_key")
        if run.steps and run.steps[0].tool_calls
        else None
    )
    account = str(subject) if subject else "（未从轨迹中识别）"

    params = dict(item.parameters or {})
    params_line = "　·　".join(f"{escape(str(k))}={escape(str(v))}" for k, v in params.items())

    # ⚠️ 证据**从工具结果里抽取**，而不是重新去图里查一遍。
    #
    # 这份报告要展示的是「**Agent 当时看到了什么**」——因为它就是基于那些东西
    # 下的结论。重新查一遍得到的可能是另一个时刻、另一组参数下的数据，
    # 那样报告就不再是"这次调查的证据"，而是一份"关于这个账户的资料"。
    # **两者看起来很像，但后者无法用来核对结论。**
    #
    # 而三类证据在轨迹里的出现次序并不固定，所以按**类别**聚合，
    # 让报告的结构对齐「可信度分层」，而不是对齐「执行顺序」。
    relations: list[Mapping[str, Any]] = []
    patterns: list[Mapping[str, Any]] = []
    signals: dict[str, Any] = {}
    missing: list[str] = []
    for step in run.steps:
        for result in step.results:
            payload: Mapping[str, Any] = result.payload
            name = result.name
            if name == "investigate_account":
                relations = [r for r in payload.get("relations", []) if isinstance(r, Mapping)]
                if payload.get("truncated"):
                    missing.append(str(payload["truncated"]))
            elif name == "match_patterns":
                found = payload.get("matches", [])
                patterns = [m for m in found if isinstance(m, Mapping)]
                if payload.get("note"):
                    missing.append(str(payload["note"]))
            elif name == "get_detector_scores":
                found_signals = payload.get("measuredSignals", {})
                if isinstance(found_signals, Mapping):
                    signals = dict(found_signals)
            elif payload.get("note"):
                missing.append(str(payload["note"]))

    # 各信号的实测可靠度：优先用工具单独返回过的全文版。
    # 一次都没查过时，从**基线内联的那份（只有等级）**兜底——
    # 报告里宁可只有等级，也不能凭空少掉这一节。
    if not signals and item.baseline is not None:
        signals = dict(item.baseline.as_payload()["measuredSignals"])

    sections = [
        _verdict_block(item),
        _baseline_block(item),
        _facts_block(relations),
        _findings_block(signals),
        _knowledge_block(patterns),
        _missing_block(missing),
        _steps_block(run),
    ]
    body = "".join(section for section in sections if section)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)} · {escape(account)}</title>
<style>{STYLE}</style>
</head>
<body>
<main>
  <h1>{escape(title)}</h1>
  <p class="sub">调查对象 <strong>{escape(account)}</strong></p>
  {f'<p class="sub">{params_line}</p>' if params_line else ""}
  {body}
  <footer>本报告由本地模型与图数据库生成，<strong>自包含、无外部依赖</strong>。
  三类证据按可信度分层展示；<strong>分位是相对排名，不是判断</strong>。</footer>
</main>
</body>
</html>
"""
