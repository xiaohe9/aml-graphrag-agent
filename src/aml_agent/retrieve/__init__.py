"""检索与融合层。

## 三个子模块，各有明确职责

| 模块 | 职责 |
|---|---|
| `matches` | 公共数据类型（`PatternMatch`）——单独放是为了**打断循环导入** |
| `semantic` | 语义检索：结构 → 已知手法（**不引入向量数据库**，理由见模块文档） |
| `fusion` | 融合：把三类异构证据装配成 LLM 上下文，**并标注各自的实测可信度** |

## 这里是"GraphRAG"真正成立的地方

两路检索本身都不新鲜。**新鲜的是融合**：

> 把「**结构化事实** + **算法判断** + **语义知识**」装配起来，
> 而且**明确区分它们的可信度**——事实是硬数据，算法判断有阈值依赖，
> 语义匹配只是"像"。

详见 `fusion` 模块文档与 `docs/13`。
"""

from aml_agent.retrieve.fusion import (
    MAX_FACTS_RENDERED,
    MEASURED_SIGNALS,
    EdgeGroup,
    Finding,
    InvestigationContext,
    derive_findings,
    group_facts,
    render,
    signal_reliability,
)
from aml_agent.retrieve.matches import DEFAULT_TOP_K, PatternMatch
from aml_agent.retrieve.semantic import (
    build_index,
    load_index,
    save_index,
    search,
)

__all__ = [
    "DEFAULT_TOP_K",
    "MAX_FACTS_RENDERED",
    "MEASURED_SIGNALS",
    "EdgeGroup",
    "Finding",
    "InvestigationContext",
    "PatternMatch",
    "build_index",
    "derive_findings",
    "group_facts",
    "load_index",
    "render",
    "save_index",
    "search",
    "signal_reliability",
]
