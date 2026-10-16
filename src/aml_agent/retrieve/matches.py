"""检索层的公共数据类型。

**单独成模块是为了打断循环导入**：
`fusion` 需要 `PatternMatch`，而 `retrieve/__init__` 又要导出 `fusion`——
如果 `PatternMatch` 定义在 `__init__` 里，就成了环。
"""

from __future__ import annotations

from dataclasses import dataclass

from aml_agent.knowledge import Pattern

#: 默认返回几条。**不是越多越好**——它们会被拼进 LLM 的上下文，
#: 上下文塞得越满，模型对每一条的注意力就越弱。
DEFAULT_TOP_K = 5


@dataclass(frozen=True, slots=True)
class PatternMatch:
    """一次"结构 → 手法"的语义匹配。"""

    pattern: Pattern
    #: 余弦相似度。**只是"像"，不是"就是"**——融合层会把这个性质说清楚。
    similarity: float

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.pattern.id,
            "name": self.pattern.name,
            "category": self.pattern.category,
            "similarity": round(self.similarity, 4),
        }
