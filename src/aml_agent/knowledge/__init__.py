"""可疑模式知识库：加载 `knowledge/patterns.yaml`。

## 为什么知识库是"数据文件"而不是"代码"

它是**领域知识**，不是逻辑。放在 YAML 里：

- 非程序员也能改；
- 改动**会出现在 diff 里**（藏在 Python 字面量里容易被忽略）；
- 可以整份替换成外部的类型学材料。

## 为什么每条都要标来源

编造洗钱类型学会让整个检索层失去意义——
**它检索出来的"知识"必须是可信的，否则融合进 LLM 上下文只是把幻觉放大。**
所以 `source` 是**必填字段**，加载时校验。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATH = PROJECT_ROOT / "knowledge" / "patterns.yaml"

#: 两个分组的键名。分开是因为**它们的用途不同**：
#: 结构模式服务于"检测"，类型学服务于"解释"。
STRUCTURAL_KEY = "structural"
TYPOLOGY_KEY = "typologies"


@dataclass(frozen=True, slots=True)
class Pattern:
    """一条可疑模式。"""

    id: str
    name: str
    aliases: tuple[str, ...]
    definition: str
    source: str
    #: `structural`（图结构）或 `typology`（资金流转类型学）
    category: str
    #: 其余字段原样保留——不同分组有不同的可选字段
    extras: dict[str, Any]

    def embed_text(self) -> str:
        """拼出**用来算嵌入**的文本。

        ## 为什么不是简单地把所有字段连起来

        嵌入模型对**过长、信息混杂**的文本效果会下降。
        这里刻意只取三样：**名称 + 别名 + 定义**。

        别名很重要——同一个模式在不同机构叫法不同（"拆分"/"化整为零"/"smurfing"），
        把别名并进文本里，**检索才能跨过术语差异**。这正是语义检索相对关键词检索的价值。

        `why_suspicious` 和 `detection_hint` 刻意**不参与嵌入**：
        它们更长、更"解释性"，会把定义的信号稀释掉。
        它们留给融合层（拼上下文时用），不参与相似度计算。
        """
        parts = [self.name, "；".join(self.aliases), self.definition]
        return "\n".join(part for part in parts if part)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "definition": self.definition,
            "source": self.source,
            **self.extras,
        }


class KnowledgeError(ValueError):
    """知识库格式不对。

    **失败要响亮**：一条缺来源的模式如果能静默加载，
    它就会一路混进 LLM 的上下文里，而**没有任何东西提示它不可信**。
    """


def _require(entry: dict[str, Any], field: str, *, index: int, group: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeError(
            f"{group}[{index}] 缺少必填字段 {field!r}（或它不是非空字符串）。\n完整条目：{entry}"
        )
    return value


def _parse(entry: dict[str, Any], *, index: int, group: str) -> Pattern:
    if not isinstance(entry, dict):
        raise KnowledgeError(f"{group}[{index}] 不是映射：{entry!r}")

    known = {"id", "name", "aliases", "definition", "source"}
    aliases = entry.get("aliases") or []
    if not isinstance(aliases, list):
        raise KnowledgeError(
            f"{group}[{index}] 的 aliases 应该是列表，收到 {type(aliases).__name__}"
        )

    return Pattern(
        id=_require(entry, "id", index=index, group=group),
        name=_require(entry, "name", index=index, group=group),
        aliases=tuple(str(alias) for alias in aliases),
        definition=_require(entry, "definition", index=index, group=group),
        # ⚠️ 来源必填——见模块文档
        source=_require(entry, "source", index=index, group=group),
        category=group,
        extras={key: value for key, value in entry.items() if key not in known},
    )


def load_patterns(path: Path | None = None) -> tuple[Pattern, ...]:
    """读取知识库。"""
    target = DEFAULT_PATH if path is None else path
    if not target.is_file():
        raise KnowledgeError(f"知识库文件不存在：{target}")

    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise KnowledgeError(f"{target} 的顶层应该是映射")

    patterns: list[Pattern] = []
    for group in (STRUCTURAL_KEY, TYPOLOGY_KEY):
        entries = raw.get(group) or []
        if not isinstance(entries, list):
            raise KnowledgeError(f"{group} 应该是列表")
        patterns.extend(
            _parse(entry, index=index, group=group) for index, entry in enumerate(entries)
        )

    if not patterns:
        raise KnowledgeError(f"{target} 里一条模式都没有")

    ids = [pattern.id for pattern in patterns]
    duplicates = {item for item in ids if ids.count(item) > 1}
    if duplicates:
        raise KnowledgeError(f"模式 id 重复：{sorted(duplicates)}")

    return tuple(patterns)


@lru_cache(maxsize=1)
def load_cached() -> tuple[Pattern, ...]:
    """带缓存——知识库是只读的，反复解析没必要。"""
    return load_patterns()
