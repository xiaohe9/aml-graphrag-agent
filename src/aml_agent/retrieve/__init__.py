"""语义检索：把"看到的可疑结构"匹配到"已知的手法"。

## ⚠️ 这里**没有向量数据库**——这是测量后的决定，不是偷懒

知识库只有 **18 条**模式（8 条图结构 + 10 条类型学，见 `knowledge/patterns.yaml`）。

引入一个向量数据库来检索 18 条数据是什么概念：

| | 18 条 × 1024 维 |
|---|---|
| 矩阵大小 | 18 × 1024 × 4 字节 = **72 KB**（实测） |
| 一次全量余弦相似度 | `18 × 1024` = **1.8 万次浮点运算**，微秒级 |
| 向量数据库要做的事 | 起进程、建索引、序列化、网络往返 |

**向量数据库是给"百万级向量、要毫秒级召回"的场景准备的。**
在这里用它，是**为了显得高级而引入的复杂度**——而且会让 CI 多一个要起的东西。

**但接口按"将来会长大"设计**：`search()` 只依赖"一个矩阵 + 一个查询向量"，
真涨到几万条时，把这一处换成向量库即可，**调用方不用动**。

详见 `docs/13-GraphRAG检索与融合.md`。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aml_agent.embed import Embedder, EmbeddingError, PatternIndex
from aml_agent.knowledge import Pattern

#: 默认返回几条。**不是越多越好**——它们会被拼进 LLM 的上下文，
#: 上下文塞得越满，模型对每一条的注意力就越弱。
DEFAULT_TOP_K = 5


@dataclass(frozen=True, slots=True)
class PatternMatch:
    """一次匹配结果。"""

    pattern: Pattern
    similarity: float

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.pattern.id,
            "name": self.pattern.name,
            "category": self.pattern.category,
            "similarity": round(self.similarity, 4),
        }


def build_index(patterns: Sequence[Pattern], embedder: Embedder) -> PatternIndex:
    """把知识库编码成索引。

    **这是构建期的一次性成本**（18 条 × 约 0.8 秒/条，实测 14.4 秒），
    不该出现在每次查询里。
    """
    if not patterns:
        raise EmbeddingError("知识库是空的，无法建索引")

    texts = tuple(pattern.embed_text() for pattern in patterns)
    vectors = embedder.embed(texts)
    return PatternIndex(
        pattern_ids=tuple(pattern.id for pattern in patterns),
        vectors=vectors,
        dimensions=embedder.dimensions,
        embedder_name=type(embedder).__name__,
        texts=texts,
    )


def search(
    query: str,
    index: PatternIndex,
    patterns: Sequence[Pattern],
    embedder: Embedder,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = 0.0,
) -> list[PatternMatch]:
    """把一段"观察到的结构描述"匹配到已知模式。

    ## 为什么要 `min_similarity`

    **余弦相似度永远会返回一个"最像"的结果**，哪怕它一点都不像。
    没有下限的话，一个完全无关的查询也会拿到 5 条"匹配"，
    然后它们会被当成证据拼进上下文——**这就是幻觉的入口**。

    有下限时，返回空列表是**有价值的答案**：
    它说的是"**这不像任何已知手法**"，而不是"没查到"。

    ## 索引与模式表为什么要分别传

    索引里只有向量和 id（那是持久化的部分）；
    模式表的完整内容（定义、来源、解释）从 YAML 加载。
    分开是为了**索引文件可以很小、且不怕知识库改文案**——
    只要 `embed_text()` 没变，索引就还有效。
    """
    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity 必须落在 [0, 1]")

    if not query.strip():
        raise ValueError("查询文本不能为空——空查询会匹配到任意模式")

    by_id = {pattern.id: pattern for pattern in patterns}
    unknown = [pid for pid in index.pattern_ids if pid not in by_id]
    if unknown:
        raise EmbeddingError(
            f"索引里有知识库里没有的模式：{unknown}。\n"
            "知识库改了就要重建索引——否则会引用到不存在的条目。"
        )

    query_vector = embedder.embed([query])
    if query_vector.shape[1] != index.dimensions:
        raise EmbeddingError(
            f"维度不匹配：查询向量 {query_vector.shape[1]}，索引 {index.dimensions}。\n"
            "换了嵌入模型就必须重建索引。"
        )

    # 两个矩阵都已归一化，所以内积就是余弦相似度
    similarities = (index.vectors @ query_vector[0]).astype(np.float64)

    # 稳定排序：同分时按索引里的原始顺序，保证**可复现**
    order = np.lexsort((np.arange(len(similarities)), -similarities))

    matches: list[PatternMatch] = []
    for position in order[: max(0, top_k)]:
        score = float(similarities[position])
        if score < min_similarity:
            # 已经降序，后面只会更低
            break
        matches.append(PatternMatch(pattern=by_id[index.pattern_ids[position]], similarity=score))
    return matches


# --------------------------------------------------------------------------- #
#  索引持久化
# --------------------------------------------------------------------------- #


def save_index(index: PatternIndex, path: Path) -> None:
    """存索引。

    存的是**三个东西**：向量、id 顺序、以及**元数据**。
    元数据不能省——没有它，换了模型之后会拿旧索引算出垃圾相似度，
    **而结果看起来完全正常**。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        vectors=index.vectors,
        pattern_ids=np.asarray(index.pattern_ids),
        dimensions=np.asarray(index.dimensions),
        embedder_name=np.asarray(index.embedder_name),
    )


def load_index(path: Path, *, embedder: Embedder) -> PatternIndex:
    """读索引，**并校验它和当前嵌入器是否匹配**。"""
    if not path.is_file():
        raise EmbeddingError(
            f"索引不存在：{path}\n建索引：uv run python scripts/build_pattern_index.py"
        )

    with np.load(path, allow_pickle=False) as data:
        dimensions = int(data["dimensions"])
        embedder_name = str(data["embedder_name"])

        if dimensions != embedder.dimensions:
            raise EmbeddingError(
                f"索引维度 {dimensions} 与当前嵌入器 {embedder.dimensions} 不一致。\n"
                "换了模型就必须重建索引——**旧索引和新向量算出的相似度是垃圾，"
                "而且看起来完全正常**。"
            )
        if embedder_name != type(embedder).__name__:
            raise EmbeddingError(
                f"索引由 {embedder_name} 建立，当前是 {type(embedder).__name__}。\n"
                "两套嵌入空间的向量不可比——请重建索引。"
            )

        return PatternIndex(
            pattern_ids=tuple(str(item) for item in data["pattern_ids"]),
            vectors=np.asarray(data["vectors"], dtype=np.float32),
            dimensions=dimensions,
            embedder_name=embedder_name,
        )
