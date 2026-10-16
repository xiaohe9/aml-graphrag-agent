"""知识库加载与嵌入索引的测试。

**不依赖 Ollama**——用 `HashingEmbedder`。
理由见 `aml_agent.embed` 模块文档：依赖外部服务的测试在 CI 里会一直跳过，
而**"一直跳过"等于没有测试**。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from aml_agent.embed import (
    BGE_M3_DIMENSIONS,
    EmbeddingError,
    HashingEmbedder,
    PatternIndex,
)
from aml_agent.knowledge import (
    DEFAULT_PATH,
    KnowledgeError,
    Pattern,
    load_cached,
    load_patterns,
)
from aml_agent.retrieve import (
    build_index,
    load_index,
    save_index,
    search,
)

# --------------------------------------------------------------------------- #
#  1. 知识库本身
# --------------------------------------------------------------------------- #


def test_knowledge_base_loads() -> None:
    patterns = load_patterns()
    assert len(patterns) >= 15, f"知识库只剩 {len(patterns)} 条，太少"


def test_every_entry_declares_a_source() -> None:
    """**来源是必填的。**

    编造洗钱类型学会让整个检索层失去意义——
    **它检索出来的"知识"必须可信，否则融合进上下文只是把幻觉放大。**
    """
    for pattern in load_patterns():
        assert pattern.source.strip(), f"{pattern.id} 没有来源"
        assert len(pattern.source) > 5, f"{pattern.id} 的来源太短，像是占位符"


def test_ids_are_unique() -> None:
    ids = [pattern.id for pattern in load_patterns()]
    assert len(ids) == len(set(ids))


def test_both_categories_are_present() -> None:
    """两个分组各有用途：结构模式管**检测**，类型学管**解释**。

    缺了前者检索不出可比的结构；缺了后者解释不了"这属于哪一类可疑活动"。
    """
    categories = {pattern.category for pattern in load_patterns()}
    assert categories == {"structural", "typologies"}


def test_the_two_dataset_labels_exist() -> None:
    """数据集带标注的两种模式必须在知识库里——否则评测真值对不上语义。"""
    by_label = {p.extras.get("dataset_label") for p in load_patterns()}
    assert "cycle" in by_label
    assert "fan_in" in by_label


def test_missing_source_is_rejected(tmp_path: Path) -> None:
    """缺来源必须**加载时就报错**，而不是静默通过。"""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump(
            {"structural": [{"id": "x", "name": "X", "definition": "d"}]},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(KnowledgeError, match="source"):
        load_patterns(bad)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "dup.yaml"
    entry = {"id": "same", "name": "N", "definition": "d", "source": "s"}
    bad.write_text(
        yaml.safe_dump({"structural": [entry, entry]}, allow_unicode=True), encoding="utf-8"
    )
    with pytest.raises(KnowledgeError, match="重复"):
        load_patterns(bad)


def test_embed_text_includes_aliases_but_not_explanations() -> None:
    """别名要进嵌入文本（跨术语检索靠它），解释性字段**不进**（会稀释信号）。"""
    pattern = Pattern(
        id="x",
        name="回流环路",
        aliases=("cycle", "环形转账"),
        definition="资金绕一圈回到起点。",
        source="s",
        category="structural",
        extras={"why_suspicious": "因为正常交易不会原路返回", "detection_hint": "找环"},
    )
    text = pattern.embed_text()
    assert "回流环路" in text
    assert "cycle" in text and "环形转账" in text
    assert "正常交易不会原路返回" not in text
    assert "找环" not in text


def test_the_knowledge_file_is_where_we_think_it_is() -> None:
    assert DEFAULT_PATH.is_file(), f"知识库不在 {DEFAULT_PATH}"


# --------------------------------------------------------------------------- #
#  2. 嵌入器
# --------------------------------------------------------------------------- #


def test_hashing_embedder_is_deterministic() -> None:
    """同一段文本必须永远得到同一个向量——否则索引就白建了。"""
    embedder = HashingEmbedder()
    first = embedder.embed(["回流环路"])
    second = embedder.embed(["回流环路"])
    assert np.array_equal(first, second)


def test_hashing_embedder_normalises() -> None:
    """约定返回已归一化的向量，这样余弦相似度就是内积。

    调用方不需要每处都手动归一化——**漏一处就会算错，而且错得不明显**。
    """
    vectors = HashingEmbedder().embed(["回流环路", "集中转入"])
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_hashing_embedder_has_some_semantic_behaviour() -> None:
    """**它不是随机数**：共享子串越多，相似度越高。

    这一点很重要——测试因此可以断言"相似文本得分更高"，
    从而验证整条检索链路真的**在按相似度排序**，
    而不是只验证"函数能跑通"。
    """
    embedder = HashingEmbedder()
    vectors = embedder.embed(["资金回流环路", "资金回流环路形成闭环", "今天天气不错"])
    similar = float(vectors[0] @ vectors[1])
    unrelated = float(vectors[0] @ vectors[2])
    assert similar > unrelated, f"相似 {similar:.3f} 应该大于无关 {unrelated:.3f}"


def test_empty_input_gives_empty_output() -> None:
    """空输入不能崩，也不能返回形状不对的东西。"""
    matrix = HashingEmbedder().embed([])
    assert matrix.shape == (0, HashingEmbedder().dimensions)


def test_zero_vector_does_not_produce_nan() -> None:
    """零向量的范数是 0，除它会得到 NaN——**而 NaN 会让排序变得不确定**。"""
    vectors = HashingEmbedder().embed(["   "])  # 全是空白字符
    assert not np.isnan(vectors).any(), "零向量被归一化成了 NaN"


# --------------------------------------------------------------------------- #
#  3. 索引
# --------------------------------------------------------------------------- #


def test_build_index_shape() -> None:
    patterns = load_patterns()
    embedder = HashingEmbedder(dimensions=64)
    index = build_index(patterns, embedder)
    assert index.vectors.shape == (len(patterns), 64)
    assert index.pattern_ids == tuple(p.id for p in patterns)
    assert len(index.texts) == len(patterns)


def test_index_rejects_mismatched_shape() -> None:
    """形状不对必须报错——错位的相似度**看起来完全正常**。"""
    with pytest.raises(EmbeddingError, match="形状不对"):
        PatternIndex(
            pattern_ids=("a", "b"),
            vectors=np.zeros((3, 8), dtype=np.float32),
            dimensions=8,
            embedder_name="X",
        )


def test_index_round_trip(tmp_path: Path) -> None:
    patterns = load_patterns()
    embedder = HashingEmbedder(dimensions=64)
    index = build_index(patterns, embedder)
    path = tmp_path / "index.npz"
    save_index(index, path)

    reloaded = load_index(path, embedder=embedder)
    assert reloaded.pattern_ids == index.pattern_ids
    assert np.allclose(reloaded.vectors, index.vectors)


def test_index_rejects_a_different_embedder(tmp_path: Path) -> None:
    """换了嵌入模型就必须重建索引。

    **旧索引和新向量算出的相似度是垃圾，而且看起来完全正常**——
    所以这个校验不能省。
    """
    patterns = load_patterns()
    index = build_index(patterns, HashingEmbedder(dimensions=64))
    path = tmp_path / "index.npz"
    save_index(index, path)

    with pytest.raises(EmbeddingError, match="维度"):
        load_index(path, embedder=HashingEmbedder(dimensions=128))


def test_load_index_gives_actionable_error_when_missing(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingError, match="build_pattern_index"):
        load_index(tmp_path / "nope.npz", embedder=HashingEmbedder())


def test_bge_m3_dimension_constant_is_not_a_guess() -> None:
    """1024 是实测出来的（真跑过一次 bge-m3），不是抄文档。

    它参与索引文件的格式——**写错会让索引与向量静默错位**。
    """
    assert BGE_M3_DIMENSIONS == 1024


# --------------------------------------------------------------------------- #
#  4. 检索
# --------------------------------------------------------------------------- #


def _index_and_patterns() -> tuple[PatternIndex, tuple[Pattern, ...]]:
    patterns = load_patterns()
    embedder = HashingEmbedder(dimensions=256)
    return build_index(patterns, embedder), patterns


def test_search_returns_ranked_matches() -> None:
    index, patterns = _index_and_patterns()
    matches = search("回流环路 环形转账 cycle", index, patterns, HashingEmbedder(dimensions=256))
    assert matches, "应该至少匹配到一条"
    scores = [m.similarity for m in matches]
    assert scores == sorted(scores, reverse=True), "必须按相似度降序"


def test_search_respects_top_k() -> None:
    index, patterns = _index_and_patterns()
    matches = search("资金转移", index, patterns, HashingEmbedder(dimensions=256), top_k=3)
    assert len(matches) <= 3


def test_min_similarity_can_return_nothing() -> None:
    """**返回空列表是有价值的答案。**

    余弦相似度永远会给出一个"最像"的结果，哪怕它一点都不像。
    没有下限的话，一个完全无关的查询也会拿到几条"匹配"，
    然后被当成证据拼进上下文——**这就是幻觉的入口**。
    """
    index, patterns = _index_and_patterns()
    matches = search(
        "完全无关的一段话：今天午饭吃什么",
        index,
        patterns,
        HashingEmbedder(dimensions=256),
        min_similarity=0.99,
    )
    assert matches == [], "相似度下限很高时应该返回空"


def test_search_rejects_empty_query() -> None:
    """空查询会匹配到任意模式——必须拦住。"""
    index, patterns = _index_and_patterns()
    with pytest.raises(ValueError, match="不能为空"):
        search("   ", index, patterns, HashingEmbedder(dimensions=256))


def test_search_rejects_invalid_min_similarity() -> None:
    index, patterns = _index_and_patterns()
    with pytest.raises(ValueError, match="min_similarity"):
        search("x", index, patterns, HashingEmbedder(dimensions=256), min_similarity=1.5)


def test_search_detects_index_knowledge_mismatch() -> None:
    """索引里有、知识库里没有的 id —— 知识库改了没重建索引。"""
    patterns = load_patterns()[:3]
    embedder = HashingEmbedder(dimensions=64)
    index = build_index(patterns, embedder)
    stale = PatternIndex(
        pattern_ids=(*index.pattern_ids, "已经被删掉的模式"),
        vectors=np.vstack([index.vectors, index.vectors[:1]]),
        dimensions=64,
        embedder_name="HashingEmbedder",
    )
    with pytest.raises(EmbeddingError, match="知识库里没有"):
        search("x", stale, patterns, embedder)


def test_search_is_reproducible() -> None:
    """同一输入必须同一输出（NFR-5）。"""
    index, patterns = _index_and_patterns()
    first = search("回流环路", index, patterns, HashingEmbedder(dimensions=256))
    second = search("回流环路", index, patterns, HashingEmbedder(dimensions=256))
    assert [m.pattern.id for m in first] == [m.pattern.id for m in second]


def test_all_patterns_are_reachable_by_their_own_name() -> None:
    """**每条模式至少能被自己的名字检索到。**

    这是对索引质量的最低要求——如果连用自己的名字都排不进前排，
    说明 `embed_text()` 拼得有问题。
    """
    index, patterns = _index_and_patterns()
    embedder = HashingEmbedder(dimensions=256)
    misses: list[str] = []
    for pattern in patterns:
        matches = search(pattern.name, index, patterns, embedder, top_k=5)
        if pattern.id not in {m.pattern.id for m in matches}:
            misses.append(pattern.id)
    assert not misses, f"这些模式用自己的名字都检索不到：{misses}"


def test_load_cached_returns_the_same_object() -> None:
    """缓存过的东西应该是同一个对象——否则 `lru_cache` 没生效。"""
    assert load_cached() is load_cached()
