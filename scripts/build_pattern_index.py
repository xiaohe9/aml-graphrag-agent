"""建立可疑模式知识库的嵌入索引。

    uv run python scripts/build_pattern_index.py            # 用 Ollama（真实语义）
    uv run python scripts/build_pattern_index.py --offline  # 用确定性哈希（CI/无 Ollama）
    uv run python scripts/build_pattern_index.py --query "资金在两个账户之间来回转"

**索引是构建期产物**：22 条模式各编码一次，之后查询只编码"查询本身"一条。
所以建索引慢一点没关系，**查询快才是关键**。

## 这个脚本还负责回答一个问题：要不要向量数据库？

它会打印知识库规模、矩阵大小、单次查询耗时——
**用数字回答"引入向量数据库是不是过度设计"**，而不是凭感觉。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from aml_agent import paths  # noqa: E402
from aml_agent.embed import EmbeddingError, HashingEmbedder, default_embedder  # noqa: E402
from aml_agent.knowledge import load_patterns  # noqa: E402
from aml_agent.retrieve import build_index, load_index, save_index, search  # noqa: E402


def _report_scale(
    pattern_count: int, dimensions: int, build_seconds: float, query_ms: float
) -> None:
    """把"要不要向量数据库"这件事**用数字说清楚**。"""
    matrix_bytes = pattern_count * dimensions * 4
    operations = pattern_count * dimensions
    print("\n=== 规模与选型 ===")
    print(f"  知识库            {pattern_count} 条")
    print(f"  向量维度          {dimensions}")
    print(
        f"  索引矩阵          {pattern_count} × {dimensions} × 4 字节"
        f" = {matrix_bytes / 1024:.0f} KB"
    )
    print(f"  建索引（一次性）   {build_seconds:.1f} 秒")
    print(f"  单次查询（含编码） {query_ms:.0f} 毫秒")
    print()
    print(
        f"  一次全量余弦相似度 = {pattern_count} × {dimensions}"
        f" = {operations / 10_000:.0f} 万次浮点运算"
    )
    print()
    print("  ── 结论 ──")
    print("  ❌ **不引入向量数据库。**")
    print("     它的适用场景是「百万级向量 + 毫秒级召回」；")
    print(f"     而这里是 {pattern_count} 条、{matrix_bytes / 1024:.0f} KB、微秒级算完。")
    print()
    print("     用向量数据库意味着：多起一个进程、多一份索引文件、")
    print("     CI 里多一个要拉起来的服务——**为了显得高级而引入的复杂度**。")
    print()
    print("  ✅ 但接口按「将来会长大」设计：")
    print("     `retrieve.search()` 只依赖「一个矩阵 + 一个查询向量」，")
    print("     真涨到几万条时换掉这一处即可，**调用方不用动**。")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="建立可疑模式知识库的嵌入索引。")
    parser.add_argument("--offline", action="store_true", help="用确定性哈希嵌入，不连 Ollama")
    parser.add_argument("--query", default=None, help="建完索引后顺手查一条")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help="相似度下限。**低于它返回空列表**，而不是硬凑几条",
    )
    args = parser.parse_args(argv)

    patterns = load_patterns()
    print(f"知识库：{len(patterns)} 条模式")
    for category in ("structural", "typologies"):
        names = [p.name for p in patterns if p.category == category]
        suffix = "…" if len(names) > 5 else ""
        print(f"  {category:<12} {len(names):>2} 条：{'、'.join(names[:5])}{suffix}")

    embedder = HashingEmbedder() if args.offline else default_embedder()
    print(f"\n嵌入器：{type(embedder).__name__}（{embedder.dimensions} 维）")

    print("\n建索引…")
    started = time.monotonic()
    index = build_index(patterns, embedder)
    build_seconds = time.monotonic() - started
    print(f"  完成，用时 {build_seconds:.1f} 秒（{build_seconds / len(patterns):.2f} 秒/条）")

    target = paths.cache_path(paths.PATTERN_INDEX)
    save_index(index, target)
    print(f"  已保存到 {target}")

    # 查询耗时：**这才是关键指标**——建索引是构建期的一次性成本
    query = args.query or "资金经过多个账户层层转移后回到起点"
    print(f"\n查询：{query!r}")
    # 先热一次（Ollama 首次调用有冷启动，实测约 26 秒）
    search(query, index, patterns, embedder, top_k=args.top_k)
    started = time.monotonic()
    matches = search(
        query,
        index,
        patterns,
        embedder,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
    )
    query_ms = (time.monotonic() - started) * 1000

    if not matches:
        print("  （没有达到相似度下限的匹配——这本身是有价值的答案：不像任何已知手法）")
    for rank, match in enumerate(matches, start=1):
        print(f"  {rank}. {match.similarity:.4f}  [{match.pattern.category}] {match.pattern.name}")
        print(f"       {match.pattern.definition[:60]}…")

    # 验证存回来的索引能读——"保存了但读不出"会在别处才暴露
    #
    # 用显式 raise 而不是 `assert`：`assert` 在 `python -O` 下会被整个去掉，
    # 于是这条校验**在优化模式下静默消失**。
    reloaded = load_index(target, embedder=embedder)
    if reloaded.pattern_ids != index.pattern_ids:
        raise EmbeddingError(
            f"存回来的索引与内存里的不一致：{reloaded.pattern_ids} vs {index.pattern_ids}"
        )

    _report_scale(len(patterns), index.dimensions, build_seconds, query_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
