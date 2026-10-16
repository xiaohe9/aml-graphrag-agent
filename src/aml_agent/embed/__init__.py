"""文本嵌入：**接口 + 两个实现**。

## 为什么必须是接口（这不是"为了好看"）

真实实现走 **Ollama**（本地、零 API 成本），但 **CI 里跑不了它**：
要下 1.2 GB 模型，CPU 推理每条约 0.45 秒。

如果测试直接依赖 Ollama，**那些测试在 CI 里就会一直跳过**——
而**"一直跳过"等于没有测试**。

所以这里有两个实现：

| 实现 | 用途 | 特点 |
|---|---|---|
| `OllamaEmbedder` | 真实检索 | 本地推理，零 API 成本 |
| `HashingEmbedder` | **CI 与单元测试** | 确定性、无网络、毫秒级 |

## 实测数据（决定了容量规划）

在本机（CPU 推理）实测 `bge-m3`：

```text
冷启动（模型首次载入）   25.8 秒   ← 只有第一次
热调用（单条）            0.41 秒
批量 20 条               10.56 秒   = 528 毫秒/条
```

⚠️ **批量传 20 条，平均每条反而更慢**（528ms vs 440ms）——
**"批量更快"这个直觉在 CPU 推理上不成立**，每条文本是独立编码的，
批量只多了序列化开销。**按 0.45 秒/条做容量规划才靠谱。**
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

#: `bge-m3` 的输出维度。硬编码是有意的——
#: 它参与索引文件的格式，变了要让索引失效，而不是悄悄错位。
BGE_M3_DIMENSIONS = 1024

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "bge-m3"


class EmbeddingError(RuntimeError):
    """嵌入服务不可用或返回了意料之外的形状。

    **失败要响亮**：如果嵌入悄悄退化成一堆零向量，
    检索会返回"看起来正常"的结果——**而它们全是错的**。
    """


@runtime_checkable
class Embedder(Protocol):
    """嵌入器的接口。

    `dimensions` 是协议的一部分，因为**索引文件要记录它**：
    换了模型（维度变了）却用旧索引，得到的相似度是垃圾。
    """

    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """把文本编码成 `(len(texts), dimensions)` 的 float32 矩阵。

        **约定返回已归一化的向量**——这样余弦相似度就是内积，
        调用方不需要每处都手动归一化（漏一处就会算错）。
        """
        ...


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化。

    零向量的范数是 0，除它会得到 NaN——**而 NaN 会一路传播到排序里，
    让顺序变得不确定**。这里把零向量原样保留（它的相似度恒为 0，语义正确）。
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return np.asarray(matrix / safe, dtype=np.float32)


# --------------------------------------------------------------------------- #
#  真实实现：Ollama
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class OllamaEmbedder:
    """走本地 Ollama 的嵌入器。**零 API 成本、完全离线。**

    用 stdlib 的 `urllib` 而不是 `requests`：少一个依赖，而且这里只需要一个 POST。
    """

    model: str = DEFAULT_OLLAMA_MODEL
    host: str = DEFAULT_OLLAMA_HOST
    timeout: float = 180.0
    dimensions: int = BGE_M3_DIMENSIONS

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        payload = json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - host 是本机常量
            f"{self.host}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise EmbeddingError(
                f"连不上本地 Ollama（{self.host}）：{exc}\n"
                "先确认服务在跑：ollama serve\n"
                f"再确认模型已拉取：ollama pull {self.model}"
            ) from exc

        vectors = body.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise EmbeddingError(
                f"Ollama 返回的向量数量不对：期望 {len(texts)}，"
                f"收到 {len(vectors) if isinstance(vectors, list) else type(vectors).__name__}"
            )

        matrix = np.asarray(vectors, dtype=np.float32)
        # 显式标注成 `tuple[int, ...]`：numpy 的 `shape` 在 mypy 那里会被收窄成
        # `tuple[int]`（它从 `np.asarray(list)` 推断成了 1 维），
        # 于是和一个 2 元组比较会被判成"不可能相等"。
        # 标注一下既说清意图，也让类型检查通过。
        shape: tuple[int, ...] = matrix.shape
        if shape != (len(texts), self.dimensions):
            raise EmbeddingError(
                f"向量形状不对：期望 ({len(texts)}, {self.dimensions})，收到 {shape}。\n"
                "模型换了就必须重建索引——旧索引和新向量的相似度是垃圾。"
            )
        return _l2_normalise(matrix)


# --------------------------------------------------------------------------- #
#  测试实现：确定性哈希
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class HashingEmbedder:
    """确定性假嵌入。**CI 与单元测试用这个。**

    ## 为什么它不是随机数

    用**字符 n-gram 哈希**（类似 sklearn 的 HashingVectorizer），
    所以**共享子串的文本相似度更高**。

    这一点很关键：测试因此可以断言"相似文本得分更高"，
    从而验证整条检索链路**真的在按相似度排序**，
    而不是只验证"函数能跑通、没抛异常"。

    中文用字符 n-gram 特别合适——不需要分词，共享二字词就会产生相似度。

    ## 它**不能**用来评估检索质量

    它不理解语义。"拆分"和"smurfing"在它眼里毫无关系。
    **真正的语义质量只能用真实模型测。**
    """

    dimensions: int = 256
    ngrams: tuple[int, ...] = (2, 3)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for n in self.ngrams:
                for start in range(max(1, len(text) - n + 1)):
                    gram = text[start : start + n]
                    if not gram.strip():
                        continue
                    digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                    bucket = int.from_bytes(digest, "big") % self.dimensions
                    matrix[row, bucket] += 1.0
        return _l2_normalise(matrix)


# --------------------------------------------------------------------------- #
#  选择实现
# --------------------------------------------------------------------------- #


def default_embedder(*, allow_network: bool = True) -> Embedder:
    """挑一个能用的嵌入器。

    先试 Ollama；连不上就**降级到哈希实现并说明原因**——
    这样"没装 Ollama 的人 clone 下来也能跑通全流程"，
    而不是撞在一个连接错误上。

    ⚠️ 降级会**明显影响检索质量**，所以必须把这件事说出来，
    不能悄悄降级（那就是在骗人）。
    """
    if not allow_network:
        return HashingEmbedder()

    probe = OllamaEmbedder()
    try:
        probe.embed(["探活"])
    except EmbeddingError as exc:
        print(
            f"⚠️  Ollama 不可用（{type(exc).__name__}），**降级到确定性哈希嵌入**。\n"
            "    流程能跑通，但**语义检索质量会明显下降**——\n"
            "    它不理解语义，'拆分'和'smurfing'在它眼里毫无关系。\n"
            "    要看真实效果：`ollama serve` 并 `ollama pull bge-m3`。"
        )
        return HashingEmbedder()
    return probe


@dataclass(frozen=True, slots=True)
class PatternIndex:
    """已经把知识库编码好的索引。

    **把向量和"它属于哪条模式"绑在一起**，而不是各自存一份列表——
    分开存迟早会错位，而错位的相似度**看起来完全正常**。
    """

    pattern_ids: tuple[str, ...]
    vectors: np.ndarray
    dimensions: int
    embedder_name: str
    #: 嵌入用的文本，留着便于排查"为什么这条没被检索到"
    texts: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.vectors.shape != (len(self.pattern_ids), self.dimensions):
            raise EmbeddingError(
                f"索引形状不对：向量 {self.vectors.shape}，"
                f"模式 {len(self.pattern_ids)} 条，维度 {self.dimensions}"
            )
        if self.texts and len(self.texts) != len(self.pattern_ids):
            raise EmbeddingError("索引里的文本条数和模式条数不一致")
