"""评测层。

**先于检测算法存在**，理由见 `metrics` 模块文档。
"""

from aml_agent.eval.metrics import (
    ConfusionMatrix,
    PrecisionAtRecall,
    Ranking,
    confusion_at,
    confusion_at_k,
    precision_at_k,
    precision_at_recall,
    summarize,
)

__all__ = [
    "ConfusionMatrix",
    "PrecisionAtRecall",
    "Ranking",
    "confusion_at",
    "confusion_at_k",
    "precision_at_k",
    "precision_at_recall",
    "summarize",
]
