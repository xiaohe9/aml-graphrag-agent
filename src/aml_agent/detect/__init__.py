"""检测层：便宜的结构信号 + 规则基线。

**分层是刻意的**（见 `docs/03-概要设计`）：

- **第一层（本包）**：可全图运行的便宜信号，负责**粗筛**；
- **第二层（Agent）**：对少数候选做有预算的定向深挖。

这么分的直接原因见 `structural` 模块文档：
全局环枚举**不可行**，而且「节点在环上」**没有区分度**。
"""

from aml_agent.detect import baseline, structural
from aml_agent.detect.baseline import (
    BASELINES,
    amount_threshold_score,
    random_score,
    small_amount_score,
)
from aml_agent.detect.structural import (
    FEATURE_COLUMNS,
    combined_score,
    compute_edge_features,
    fuse_max,
    fuse_sum,
    reciprocal_score,
    small_amount_reference_score,
    two_detector_score,
    two_detector_scores,
)

__all__ = [
    "BASELINES",
    "FEATURE_COLUMNS",
    "amount_threshold_score",
    "baseline",
    "combined_score",
    "compute_edge_features",
    "fuse_max",
    "fuse_sum",
    "random_score",
    "reciprocal_score",
    "small_amount_reference_score",
    "small_amount_score",
    "structural",
    "two_detector_score",
    "two_detector_scores",
]
