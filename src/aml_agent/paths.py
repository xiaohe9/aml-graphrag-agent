"""项目里的路径约定。

## 为什么要有这个模块

"派生数据放哪"这件事一旦在两个地方各写一遍，**就会漂移**——
而漂移的后果是"某个地方读的是旧缓存"，**这种错很难查**。

所以缓存目录的逻辑只有这一份实现。

## 为什么缓存目录可配置

默认写在 `<项目>/data/cache`。但本机沙箱不允许**子进程**写项目目录，
所以允许用环境变量 `AML_CACHE_DIR` 指到别处。

**这不是为了本机方便，而是"派生数据的位置不该写死在代码里"这条原则的体现。**
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache"

#: 拉出来的边表缓存
EDGES_CACHE = "edges.parquet"

#: 模式知识库的嵌入索引
PATTERN_INDEX = "pattern_index.npz"


def cache_dir() -> Path:
    """派生数据的存放目录。"""
    override = os.environ.get("AML_CACHE_DIR")
    return Path(override) if override else DEFAULT_CACHE_DIR


def cache_path(name: str) -> Path:
    return cache_dir() / name
