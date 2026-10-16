"""图谱层：Cypher 语句、建图、**安全读路径**、一致性校验。

⚠️ **这里的函数叫 `build_graph` / `verify_graph`，而不是 `build` / `verify`。**

原因：包 `aml_agent.graph` 里有同名子模块 `build.py` 和 `verify.py`。
如果在 `__init__.py` 里导出同名的函数，那个函数会**覆盖掉同名的子模块属性**——
于是 `from aml_agent.graph import build` 拿到的是**函数**而不是**模块**，
`build.DEFAULT_URI` 这类访问全部失效。

**这个坑是 mypy 抓出来的**（6 个 `attr-defined` 报错）。
用 `as` 起别名能绕过，但会触发 ruff 的别名排序规则，而且**调用方得记住这件事**。
**在源头把名字改清楚，是唯一不需要调用方小心的解法。**

## 关于 `queries` 与 `guard`

`queries` 是**唯一允许从图里读数据的地方**，它所有查询的 `RETURN` 子句
都从白名单生成；`guard` 是它的自动审计。三层防御的说明见 `guard` 模块文档。
"""

from aml_agent.graph import cypher, guard, queries
from aml_agent.graph.build import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_PASSWORD,
    DEFAULT_URI,
    DEFAULT_USER,
    INSTITUTION,
    BuildStats,
    GraphNotEmptyError,
    build_graph,
    clear,
    connect,
    counts,
    setup,
)
from aml_agent.graph.queries import (
    MAX_EDGE_LIMIT,
    MAX_HOPS,
    MAX_PATH_LENGTH,
    Neighborhood,
    TransferEdge,
    neighborhood,
    property_names,
    shortest_path,
    transactions_between,
)
from aml_agent.graph.verify import Check, verify_and_report, verify_graph

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_PASSWORD",
    "DEFAULT_URI",
    "DEFAULT_USER",
    "INSTITUTION",
    "MAX_EDGE_LIMIT",
    "MAX_HOPS",
    "MAX_PATH_LENGTH",
    "BuildStats",
    "Check",
    "GraphNotEmptyError",
    "Neighborhood",
    "TransferEdge",
    "build_graph",
    "clear",
    "connect",
    "counts",
    "cypher",
    "guard",
    "neighborhood",
    "property_names",
    "queries",
    "setup",
    "shortest_path",
    "transactions_between",
    "verify_and_report",
    "verify_graph",
]
