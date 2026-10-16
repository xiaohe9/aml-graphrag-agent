"""校验仓库里**所有** YAML 文件能被真正解析。

## 为什么需要这个测试（**本项目第三次踩同一类坑**）

| # | 文件 | 写错的写法 | YAML 当成什么 |
|---|---|---|---|
| 1 | 上个项目的 `ci.yml` | `- name: ... (not committed: 5 MB)` | 冒号后是**映射值** |
| 2 | 本项目的 `ci.yml` | 开头写 `> **设计要点…**` | **块标量指示符** |
| 3 | **`knowledge/patterns.yaml`** | `definition: **同一个顶点**…` | **别名引用**（`*` 开头） |

三次都是同一件事：**用写文档的习惯写数据文件**。
而中文技术写作里加粗、冒号、破折号都是家常便饭。

## 为什么必须扫"全仓库"

前两次的检查只覆盖了 `.github/workflows/*.yml`——
**而第三次的漏洞就在没扫到的地方。**

这和 `test_label_guard.py` 里那条教训完全一样：
**没扫到的文件就是漏洞。** 所以这里扫的是整个仓库。

## 为什么用真解析器

`Select-String`（正则）看不出语法错误——**正则不是解析器**。
第 1 次踩坑时，正是用正则"检查"过、显示没问题，结果 GitHub 整份拒收 workflow。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 不扫这些目录。
#:
#: `data/` 是下载的数据集，`_probe/` 是一次性脚本，`_tools/` 是外部工具——
#: 它们都不该被 lint 或测试，而且体积大。
#: （它们同时也在 `.gitignore` 里，所以不会出现在新克隆的仓库中。）
EXCLUDED_DIRS = frozenset(
    {"data", "_probe", "_tools", ".git", ".venv", "venv", "node_modules", "__pycache__"}
)

#: 解析失败时要给出的**可操作**提示。
#:
#: 光说"YAML 语法错误"没用——得告诉人**哪种中文写作习惯会踩坑**。
HINTS = (
    "以 `*` 开头的值会被当成【别名引用】，以 `&` 开头是【锚点】",
    "含 `: `（冒号加空格）的裸值会被当成【嵌套映射】",
    "以 `>`、`|` 开头会被当成【块标量指示符】",
    "以 `-` 开头（后面跟空格）会被当成【列表项】",
    "以 `%`、`@`、反引号 开头是保留字符",
)


def _yaml_files() -> list[Path]:
    """找出仓库里所有 YAML 文件（已跟踪区域内的）。"""
    found: list[Path] = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml"}:
            continue
        if any(part in EXCLUDED_DIRS for part in path.relative_to(PROJECT_ROOT).parts):
            continue
        found.append(path)
    return sorted(found)


def test_the_scan_actually_finds_files() -> None:
    """防止"扫描逻辑坏了但测试还绿"这种假安全。"""
    files = _yaml_files()
    assert files, "一个 YAML 文件都没扫到——扫描逻辑坏了"
    names = {path.name for path in files}
    # 至少要有 CI 配置和知识库这两类
    assert "ci.yml" in names, f"没扫到 ci.yml：{names}"
    assert "patterns.yaml" in names, f"没扫到知识库：{names}"


@pytest.mark.parametrize("path", _yaml_files(), ids=lambda p: str(p.relative_to(PROJECT_ROOT)))
def test_yaml_file_parses(path: Path) -> None:
    """**用真正的解析器**，不是正则。"""
    text = path.read_text(encoding="utf-8")
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        relative = path.relative_to(PROJECT_ROOT)
        pytest.fail(
            f"{relative} 不是合法的 YAML。\n\n{exc}\n\n"
            "常见原因（本项目三次都栽在这上面）：\n  - "
            + "\n  - ".join(HINTS)
            + "\n\n修法：散文类字段一律用块标量 `>-`，把文本原样保留，"
            "绕开所有这些陷阱。"
        )
    assert parsed is not None, f"{relative} 解析结果是空的"


def test_no_plain_scalar_starts_with_a_reserved_character() -> None:
    """即使侥幸解析通过，**裸值以保留字符开头也是隐患**。

    这条是补充性的：`*` 开头的值通常直接报错，但有些情况（比如后面正好
    跟着一个已定义的锚点名）会**静默地解析成别的东西**——那更危险。
    """
    offenders: list[str] = []
    for path in _yaml_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("- "):
                continue
            if ":" not in stripped:
                continue
            _, _, value = stripped.partition(":")
            value = value.strip()
            if value and value[0] in "*&|>%@`" and value not in {"|", "|-", ">", ">-"}:
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{number}  {stripped[:70]}")

    assert not offenders, (
        "这些裸值以 YAML 保留字符开头：\n  "
        + "\n  ".join(offenders)
        + "\n\n用块标量 `>-` 或加引号。"
    )
