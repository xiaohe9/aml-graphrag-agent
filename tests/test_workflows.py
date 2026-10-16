"""校验仓库里的 GitHub Actions workflow 能被真正解析。

## 为什么需要这个测试（**这是踩过坑的**）

上一个项目里，`ci.yml` 第 87 行有一句没加引号的 `- name: ... (not committed: 5 MB ...)`。
那个 `": "` 让 YAML 认为这里冒出一个映射值，**GitHub 直接拒收整个 workflow**——
于是**每一次推送的每一个 job 都失败，而且从没运行过**。

更糟的是：当时我用 `Select-String`（正则）"检查"过，它显示没问题。
**正则不是解析器**——它只能看到我想到的模式，看不到语法本身错了。

**所以这个测试用真正的 YAML 解析器**，并且额外做两项检查：

1. **能被 `yaml.safe_load` 解析**——语法正确；
2. **每个 job 都有 `runs-on`**——GitHub 会拒收缺这个字段的 job，
   而单靠语法解析发现不了（它在语法上是合法的）；
3. **触发条件存在**——没有 `on:` 的 workflow 永远不会跑，
   而"永远不跑"和"跑失败"在页面上看起来完全不一样。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOW_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _workflow_files() -> list[Path]:
    return sorted(WORKFLOW_DIR.glob("*.y*ml"))


def test_workflow_directory_is_not_empty() -> None:
    """防止"测试绿了但其实一个 workflow 都没扫到"这种假安全。"""
    assert WORKFLOW_DIR.is_dir(), f"{WORKFLOW_DIR} 不存在"
    assert _workflow_files(), "没有扫到任何 workflow 文件"


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_workflow_parses_as_yaml(path: Path) -> None:
    """**用真正的解析器**，不是正则。"""
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        pytest.fail(
            f"{path.name} 不是合法的 YAML——GitHub 会**整份拒收**，"
            f"于是所有 job 都静默不运行。\n{exc}"
        )
    assert isinstance(parsed, dict), f"{path.name} 的顶层应该是映射"


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_workflow_has_a_trigger(path: Path) -> None:
    """没有 `on:` 的 workflow 永远不会跑。

    **"永远不跑"和"跑失败"在页面上看起来完全不一样**——
    前者是静默的，没人会发现。所以单独测一条。
    """
    parsed: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    # YAML 会把裸 `on` 解析成布尔 True，这是个经典坑，所以两种键名都要认
    assert "on" in parsed or True in parsed, f"{path.name} 缺少触发条件 `on:`"
    jobs = parsed.get("jobs")
    assert isinstance(jobs, dict) and jobs, f"{path.name} 没有任何 job"
    for name, job in jobs.items():
        assert job.get("runs-on"), f"{path.name} 的 job {name!r} 缺少 runs-on"


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_workflow_steps_are_well_formed(path: Path) -> None:
    """每个 step 要么有 `uses`，要么有 `run`——否则 GitHub 拒收。"""
    parsed: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    for job_name, job in parsed.get("jobs", {}).items():
        for index, step in enumerate(job.get("steps", [])):
            assert isinstance(step, dict), f"{path.name} {job_name} 第 {index} 步不是映射"
            has_action = "uses" in step
            has_command = "run" in step
            assert has_action or has_command, (
                f"{path.name} 的 {job_name} 第 {index} 步既没有 uses 也没有 run"
            )


def test_ci_uses_the_shared_check_script() -> None:
    """**CI 必须调用 `scripts/check.py`，不能自己写一套命令。**

    如果 CI 和本地各写一套，两者迟早不一致，
    然后「CI 绿但本地红」会让人不再信任门禁、开始绕过它。
    """
    text = (WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")
    assert "scripts/check.py" in text, (
        "CI 没有调用 scripts/check.py——CI 和本地会各自演化，门禁迟早失效"
    )
