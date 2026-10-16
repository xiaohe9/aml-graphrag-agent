"""仓库层面的自动审计。

**这些检查的对象不是代码，是仓库本身。** 它们拦的是"人忘了"这一类错误——
而这类错误**一旦发生就是公开的**，改不回来。

## 三件被自动守住的事

### 1. 隐私红线

上个项目里 `.gitignore` 少了简历相关的规则，
**一次 `git add -A` 就会把简历副本推到公开仓库**——当时没有任何东西拦着它。

**规则写在 `.gitignore` 里只是"声明"，`git ls-files` 才能证明"确实没进去"。**
所以这里直接查**已跟踪的文件列表**。

### 2. 依赖源不能被写死成第三方镜像

生成 `uv.lock` 时如果带了国内镜像参数，**镜像地址会被烧进 lock 文件**——
于是 GitHub 的 CI 机器会跑去中国高校镜像下依赖：慢、可能限流、
而且给公开仓库引入了不必要的第三方依赖。

**这个坑是靠肉眼看出来的。** 加了测试之后就不必再靠眼睛。

### 3. 必须入库的东西不能少

`uv.lock` 少一个，CI 的 `--frozen` 就装不了依赖。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 匹配这些模式的路径**绝不能**出现在已跟踪文件里。
#:
#: 刻意写得宽：**漏掉一条的代价（隐私泄露到公开仓库）远大于多拦一条的代价（改个文件名）。**
PRIVACY_PATTERNS: tuple[str, ...] = (
    ".docx",
    ".doc",
    ".pdf",
    ".pem",
    ".key",
    ".env",
    "resume",
    "简历",
    "data/extracted",
    "_probe",
    "_tools",
)

#: 必须入库的文件——少一个 CI 就跑不起来。
REQUIRED_TRACKED: tuple[str, ...] = ("uv.lock", ".gitignore", "pyproject.toml")


def _tracked_files() -> list[str]:
    """已跟踪的文件列表。不在 git 仓库里就跳过。"""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("不是 git 仓库")
    return [line for line in result.stdout.splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
#  1. 隐私红线
# --------------------------------------------------------------------------- #


def test_no_private_files_are_tracked() -> None:
    """**已跟踪的文件里不能有隐私文件。**

    查 `git ls-files` 而不是读 `.gitignore`：
    `.gitignore` 只能说明"打算忽略什么"，
    **只有已跟踪列表能证明"确实没进去"**。
    """
    tracked = _tracked_files()
    assert tracked, "已跟踪文件列表为空——检查逻辑或仓库状态有问题"

    offenders = [
        path
        for path in tracked
        if any(pattern.lower() in path.lower() for pattern in PRIVACY_PATTERNS)
    ]
    assert not offenders, (
        "这些隐私文件被跟踪了，**已经进入公开仓库历史**：\n  "
        + "\n  ".join(offenders)
        + "\n\n如果是刚 add 还没 commit，用 `git rm --cached <路径>` 撤出暂存区；"
        "\n如果已经 commit，需要改写历史——**所以这个检查必须跑在提交之前**。"
    )


def test_gitignore_declares_the_privacy_red_lines() -> None:
    """`.gitignore` 必须**声明**这些红线。

    这条和上一条是两回事：
    上一条查"实际有没有进去"，这一条查"规则有没有写"。
    规则没写，下次有人建一个 `report.docx` 时就不会被拦。
    """
    text = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    must_declare = ("*.docx", ".env", "data/", "resume/", "_probe/")
    missing = [pattern for pattern in must_declare if pattern not in text]
    assert not missing, f".gitignore 缺少这些红线声明：{missing}"


# --------------------------------------------------------------------------- #
#  2. 依赖源
# --------------------------------------------------------------------------- #


def test_lock_file_does_not_pin_a_third_party_mirror() -> None:
    """`uv.lock` **不能**把国内镜像写死。

    若生成 lock 时带了 `--index-url <镜像>`，**镜像地址会被烧进 lock**，
    于是 GitHub 的 CI 机器会去中国高校镜像下依赖。

    **这个坑是靠肉眼看出来的**——现在不用了。
    """
    lock_path = PROJECT_ROOT / "uv.lock"
    assert lock_path.is_file(), "uv.lock 不存在——CI 的 `uv sync --frozen` 会失败"

    text = lock_path.read_text(encoding="utf-8")
    mirrors = ("tuna.tsinghua", "aliyun", "douban", "ustc.edu", "mirrors.cloud")
    found = [mirror for mirror in mirrors if mirror in text]
    assert not found, (
        f"uv.lock 里写死了第三方镜像：{found}\n"
        "用官方源重新生成：`uv lock --default-index https://pypi.org/simple`"
    )
    # 反向确认：确实指向了官方源（否则可能是空 lock 或格式变了）
    assert "pypi.org" in text or "files.pythonhosted.org" in text


# --------------------------------------------------------------------------- #
#  3. 必需文件
# --------------------------------------------------------------------------- #


def test_required_files_are_tracked() -> None:
    """必需文件必须在版本控制里——否则 CI 上就是另一个世界。"""
    tracked = set(_tracked_files())
    missing = [name for name in REQUIRED_TRACKED if name not in tracked]
    assert not missing, f"这些文件必须入库但没被跟踪：{missing}"


def test_workflow_is_tracked() -> None:
    """CI 配置本身必须入库，不然 GitHub 不会跑任何东西。"""
    tracked = _tracked_files()
    workflows = [path for path in tracked if path.startswith(".github/workflows/")]
    assert workflows, "没有任何 workflow 被跟踪——GitHub 上不会跑 CI"
