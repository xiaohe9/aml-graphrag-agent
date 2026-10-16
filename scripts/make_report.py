"""跑一次真实调查，把**证据链**渲染成自包含的 HTML 报告。

    uv run python scripts/make_report.py amlsim:2454 --out docs/demo-report.html
    uv run python scripts/make_report.py --fraud-sample 1 --alert-type cycle

## 为什么报告是"自包含 HTML"

**一个文件，双击就开，能直接发给别人。**
不需要 clone、不需要起服务、不需要配环境。

而这个项目的证据展示**本质上是一份可核对的报告**，不是交互工具。
所以可移植性比交互性重要。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from aml_agent import paths  # noqa: E402
from aml_agent.agent import (  # noqa: E402
    DEFAULT_MAX_STEPS,
    BaselineIndex,
    OllamaChat,
    Toolbox,
    run,
)
from aml_agent.data import loader  # noqa: E402
from aml_agent.embed import EmbeddingError, HashingEmbedder  # noqa: E402
from aml_agent.graph import INSTITUTION, connect, load  # noqa: E402
from aml_agent.knowledge import load_patterns  # noqa: E402
from aml_agent.report import ReportInput, render_html  # noqa: E402
from aml_agent.retrieve import build_index, load_index  # noqa: E402


def pick_fraud_account(alert_type: str | None) -> str:
    """从**已确认洗钱**的告警里挑一个账户。

    标签从 CSV 读，不从图里读（ADR-0002）——图里根本没有标签。
    """
    dataset = loader.load()
    alerts = dataset.alerts
    if alert_type:
        alerts = alerts[alerts["ALERT_TYPE"] == alert_type]
    if alerts.empty:
        raise SystemExit(f"没有 ALERT_TYPE={alert_type!r} 的告警")
    first = int(alerts["SENDER_ACCOUNT_ID"].iloc[0])
    return f"{INSTITUTION}:{first}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成自包含的 HTML 调查报告。")
    parser.add_argument("account", nargs="?", default=None)
    parser.add_argument("--alert-type", default="cycle", help="--fraud-sample 时用哪种模式")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--min-similarity", type=float, default=0.35)
    parser.add_argument("--out", default="docs/demo-report.html")
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="不跑模型，只渲染一份说明性的样例报告（用于快速验证渲染）",
    )
    args = parser.parse_args(argv)

    account = args.account or pick_fraud_account(args.alert_type)
    print(f"调查对象：{account}")

    patterns = load_patterns()
    embedder = HashingEmbedder(dimensions=256)
    index_path = paths.cache_path(paths.PATTERN_INDEX)
    try:
        index = load_index(index_path, embedder=embedder)
    except EmbeddingError:
        index = build_index(patterns, embedder)

    print("构造全图基线…", flush=True)
    baseline_index = BaselineIndex.build(load.load_edges())
    baseline = baseline_index.lookup(account)
    print(f"  覆盖 {baseline_index.account_count} 个账户")

    driver = connect()
    try:
        toolbox = Toolbox(
            driver=driver,
            patterns=patterns,
            index=index,
            embedder=embedder,
            min_similarity=args.min_similarity,
            baseline=baseline_index,
        )

        if args.skip_agent:
            print("（--skip-agent：不跑模型）")
            print("  但报告需要一次真实的 Agent 运行才能渲染证据链。")
            print("  去掉这个参数即可（约 1–5 分钟）。")
            return 0

        question = (
            f"帮我查一下账户 {account} 有没有可疑的资金往来。"
            "结论里请引用你实际查到的数据（笔数、金额、时间跨度、分位）。"
        )
        print("跑 Agent …", flush=True)
        agent_run = run(
            question,
            chat=OllamaChat(temperature=args.temperature),
            toolbox=toolbox,
            max_steps=args.max_steps,
        )
    finally:
        driver.close()

    verdict = (
        "未产出结论"
        if agent_run.conclusion is None
        else ("可疑" if agent_run.conclusion.suspicious else "未发现可疑")
    )
    print(f"  判定：{verdict}　{len(agent_run.steps)} 步　{agent_run.seconds:.0f} 秒")

    html = render_html(
        ReportInput(
            run=agent_run,
            baseline=baseline,
            parameters={
                "temperature": args.temperature,
                "max_steps": args.max_steps,
                "model": OllamaChat().model,
            },
        )
    )

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    print(f"\n报告已写入 {target}（{len(html) / 1024:.0f} KB，自包含）")
    print("  用浏览器直接打开即可——不需要任何服务。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
