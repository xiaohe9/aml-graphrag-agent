"""真跑一次 Agent（本地 qwen3:4b + Neo4j）。

    uv run python scripts/run_agent.py amlsim:4122
    uv run python scripts/run_agent.py --fraud-sample 3     # 从已确认洗钱的交易里挑账户
    uv run python scripts/run_agent.py --question "帮我查一下 amlsim:4122"

## 这个脚本会打印什么

每一步调了什么工具、工具返回了什么（截断显示）、最后结论是什么，
以及**三个必须单独看的字段**：

```text
exhausted   步数耗尽、从未给结论   ← 工程缺陷，不是"无异常"
forced      结论来自强制产出步     ← 说明自由步数不够用
premature   没做任何侦查就给结论   ← 实测发现模型会这样
```

## ⚠️ 单次调用 10–17 秒，所以一个案例约 1–2 分钟

这不是"慢"的问题——反洗钱是批处理业务。但**它意味着不能反复跑大评测**。
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
from aml_agent.agent import DEFAULT_MAX_STEPS, OllamaChat, Toolbox, run  # noqa: E402
from aml_agent.data import loader  # noqa: E402
from aml_agent.embed import EmbeddingError, HashingEmbedder  # noqa: E402
from aml_agent.graph import connect  # noqa: E402
from aml_agent.knowledge import load_patterns  # noqa: E402
from aml_agent.retrieve import build_index, load_index  # noqa: E402


def pick_fraud_accounts(count: int, alert_type: str | None = None) -> list[str]:
    """从**已确认洗钱**的告警里挑几个账户。

    **标签从 CSV 读，不从图里读**（ADR-0002）——图里根本没有标签。
    这是"标签不写进图"这条设计在调用侧的体现。

    `alerts.csv` 自带 `ALERT_TYPE`（`fan_in` / `cycle`），
    所以可以**按模式类型挑样本**——这样 demo 才有针对性。
    """
    from aml_agent.graph import INSTITUTION

    dataset = loader.load()
    alerts = dataset.alerts
    if alert_type:
        alerts = alerts[alerts["ALERT_TYPE"] == alert_type]
    if alerts.empty:
        raise SystemExit(f"没有 ALERT_TYPE={alert_type!r} 的告警，无法取样")

    account_ids = alerts["SENDER_ACCOUNT_ID"].drop_duplicates().head(count).tolist()
    return [f"{INSTITUTION}:{int(account_id)}" for account_id in account_ids]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="真跑一次 Agent。")
    parser.add_argument("account", nargs="?", default=None, help="账户键，形如 amlsim:4122")
    parser.add_argument("--question", default=None, help="自定义问题")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--fraud-sample", type=int, default=0, help="从确认洗钱的告警里挑 N 个账户")
    parser.add_argument(
        "--alert-type",
        default=None,
        help="只挑这种模式的告警（如 cycle / fan_in）",
    )
    parser.add_argument("--min-similarity", type=float, default=0.35)
    args = parser.parse_args(argv)

    if args.fraud_sample:
        accounts = pick_fraud_accounts(args.fraud_sample, args.alert_type)
        label = f"（ALERT_TYPE={args.alert_type}）" if args.alert_type else ""
        print(f"从已确认洗钱的告警里挑出 {len(accounts)} 个账户{label}：{accounts}")
    elif args.account:
        accounts = [args.account]
    else:
        accounts = ["amlsim:4122"]
    print()

    patterns = load_patterns()
    # ⚠️ 这里用哈希嵌入而不是 Ollama——**省掉 18 条 × 0.8 秒的建索引时间**。
    # 代价是语义检索质量下降，但本脚本要看的是【Agent 编排】而不是检索质量。
    embedder = HashingEmbedder(dimensions=256)
    index_path = paths.cache_path(paths.PATTERN_INDEX)
    try:
        index = load_index(index_path, embedder=embedder)
    except EmbeddingError:
        # 索引不存在、或者它是用**另一个嵌入器**建的（本脚本用哈希，而
        # `build_pattern_index.py` 默认用 Ollama 的 bge-m3，维度就不同）。
        # **这不是错误，是预期内的分支**——所以只捕获 `EmbeddingError`，
        # 而不是所有异常（那会把真正的 bug 也吞掉）。
        print("建检索索引…")
        index = build_index(patterns, embedder)
        print("  完成")

    driver = connect()
    chat = OllamaChat()
    print(f"模型：{chat.model}    步数预算：{args.max_steps}")
    print("=" * 78)

    try:
        for position, account in enumerate(accounts, start=1):
            question = args.question or (
                f"帮我查一下账户 {account} 有没有可疑的资金往来。"
                "结论里请引用你实际查到的数据（笔数、金额、时间跨度）。"
            )
            print(f"\n【案例 {position}/{len(accounts)}】{question}\n")

            toolbox = Toolbox(
                driver=driver,
                patterns=patterns,
                index=index,
                embedder=embedder,
                min_similarity=args.min_similarity,
            )
            started = time.monotonic()
            result = run(question, chat=chat, toolbox=toolbox, max_steps=args.max_steps)
            elapsed = time.monotonic() - started

            for step in result.steps:
                marker = " [强制产出步]" if step.forced else ""
                print(f"  第 {step.index + 1} 步（{step.seconds:.1f} 秒）{marker}")
                if not step.tool_calls:
                    print(f"    只说话：{step.content[:90]}")
                for call in step.tool_calls:
                    print(f"    → {call.name}({dict(call.arguments)})")
                for tool_result in step.results:
                    status = "ok" if tool_result.ok else "失败"
                    print(f"      ← [{status}] {tool_result.content[:150]}")

            print()
            if result.conclusion is None:
                print("  ❌ **未产出结论**（步数耗尽）—— 这一条必须单独统计，不能算作'无异常'")
            else:
                conclusion = result.conclusion
                verdict = "可疑" if conclusion.suspicious else "未发现可疑"
                print(f"  结论：{verdict}    置信度 {conclusion.confidence:.2f}")
                print(f"  理由：{conclusion.reason}")
                if conclusion.missing:
                    print(f"  还缺：{conclusion.missing}")
                if result.forced:
                    print("  ⚠️ 结论来自强制产出步——自由步数不够用")
                if result.premature:
                    print("  ⚠️ **没做任何侦查就给了结论**")

            print(
                f"  用时 {elapsed:.1f} 秒，{len(result.steps)} 步，"
                f"工具调用 {dict(result.tool_counts)}"
            )
            print("=" * 78)
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
