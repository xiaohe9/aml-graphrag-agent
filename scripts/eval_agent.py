"""Agent 量化评测：在**同时含正例与负例**的样本上跑，输出混淆矩阵。

    uv run python scripts/eval_agent.py --per-class 5
    uv run python scripts/eval_agent.py --per-class 10 --max-steps 6 --temperature 0.1

## ⚠️ 为什么必须有负例

只测洗钱案例的话，**一个"见谁都说可疑"的 Agent 会拿满分**。
所以两类各抽同样多，而且报告两类的活跃度是否可比。

## ⚠️ 耗时

实测约 **15 秒/步**，平均 2–3 步 → 约 1 分钟/案例。
`--per-class 5` 就是 10 个案例、约 10–15 分钟。**别一次跑太多。**
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from aml_agent import paths  # noqa: E402
from aml_agent.agent import (  # noqa: E402
    DEFAULT_MAX_STEPS,
    AgentRun,
    BaselineIndex,
    OllamaChat,
    Toolbox,
    run,
)
from aml_agent.data import loader  # noqa: E402
from aml_agent.embed import EmbeddingError, HashingEmbedder  # noqa: E402
from aml_agent.eval.agent_eval import (  # noqa: E402
    AgentEvalResult,
    CaseOutcome,
    describe_balance,
    question_for,
    sample_cases,
)
from aml_agent.graph import connect, load  # noqa: E402
from aml_agent.knowledge import load_patterns  # noqa: E402
from aml_agent.retrieve import build_index, load_index  # noqa: E402


def _activity(frame: pd.DataFrame) -> dict[int, int]:
    """每个账户的交易笔数（进出都算）——用来检查两类样本是否可比。"""
    out = frame["SENDER_ACCOUNT_ID"].value_counts()
    incoming = frame["RECEIVER_ACCOUNT_ID"].value_counts()
    total = out.add(incoming, fill_value=0)
    # 键是 pandas 的 `Hashable`，但账户 ID 在这个数据集里一定是整数。
    # 经 `str` 再转 `int`：对 mypy 干净，而且**遇到非数字键会立刻报错**，
    # 不会悄悄产出一个错的字典。
    return {int(str(key)): int(value) for key, value in total.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent 量化评测。")
    parser.add_argument("--per-class", type=int, default=5, help="每类抽几个（默认 5）")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--min-similarity", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument(
        "--no-match",
        action="store_true",
        help="关掉活跃度配对抽样（用来对比配对前后的差异）",
    )
    parser.add_argument("--out", default=None, help="把逐案例明细写到 JSON")
    parser.add_argument("--dry-run", action="store_true", help="只抽样，不跑模型")
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="不加载全图基线（用来对比'接上基线前后'的差异）",
    )
    args = parser.parse_args(argv)

    dataset = loader.load()
    alerts = dataset.alerts
    activity = _activity(dataset.transactions)

    cases = sample_cases(
        accounts=dataset.accounts["ACCOUNT_ID"].tolist(),
        # ⚠️ 真值来自 CSV，**不来自图**（ADR-0002）
        fraud_accounts=alerts["SENDER_ACCOUNT_ID"].tolist(),
        activity=activity,
        per_class=args.per_class,
        seed=args.seed,
        match_activity=not args.no_match,
    )

    print(f"样本 {len(cases)} 个（每类 {args.per_class}），种子 {args.seed}")
    for case in cases:
        mark = "洗钱" if case.is_fraud else "正常"
        print(f"  [{mark}] {case.account_key:<18} 活跃度 {case.activity:>5} 笔")
    notes = describe_balance(cases)
    print()
    for note in notes:
        print(f"  {note}")

    if args.dry_run:
        print("\n（--dry-run，不跑模型）")
        return 0

    patterns = load_patterns()
    embedder = HashingEmbedder(dimensions=256)
    index_path = paths.cache_path(paths.PATTERN_INDEX)
    try:
        index = load_index(index_path, embedder=embedder)
    except EmbeddingError:
        index = build_index(patterns, embedder)

    chat = OllamaChat(temperature=args.temperature)
    driver = connect()
    outcomes: list[CaseOutcome] = []
    started = time.monotonic()

    # ⚠️ 基线：**一次性算全图分数与分位**（约 2–4 秒），之后每个案例查表 O(1)。
    #
    # 这是 M7 评测实测出来的缺口的修法：Agent 之前只看得到「180 笔」，
    # 看不到「全图第 62 分位」——**不是判断错了，是没有基线可依**。
    baseline = None
    if not args.no_baseline:
        print("\n构造全图基线（检测分数 + 特征分位）…", flush=True)
        baseline_started = time.monotonic()
        baseline = BaselineIndex.build(load.load_edges())
        print(
            f"  完成，覆盖 {baseline.account_count} 个账户"
            f"（{time.monotonic() - baseline_started:.1f} 秒）"
        )

    try:
        for position, case in enumerate(cases, start=1):
            toolbox = Toolbox(
                driver=driver,
                patterns=patterns,
                index=index,
                embedder=embedder,
                min_similarity=args.min_similarity,
                baseline=baseline,
            )
            print(f"\n[{position}/{len(cases)}] {case.account_key} …", flush=True)
            agent_run: AgentRun = run(
                question_for(case), chat=chat, toolbox=toolbox, max_steps=args.max_steps
            )
            outcome = CaseOutcome(case=case, run=agent_run)
            outcomes.append(outcome)

            if outcome.verdict is None:
                verdict = "**未产出结论**"
            elif outcome.verdict:
                verdict = f"可疑（{outcome.confidence:.2f}）"
            else:
                verdict = f"未发现可疑（{outcome.confidence:.2f}）"
            truth = "洗钱" if case.is_fraud else "正常"
            hit = (
                "✅"
                if outcome.verdict is not None and outcome.verdict == case.is_fraud
                else ("—" if outcome.verdict is None else "❌")
            )
            print(
                f"  真值 {truth} / 判定 {verdict} {hit}"
                f"   {len(agent_run.steps)} 步 {agent_run.seconds:.0f} 秒"
            )
    finally:
        driver.close()

    result = AgentEvalResult(
        outcomes=tuple(outcomes),
        seconds=time.monotonic() - started,
        temperature=args.temperature,
        max_steps=args.max_steps,
        notes=(
            *notes,
            f"基线：{'已接（get_detector_scores 可用）' if baseline else '未接（--no-baseline）'}",
        ),
    )
    print()
    print(result.render())

    if args.out:
        target = Path(args.out)
        target.write_text(
            json.dumps(
                {
                    "summary": result.summary(),
                    "cases": [outcome.as_dict() for outcome in outcomes],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n明细已写入 {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
