"""Agent 评测的测试。

**不跑模型**——全部用构造好的 `AgentRun` / `CaseOutcome`。
理由同前：依赖真实模型的测试在 CI 里会一直跳过。

## 这个文件守的两件事

### 1. **必须有负例**

只测洗钱案例的话，**一个"见谁都说可疑"的 Agent 会拿满分**。

### 2. **「未产出结论」不能折进 `fn` 或 `tn`**

折进 `fn` → 虚假地压低召回率？不——折进 `tn` 才是灾难：
它同时**压低检出率**（本该算漏报）又**虚假地提高精确率**。
所以它必须落在 `positive_silent` / `negative_silent` 里。
"""

from __future__ import annotations

import pytest

from aml_agent.agent import AgentRun, Conclusion
from aml_agent.eval.agent_eval import (
    AgentCase,
    AgentEvalResult,
    CaseOutcome,
    describe_balance,
    question_for,
    sample_cases,
)

# --------------------------------------------------------------------------- #
#  构造替身
# --------------------------------------------------------------------------- #


def _run(
    *, suspicious: bool | None, steps: int = 2, exhausted: bool = False, forced: bool = False
) -> AgentRun:
    conclusion = (
        None
        if suspicious is None
        else Conclusion(suspicious=suspicious, confidence=0.8, reason="因为查到了 X")
    )
    return AgentRun(
        question="q",
        conclusion=conclusion,
        steps=(),
        exhausted=exhausted or conclusion is None,
        forced=forced,
        premature=False,
        seconds=10.0 * steps,
        tool_counts={},
    )


def _outcome(
    account: str, *, is_fraud: bool, suspicious: bool | None, **kwargs: object
) -> CaseOutcome:
    return CaseOutcome(
        case=AgentCase(account_key=account, is_fraud=is_fraud, activity=100),
        run=_run(suspicious=suspicious, **kwargs),  # type: ignore[arg-type]
    )


def _result(outcomes: list[CaseOutcome]) -> AgentEvalResult:
    return AgentEvalResult(outcomes=tuple(outcomes), seconds=100.0, temperature=0.1, max_steps=6)


# --------------------------------------------------------------------------- #
#  1. 混淆矩阵的四格
# --------------------------------------------------------------------------- #


def test_confusion_counts_the_four_ordinary_cells() -> None:
    result = _result(
        [
            _outcome("a", is_fraud=True, suspicious=True),  # TP
            _outcome("b", is_fraud=True, suspicious=False),  # FN
            _outcome("c", is_fraud=False, suspicious=True),  # FP
            _outcome("d", is_fraud=False, suspicious=False),  # TN
        ]
    )
    confusion = result.confusion
    assert (confusion.tp, confusion.fn, confusion.fp, confusion.tn) == (1, 1, 1, 1)
    assert confusion.no_answer == 0


# --------------------------------------------------------------------------- #
#  2. ★「未产出结论」必须单独一格
# --------------------------------------------------------------------------- #


def test_no_conclusion_lands_in_silent_not_in_fn_or_tn() -> None:
    """**这是本项目最重要的一条口径。**

    一个没给出结论的洗钱案例，**既不是"正确地没报警"（tn），
    也不是"给了结论但漏了"（fn）**——它是 `positive_silent`。

    折进 `tn` 会同时**压低检出率**又**虚假地提高精确率**。
    """
    result = _result(
        [
            _outcome("a", is_fraud=True, suspicious=None),  # 洗钱 + 没给结论
            _outcome("b", is_fraud=False, suspicious=None),  # 正常 + 没给结论
        ]
    )
    confusion = result.confusion
    assert confusion.positive_silent == 1
    assert confusion.negative_silent == 1
    assert confusion.fn == 0, "「没给结论」不是「给了结论但漏报」"
    assert confusion.tn == 0, "「没给结论」更不是「正确地没报警」"
    assert confusion.no_answer == 2


def test_recall_denominator_includes_silent_positives() -> None:
    """召回率的分母**必须包含「没给结论」的洗钱案例**。

    否则"什么都不答"能拿到一个不难看的召回率。
    """
    result = _result(
        [
            _outcome("a", is_fraud=True, suspicious=True),
            _outcome("b", is_fraud=True, suspicious=None),
            _outcome("c", is_fraud=True, suspicious=None),
        ]
    )
    # TP=1，分母 = 1 + 0 + 2 = 3
    assert result.confusion.recall == pytest.approx(1 / 3)


def test_silent_negatives_do_not_inflate_precision() -> None:
    """没给结论的正常案例**不能**算进"没报警"，否则精确率会虚高。"""
    silent = _result(
        [
            _outcome("a", is_fraud=True, suspicious=True),
            _outcome("b", is_fraud=False, suspicious=None),
            _outcome("c", is_fraud=False, suspicious=None),
        ]
    )
    # 报了一个警，其中 1 个是真的 → 精确率 100%（分母只有报出来的）
    assert silent.confusion.precision == 1.0
    # 但"未产出结论"有 2 个，召回的分子分母不受影响
    assert silent.confusion.no_answer == 2


# --------------------------------------------------------------------------- #
#  3. ★ 一个"见谁都说可疑"的 Agent 不能拿高分
# --------------------------------------------------------------------------- #


def test_a_always_suspicious_agent_is_exposed_by_the_negatives() -> None:
    """**这就是必须有负例的理由。**

    如果样本全是洗钱案例，这个 Agent 的检出率是 100%——
    看起来完美，实际上毫无分辨力。
    """
    result = _result(
        [_outcome(f"p{i}", is_fraud=True, suspicious=True) for i in range(5)]
        + [_outcome(f"n{i}", is_fraud=False, suspicious=True) for i in range(5)]
    )
    assert result.confusion.recall == 1.0, "召回率确实 100%"
    assert result.confusion.precision == pytest.approx(0.5), "但精确率只有 50%"


def test_a_always_normal_agent_is_exposed_by_the_positives() -> None:
    result = _result(
        [_outcome(f"p{i}", is_fraud=True, suspicious=False) for i in range(5)]
        + [_outcome(f"n{i}", is_fraud=False, suspicious=False) for i in range(5)]
    )
    assert result.confusion.recall == 0.0
    assert result.confusion.precision == 0.0


# --------------------------------------------------------------------------- #
#  4. 运行健康度
# --------------------------------------------------------------------------- #


def test_health_rates_are_computed_over_all_cases() -> None:
    result = _result(
        [
            _outcome("a", is_fraud=True, suspicious=True),
            _outcome("b", is_fraud=False, suspicious=None),  # exhausted
            _outcome("c", is_fraud=True, suspicious=True, forced=True),
            _outcome("d", is_fraud=False, suspicious=False),
        ]
    )
    assert result.exhausted_rate == pytest.approx(0.25)
    assert result.produced_conclusion_rate == pytest.approx(0.75)
    assert result.forced_rate == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
#  5. ★ 抽样：必须有负例，而且要按活跃度配对
# --------------------------------------------------------------------------- #


def _activity_map(accounts: list[int], base: int = 100) -> dict[int, int]:
    return {account: base + account for account in accounts}


def test_sampling_takes_both_classes_in_equal_numbers() -> None:
    accounts = list(range(1, 41))
    fraud = list(range(1, 11))
    cases = sample_cases(accounts, fraud, _activity_map(accounts), per_class=5, seed=1)
    assert sum(1 for case in cases if case.is_fraud) == 5
    assert sum(1 for case in cases if not case.is_fraud) == 5


def test_sampling_is_reproducible_for_a_fixed_seed() -> None:
    """同一粒种子必须抽出同一批案例——否则数字没法横向比。"""
    accounts = list(range(1, 41))
    fraud = list(range(1, 11))
    activity = _activity_map(accounts)
    first = sample_cases(accounts, fraud, activity, per_class=5, seed=7)
    second = sample_cases(accounts, fraud, activity, per_class=5, seed=7)
    assert [case.account_key for case in first] == [case.account_key for case in second]


def test_activity_matching_brings_the_two_classes_close() -> None:
    """**dry-run 抓到的真问题**：两类活跃度原本相差 3.3 倍。

    不配对的话，Agent 可能靠"这个账户数据多不多"就蒙对，
    **那评测出来的不是我们要测的能力**。
    """
    # 人为构造偏差：正例活跃度普遍高得多，
    # **但负例里也有一小撮高活跃度的**——否则配对无从下手。
    #
    # （第一版我把负例活跃度全设成一样的 50，于是"配不配对"毫无区别，
    #   测试因此失败。**要测配对，样本里就必须存在可配的对象。**）
    activity = dict.fromkeys(range(100, 200), 50)  # 大多数负例：低活跃度
    activity.update(dict.fromkeys(range(200, 210), 500))  # 少数负例：与正例相当
    activity.update(dict.fromkeys(range(1, 11), 500))  # 正例
    accounts = list(range(1, 210))
    fraud = list(range(1, 11))

    matched = sample_cases(accounts, fraud, activity, per_class=5, seed=3, match_activity=True)
    unmatched = sample_cases(accounts, fraud, activity, per_class=5, seed=3, match_activity=False)

    def spread(cases: list[AgentCase]) -> int:
        """两类活跃度中位数的差距——**越小说明越可比**。"""
        positives = sorted(c.activity for c in cases if c.is_fraud)
        negatives = sorted(c.activity for c in cases if not c.is_fraud)
        return abs(positives[len(positives) // 2] - negatives[len(negatives) // 2])

    assert spread(matched) < spread(unmatched), "配对之后两类的活跃度应该更接近"


def test_balance_note_flags_a_badly_matched_sample() -> None:
    cases = [
        AgentCase(account_key="p", is_fraud=True, activity=500),
        AgentCase(account_key="n", is_fraud=False, activity=50),
    ]
    notes = describe_balance(cases)
    assert any("相差" in note for note in notes), "不可比必须说出来"
    assert any("打折" in note for note in notes)


def test_balance_note_confirms_a_good_sample() -> None:
    cases = [
        AgentCase(account_key="p", is_fraud=True, activity=200),
        AgentCase(account_key="n", is_fraud=False, activity=204),
    ]
    notes = describe_balance(cases)
    assert any("可比" in note for note in notes)


def test_balance_note_flags_a_single_class_sample() -> None:
    """**样本里只有一类，评测没有说服力**——必须直说。"""
    notes = describe_balance([AgentCase(account_key="p", is_fraud=True, activity=10)])
    assert any("缺一类" in note for note in notes)


def test_sampling_rejects_a_sample_too_small_to_balance() -> None:
    """两类各要 5 个，任何一类不够都要**响亮地报错**。

    ⚠️ 第二版我写错了：`accounts=[1,2,3]` 且 `fraud=[1,2,3]` 意味着
    **所有账户都是正例**，所以先撞上的是「正例不够」。
    要触发「负例不够」，必须**正例够、负例不够**——
    这里正例 5 个、负例 0 个。
    """
    with pytest.raises(ValueError, match="正例不够"):
        sample_cases(accounts=[1, 2, 3], fraud_accounts=[1], activity={}, per_class=5)
    with pytest.raises(ValueError, match="负例不够"):
        sample_cases(
            accounts=[1, 2, 3, 4, 5], fraud_accounts=[1, 2, 3, 4, 5], activity={}, per_class=3
        )


# --------------------------------------------------------------------------- #
#  6. 报告
# --------------------------------------------------------------------------- #


def test_question_wording_is_fixed() -> None:
    """问题措辞**必须统一**——一变，结果就没法横向比。"""
    first = question_for(AgentCase(account_key="amlsim:1", is_fraud=True))
    second = question_for(AgentCase(account_key="amlsim:2", is_fraud=False))
    assert "amlsim:1" in first and "amlsim:2" in second
    assert first.replace("amlsim:1", "X") == second.replace("amlsim:2", "X")


def test_report_separates_silent_from_the_four_cells() -> None:
    """报告里「未产出结论」必须**单独成节**，不能混进四格。"""
    result = _result(
        [
            _outcome("a", is_fraud=True, suspicious=True),
            _outcome("b", is_fraud=True, suspicious=None),
            _outcome("c", is_fraud=False, suspicious=None),
        ]
    )
    text = result.render()
    assert "未产出结论" in text
    assert "既不算漏报，也不算检出" in text
    assert "工程缺陷指标" in text


def test_report_distinguishes_alert_false_rate_from_fpr() -> None:
    """**`1 − 精确率` 不是 FPR。** 两个都要报，而且要标清楚。"""
    result = _result(
        [
            _outcome("a", is_fraud=True, suspicious=True),
            _outcome("b", is_fraud=False, suspicious=True),
            _outcome("c", is_fraud=False, suspicious=False),
        ]
    )
    text = result.render()
    assert "误报率" in text
    assert "FPR 是另一个数" in text


def test_summary_is_json_serialisable() -> None:
    import json

    result = _result([_outcome("a", is_fraud=True, suspicious=True)])
    json.dumps(result.summary(), ensure_ascii=False)
    json.dumps(result.outcomes[0].as_dict(), ensure_ascii=False)
