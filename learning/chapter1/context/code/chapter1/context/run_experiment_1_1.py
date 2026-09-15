#!/usr/bin/env python3
"""运行来自 book/chapter1.md 的完整五分支上下文消融实验。

与传统的演示表格不同，该运行器会持久化保存每次免除凭据的 API 请求与响应。
这使得能够证明每次推理中具体移除了哪个上下文组件，而非在事后通过命令行参数去推测消融配置。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from agent import HIDDEN_RESULT_STYLES, ContextAwareAgent, ContextMode
from config import PROVIDERS, SUPPORTED_PROVIDERS, canonical_provider
from grounding import assess_groundedness, observation_quantities


EXPERIMENT_ID = "1-1"
MODES = list(ContextMode)
CANONICAL_TASK = """According to the company's quarterly revenue:
- Q1: 2.5 million USD
- Q2: 2.1 million EUR
- Q3: 1.8 million GBP
- Q4: 380 million JPY

Use the available currency-conversion and calculation tools to convert every
non-USD quarter to USD, then calculate the annual total and quarterly average.
Report both values rounded to two decimal places. Do not estimate exchange
rates yourself; use the tool observations."""

EXPECTED_NUMBERS = ("9602895.73", "2400723.93")

# 上方禁止自行估算汇率的语句是一个防护提示（guard），它的存在与否会改变无工具定义组的行为：
# 有该提示时，无法换算的模型会直接说明；没有该提示时，部分模型会转而凭记忆给出汇率。
# 两种情况都值得运行，因此该防护提示设计为一个标志位，而非固定内嵌在任务中。
# 只有带防护提示的任务才是标准基准，因为书稿中描述的正是这一任务。
ESTIMATION_GUARD = """Do not estimate exchange
rates yourself; use the tool observations."""
UNGUARDED_TASK = CANONICAL_TASK.replace(ESTIMATION_GUARD, "").rstrip()
TASK_VARIANTS = {"guarded": CANONICAL_TASK, "unguarded": UNGUARDED_TASK}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_version(distribution: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(distribution)
    except Exception:
        return None


def provider_spec(provider: str):
    """在共享注册表中查找提供商。

    Args:
        provider: 被 :mod:`agentbook.providers` 接受的提供商名称或别名。

    Returns:
        已注册的提供商规范对象。

    Raises:
        RuntimeError: 若该名称未解析到任何注册表条目。
    """
    try:
        return PROVIDERS[canonical_provider(provider)]
    except KeyError:
        raise RuntimeError(f"Unknown provider: {provider!r}") from None


def resolve_key(provider: str) -> tuple[str, str]:
    """查找该提供商自身的凭证，拒绝自动兜底。

    注册表的 :func:`resolve_backend` 在提供商密钥缺失时会直接通过 OpenRouter 重新路由。
    这对于运行 demo 的读者是合适的，但在此处是不恰当的：证据文件声明的是直连 API 的真实溯源，
    因此密钥缺失必须中止运行，而不是悄悄改变实际响应的端点。

    Args:
        provider: 提供商名称或别名。

    Returns:
        密钥及对应的环境变量名称元组。

    Raises:
        RuntimeError: 若未设置该提供商的任何密钥环境变量。
    """
    spec = provider_spec(provider)
    for name in spec.key_vars:
        value = os.getenv(name)
        if value:
            return value, name
    raise RuntimeError(
        f"No direct credential for {provider}; expected one of "
        f"{', '.join(spec.key_vars) or '(none)'}"
    )


def tool_call_dict(call: Any) -> Dict[str, Any]:
    return {
        "tool_name": call.tool_name,
        "arguments": call.arguments,
        "result": call.result,
        "timestamp": call.timestamp,
    }


def call_signatures(tool_calls: Iterable[Dict[str, Any]]) -> List[str]:
    signatures = []
    for call in tool_calls:
        signatures.append(
            f"{call['tool_name']}:"
            + json.dumps(call.get("arguments", {}), sort_keys=True, ensure_ascii=False)
        )
    return signatures


def response_message(turn: Dict[str, Any]) -> Dict[str, Any]:
    choices = turn.get("response", {}).get("choices") or []
    return (choices[0].get("message") or {}) if choices else {}


def request_roles(turn: Dict[str, Any]) -> List[str]:
    return [message.get("role") for message in turn.get("request", {}).get("messages", [])]


def evaluate_context_contract(
    mode: str,
    turns: List[Dict[str, Any]],
    hidden_result_content: str = HIDDEN_RESULT_STYLES["empty"],
) -> Dict[str, Any]:
    """验证实际发送给提供商的请求，而非请求的 CLI 模式。

    Args:
        mode: 正在检查的消融实验分支。
        turns: 该分支记录的 API 交互轮次。
        hidden_result_content: 无工具结果分支中用于替代观测结果的内容。
            进行精确检查，因此真实结果的泄漏仍会导致契约校验失败。

    Returns:
        包含逐项检查详情及 ``passed`` 结论的字典。

    Raises:
        ValueError: 若 ``mode`` 为未知的分支名称。
    """
    requests = [turn.get("request", {}) for turn in turns if turn.get("request")]
    real_responses = [turn for turn in turns if turn.get("response", {}).get("id")]
    details: Dict[str, Any] = {
        "has_provider_response_ids": len(real_responses) == len(turns) and bool(turns),
        "turn_count": len(turns),
        "request_roles": [request_roles(turn) for turn in turns],
    }

    if mode == ContextMode.FULL.value:
        details.update(
            {
                "tools_present_every_turn": all(bool(r.get("tools")) for r in requests),
                "history_present_after_first_turn": len(requests) > 1
                and all(
                    "assistant" in [m.get("role") for m in r.get("messages", [])]
                    and "tool" in [m.get("role") for m in r.get("messages", [])]
                    for r in requests[1:]
                ),
                "reasoning_retained_after_first_turn": len(requests) > 1
                and any(
                    bool(m.get("reasoning_content"))
                    for m in requests[1].get("messages", [])
                    if m.get("role") == "assistant"
                ),
            }
        )
        required = (
            "has_provider_response_ids",
            "tools_present_every_turn",
            "history_present_after_first_turn",
            "reasoning_retained_after_first_turn",
        )
    elif mode == ContextMode.NO_TOOL_CALLS.value:
        details.update(
            {
                "tools_absent_every_turn": all(
                    "tools" not in r and "tool_choice" not in r for r in requests
                ),
            }
        )
        required = ("has_provider_response_ids", "tools_absent_every_turn")
    elif mode == ContextMode.NO_TOOL_RESULTS.value:
        tool_messages = [
            m
            for r in requests[1:]
            for m in r.get("messages", [])
            if m.get("role") == "tool"
        ]
        details.update(
            {
                "tool_calls_retained": any(
                    m.get("role") == "assistant" and m.get("tool_calls")
                    for r in requests[1:]
                    for m in r.get("messages", [])
                ),
                "tool_results_hidden": bool(tool_messages)
                and all(
                    m.get("content") == hidden_result_content for m in tool_messages
                ),
            }
        )
        required = (
            "has_provider_response_ids",
            "tool_calls_retained",
            "tool_results_hidden",
        )
    elif mode == ContextMode.NO_REASONING.value:
        assistant_history = [
            m
            for r in requests[1:]
            for m in r.get("messages", [])
            if m.get("role") == "assistant"
        ]
        provider_reasoning = [
            response_message(turn).get("reasoning_content") for turn in turns
        ]
        details.update(
            {
                "provider_generated_reasoning": any(provider_reasoning),
                "reasoning_removed_from_history": bool(assistant_history)
                and all(not m.get("reasoning_content") for m in assistant_history),
                "tool_and_result_history_retained": any(
                    "tool" in request_roles(turn) for turn in turns[1:]
                ),
            }
        )
        required = (
            "has_provider_response_ids",
            "provider_generated_reasoning",
            "reasoning_removed_from_history",
            "tool_and_result_history_retained",
        )
    elif mode == ContextMode.NO_HISTORY.value:
        details.update(
            {
                "only_static_prefix_and_user_every_turn": bool(requests)
                and all(
                    [m.get("role") for m in r.get("messages", [])]
                    == ["system", "user"]
                    for r in requests
                ),
                "tools_still_present": all(bool(r.get("tools")) for r in requests),
            }
        )
        required = (
            "has_provider_response_ids",
            "only_static_prefix_and_user_every_turn",
            "tools_still_present",
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    details["required_checks"] = list(required)
    details["passed"] = all(details[name] is True for name in required)
    return details


def normalized_number_text(value: str | None) -> str:
    return (value or "").replace(",", "").replace("$", "").replace(" ", "")


def canonical_answer_correct(final_answer: str | None) -> bool:
    """评估实验 1-1 基准任务的已知数值评分标准。

    这特意保留在 ``ContextAwareAgent`` 之外。通用的 Agent 无法从任意自然语言任务中推断正确性，
    而本实验具有明确的答案细则。
    """
    normalized = normalized_number_text(final_answer)
    return bool(final_answer) and all(number in normalized for number in EXPECTED_NUMBERS)


def arm_outcome(completed: bool, correct: bool, verdict: str) -> str:
    """将实验分支的表现归纳为消融表格中应展示的一个词。

    ``Completed`` 无法区分消融分支在没有得到正确答案的情况下结束的两种方式，且两者的危害程度截然不同：
    一个从未声称任何未获给出的数字的模型属于安全失败；而凭记忆提供汇率的模型则产生了一个读起来与正确数字一模一样的错误答案。

    这些标签仅描述测量到的现象：
    ``no_unsupported_numbers`` 涵盖了原则性拒绝作答以及仅宣布打算做什么便停下的轮次——区分这两者属于对意图的判断，本测试框架无从推断。

    Args:
        completed: 模型是否返回了终止响应。
        correct: 该响应是否满足任务的答案评分细则。
        verdict: 来自 :func:`grounding.assess_groundedness` 的数据依据性判定。

    Returns:
        ``no_terminal_response``、``correct``、``unsupported_numbers``、
        ``no_unsupported_numbers`` 或 ``incorrect`` 之一。
    """
    if not completed:
        return "no_terminal_response"
    if correct:
        return "correct"
    if verdict == "ungrounded":
        return "unsupported_numbers"
    if verdict in ("grounded", "no_quantities"):
        return "no_unsupported_numbers"
    return "incorrect"


def sent_messages(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """扁平化实际发送至网络的所有消息列表。

    Args:
        turns: 该分支记录的 API 交互轮次。

    Returns:
        按交互顺序拼接的请求消息列表。
    """
    return [
        message
        for turn in turns
        for message in (turn.get("request") or {}).get("messages", [])
    ]


def summarize_arm(
    mode: ContextMode,
    result: Dict[str, Any],
    elapsed: float,
    task_text: str = CANONICAL_TASK,
    hidden_result_content: str = HIDDEN_RESULT_STYLES["empty"],
) -> Dict[str, Any]:
    trajectory = result["trajectory"]
    tool_calls = [tool_call_dict(call) for call in trajectory.tool_calls]
    signatures = call_signatures(tool_calls)
    repeats = len(signatures) - len(set(signatures))
    final_answer = result.get("final_answer")
    completed = bool(result.get("completed", result.get("success", False)))
    task_success = canonical_answer_correct(final_answer)
    # 数据依据性是从模型接收到的消息中读取的，而非从测试框架计算的工具结果中读取。
    # 在无工具结果组中，两者在设计上是不同的，只有前者才是模型进行推理的依据。
    observations = observation_quantities(sent_messages(trajectory.api_turns))
    groundedness = assess_groundedness(final_answer, task_text, observations)
    arm = {
        "mode": mode.value,
        "provider": result.get("provider"),
        "model": result.get("model"),
        "base_url": result.get("base_url"),
        "using_openrouter": result.get("using_openrouter", False),
        "started_at": None,
        "elapsed_seconds": round(elapsed, 6),
        # 保留 ``success`` 以兼容现有证据；它代表终止响应/完成状态，而非任务正确性。
        "success": completed,
        "completed": completed,
        "task_success": task_success,
        "iterations": result.get("iterations", 0),
        "error": result.get("error"),
        "final_answer": final_answer,
        "tool_calls": tool_calls,
        "tool_call_signatures": signatures,
        "repeated_tool_calls": repeats,
        "reasoning_steps": trajectory.reasoning_steps,
        "api_turns": trajectory.api_turns,
    }
    arm["context_contract"] = evaluate_context_contract(
        mode.value, trajectory.api_turns, hidden_result_content
    )
    arm["groundedness"] = groundedness
    arm["outcome"] = arm_outcome(completed, task_success, groundedness["verdict"])
    arm["behavior"] = {
        "tool_action_count": len(tool_calls),
        "has_repeated_tool_action": repeats > 0,
        "hit_iteration_ceiling": result.get("iterations") >= 5 and not completed,
        "canonical_answer_correct": task_success,
        "stated_unsupported_numbers": groundedness["verdict"] == "ungrounded",
    }
    return arm


def token_usage(arms: List[Dict[str, Any]]) -> Dict[str, int]:
    prompt = completion = cached = reasoning = 0
    for arm in arms:
        for turn in arm["api_turns"]:
            usage = turn.get("response", {}).get("usage") or {}
            prompt += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            completion += int(
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            )
            prompt_details = usage.get("prompt_tokens_details") or usage.get(
                "input_tokens_details"
            ) or {}
            completion_details = usage.get("completion_tokens_details") or usage.get(
                "output_tokens_details"
            ) or {}
            cached += int(prompt_details.get("cached_tokens") or 0)
            reasoning += int(completion_details.get("reasoning_tokens") or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "cached_prompt_tokens": cached,
        "reasoning_tokens": reasoning,
    }


def arm_produced_inference(arm: Dict[str, Any] | None) -> bool:
    """报告该实验分支是否确实从提供商处获得了回答。

    书稿中的四个行为结论有两个表述为“未发生”——无工具行为、无正确答案——而从未到达提供商的分支同时满足这两点。
    若无此项检查，一次所有请求均返回 402 的运行会将四个结论中的三个报告为“已观察到”，这是证据文件绝不能出现的错误。

    Args:
        arm: 汇总后的实验分支数据，若未运行该模式则为 ``None``。

    Returns:
        若记录的每个轮次均携带提供商响应 ID 且该分支未记录传输错误，则返回 ``True``。
    """
    if not arm or arm.get("error") or not arm.get("api_turns"):
        return False
    return all(turn.get("response", {}).get("id") for turn in arm["api_turns"])


CLAIM_QUALIFICATIONS = {
    "without_tool_definitions_no_tool_action": (
        "结构性必然：请求中不包含工具定义，因此提供商无法发出工具调用。"
        "不同模型之间的区别在于它们转而采取的行动——参见 arm_outcomes.no_tool_calls，"
        "它区分了不声称未给予的数据与基于自身提供的汇率构建答案这两种情况。"
    ),
    "without_reasoning_degraded": (
        "从基准正确性的丧失中推断得出，并请注意该分支移除了什么：历史中保留的思考过程被剥离，"
        "而模型在每一轮依然会重新进行思考。因此它测试的是向前传递先前的思考过程是否重要，"
        "当每一步已经由先前的观测确定时，这并不一定重要。更强的断言——即消融会导致相互矛盾的决策——"
        "不是测试框架所能强制促成的，且未曾被观察到。"
    ),
}


def claim(value: bool, evaluable: bool) -> bool | None:
    """返回观测结果，若无可观测对象则返回 ``None``。

    Args:
        value: 从实验分支计算出的断言值。
        evaluable: 该分支是否产生了真实推理。

    Returns:
        当分支可评估时返回 ``value``，否则返回 ``None``。
    """
    return value if evaluable else None


def analyze(arms: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_mode = {arm["mode"]: arm for arm in arms}
    exact_five_arms = set(by_mode) == {mode.value for mode in MODES}
    contracts_pass = exact_five_arms and all(
        arm["context_contract"]["passed"] for arm in arms
    )
    direct_real_api = bool(arms) and all(
        not arm["using_openrouter"] and arm_produced_inference(arm) for arm in arms
    )
    live = {mode: arm_produced_inference(by_mode.get(mode)) for mode in by_mode}

    def behavior_of(mode: str, key: str, default: Any = False) -> Any:
        return by_mode.get(mode, {}).get("behavior", {}).get(key, default)

    behavior = {
        "full_baseline_correct": claim(
            bool(behavior_of("full", "canonical_answer_correct")), live.get("full", False)
        ),
        "without_tool_definitions_no_tool_action": claim(
            behavior_of("no_tool_calls", "tool_action_count", None) == 0,
            live.get("no_tool_calls", False),
        ),
        "without_tool_results_repeated_action": claim(
            bool(behavior_of("no_tool_results", "has_repeated_tool_action")),
            live.get("no_tool_results", False),
        ),
        "without_history_repeated_action": claim(
            bool(behavior_of("no_history", "has_repeated_tool_action")),
            live.get("no_history", False),
        ),
        # 产生矛盾是一个经验性结果，而非测试框架可以合理强制要求的现象。
        # 我们报告无推理答案是否丧失了基准正确性，并将其与执行有效性分开。
        "without_reasoning_degraded": claim(
            not behavior_of("no_reasoning", "canonical_answer_correct"),
            live.get("no_reasoning", False),
        ),
    }
    behavior["all_manuscript_behavior_claims_observed"] = all(
        value is True for value in behavior.values()
    )
    return {
        "exact_five_arms_present": exact_five_arms,
        "all_context_contracts_passed": contracts_pass,
        "direct_real_api_evidence": direct_real_api,
        "experiment_execution_accepted": bool(
            exact_five_arms
            and contracts_pass
            and direct_real_api
            and behavior["full_baseline_correct"]
        ),
        "manuscript_behavior_claims": behavior,
        "claim_qualifications": CLAIM_QUALIFICATIONS,
        # 每个分支的实际行为（正是 ``Completed`` 所隐藏的部分）：
        # 拒绝作答与凭记忆汇率拼凑出的答案都属于终止响应。
        "arm_outcomes": {arm["mode"]: arm["outcome"] for arm in arms},
        "arms_stating_unsupported_numbers": [
            arm["mode"] for arm in arms if arm["groundedness"]["verdict"] == "ungrounded"
        ],
        "usage": token_usage(arms),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_terminal_summary(
    evidence: Dict[str, Any],
    evidence_path: Path,
    digest: str,
    *,
    promoted: bool,
) -> None:
    """用带中文说明的区块输出实验结果，方便学习时逐项观察。"""

    analysis = evidence["analysis"]
    print("\n【实验结果汇总】以下 JSON 汇总各分支的结果；优先查看 arm_outcomes 与 experiment_execution_accepted。")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    print(f"\n【证据文件】逐轮请求、响应与上下文契约已保存到：{evidence_path}")
    print(f"【完整性校验】evidence.json 的 SHA-256：{digest}")
    if promoted:
        print("【台账更新状态】本次为标准且已验收的运行，已更新 validation/latest.json。")
    else:
        print(
            "【台账更新状态】本次未更新 validation/latest.json "
            f"（canonical_run={evidence['canonical_run']}，"
            f"accepted={analysis['experiment_execution_accepted']}）。"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        type=canonical_provider,
        default="moonshot",
        choices=SUPPORTED_PROVIDERS,
        help="待运行所有实验分支的提供商（默认：moonshot；kimi 是其别名）。",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="模型 ID。默认为所选提供商在注册表中的默认模型，因此单独指定 --provider 即可；"
             "指定其他提供商的模型会导致所有分支报 400 失败。",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[mode.value for mode in MODES],
        help="待运行的实验分支（默认：全部五种）。子集属于探测性运行，非基准运行，"
             "绝不会被推送到 validation/latest.json。",
    )
    parser.add_argument(
        "--task",
        default="guarded",
        choices=sorted(TASK_VARIANTS),
        help="guarded（默认，基准）禁止自行估算汇率；unguarded 去掉该限制语句，"
             "以观察在未限制猜测时模型的行为。",
    )
    parser.add_argument(
        "--hidden-result",
        default="empty",
        choices=sorted(HIDDEN_RESULT_STYLES),
        help="无工具结果分支扣留观测数据的方式。empty（默认，基准）静默扣留，"
             "即真正移除工具结果；marker 保留可见遮盖标记，会引入本应被消融掉的信号，"
             "让模型察觉并停止。",
    )
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.max_iterations < 2:
        parser.error("--max-iterations must be at least 2")

    # 未设置 --model 表示“使用该提供商的默认模型”，从共享注册表解析，
    # 而非从恰好命名了某个提供商模型的常量中解析。
    model = args.model or provider_spec(args.provider).default_model
    modes = [ContextMode(name) for name in args.modes] if args.modes else list(MODES)
    task = TASK_VARIANTS[args.task]
    hidden_result_content = HIDDEN_RESULT_STYLES[args.hidden_result]
    canonical_run = (
        args.task == "guarded"
        and args.hidden_result == "empty"
        and set(modes) == set(MODES)
    )

    key, key_env = resolve_key(args.provider)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path("validation") / f"real_{stamp}"
    command = [
        sys.executable,
        Path(__file__).name,
        "--provider",
        args.provider,
        "--model",
        model,
        "--task",
        args.task,
        "--hidden-result",
        args.hidden_result,
        "--modes",
        *[mode.value for mode in modes],
        "--max-iterations",
        str(args.max_iterations),
        "--output-dir",
        str(output_dir),
    ]

    arms = []
    for mode in modes:
        started = utc_now()
        agent = ContextAwareAgent(
            key,
            context_mode=mode,
            provider=args.provider,
            model=model,
            verbose=False,
            hidden_result_content=hidden_result_content,
        )
        begin = time.monotonic()
        result = agent.execute_task(task, max_iterations=args.max_iterations)
        arm = summarize_arm(
            mode,
            result,
            time.monotonic() - begin,
            task_text=task,
            hidden_result_content=hidden_result_content,
        )
        arm["started_at"] = started
        # 重新计算配置的上限，而非保留纯摘要生成器中的默认值（单元测试也会用到该摘要器）。
        arm["behavior"]["hit_iteration_ceiling"] = (
            result.get("iterations") >= args.max_iterations and not result.get("success")
        )
        arms.append(arm)

    evidence: Dict[str, Any] = {
        "schema_version": "1.1",
        "experiment_id": EXPERIMENT_ID,
        "evidence_mode": "real_api",
        "created_at": utc_now(),
        "canonical_source": "book/chapter1.md#实验-1-1-上下文的关键作用",
        "task": task,
        "task_variant": args.task,
        "hidden_result_style": args.hidden_result,
        "canonical_run": canonical_run,
        "expected_numbers": list(EXPECTED_NUMBERS),
        "command": command,
        "credential_source_env": key_env,
        "credential_value_recorded": False,
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "machine": platform.machine(),
        },
        "dependencies": {
            "openai": package_version("openai"),
            "requests": package_version("requests"),
        },
        "repository": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "worktree_dirty": bool(git_value("status", "--porcelain")),
        },
        "arms": arms,
    }
    evidence["analysis"] = analyze(arms)
    evidence_path = output_dir / "evidence.json"
    write_json(evidence_path, evidence)
    digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    (output_dir / "evidence.sha256").write_text(
        f"{digest}  evidence.json\n", encoding="utf-8"
    )
    # validation/latest.json 是台账所引用的文件，因此只有既属于基准运行且被验收的运行才能替换它。
    # 探索性测试——部分分支运行、未加防护的任务或请求未成功送达的运行——保留各自带时间戳的目录，不触动所引用的证据文件。
    promoted = canonical_run and evidence["analysis"]["experiment_execution_accepted"]
    if promoted:
        latest = Path("validation/latest.json")
        latest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(evidence_path, latest)

    print_terminal_summary(
        evidence,
        evidence_path,
        digest,
        promoted=promoted,
    )
    return 0 if evidence["analysis"]["experiment_execution_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
