"""
上下文感知 Agent 的主入口模块
"""

import os
import sys
import argparse
import logging
from agent import ContextAwareAgent, ContextMode
from config import PROVIDERS, SUPPORTED_PROVIDERS, canonical_provider, resolve_backend
from grounding import assess_groundedness, observation_quantities
import json
from pathlib import Path
import subprocess
import time
from typing import Dict, Any, List
from tabulate import tabulate
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _completed(result: Dict[str, Any]) -> bool:
    """返回终止响应状态，兼容旧版结果。"""
    return bool(result.get("completed", result.get("success", False)))


def print_task_result(result: Dict[str, Any], *, context_mode: str) -> None:
    """以带中文说明的区块打印单任务或交互任务的结果。"""

    trajectory = result.get("trajectory")
    tool_call_count = len(getattr(trajectory, "tool_calls", []))
    print("\n" + "=" * 60)
    print("【任务执行结果】下面依次说明本次任务的终止状态、轮次、工具调用和回答。")
    print("=" * 60)
    print(f"【上下文模式】{context_mode}")
    print(
        f"【终止响应】{_completed(result)}"
        "（表示模型是否返回文本，不等同于任务是否正确）。"
    )
    print(f"【执行轮次】{result.get('iterations', 0)}")
    print(f"【工具调用】{tool_call_count} 次")

    if result.get("final_answer"):
        print("\n【最终回答】以下为模型返回的文本：")
        print("-" * 40)
        print(result["final_answer"])

    if result.get("error"):
        print(f"\n【任务错误】{result['error']}")


def _outcome(result: Dict[str, Any]) -> str:
    """说明实验分支的具体表现，而不仅仅是是否终止。

    对于消融分支在没有得到答案的情况下结束的两种可能方式，``Completed`` 均为 true，
    但它们并非同一事件。要求模型在没有换算工具的情况下换算货币，模型可能会说自己无法完成，
    也可能凭记忆提供汇率并呈现算术过程（就好像它真的查询过汇率一样）。
    第二种情况在以往打印的每一列中都会被判定为完全成功。

    Args:
        result: 单个测试结果。

    Returns:
        ``no_terminal_response``、``unsupported_numbers`` 或 ``completed``。
    """
    if not _completed(result):
        return "no_terminal_response"
    if result.get("grounding_verdict") == "ungrounded":
        return "unsupported_numbers"
    return "completed"


# 加载 .env 文件，以便通过 os.getenv 获取配置的 API 密钥。
# （config.py 也会调用 load_dotenv()，但 main.py 仅导入 agent，
#  而 agent 未导入 config，因此在此处显式触发加载。）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# 数据依据性（groundedness）判定在表格单元格中的展示标签。
_GROUNDING_LABEL = {
    "grounded": "from task",
    "ungrounded": "UNSUPPORTED",
    "not_assessable": "saw tools",
    "no_quantities": "none",
    "no_answer": "-",
}

_OUTCOME_MARK = {
    "completed": "✓",
    "unsupported_numbers": "⚠",
    "no_terminal_response": "✗",
}


class AblationTestSuite:
    """用于通过消融实验探索上下文重要性的测试套件"""
    
    def __init__(self, api_key: str, provider: str = "siliconflow", model: str = None):
        """
        初始化测试套件
        
        Args:
            api_key: LLM 提供商的 API 密钥
            provider: 要使用的 LLM 提供商
            model: 可选的模型重载
        """
        self.api_key = api_key
        self.provider = provider
        self.model = model
        self.test_results = []
        
    def create_complex_financial_task(self) -> str:
        """
        创建需要多次工具调用和推理的复杂任务
        
        Returns:
            任务描述
        """
        return """Analyze the financial report from this PDF: https://www.berkshirehathaway.com/qtrly/1stqtr23.pdf

Please complete the following analysis:
1. Extract the total revenue figures from Q1 2023
2. Convert the revenue from USD to EUR, GBP, and JPY
3. Calculate the following metrics:
   - Average revenue across the three converted currencies
   - Percentage difference between highest and lowest converted values
   - If the company maintains a 15% profit margin, what would be the profit in each currency?

Provide a comprehensive financial summary with all calculations shown."""
    
    def create_multinational_budget_task(self) -> str:
        """
        创建需要多次货币换算和计算的任务
        
        Returns:
            任务描述  
        """
        return """A multinational company has the following Q1 2024 expenses documented in this report:
https://raw.githubusercontent.com/adobe/pdfservices-node-sdk-samples/master/resources/extractPDFInput.pdf

Tasks to complete:
1. Parse the PDF and extract all monetary values mentioned
2. The company operates in 5 regions with expenses in different currencies:
   - US Office: $2,500,000 USD
   - UK Office: £1,800,000 GBP  
   - Japan Office: ¥380,000,000 JPY
   - EU Office: €2,100,000 EUR
   - Singapore Office: $3,200,000 SGD
3. Convert all expenses to USD for consolidation
4. Calculate:
   - Total global expenses in USD
   - Average expense per region
   - What percentage each region represents of total expenses
   - If we apply a 8% cost reduction uniformly, what would be the new expense for each region in their local currency?

Present a detailed financial analysis with all conversions and calculations."""
    
    def run_single_test(self, task: str, context_mode: ContextMode, test_name: str,
                        case_name: str = "default") -> Dict[str, Any]:
        """
        运行单个消融测试

        Args:
            task: 待执行的任务
            context_mode: 待测试的上下文模式
            test_name: 测试名称
            case_name: 该测试所属的用例/任务名称

        Returns:
            测试结果
        """
        logger.info(
            f"【单组消融开始】测试={test_name}；用例={case_name}；"
            f"上下文模式={context_mode.value}。"
        )

        agent = ContextAwareAgent(self.api_key, context_mode, provider=self.provider, model=self.model)

        start_time = time.time()
        result = agent.execute_task(task)
        execution_time = time.time() - start_time
        completed = _completed(result)

        # 从实际发送的消息中读取观测结果。在无工具结果组中，测试框架执行了工具但向模型展示的是占位符，因此模型看到的数字为空。
        sent = [
            message
            for turn in result["trajectory"].api_turns
            for message in (turn.get("request") or {}).get("messages", [])
        ]
        grounding = assess_groundedness(
            result.get("final_answer"), task, observation_quantities(sent)
        )

        # 分析测试结果
        test_result = {
            "test_name": test_name,
            "case_name": case_name,
            "context_mode": context_mode.value,
            "execution_time": round(execution_time, 2),
            "iterations": result.get("iterations", 0),
            "num_tool_calls": len(result["trajectory"].tool_calls),
            # 该传统测试套件没有特定任务的正确性评分细则。保留 ``success`` 作为兼容性别名，但单独报告完成状态，以免将拒绝作答呈现为任务成功。
            "completed": completed,
            "success": completed,
            "task_success": None,
            "has_final_answer": result.get("final_answer") is not None,
            # 此处没有任务细则，因此正确性保持未知——但答案中的数字是否有数据来源则无需细则即可判定。
            "grounding_verdict": grounding["verdict"],
            "observation_count": grounding["observation_count"],
            "unsupported_quantities": grounding["unsupported_quantities"],
            "error": result.get("error"),
            "reasoning_steps": len(result["trajectory"].reasoning_steps),
            "final_answer_preview": (result.get("final_answer", "")[:200] + "...") if result.get("final_answer") else None
        }
        
        # 记录摘要
        logger.info(f"【单组消融耗时】{test_result['execution_time']}s")
        logger.info(
            f"【终止响应】{test_result['completed']}（不等同于任务数值正确）。"
        )
        if test_result["grounding_verdict"] == "ungrounded":
            logger.warning(
                "【数字依据警告】回答含 %d 个没有工具观测依据的数字：%s",
                len(test_result["unsupported_quantities"]),
                test_result["unsupported_quantities"],
            )
        logger.info(f"【工具调用数】{test_result['num_tool_calls']}")
        logger.info(f"【执行轮次】{test_result['iterations']}")
        
        if test_result["error"]:
            logger.error(f"【单组消融错误】{test_result['error']}")
        
        return test_result
    
    def run_ablation_study(self, context_modes: List[ContextMode] = None,
                          cases: List[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        在指定的上下文模式以及一个或多个用例上运行消融实验。

        Args:
            context_modes: 待测试的上下文模式列表（默认为所有模式）
            cases: 包含 {"name", "task"} 字典的列表，用于针对每个模式运行。
                   默认仅运行单一跨国预算案例，保持原有的单任务行为。

        Returns:
            扁平的测试结果列表（每个用例 x 模式对应一项）。
        """
        # 默认：单个跨国预算用例（保持原有行为）。
        if cases is None:
            cases = [{
                "name": "Multinational Budget",
                "task": self.create_multinational_budget_task()
            }]

        # 将上下文模式映射到测试名称
        mode_names = {
            ContextMode.FULL: "Baseline - Full Context",
            ContextMode.NO_HISTORY: "Ablation 1 - No Historical Tool Calls",
            ContextMode.NO_REASONING: "Ablation 2 - No Reasoning Process",
            ContextMode.NO_TOOL_CALLS: "Ablation 3 - No Tool Call Commands",
            ContextMode.NO_TOOL_RESULTS: "Ablation 4 - No Tool Call Results"
        }

        if context_modes is None:
            context_modes = list(mode_names.keys())

        results = []
        for case in cases:
            case_name = case["name"]
            task = case["task"]
            for context_mode in context_modes:
                test_name = mode_names[context_mode]
                try:
                    result = self.run_single_test(task, context_mode, test_name, case_name=case_name)
                    results.append(result)
                    self.test_results.append(result)

                    # 在测试之间添加延时以避免触发速率限制
                    time.sleep(2)

                except Exception as e:
                    logger.error(
                        f"【单组消融失败】测试={test_name}；用例={case_name}；错误={str(e)}"
                    )
                    results.append({
                        "test_name": test_name,
                        "case_name": case_name,
                        "context_mode": context_mode.value,
                        "error": str(e),
                        "completed": False,
                        "success": False,
                        "task_success": False,
                    })

        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析消融实验结果
        
        Args:
            results: 测试结果列表
            
        Returns:
            分析摘要
        """
        analysis = {
            "total_tests": len(results),
            "completed_tests": sum(1 for r in results if _completed(r)),
            # 现有报告读取者的兼容键。现仅统计终止响应数量，而非经过验证的任务成功数。
            "successful_tests": sum(1 for r in results if _completed(r)),
            # 其数字无任何观测依据的终止响应。没有细则本测试套件无法判定答案错误，但可以明确指出模型没有任何计算依据。
            "unsupported_number_tests": sum(
                1 for r in results if r.get("grounding_verdict") == "ungrounded"
            ),
            "unsupported_number_modes": sorted(
                {
                    r["context_mode"]
                    for r in results
                    if r.get("grounding_verdict") == "ungrounded"
                }
            ),
            "context_mode_impact": {}
        }
        
        # 分析各消融模式的影响
        baseline = next((r for r in results if r["context_mode"] == "full"), None)
        
        if baseline:
            for result in results:
                if result["context_mode"] != "full":
                    mode_analysis = {
                        "completion_maintained": _completed(result),
                        "success_maintained": _completed(result),
                        "stated_unsupported_numbers": result.get("grounding_verdict")
                        == "ungrounded",
                        "execution_time_delta": result.get("execution_time", 0) - baseline.get("execution_time", 0),
                        "iteration_delta": result.get("iterations", 0) - baseline.get("iterations", 0),
                        "tool_call_delta": result.get("num_tool_calls", 0) - baseline.get("num_tool_calls", 0),
                        "failure_reason": None
                    }
                    
                    # 识别失败原因
                    if result.get("grounding_verdict") == "ungrounded":
                        mode_analysis["failure_reason"] = (
                            "Answered with figures no tool observation supports"
                        )
                    elif not _completed(result):
                        if result["context_mode"] == "no_tool_calls":
                            mode_analysis["failure_reason"] = "Cannot execute tools without tool call capability"
                        elif result["context_mode"] == "no_tool_results":
                            mode_analysis["failure_reason"] = "Cannot make informed decisions without tool results"
                        elif result["context_mode"] == "no_history":
                            mode_analysis["failure_reason"] = "May repeat actions or lose track of progress"
                        elif result["context_mode"] == "no_reasoning":
                            mode_analysis["failure_reason"] = "Lacks planning and strategic thinking"
                    
                    analysis["context_mode_impact"][result["context_mode"]] = mode_analysis
        
        return analysis
    
    def print_results_table(self, results: List[Dict[str, Any]]):
        """
        以格式化表格打印结果
        
        Args:
            results: 测试结果列表
        """
        # 准备制表数据
        table_data = []
        for result in results:
            table_data.append([
                result.get("case_name", "default"),
                result["context_mode"],
                "✓" if _completed(result) else "✗",
                _GROUNDING_LABEL.get(result.get("grounding_verdict"), "-"),
                f"{result.get('execution_time', 0)}s",
                result.get("iterations", 0),
                result.get("num_tool_calls", 0),
                result.get("reasoning_steps", 0),
                "Yes" if result.get("has_final_answer", False) else "No"
            ])

        headers = ["用例", "上下文模式", "终止响应", "数字依据", "耗时", "轮次", "工具调用", "推理步骤", "最终回答"]

        print("\n" + "="*80)
        print("【消融结果表】逐行展示每个用例和上下文模式的运行结果。")
        print("="*80)
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    def print_comparison_matrix(self, results: List[Dict[str, Any]]):
        """
        打印模式 x 用例对照矩阵，以便在所有用例上一目了然地查看各上下文组件的影响。

        Args:
            results: 测试结果列表
        """
        cases = []
        for r in results:
            c = r.get("case_name", "default")
            if c not in cases:
                cases.append(c)

        modes = []
        for r in results:
            m = r["context_mode"]
            if m not in modes:
                modes.append(m)

        # 按 (mode, case) 建立结果索引以供快速查找。
        by_key = {(r["context_mode"], r.get("case_name", "default")): r for r in results}

        table_data = []
        for mode in modes:
            row = [mode]
            for case in cases:
                r = by_key.get((mode, case))
                if r is None:
                    row.append("-")
                else:
                    counts = f"{r.get('iterations', 0)}it/{r.get('num_tool_calls', 0)}tc"
                    row.append(f"{_OUTCOME_MARK[_outcome(r)]} {counts}")
            table_data.append(row)

        headers = ["上下文模式"] + cases
        print("\n" + "="*80)
        print("【对照矩阵】行=上下文模式，列=用例；单元格依次为 outcome、it=轮次、tc=工具调用。")
        print("说明：✓ 有终止回答；⚠ 回答含无观测依据的数字；✗ 没有终止回答。")
        print("="*80)
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    def visualize_results(self, results: List[Dict[str, Any]]):
        """
        生成消融实验结果的可视化图表
        
        Args:
            results: 测试结果列表
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("【可视化跳过】未安装 Matplotlib，跳过图片生成。")
            return
            
        # 提取用于可视化的数据
        modes = [r["context_mode"] for r in results]
        iterations = [r.get("iterations", 0) for r in results]
        tool_calls = [r.get("num_tool_calls", 0) for r in results]
        exec_times = [r.get("execution_time", 0) for r in results]
        success = [1 if _completed(r) else 0 for r in results]
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Ablation Study: Impact of Context Components", fontsize=16)
        
        # 图 1：终止响应完成率
        axes[0, 0].bar(modes, success, color=['green' if s else 'red' for s in success])
        axes[0, 0].set_title("Terminal Responses by Context Mode")
        axes[0, 0].set_ylabel("Completed (1) / No response (0)")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 图 2：所需迭代次数
        axes[0, 1].bar(modes, iterations, color='blue')
        axes[0, 1].set_title("Iterations Required")
        axes[0, 1].set_ylabel("Number of Iterations")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 图 3：工具调用次数
        axes[1, 0].bar(modes, tool_calls, color='orange')
        axes[1, 0].set_title("Tool Calls Made")
        axes[1, 0].set_ylabel("Number of Tool Calls")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 图 4：执行耗时
        axes[1, 1].bar(modes, exec_times, color='purple')
        axes[1, 1].set_title("Execution Time")
        axes[1, 1].set_ylabel("Time (seconds)")
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig("ablation_study_results.png", dpi=150, bbox_inches='tight')
        logger.info("【可视化保存】已生成 ablation_study_results.png。")
    
    def generate_report(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> str:
        """
        生成消融实验的综合报告
        
        Args:
            results: 测试结果列表
            analysis: 分析摘要
            
        Returns:
            报告文本
        """
        report = """
# Context Ablation Study Report

## Executive Summary
This ablation study explores the effect of different context components on AI
agent behavior. This legacy suite reports terminal-response completion and
tool-use behavior; it does not claim task correctness without a task-specific
rubric.

## Test Configuration
- **Provider**: {provider}
- **Model**: {model}
- **Task**: Complex financial analysis requiring PDF parsing, currency conversion, and calculations
- **Context Modes Tested**: {num_modes}

## Key Findings

### 1. Complete Lack of Historical Tool Calls (NO_HISTORY)
**Impact**: Agent loses track of previous actions and may repeat operations unnecessarily.
- **Behavior**: Agent cannot reference past tool executions, leading to redundant API calls
- **Performance**: {no_history_perf}
- **Critical for**: Multi-step tasks requiring sequential dependencies

### 2. Lack of Reasoning Process (NO_REASONING)
**Impact**: Agent operates without strategic planning or step-by-step thinking.
- **Behavior**: Direct execution without planning leads to inefficient or incorrect solutions
- **Performance**: {no_reasoning_perf}
- **Critical for**: Complex tasks requiring logical decomposition

### 3. Lack of Tool Call Commands (NO_TOOL_CALLS)
**Impact**: Agent cannot execute any external tools.
- **Behavior**: The model may return a refusal or describe the missing tools;
  a terminal response is not evidence that the financial task was completed.
- **Performance**: {no_tool_calls_perf}
- **Critical for**: Any task requiring external data or computation

### 4. Lack of Tool Call Results (NO_TOOL_RESULTS)
**Impact**: Agent operates blind to the outcomes of its actions.
- **Behavior**: The model may stop with a warning, repeat actions, or produce an
  incorrect conclusion; correctness must be checked by a task-specific rubric.
- **Performance**: {no_tool_results_perf}
- **Critical for**: Tasks requiring iterative refinement or result validation

## Statistical Summary
- **Total Tests Run**: {total_tests}
- **Terminal Responses**: {successful_tests}
- **Answers Stating Unsupported Figures**: {unsupported_number_tests} {unsupported_number_modes}
- **Average Execution Time (Full Context)**: {avg_exec_time}s
- **Average Tool Calls (Full Context)**: {avg_tool_calls}

## Conclusion
The ablation study records how each context component changes execution behavior:
1. **Tool calls** determine whether the agent can interact with external tools
2. **Tool results** provide feedback for decision-making
3. **Reasoning** can affect planning and execution efficiency
4. **History** can prevent redundant actions and maintain task coherence

Task-level claims require a separate evaluator, such as the canonical numeric
rubric used by `run_experiment_1_1.py`.

## Recommendations
- Always maintain complete context for production agents
- Consider context windowing rather than removal for memory optimization
- Implement fallback mechanisms when context components are unavailable
"""
        
        # 填入性能指标
        def get_perf_string(mode):
            mode_result = next((r for r in results if r["context_mode"] == mode), None)
            if mode_result:
                return f"{'COMPLETED' if _completed(mode_result) else 'NO TERMINAL RESPONSE'} - {mode_result.get('iterations', 0)} iterations, {mode_result.get('execution_time', 0)}s"
            return "N/A"
        
        # 计算基准指标
        baseline = next((r for r in results if r["context_mode"] == "full"), None)
        avg_exec_time = baseline.get("execution_time", 0) if baseline else 0
        avg_tool_calls = baseline.get("num_tool_calls", 0) if baseline else 0
        
        report = report.format(
            provider=self.provider,
            model=self.model or "default",
            num_modes=len(results),
            no_history_perf=get_perf_string("no_history"),
            no_reasoning_perf=get_perf_string("no_reasoning"),
            no_tool_calls_perf=get_perf_string("no_tool_calls"),
            no_tool_results_perf=get_perf_string("no_tool_results"),
            total_tests=analysis["total_tests"],
            successful_tests=analysis["successful_tests"],
            unsupported_number_tests=analysis.get("unsupported_number_tests", 0),
            unsupported_number_modes=analysis.get("unsupported_number_modes", []) or "",
            avg_exec_time=avg_exec_time,
            avg_tool_calls=avg_tool_calls
        )
        
        return report


def run_single_task(api_key: str, task: str, context_mode: str = "full", provider: str = "siliconflow", model: str = None, output: str = None):
    """
    使用 Agent 运行单个任务

    Args:
        api_key: LLM 提供商的 API 密钥
        task: 任务描述
        context_mode: 要使用的上下文模式
        provider: 要使用的 LLM 提供商
        model: 可选的模型重载
        output: JSON 结果的可选保存路径（默认为 task_result_{mode}.json）
    """
    # 解析上下文模式
    mode_map = {
        "full": ContextMode.FULL,
        "no_history": ContextMode.NO_HISTORY,
        "no_reasoning": ContextMode.NO_REASONING,
        "no_tool_calls": ContextMode.NO_TOOL_CALLS,
        "no_tool_results": ContextMode.NO_TOOL_RESULTS
    }
    
    if context_mode not in mode_map:
        logger.error(f"【参数错误】无效的上下文模式：{context_mode}")
        logger.info(f"【可选模式】{', '.join(mode_map.keys())}")
        return
    
    # 创建 Agent
    agent = ContextAwareAgent(api_key, mode_map[context_mode], provider=provider, model=model)
    
    logger.info(f"【任务启动】上下文模式：{context_mode}")
    logger.info(f"【任务内容】前 100 字符：{task[:100]}...")
    
    # 执行任务
    result = agent.execute_task(task)
    
    print_task_result(result, context_mode=context_mode)
    
    # 保存详细结果
    output_file = output or f"task_result_{context_mode}.json"
    with open(output_file, 'w') as f:
        # 将轨迹转换为可序列化格式
        serializable_result = {
            "completed": _completed(result),
            "task_success": result.get("task_success"),
            # 供旧版结果读取代码使用的向后兼容别名。
            "success": _completed(result),
            "iterations": result.get("iterations", 0),
            "final_answer": result.get("final_answer"),
            "error": result.get("error"),
            "context_mode": context_mode,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "result": tc.result,
                    "timestamp": tc.timestamp
                }
                for tc in result["trajectory"].tool_calls
            ],
            "reasoning_steps": result["trajectory"].reasoning_steps,
            # 去除凭据的提供商原生请求/响应证据。消融实验关乎每次推理时可见的上下文，因此事后摘要不足以验证实验 1-1。
            "api_turns": result["trajectory"].api_turns,
            "provider": result.get("provider", provider),
            "model": result.get("model", model),
            "base_url": result.get("base_url"),
            "using_openrouter": result.get("using_openrouter", False),
        }
        json.dump(serializable_result, f, indent=2)
    
    logger.info(f"【结果保存】详细结果已保存到：{output_file}")


def ensure_sample_pdfs():
    """
    确保示例 PDF 文件存在，若不存在则创建
    
    Returns:
        bool: 若 PDF 可用返回 True
    """
    pdf_dir = Path("fixtures/pdfs")
    sample_pdf = pdf_dir / "simple_expense_report.pdf"
    
    if not pdf_dir.exists() or not sample_pdf.exists():
        print("\n【样例 PDF】未找到样例文件，正在创建。")
        try:
            # 运行 PDF 创建脚本
            result = subprocess.run(
                [sys.executable, "create_sample_pdf.py"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print("【样例 PDF】创建完成，文件位于 fixtures/pdfs/。")
                return True
            else:
                print(f"【样例 PDF 创建失败】{result.stderr}")
                return False
        except Exception as e:
            print(f"【样例 PDF 创建错误】{str(e)}")
            return False
    return True


def get_sample_tasks():
    """
    获取用于测试的示例任务
    
    Returns:
        list: 示例任务字典列表
    """
    # 检查是在本地运行还是需要使用在线 PDF
    local_pdfs = Path("fixtures/pdfs").exists()
    
    if local_pdfs:
        pdf_path = "file://" + str(Path.cwd() / "fixtures/pdfs" / "simple_expense_report.pdf")
        pdf_note = "使用本地 PDF 样例"
    else:
        # 使用公开可用的 PDF 进行测试
        pdf_path = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        pdf_note = "使用在线 PDF 样例"
    
    return [
        {
            "name": "📊 货币换算任务",
            "description": "在多种货币之间进行换算",
            "task": """将 1000 USD 分别换算为 EUR、GBP 和 JPY。
然后计算这三种换算结果的平均值。"""
        },
        {
            "name": "📄 PDF 分析任务",
            "description": f"从 PDF 中提取并分析数据（{pdf_note}）",
            "task": f"""请分析这份 PDF 文档：{pdf_path}
提取其中的文本内容，并概述你发现的信息。"""
        },
        {
            "name": "💰 复杂财务分析",
            "description": "多步骤财务计算",
            "task": """某公司各季度收入如下：
- Q1: $2,500,000 USD
- Q2: €2,100,000 EUR
- Q3: £1,800,000 GBP
- Q4: ¥380,000,000 JPY

请完成：
1. 将全部收入换算为 USD
2. 计算以 USD 计的全年总收入
3. 计算季度平均收入
4. 找出收入最高的季度
5. 若公司利润率为 20%，计算以 USD 计的全年利润"""
        },
        {
            "name": "🌍 多币种预算规划",
            "description": "国际预算计算",
            "task": """一场国际会议的预算分配如下：
- 场地（英国）：£45,000
- 演讲嘉宾（美国）：$75,000
- 餐饮（法国）：€38,000
- 技术（日本）：¥8,500,000
- 市场推广（新加坡）：S$25,000

任务：
1. 将全部金额换算为 USD
2. 计算总预算
3. 计算每个类别占总预算的百分比
4. 如果总预算需要削减 15%，各类别应削减多少（按其原始币种表示）？"""
        },
        {
            "name": "📈 投资组合分析",
            "description": "分析国际投资回报",
            "task": """某投资者持有以下国际投资，其当前价值如下：
- 美国科技股：$125,000（买入价：$100,000）
- 欧洲债券：€85,000（买入价：€90,000）
- 英国房地产：£200,000（买入价：£175,000）
- 日本 ETF：¥15,000,000（买入价：¥12,000,000）

请计算：
1. 将全部当前价值换算为 USD
2. 将全部买入价格换算为 USD（为简化起见，使用当前汇率）
3. 计算每项投资以 USD 计的盈亏
4. 计算投资组合总价值和整体回报率
5. 哪项投资的百分比表现最好？"""
        }
    ]


def get_ablation_cases(num_cases: int = 1) -> List[Dict[str, str]]:
    """
    构建消融实验的独立案例列表。

    复用预定义的示例任务，跳过需要网络访问的 PDF 任务，
    以便除了 LLM 调用本身之外，整个实验可在离线状态下复现。
    num_cases=1 返回单个案例；更大的数值将添加更多案例。

    Args:
        num_cases: 包含的案例数量（限制在 [1, 可用数量] 范围内）。

    Returns:
        包含 {"name", "task"} 字典的列表。
    """
    samples = get_sample_tasks()
    # 索引 0/2/3/4 为货币/金融任务（索引 1 为 PDF 任务）。
    candidates = [samples[0], samples[2], samples[3], samples[4]]
    num_cases = max(1, min(num_cases, len(candidates)))
    return [{"name": c["name"], "task": c["task"]} for c in candidates[:num_cases]]


def run_ablation_study(api_key: str, provider: str = "siliconflow", model: str = None,
                       context_modes: List[str] = None, num_cases: int = 1,
                       output: str = None):
    """
    运行消融实验以测试上下文的重要性

    Args:
        api_key: LLM 提供商的 API 密钥
        provider: 要使用的 LLM 提供商
        model: 可选的模型重载
        context_modes: 待测试的上下文模式名称列表（默认为全部）
        num_cases: 针对每个模式运行的案例数。1（默认）保持原有的单个跨国预算任务；
            >1 时在多个示例任务上跨模式对比。
        output: 原始 JSON 结果的可选保存路径（默认 ablation_results.json）
    """
    # 若提供了上下文模式则进行解析
    mode_map = {
        "full": ContextMode.FULL,
        "no_history": ContextMode.NO_HISTORY,
        "no_reasoning": ContextMode.NO_REASONING,
        "no_tool_calls": ContextMode.NO_TOOL_CALLS,
        "no_tool_results": ContextMode.NO_TOOL_RESULTS
    }

    modes_to_test = None
    if context_modes:
        modes_to_test = []
        for mode_name in context_modes:
            if mode_name in mode_map:
                modes_to_test.append(mode_map[mode_name])
            else:
                logger.error(f"【参数错误】无效的上下文模式：{mode_name}")
                logger.info(f"【可选模式】{', '.join(mode_map.keys())}")
                return

    # 构建案例。num_cases <= 1 保持原有的单个案例行为。
    cases = None
    if num_cases and num_cases > 1:
        cases = get_ablation_cases(num_cases)

    test_suite = AblationTestSuite(api_key, provider=provider, model=model)

    logger.info("【消融实验启动】即将依次运行配置的上下文模式。")
    if modes_to_test:
        logger.info(f"【实验分支】{', '.join(context_modes)}")
    else:
        logger.info("【实验分支】将运行全部上下文模式。")
    logger.info(f"【实验用例数】{len(cases) if cases else 1}")

    results = test_suite.run_ablation_study(modes_to_test, cases=cases)

    # 分析结果
    analysis = test_suite.analyze_results(results)

    # 打印结果表格
    test_suite.print_results_table(results)

    # 打印模式 x 用例对照矩阵
    test_suite.print_comparison_matrix(results)

    # 生成可视化图表
    try:
        test_suite.visualize_results(results)
    except Exception as e:
        logger.warning(f"【可视化生成失败】{str(e)}")

    # 生成并保存报告
    report = test_suite.generate_report(results, analysis)
    with open("ablation_study_report.md", "w") as f:
        f.write(report)
    logger.info("【报告保存】已生成 ablation_study_report.md。")

    # 保存原始结果
    output_file = output or "ablation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"【原始结果保存】已保存到：{output_file}")
    
    # 打印分析摘要
    print("\n" + "="*80)
    print("【消融分析摘要】先看终止回答比例，再比较各模式与基线的差异。")
    print("="*80)
    print(f"【终止回答比例】{analysis['completed_tests']}/{analysis['total_tests']}")
    print("\n【模式影响】以下数值相对完整上下文基线计算：")
    for mode, impact in analysis["context_mode_impact"].items():
        print(f"\n【模式：{mode.upper()}】")
        print(f"  【终止回答保持】{impact['completion_maintained']}")
        print(f"  【耗时差】{impact['execution_time_delta']:.2f}s")
        print(f"  【轮次差】{impact['iteration_delta']}")
        print(f"  【工具调用差】{impact['tool_call_delta']}")
        if impact['failure_reason']:
            print(f"  【失败原因】{impact['failure_reason']}")


def interactive_mode(api_key: str, provider: str = "siliconflow", model: str = None):
    """
    以交互模式运行 Agent
    
    Args:
        api_key: LLM 提供商的 API 密钥
        provider: 要使用的 LLM 提供商
        model: 可选的模型重载
    """
    # 保存当前提供商和模型
    current_provider = provider
    current_model = model
    current_api_key = api_key
    
    # 可用提供商
    available_providers = list(SUPPORTED_PROVIDERS)
    
    print("\n" + "="*60)
    print("【交互模式】Context-Aware Agent；可输入任务或下方命令。")
    print(f"【当前模型】提供商={current_provider.upper()}；模型={current_model or 'default'}")
    print("="*60)
    print("【可用命令】")
    print("  - 直接输入任务/问题：让 Agent 执行")
    print("  - 'samples'：查看样例任务")
    print("  - 'sample <number>'：运行一个样例任务")
    print("  - 'create_pdfs'：创建样例 PDF")
    print("  - 'providers'：查看可用提供商")
    print("  - 'provider <name>'：切换提供商")
    print("  - 'modes'：查看上下文模式")
    print("  - 'mode <mode_name>'：切换上下文模式（也兼容 'modes <mode_name>'）")
    print("  - 'reset'：清空当前 Agent 轨迹")
    print("  - 'status'：查看当前配置和轨迹计数")
    print("  - 'help'：再次显示命令说明")
    print("  - 'quit'：退出交互模式")
    print("-"*60)
    
    # 确保示例 PDF 存在
    ensure_sample_pdfs()
    
    # 获取示例任务
    sample_tasks = get_sample_tasks()
    
    # 初始化使用完整上下文的 Agent
    mode_map = {
        "full": ContextMode.FULL,
        "no_history": ContextMode.NO_HISTORY,
        "no_reasoning": ContextMode.NO_REASONING,
        "no_tool_calls": ContextMode.NO_TOOL_CALLS,
        "no_tool_results": ContextMode.NO_TOOL_RESULTS
    }
    
    current_mode = ContextMode.FULL
    agent = ContextAwareAgent(current_api_key, current_mode, provider=current_provider, model=current_model)
    
    while True:
        try:
            # 在提示符中显示当前提供商
            prompt = f"\n【{current_provider.upper()}｜输入任务或命令】> "
            user_input = input(prompt).strip()
            
            if user_input.lower() == 'quit':
                print("【退出交互模式】会话结束。")
                break
            
            elif user_input.lower() == 'help':
                print("\n【命令帮助】")
                print("  samples          - 显示全部样例任务")
                print("  sample <n>       - 运行第 n 个样例任务")
                print("  providers        - 显示可用 LLM 提供商")
                print("  provider <name>  - 切换提供商")
                print("  modes            - 显示上下文模式")
                print("  mode <name>      - 切换上下文模式（也兼容 modes <name>）")
                print("  status           - 显示当前配置和轨迹")
                print("  reset            - 重置 Agent 轨迹")
                print("  create_pdfs      - 生成样例 PDF")
                print("  help             - 显示本帮助")
                print("  quit             - 退出交互模式")
                print("\n【任务输入】也可以直接输入任意任务/问题。")
            
            elif user_input.lower() == 'samples':
                print("\n【样例任务】以下任务可直接用 sample <编号> 运行：")
                for i, sample in enumerate(sample_tasks, 1):
                    print(f"\n{i}. {sample['name']}")
                    print(f"   {sample['description']}")
            
            elif user_input.lower().startswith('sample '):
                try:
                    sample_num = int(user_input.split()[1])
                    if 1 <= sample_num <= len(sample_tasks):
                        sample = sample_tasks[sample_num - 1]
                        print(f"\n【样例任务启动】{sample['name']}")
                        print(f"【任务内容】{sample['task']}")
                        print("【任务执行】正在等待 Agent 返回结果……")
                        
                        result = agent.execute_task(sample['task'])
                        
                        print_task_result(result, context_mode=current_mode.value)
                    else:
                        print(f"【输入错误】样例编号应在 1 到 {len(sample_tasks)} 之间。")
                except (ValueError, IndexError):
                    print(f"【输入错误】样例编号应在 1 到 {len(sample_tasks)} 之间。")
            
            elif user_input.lower() == 'create_pdfs':
                print("\n【样例 PDF】正在创建测试所需的 PDF 文件。")
                try:
                    result = subprocess.run(
                        [sys.executable, "create_sample_pdf.py"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        print("【样例 PDF】创建成功，文件位于 fixtures/pdfs/。")
                        # 使用新的 PDF 路径更新示例任务
                        sample_tasks = get_sample_tasks()
                    else:
                        print(f"【样例 PDF 创建失败】{result.stderr}")
                except Exception as e:
                    print(f"【样例 PDF 创建错误】{str(e)}")
            
            elif user_input.lower() == 'providers':
                print("\n【可用提供商】括号内仅显示密钥是否已配置，不显示密钥内容：")
                for p in available_providers:
                    status = "（当前）" if p == current_provider else ""
                    spec = PROVIDERS.get(canonical_provider(p))
                    if spec is None:
                        print(f"  - {p}{status}")
                        continue
                    # 从注册表中派生，因此新增项会直接在此显示而无需修改本命令。
                    if not spec.requires_key:
                        keys = "无需 API Key"
                    else:
                        keys = " / ".join(spec.key_vars)
                        keys += "（已配置）" if spec.api_key() else "（未配置）"
                    print(f"  - {p}: {spec.default_model} [{keys}]{status}")
            
            elif user_input.lower().startswith('provider '):
                new_provider = canonical_provider(user_input[9:].strip())
                if new_provider in available_providers:
                    # 通过注册表解析：它掌握各提供商的密钥变量、哪些提供商不需要密钥（ollama）以及 OpenRouter 兜底，并在无可用的配置时精确报告应设置的具体变量。
                    try:
                        new_backend = resolve_backend(new_provider)
                    except ValueError as exc:
                        print(f"【提供商切换失败】{exc}")
                        continue
                    # ContextAwareAgent 会重新解析，因此仅向其传递会被视为该提供商自身拥有的密钥。在兜底路径上 new_backend.api_key 是 OpenRouter 密钥；传递它会向提供商自身端点发送 OpenRouter 密钥。
                    new_api_key = "" if new_backend.using_openrouter else new_backend.api_key

                    # 更新当前设置
                    current_provider = new_provider
                    current_api_key = new_api_key
                    current_model = None  # 重置为使用新提供商的默认模型
                    
                    # 创建带有新提供商的新 Agent
                    agent = ContextAwareAgent(current_api_key, current_mode, provider=current_provider, model=current_model)
                    
                    # 从配置中获取默认模型名称
                    from config import Config
                    default_model = Config.get_default_model(current_provider)
                    
                    print(f"【提供商已切换】{current_provider}")
                    print(f"【当前模型】{default_model}")
                else:
                    print(f"【输入错误】无效提供商；可选：{', '.join(available_providers)}")
            
            elif user_input.lower() == 'modes':
                print("\n【上下文模式】以下模式用于控制每轮发送给模型的上下文：")
                for mode in mode_map.keys():
                    print(f"  - {mode}")
            
            elif user_input.lower().startswith(('mode ', 'modes ')):
                plural_alias = user_input.lower().startswith('modes ')
                new_mode = user_input[6 if plural_alias else 5:].strip()
                if new_mode in mode_map:
                    if plural_alias:
                        print(
                            "【命令兼容】已将 'modes <模式名>' 识别为模式切换；"
                            "推荐写法仍是 'mode <模式名>'。"
                        )
                    current_mode = mode_map[new_mode]
                    agent = ContextAwareAgent(current_api_key, current_mode, provider=current_provider, model=current_model)
                    print(f"【上下文模式已切换】{current_mode.value}")
                    if current_mode != ContextMode.FULL:
                        print("【实验提示】该模式会刻意移除部分上下文，仅用于观察消融现象。")
                else:
                    print(f"【输入错误】无效模式；可选：{', '.join(mode_map.keys())}")
            
            elif user_input.lower() == 'reset':
                agent.reset()
                print("【会话重置】已清空 Agent 轨迹和对话历史。")
            
            elif user_input.lower() == 'status':
                from config import Config
                model_name = current_model or Config.get_default_model(current_provider)
                print("\n【当前配置】以下信息用于确认本轮实验到底使用了什么设置：")
                print(f"  【提供商】{current_provider.upper()}")
                print(f"  【模型】{model_name}")
                print(f"  【上下文模式】{current_mode.value}")
                print(f"  【对话历史】{len(agent.conversation_history)} 条消息")
                print(f"  【工具调用】{len(agent.trajectory.tool_calls)} 次")
                
                # API 密钥状态从注册表中派生，因此每个可选择的提供商都会显示对应状态。
                spec = PROVIDERS.get(canonical_provider(current_provider))
                if spec is None:
                    pass
                elif not spec.requires_key:
                    print("  【API Key】本地运行时不需要")
                else:
                    names = " / ".join(spec.key_vars)
                    key_status = "已配置" if spec.api_key() else "未配置"
                    print(f"  【API Key 状态】{names}：{key_status}")
                    if not spec.api_key() and os.getenv("OPENROUTER_API_KEY"):
                        print("  【兜底路由】OPENROUTER_API_KEY 已配置，将经 OpenRouter 路由。")
            
            elif user_input:
                # 执行任务
                print("\n【任务执行】正在等待 Agent 返回结果……")
                result = agent.execute_task(user_input)
                print_task_result(result, context_mode=current_mode.value)
                    
        except KeyboardInterrupt:
            print("\n\n【已中断】输入 quit 可退出交互模式。")
        except Exception as e:
            print(f"【交互模式错误】{str(e)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="上下文感知 AI Agent（第一章 实验 1.1：上下文消融实验）",
        epilog="""示例：
  # 交互模式（默认）
  python main.py

  # 运行完整消融实验（五种上下文模式对照）
  python main.py --mode ablation

  # 多案例消融对照，输出结果到指定文件
  python main.py --mode ablation --cases 3 --output my_ablation.json

  # 只对比“完整上下文”与“无历史消息”两种模式
  python main.py --mode ablation --ablation-modes full no_history

  # 单任务执行，指定上下文模式与提供商
  python main.py --mode single --task "把 1000 美元换算成欧元、英镑、日元并求平均值" \\
      --context-mode full --provider doubao
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["single", "ablation", "interactive"],
        default="interactive",
        help="运行模式：single=单任务，ablation=消融实验，interactive=交互式（默认）"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="要执行的任务/问题（用于 single 模式；不提供则从示例任务中选择）"
    )
    parser.add_argument(
        "--context-mode",
        choices=["full", "no_history", "no_reasoning", "no_tool_calls", "no_tool_results"],
        default="full",
        help="single 模式下的上下文模式：full=完整；no_history=无历史消息；"
             "no_reasoning=无思考过程；no_tool_calls=无工具定义；no_tool_results=无工具执行结果"
    )
    parser.add_argument(
        "--ablation-modes",
        nargs="+",
        choices=["full", "no_history", "no_reasoning", "no_tool_calls", "no_tool_results"],
        help="消融实验中要测试的上下文模式（默认测试全部五种）"
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=1,
        help="消融实验运行的案例数量（默认 1；>1 时在多个示例任务上做跨模式对照）"
    )
    parser.add_argument(
        "--provider",
        type=canonical_provider,
        choices=SUPPORTED_PROVIDERS,
        default="doubao",
        help="LLM 提供商（默认：doubao；kimi 是 moonshot 的别名；openrouter 或缺失主 key 时经 OpenRouter 兜底；ollama 为本地免费）"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="使用的模型名称（可选，不指定则使用该提供商的默认模型）"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="LLM 提供商的 API Key（也可通过对应环境变量设置，例如 DASHSCOPE_API_KEY、SILICONFLOW_API_KEY、ARK_API_KEY、MOONSHOT_API_KEY、DEEPSEEK_API_KEY 或 ZHIPU_API_KEY；缺失主 key 时可用 OPENROUTER_API_KEY 兜底）"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="结果输出文件路径（single 模式为 JSON 结果，ablation 模式为原始结果 JSON）"
    )

    args = parser.parse_args()
    
    # 注册表掌握每个提供商的密钥变量、OpenRouter 兜底以及哪些提供商无需密钥，因此通过其进行解析，而非在此维护针对各个提供商的处理链。显式传入的 --api-key 仍具有最高优先级。
    try:
        backend = resolve_backend(args.provider, model=args.model, api_key=args.api_key)
    except ValueError as exc:
        logger.error(f"【配置错误】{exc}")
        sys.exit(1)

    api_key = args.api_key or ""
    if backend.using_openrouter and not args.api_key:
        logger.info(
            f"【路由说明】未设置 {args.provider} 的直连密钥；"
            "将通过 OPENROUTER_API_KEY 经 OpenRouter 调用。"
        )
    elif not args.api_key:
        # 传递解析后的密钥；ContextAwareAgent 会重新解析，空值会使其走兜底路径。
        api_key = backend.api_key
    
    # 记录提供商信息
    logger.info(
        f"【启动配置】提供商={args.provider}；模型={args.model or 'default'}；"
        f"运行模式={args.mode}。"
    )
    
    # 根据模式执行
    if args.mode == "single":
        if not args.task:
            # 提示用户选择示例任务
            print("\n" + "="*60)
            print("【单任务模式】尚未提供任务，将显示可选样例。")
            print("="*60)
            
            # 确保 PDF 存在
            ensure_sample_pdfs()
            
            # 获取并展示示例任务
            sample_tasks = get_sample_tasks()
            print("\n【样例任务】请选择一个任务；任务正文将在确认前显示：")
            for i, sample in enumerate(sample_tasks, 1):
                print(f"\n{i}. {sample['name']}")
                print(f"   {sample['description']}")
            
            print("\n" + "="*60)
            try:
                choice = input("\n【选择样例】输入 1-{}，或输入 q 退出：".format(len(sample_tasks))).strip()
                if choice.lower() == 'q':
                    sys.exit(0)
                
                task_num = int(choice)
                if 1 <= task_num <= len(sample_tasks):
                    selected_task = sample_tasks[task_num - 1]
                    print(f"\n【已选择样例】{selected_task['name']}")
                    print("\n【任务内容】")
                    print("-"*40)
                    print(selected_task['task'])
                    print("-"*40)
                    
                    confirm = input("\n【执行确认】运行此任务？(y/n)：").strip().lower()
                    if confirm == 'y':
                        run_single_task(api_key, selected_task['task'], args.context_mode,
                                      provider=args.provider, model=args.model, output=args.output)
                    else:
                        print("【已取消】未执行任务。")
                else:
                    print(f"【输入错误】请选择 1 到 {len(sample_tasks)} 之间的编号。")
                    sys.exit(1)
            except (ValueError, KeyboardInterrupt):
                print("\n【退出】已结束单任务选择。")
                sys.exit(0)
        else:
            run_single_task(api_key, args.task, args.context_mode,
                          provider=args.provider, model=args.model, output=args.output)

    elif args.mode == "ablation":
        run_ablation_study(api_key, provider=args.provider, model=args.model,
                          context_modes=args.ablation_modes, num_cases=args.cases,
                          output=args.output)
    
    else:  # 交互模式
        interactive_mode(api_key, provider=args.provider, model=args.model)


if __name__ == "__main__":
    main()
