"""
具备工具调用能力的上下文感知 AI Agent
使用来自 SiliconFlow 等平台的 Qwen 大模型，支持文档解析、货币转换与计算器等工具。
旨在通过消融实验展示上下文各组件的重要性。
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import requests
from openai import OpenAI
import pypdf
from io import BytesIO
import math
from datetime import datetime
from concurrent.futures import TimeoutError

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class _HttpStatusLabelFilter(logging.Filter):
    """为第三方 HTTP 状态日志补上面向学习者的中文说明。"""

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "_context_http_status_labelled", False):
            return True
        record.msg = f"【HTTP 通信】第三方 SDK 的网络请求状态：{record.getMessage()}"
        record.args = ()
        record._context_http_status_labelled = True
        return True


_httpx_logger = logging.getLogger("httpx")
if not any(isinstance(log_filter, _HttpStatusLabelFilter) for log_filter in _httpx_logger.filters):
    _httpx_logger.addFilter(_HttpStatusLabelFilter())


def _safe_provider_error(error: Exception) -> str:
    """Keep provider diagnostics useful without persisting credential fragments."""

    message = str(error)
    message = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [redacted]", message)
    message = re.sub(r"(?i)<(?:sk|ak|rk|pk)-[^>\s]+>", "<[redacted]>", message)
    return re.sub(r"(?i)\b(?:sk|ak|rk|pk)-[A-Za-z0-9._-]{6,}", "[redacted]", message)


def _reasoning_safe_temperature(model, requested=1.0):
    """推理模型（如 Kimi K3, GPT-5 等）仅接受 temperature=1。
    对于这些模型返回 1；否则返回请求的数值，以保持非推理模型（Doubao、DeepSeek、早期 Moonshot 等）不受影响。"""
    m = str(model or "").lower().replace("/", "-")
    return 1 if ("kimi-k3" in m or "gpt-5" in m) else requested


# 剥离工具结果的两种方式，它们代表不同的实验对照：
# MARKER 保留可见的遮盖标记：模型可以看到观测结果存在但被扣留，从而决定停止并说明此情况。
# EMPTY 则静默扣留——消息依然存在（满足 API 格式要求），但不携带任何内容，
# 这对于无法区分遮盖与工具无响应的模型而言，就表现为“缺少工具执行结果”。
HIDDEN_RESULT_MARKER = "[Tool result hidden due to context mode]"
HIDDEN_RESULT_EMPTY = ""
HIDDEN_RESULT_STYLES = {"marker": HIDDEN_RESULT_MARKER, "empty": HIDDEN_RESULT_EMPTY}


class ContextMode(Enum):
    """消融实验的不同上下文模式"""
    FULL = "full"  # 完整上下文（包含所有组件）
    NO_HISTORY = "no_history"  # 无历史工具调用
    NO_REASONING = "no_reasoning"  # 无推理/思考过程
    NO_TOOL_CALLS = "no_tool_calls"  # 无工具调用指令
    NO_TOOL_RESULTS = "no_tool_results"  # 无工具调用结果


@dataclass
class ToolCall:
    """表示单次工具调用"""
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentTrajectory:
    """记录 Agent 的执行轨迹"""
    reasoning_steps: List[str] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    # 记录每次真实模型交互的无凭据请求/响应确凿证据。
    # 这特意作为执行轨迹的一部分：实验 1-1 关乎模型在决策时能看到什么，因此事后重建请求并不是可接受的证据。
    api_turns: List[Dict[str, Any]] = field(default_factory=list)
    context_mode: ContextMode = ContextMode.FULL


class ToolRegistry:
    """可用工具注册表"""
    
    @staticmethod
    def parse_pdf(url: str) -> Dict[str, Any]:
        """
        从 URL 或本地文件下载并解析 PDF 文档
        
        Args:
            url: 待解析 PDF 文档的 URL 或文件路径
            
        Returns:
            包含解析后文本和元数据的字典
        """
        try:
            # 检查是否为本地文件
            if url.startswith('file://'):
                # 从 file:// URL 中提取文件路径
                file_path = url.replace('file://', '')
                logger.info(f"【PDF 读取】正在读取本地 PDF：{file_path}")
                
                # 直接读取文件
                with open(file_path, 'rb') as f:
                    pdf_content = f.read()
                    
            elif url.startswith('/') or url.startswith('./') or url.startswith('../') or ':\\' in url or ':/' in url[1:3]:
                # 直接文件路径（绝对路径或相对路径）
                logger.info(f"【PDF 读取】正在读取本地 PDF：{url}")
                
                # 直接读取文件
                with open(url, 'rb') as f:
                    pdf_content = f.read()
                    
            else:
                # 远程 URL，进行下载
                logger.info(f"【PDF 下载】正在下载远程 PDF：{url}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                pdf_content = response.content
            
            # 解析 PDF 内容
            pdf_file = BytesIO(pdf_content)
            pdf_reader = pypdf.PdfReader(pdf_file)
            
            text_content = []
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                text_content.append({
                    "page": page_num,
                    "text": text
                })
            
            result = {
                "url": url,
                "num_pages": len(pdf_reader.pages),
                "content": text_content,
                "metadata": pdf_reader.metadata if hasattr(pdf_reader, 'metadata') else {}
            }
            
            logger.info(f"【PDF 解析】解析完成：共 {len(pdf_reader.pages)} 页")
            return result
            
        except Exception as e:
            logger.error(f"【PDF 解析失败】{str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def convert_currency(amount: float, from_currency: str, to_currency: str) -> Dict[str, Any]:
        """
        使用静态汇率进行货币换算
        
        Args:
            amount: 待换算金额
            from_currency: 源货币代码（如 'USD'）
            to_currency: 目标货币代码（如 'EUR'）
            
        Returns:
            包含换算结果的字典
        """
        try:
            if isinstance(amount, str):
                clean_amt = amount.replace(",", "").strip()
                symbols_to_strip = sorted(
                    [
                        "USD$", "U.S.$", "US$", "$",
                        "SGD$", "SG$", "S$",
                        "AUD$", "AU$", "A$",
                        "CAD$", "CA$", "C$",
                        "€", "£", "₹",
                    ],
                    key=len,
                    reverse=True,
                )
                for sym in symbols_to_strip:
                    clean_amt = clean_amt.replace(sym, "")
                amount = float(clean_amt.strip())
            else:
                amount = float(amount)
            exchange_rates = {
                "USD": 1.0,
                "EUR": 0.92,
                "GBP": 0.79,
                "JPY": 149.50,
                "CNY": 7.24,
                "CAD": 1.36,
                "AUD": 1.53,
                "CHF": 0.88,
                "INR": 83.12,
                "SGD": 1.34
            }

            def _normalize_code(code: str) -> str:
                if not isinstance(code, str):
                    return str(code or "")
                c = code.strip().upper()
                symbols = {
                    "$": "USD",
                    "US$": "USD",
                    "U.S.$": "USD",
                    "USD$": "USD",
                    "S$": "SGD",
                    "SG$": "SGD",
                    "SGD$": "SGD",
                    "A$": "AUD",
                    "AU$": "AUD",
                    "AUD$": "AUD",
                    "C$": "CAD",
                    "CA$": "CAD",
                    "CAD$": "CAD",
                    "€": "EUR",
                    "£": "GBP",
                    "₹": "INR",
                }
                if c in symbols:
                    return symbols[c]
                if c.endswith("$"):
                    prefix = c[:-1].strip()
                    if prefix in exchange_rates:
                        return prefix
                    if prefix in ("US", "U.S."):
                        return "USD"
                    if prefix in ("AU", "A"):
                        return "AUD"
                    if prefix in ("CA", "C"):
                        return "CAD"
                return c

            from_currency = _normalize_code(from_currency)
            to_currency = _normalize_code(to_currency)
            
            logger.info(f"【货币换算】开始：{amount} {from_currency} -> {to_currency}")
            
            if from_currency not in exchange_rates or to_currency not in exchange_rates:
                return {"error": f"Unsupported currency: {from_currency} or {to_currency}"}
            
            # 先换算为 USD，再换算为目标货币
            usd_amount = amount / exchange_rates[from_currency]
            converted_amount = usd_amount * exchange_rates[to_currency]
            
            result = {
                "original_amount": amount,
                "from_currency": from_currency,
                "to_currency": to_currency,
                "converted_amount": round(converted_amount, 2),
                "exchange_rate": round(exchange_rates[to_currency] / exchange_rates[from_currency], 4),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"【货币换算】结果：{result['converted_amount']} {to_currency}")
            return result
            
        except Exception as e:
            logger.error(f"【货币换算失败】{str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def calculate(expression: str) -> Dict[str, Any]:
        """
        计算数学表达式
        
        Args:
            expression: 待求值的数学表达式
            
        Returns:
            包含计算结果的字典
        """
        try:
            logger.info(f"【计算器】正在计算表达式：{expression}")
            
            # 清洗表达式——仅允许安全的数学运算
            allowed_names = {
                k: v for k, v in math.__dict__.items() if not k.startswith("__")
            }
            allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
            
            # 替换常见运算符以保证兼容性
            expression = expression.replace("^", "**")
            
            # 执行表达式求值
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            
            return {
                "expression": expression,
                "result": result,
                "type": type(result).__name__
            }
            
        except Exception as e:
            logger.error(f"【计算器失败】{str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def code_interpreter(code: str) -> Dict[str, Any]:
        """
        执行 Python 代码以进行复杂计算与数据处理
        
        Args:
            code: 待执行的 Python 代码
            
        Returns:
            包含执行结果与输出的字典
        """
        try:
            logger.info(f"【代码解释器】正在执行 Python 代码（前 100 字符）：{code[:100]}...")
            
            # 创建包含安全内置函数的受限命名空间
            safe_namespace = {
                '__builtins__': {
                    'abs': abs,
                    'all': all,
                    'any': any,
                    'sum': sum,
                    'min': min,
                    'max': max,
                    'round': round,
                    'len': len,
                    'list': list,
                    'dict': dict,
                    'set': set,
                    'tuple': tuple,
                    'enumerate': enumerate,
                    'zip': zip,
                    'map': map,
                    'filter': filter,
                    'sorted': sorted,
                    'reversed': reversed,
                    'range': range,
                    'int': int,
                    'float': float,
                    'str': str,
                    'bool': bool,
                    'print': print,
                }
            }
            
            # 添加 math 模块
            safe_namespace['math'] = math
            
            # 捕获打印输出
            import io
            import contextlib
            
            output_buffer = io.StringIO()
            
            with contextlib.redirect_stdout(output_buffer):
                # 执行代码
                exec(code, safe_namespace)
            
            # 获取打印输出
            printed_output = output_buffer.getvalue()
            
            # 若赋值给 'result' 变量则尝试提取结果
            result = safe_namespace.get('result', None)
            
            # 同时检查常见的变量名
            if result is None:
                for var_name in ['total', 'sum', 'output', 'answer', 'final']:
                    if var_name in safe_namespace:
                        result = safe_namespace[var_name]
                        break
            
            # 获取所有定义的变量（排除内置对象与模块）
            variables = {
                k: v for k, v in safe_namespace.items() 
                if not k.startswith('__') and k not in ['math'] and not callable(v)
            }
            
            return {
                "code": code,
                "result": result,
                "output": printed_output if printed_output else None,
                "variables": variables,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"【代码解释器失败】{str(e)}")
            return {
                "code": code,
                "error": str(e),
                "success": False
            }


class ContextAwareAgent:
    """
    具备可配置 LLM 提供商与上下文消融模式的 AI Agent
    """
    
    def __init__(self, api_key: str, context_mode: ContextMode = ContextMode.FULL, 
                 provider: str = "siliconflow", model: Optional[str] = None, 
                 verbose: bool = True,
                 hidden_result_content: str = HIDDEN_RESULT_EMPTY):
        """
        初始化 Agent
        
        Args:
            api_key: LLM 提供商的 API 密钥
            context_mode: 消融实验的上下文模式
            provider: 在 ``agentbook.providers`` 中注册的任意提供商（例如
                ``dashscope``/``qwen``、``siliconflow``、``doubao``、
                ``kimi``、``deepseek`` 或 ``openrouter``）
            model: 可选的模型重载
            verbose: 若为 True 则记录完整 HTTP 请求与响应（默认：True）
            hidden_result_content: 在 NO_TOOL_RESULTS 消融模式下替换工具结果的内容。
                默认为静默扣留（即完全移除结果）；传入 :data:`HIDDEN_RESULT_MARKER`
                可保留可见的遮盖标记。详见 :data:`HIDDEN_RESULT_STYLES`。
        """
        self.provider = provider.lower()
        self.verbose = verbose
        self.hidden_result_content = hidden_result_content

        # Base URL、默认模型与密钥查询均存放在共享注册表（agentbook/providers.py）中，
        # 因此在此处添加提供商即可直接使用，无需修改本处代码。resolve_backend 还应用了通用的
        # OpenRouter 兜底：当提供商自身密钥缺失但配置了 OPENROUTER_API_KEY 时，请求将通过 OpenRouter 路由，
        # 并使用映射后的模型 ID。当提供商密钥已配置时行为保持不变。
        from config import resolve_backend
        backend = resolve_backend(self.provider, model=model, api_key=api_key)
        resolved_key = backend.api_key
        resolved_base_url = backend.base_url
        self.model = backend.model
        self.using_openrouter = backend.using_openrouter
        if self.using_openrouter:
            logger.info(
                f"【路由说明】未设置 {self.provider} 的直连密钥；改经 OpenRouter 调用 "
                f"（模型：{self.model}）。"
            )
        self.client = OpenAI(
            api_key=resolved_key,
            base_url=resolved_base_url
        )
        self.base_url = resolved_base_url
        
        self.context_mode = context_mode
        self.trajectory = AgentTrajectory(context_mode=context_mode)
        self.tools = ToolRegistry()
        
        # 初始化对话历史
        self.conversation_history = []
        self._init_system_prompt()
        
        logger.info(
            f"【Agent 初始化】提供商={self.provider}，模型={self.model}，"
            f"上下文模式={context_mode.value}，详细输出={self.verbose}。"
        )
    
    def _init_system_prompt(self):
        """初始化会话的系统提示词"""
        self.conversation_history = [
            {
                "role": "system",
                "content": """You are an intelligent assistant with access to tools. 

Your task is to solve the given problems using the available tools. Think step by step and use tools as needed.

Important: When you have gathered all necessary information and computed the final answer, clearly state "FINAL ANSWER:" followed by your answer."""
            }
        ]
    
    def _get_tools_description(self) -> List[Dict[str, Any]]:
        """获取模型的工具描述"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "parse_pdf",
                    "description": "Download and parse a PDF document from a URL or a file path to extract text content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL or file path of the PDF document to parse"
                            }
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "convert_currency",
                    "description": "Convert an amount from one currency to another using current exchange rates",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "amount": {
                                "type": "number",
                                "description": "The amount to convert"
                            },
                            "from_currency": {
                                "type": "string",
                                "description": "The source currency code (e.g., USD, EUR)"
                            },
                            "to_currency": {
                                "type": "string",
                                "description": "The target currency code (e.g., USD, EUR)"
                            }
                        },
                        "required": ["amount", "from_currency", "to_currency"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Evaluate a simple mathematical expression",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression to evaluate (e.g., '2 + 2 * 3')"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "code_interpreter",
                    "description": "Execute Python code for complex calculations, data processing, and computing totals. Use this for tasks like: summing lists of values, calculating percentages, aggregating financial data, performing multi-step calculations, or any computation requiring variables and intermediate steps.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute. Can use variables, loops, and mathematical operations. Example: 'amounts = [2500000, 2278481, 2541806, 2282609, 2388060]; total = sum(amounts); print(f\"Total: ${total:,.2f}\")"
                            }
                        },
                        "required": ["code"]
                    }
                }
            }
        ]
    
    def _prepare_assistant_message(self, message) -> Dict[str, Any]:
        """
        准备待添加至消息列表的 assistant 消息，
        在 NO_REASONING 模式下过滤掉 reasoning_content
        
        Args:
            message: assistant 消息对象
            
        Returns:
            消息的字典表示
        """
        msg_dict = message.dict() if hasattr(message, 'dict') else message.model_dump()
        
        # 在 NO_REASONING 模式下移除 reasoning_content
        if self.context_mode == ContextMode.NO_REASONING and 'reasoning_content' in msg_dict:
            msg_dict.pop('reasoning_content')
            
        return msg_dict

    @staticmethod
    def _reasoning_content(message) -> Optional[str]:
        """获取提供商的推理文本，兼容不同的 SDK 数据结构"""
        value = getattr(message, "reasoning_content", None)
        if value:
            return str(value)
        extra = getattr(message, "model_extra", None) or {}
        value = extra.get("reasoning_content") or extra.get("reasoning")
        if isinstance(value, dict):
            value = value.get("content") or value.get("text")
        return str(value) if value else None

    @staticmethod
    def _json_snapshot(value: Any) -> Any:
        """对 API 证据对象进行解耦快照，防止后续内存修改对其产生影响"""
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    
    def _build_context(self) -> str:
        """
        构建执行轨迹的可读摘要（旧版辅助方法，仅保留用于检查/调试）。

        注意：发送给模型的消息列表由 ``_prepare_messages_for_api`` 组装——那才是
        NO_HISTORY 消融实际生效的地方。本方法不处于请求路径中。

        Returns:
            面向模型的上下文文本
        """
        context_parts = []
        
        # 若未禁用则添加推理步骤
        if self.context_mode != ContextMode.NO_REASONING and self.trajectory.reasoning_steps:
            context_parts.append("## Previous Reasoning Steps:")
            for step in self.trajectory.reasoning_steps:
                context_parts.append(f"- {step}")
            context_parts.append("")
        
        # 若未禁用则添加工具调用历史
        if self.context_mode not in [ContextMode.NO_HISTORY, ContextMode.NO_TOOL_CALLS] and self.trajectory.tool_calls:
            context_parts.append("## Tool Call History:")
            for call in self.trajectory.tool_calls:
                if self.context_mode != ContextMode.NO_TOOL_CALLS:
                    context_parts.append(f"- Called {call.tool_name} with args: {json.dumps(call.arguments)}")
                if self.context_mode != ContextMode.NO_TOOL_RESULTS and call.result:
                    context_parts.append(f"  Result: {json.dumps(call.result, indent=2)}")
            context_parts.append("")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _log_request_response(self, request_data: Dict[str, Any], response_data: Any, iteration: int):
        """
        详细模式下记录完整请求与响应
        
        Args:
            request_data: 发送给 API 的请求载荷
            response_data: 从 API 收到的响应数据
            iteration: 当前迭代轮次
        """
        if not self.verbose:
            return
            
        if request_data:
            print("\n" + "="*80)
            print(f"【请求载荷｜第 {iteration} 轮】以下 JSON 是实际发给模型的请求，不含 API Key。")
            print("-"*80)
            print(json.dumps(request_data, indent=2, ensure_ascii=False))
        
        if response_data:
            print("\n" + "="*80)
            print(f"【模型响应｜第 {iteration} 轮】以下 JSON 是模型返回，用于观察工具调用或最终回答。")
            print("-"*80)
        
            # 将响应转换为字典以供展示
            if hasattr(response_data, 'model_dump'):
                response_dict = response_data.model_dump()
            elif hasattr(response_data, 'dict'):
                response_dict = response_data.dict()
            else:
                response_dict = {"raw_response": str(response_data)}

            print(json.dumps(response_dict, indent=2, ensure_ascii=False))
            print("="*80 + "\n")
    
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        执行工具并返回结果
        
        Args:
            tool_name: 待执行的工具名称
            arguments: 传入工具的参数
            
        Returns:
            工具执行结果
        """
        tool_map = {
            "parse_pdf": self.tools.parse_pdf,
            "convert_currency": self.tools.convert_currency,
            "calculate": self.tools.calculate,
            "code_interpreter": self.tools.code_interpreter
        }
        
        if tool_name not in tool_map:
            return {"error": f"Unknown tool: {tool_name}"}

        return tool_map[tool_name](**arguments)

    def _prepare_messages_for_api(self) -> List[Dict[str, Any]]:
        """
        构建当前迭代实际发送给模型的消息列表，应用 NO_HISTORY 消融。

        除 NO_HISTORY 外的所有模式均原样返回完整对话历史（累积的轨迹）。
        对于 NO_HISTORY，请求仅包含静态系统提示词和当前用户任务。
        前序轮次的 assistant 决策、工具调用或工具结果均不予保留。
        这就是实验 1-1 的字面消融：模型在每次推理时重新开始任务，因此往往会重复执行相同的首个动作。
        单步滑动窗口依然属于历史，会实质性缩小书稿中描述的实验效果。

        Returns:
            本次迭代发送给模型的消息列表。
        """
        messages = self.conversation_history
        if self.context_mode != ContextMode.NO_HISTORY:
            return messages

        # 系统提示词始终保留作为静态前缀
        windowed = [m for m in messages if m.get("role") == "system"]

        # 锚定在最新的用户任务上。其后的所有消息均不保留：那些消息正是被消融的前序轮次历史。
        user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if not user_indices:
            return windowed
        last_user_idx = user_indices[-1]
        windowed.append(messages[last_user_idx])
        return windowed

    @staticmethod
    def _extract_final_answer(content: str) -> Optional[str]:
        """若存在 FINAL ANSWER: 则提取其后的文本；否则返回 None。"""
        if not content or "FINAL ANSWER:" not in content:
            return None
        return content.split("FINAL ANSWER:", 1)[1].strip()

    def execute_task(self, task: str, max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        使用可用工具执行任务（ReAct 循环）。

        终止条件：
          1. 模型给出了纯文本回复（无 tool_calls）——无论是闲聊对话还是任务完成，
             包括像 "hi" 这样省略 FINAL ANSWER: 标记的普通回复；或
          2. 达到了 max_iterations 上限（工具调用循环的安全阈值，如 no_tool_results 消融组）。

        Args:
            task: 待执行的任务
            max_iterations: 最大 ReAct 步数（默认为 Config.MAX_ITERATIONS 或 10）。
                这是一个安全上限，并非目标轮数。

        Returns:
            任务执行结果字典

        结果语义：
          - ``completed`` 表示循环收到了非空的最终文本响应，并不代表所请求任务的结果是正确的。
          - ``task_success`` 在此处为 ``None``，因为正确性取决于特定任务，无法从任意自然语言提示中推断。
            有已知评分规则的调用方应根据最终答案和轨迹计算此值。
          - ``success`` 作为向后兼容别名保留，等价于 ``completed``。新代码应使用 ``completed``
            或特定任务的 ``task_success``。
        """
        if max_iterations is None:
            try:
                from config import Config
                max_iterations = Config.MAX_ITERATIONS
            except Exception:
                max_iterations = 10

        # 将用户消息添加至对话历史
        self.conversation_history.append({"role": "user", "content": task})
        
        # 直接使用对话历史（无需复制）
        messages = self.conversation_history
        
        iteration = 0
        final_answer = None
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(
                f"【执行轮次】第 {iteration}/{max_iterations} 轮："
                "组装当前上下文并准备请求模型。"
            )
            
            try:
                # 构建实际发送给模型的消息列表。除 NO_HISTORY 模式外均等于完整轨迹；
                # NO_HISTORY 模式下丢弃前序步骤。
                api_messages = self._prepare_messages_for_api()

                # 准备用于日志记录的请求数据
                request_data = {
                    "model": self.model,
                    "messages": api_messages,
                    "temperature": _reasoning_safe_temperature(self.model, 0.3),
                    "max_tokens": 8192
                }

                if self.context_mode != ContextMode.NO_TOOL_CALLS:
                    request_data["tools"] = self._get_tools_description()
                    request_data["tool_choice"] = "auto"

                # DeepSeek V4：启用思考功能以便 reasoning_content 存在，用于 no_reasoning 消融（对齐 Doubao/Kimi 的默认思考模式）。
                # 通过 OpenRouter 路由时跳过，因其可能不接受相同的 extra_body 结构。
                create_kwargs = {
                    "model": self.model,
                    "messages": api_messages,
                    "tools": self._get_tools_description() if self.context_mode != ContextMode.NO_TOOL_CALLS else None,
                    "tool_choice": "auto" if self.context_mode != ContextMode.NO_TOOL_CALLS else None,
                    "temperature": _reasoning_safe_temperature(self.model, 0.3),
                    "max_tokens": 8192,
                    "timeout": 180,  # 主执行过程超时设为 180 秒
                }
                if self.provider == "deepseek" and not getattr(self, "using_openrouter", False):
                    create_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
                    request_data["thinking"] = {"type": "enabled"}

                logger.info(
                    f"【模型请求】向 {self.provider} API 发送第 {iteration} 轮请求；"
                    f"上下文模式={self.context_mode.value}。"
                )

                # 携带工具调用模型
                response = self.client.chat.completions.create(**create_kwargs)

                response_dict = (
                    response.model_dump() if hasattr(response, "model_dump")
                    else response.dict() if hasattr(response, "dict")
                    else {"raw_response": str(response)}
                )
                logger.info(
                    f"【模型响应】已收到第 {iteration} 轮响应；"
                    "正在判断工具调用或最终回答。"
                )
                self.trajectory.api_turns.append({
                    "iteration": iteration,
                    "provider": self.provider,
                    "resolved_model": self.model,
                    "base_url": self.base_url,
                    "using_openrouter": bool(getattr(self, "using_openrouter", False)),
                    "request": self._json_snapshot(request_data),
                    "response": self._json_snapshot(response_dict),
                })
                
                # 详细模式下打印响应
                if self.verbose:
                    self._log_request_response(request_data, response, iteration)
                
                message = response.choices[0].message
                has_tool_calls = bool(getattr(message, "tool_calls", None))
                reasoning_content = self._reasoning_content(message)
                if reasoning_content:
                    self.trajectory.reasoning_steps.append(reasoning_content)

                # --- 终止分支：仅返回文本且无工具调用 ---
                # 正常的单轮对话（"hi" -> "Hello!"）或未带 FINAL ANSWER: 标记的任务回答均应结束 ReAct 循环。
                # 此前仅 "FINAL ANSWER:" 能跳出循环，导致普通回复会反复重发直到达到 max_iterations（浪费 API 调用）。
                if not has_tool_calls:
                    assistant_msg = self._prepare_assistant_message(message)
                    messages.append(assistant_msg)
                    content = (message.content or "").strip()
                    if content:
                        marked = self._extract_final_answer(content)
                        final_answer = marked if marked is not None else content
                        logger.info(
                            "【终止条件】模型返回纯文本且没有工具调用；"
                            f"第 {iteration} 轮结束执行。"
                        )
                    else:
                        logger.warning(
                            "【异常响应】模型返回空文本且没有工具调用；"
                            "为避免耗尽剩余轮次，停止执行。"
                        )
                    break

                # --- 继续分支：模型请求执行工具 ---
                assistant_msg = self._prepare_assistant_message(message)
                messages.append(assistant_msg)
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    raw_args = tool_call.function.arguments or "{}"
                    try:
                        function_args = json.loads(raw_args)
                    except json.JSONDecodeError as exc:
                        # 工具参数 JSON 异常时保持会话继续
                        err = (
                            f"Invalid tool arguments (not valid JSON): {exc}. "
                            f"Raw arguments: {raw_args[:500]}"
                        )
                        logger.warning(f"【工具参数错误】{err}")
                        self.trajectory.tool_calls.append(ToolCall(
                            tool_name=function_name,
                            arguments={},
                            result={"error": err},
                        ))
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps({"error": err}),
                        })
                        continue

                    logger.info(
                        f"【工具执行】模型请求调用 {function_name}；参数：{function_args}"
                    )

                    result = self._execute_tool(function_name, function_args)

                    tool_call_record = ToolCall(
                        tool_name=function_name,
                        arguments=function_args,
                        result=result
                    )
                    self.trajectory.tool_calls.append(tool_call_record)

                    if self.context_mode != ContextMode.NO_TOOL_RESULTS:
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            # default=str: code_interpreter 在 variables 中返回原始命名空间，
                            # 可能包含 set、dict views 等无法被 json 编码的对象——这绝不能中断整个任务。
                            "content": json.dumps(result, default=str)
                        }
                    else:
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": self.hidden_result_content
                        }
                    messages.append(tool_msg)

                # 若同轮中也标记了 FINAL ANSWER:（在工具调用中较罕见），仍优先在记录工具后提取
                if message.content and "FINAL ANSWER:" in message.content:
                    final_answer = self._extract_final_answer(message.content)
                    logger.info(
                        "【终止条件】模型在工具调用同轮给出 FINAL ANSWER；"
                        "已记录答案并结束执行。"
                    )
                    break

                # 注意：我们不再修改系统提示词。上下文已通过工具历史融入对话中。
                    
            except TimeoutError:
                logger.error("【请求超时】模型请求超过 60 秒，已停止当前任务。")
                return {
                    "error": "Request timed out. The model is taking too long to respond. Try a simpler task or different provider.",
                    "trajectory": self.trajectory,
                    "iterations": iteration,
                    "completed": False,
                    "task_success": False,
                    "success": False,
                    **self._backend_identity(),
                }
            except Exception as e:
                safe_error = _safe_provider_error(e)
                logger.error(f"【任务执行错误】{safe_error}")
                self.trajectory.api_turns.append({
                    "iteration": iteration,
                    "provider": self.provider,
                    "resolved_model": self.model,
                    "base_url": self.base_url,
                    "using_openrouter": bool(getattr(self, "using_openrouter", False)),
                    "error": {"class": type(e).__name__, "message": safe_error},
                })
                # 检查是否为超时相关错误
                if "timeout" in safe_error.lower() or "timed out" in safe_error.lower():
                    return {
                        "error": "Request timed out. The model is taking too long to respond. Try a simpler task or different provider.",
                        "trajectory": self.trajectory,
                        "iterations": iteration,
                        "completed": False,
                        "task_success": False,
                        "success": False,
                        **self._backend_identity(),
                    }
                return {
                    "error": safe_error,
                    "trajectory": self.trajectory,
                    "iterations": iteration,
                    "completed": False,
                    "task_success": False,
                    "success": False,
                    **self._backend_identity(),
                }
        completed = bool(final_answer and str(final_answer).strip())
        return {
            "final_answer": final_answer,
            "trajectory": self.trajectory,
            "iterations": iteration,
            "completed": completed,
            "task_success": None,
            # 向后兼容别名。这是终止响应状态，并非正确性判断。
            "success": completed,
            **self._backend_identity(),
        }
    
    def _backend_identity(self) -> Dict[str, Any]:
        """返回做出响应（或失败）的端点信息。

        失败的实验分支同样是证据，若证据未标明请求的模型则无法进行审计。
        成功路径内联报告该信息；错误路径通过此处返回，从而使错误模型 ID 引起的 404 错误
        在记录中保持清晰可辨，而非显示为 null。

        Returns:
            包含提供商、解析后模型、base URL 和 OpenRouter 标识的字典。
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "using_openrouter": bool(getattr(self, "using_openrouter", False)),
        }

    def reset(self):
        """重置 Agent 的执行轨迹与对话历史"""
        self.trajectory = AgentTrajectory(context_mode=self.context_mode)
        self._init_system_prompt()  # 使用系统提示词重新初始化会话
        logger.info("【会话重置】已清空 Agent 轨迹和对话历史。")
    
    def process(self, query: str, max_iterations: Optional[int] = None) -> str:
        """
        处理查询并以字符串形式返回最终答案
        
        Args:
            query: 待处理的查询
            max_iterations: 最大 ReAct 步数（默认取自 Config）
            
        Returns:
            作为字符串的最终答案
        """
        result = self.execute_task(query, max_iterations)
        if result.get('final_answer'):
            return result['final_answer']
        elif result.get('error'):
            return f"Error: {result['error']}"
        else:
            return "No answer found"
