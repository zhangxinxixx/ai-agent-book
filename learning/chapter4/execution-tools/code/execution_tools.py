"""通用执行工具：代码解释器（code_interpreter）与虚拟终端（virtual_terminal）。"""

import os
import subprocess
import sys
import io
import tempfile
import traceback
from typing import Dict, Any, Optional, Tuple
from contextlib import redirect_stdout, redirect_stderr
from llm_helper import LLMHelper
from config import Config
from multilang_executor import LanguageExecutor, ExecutionStatus

# 长输出处理阈值（对应书中第 4 章“长输出的截断与持久化”）。
# 当工具执行输出超过行数或字符数阈值时，上下文中仅保留前 head_lines 行和后 tail_lines 行，
# 并将完整输出持久化保存到临时文件中，防止撑爆大模型的上下文窗口（Context Window），同时便于后续按需读取。
MAX_OUTPUT_LINES = 200
MAX_OUTPUT_CHARS = 10000
HEAD_LINES = 50
TAIL_LINES = 50


def truncate_and_persist(
    text: str,
    tool_name: str = "execution",
    max_lines: int = MAX_OUTPUT_LINES,
    max_chars: int = MAX_OUTPUT_CHARS,
    head_lines: int = HEAD_LINES,
    tail_lines: int = TAIL_LINES,
) -> Tuple[str, Optional[str]]:
    """截断超长输出并将完整文本持久化保存到临时文件。

    返回元组 (处理后的文本, 保存的文件路径)。
    - 若输出未超过阈值，原样返回，保存路径为 None。
    - 若超过阈值，上下文中仅保留前 head_lines 行与后 tail_lines 行，
      并在中间插入提示信息告知完整输出落盘路径。
    该机制在不丢失任何信息的前提下保证了 Agent 上下文的边界可控，且完全不依赖 LLM 调用（离线安全）。
    """
    if text is None:
        return text, None

    lines = text.split("\n")
    if len(text) <= max_chars and len(lines) <= max_lines:
        return text, None

    # 将完整输出持久化落盘，供 Agent 后续通过 read_file 按需查阅
    fd, path = tempfile.mkstemp(prefix=f"{tool_name}_output_", suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)

    # 在 Python 中 lines[-0:] 会切出整个列表，因此当 tail_lines 为 0 时特殊处理为保留 0 行
    head_n = max(0, head_lines)
    tail_n = max(0, tail_lines)
    head_part = lines[:head_n] if head_n else []
    tail_part = lines[-tail_n:] if tail_n else []
    omitted = max(len(lines) - head_n - tail_n, 0)

    guide = f"[如需完整输出，请使用 read_file 工具读取 {path}]"
    if omitted == 0:
        # 头尾切片已完全覆盖整个内容，不再重复拼接重叠部分
        truncated = "\n".join(lines + [guide])
    else:
        middle = f"... [省略 {omitted} 行，完整输出已保存至 {path}] ..."
        truncated = "\n".join(head_part + [middle] + tail_part + [guide])
    return truncated, path


class ExecutionTools:
    """通用执行工具集：集成安全审查、语法验证与执行结果分析。"""
    
    def __init__(self, llm_helper: LLMHelper):
        """初始化执行工具，注入 LLM 辅助类并实例化多语言执行器。"""
        self.llm_helper = llm_helper
        self.lang_executor = LanguageExecutor(workspace_dir=Config.WORKSPACE_DIR)
    
    async def code_interpreter(
        self,
        code: str,
        language: str = "python",
        timeout: float = 30.0,
        stdin: Optional[str] = None,
        files: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """在沙盒环境中执行多语言代码。

        参数:
            code: 待执行的代码字符串
            language: 编程语言（python, javascript, typescript, go, java, cpp, rust, php, bash）
            timeout: 执行超时时间（秒）
            stdin: 可选的标准输入数据
            files: 可选的附加文件字典（文件名 -> 内容）

        返回:
            包含执行状态、标准输出、标准错误和分析结果的字典
        """
        if language is None:
            language = "python"
        language = language.lower()
        
        # 1. 语法预校验（本地离线快速校验，目前针对 Python 使用 compile 拦截语法错误代码）
        if Config.AUTO_VERIFY_CODE and language in ['python', 'python3']:
            is_valid, error_msg = self.llm_helper.verify_code_syntax(code, language)
            if not is_valid:
                return {
                    "success": False,
                    "error": f"语法错误：{error_msg}",
                    "verification": "failed",
                    "language": language
                }
        
        # 2. 危险模式匹配与安全审查（匹配潜在高危代码，交由 LLM 事前审批或离线安全拦截）
        if Config.REQUIRE_APPROVAL_FOR_DANGEROUS_OPS:
            dangerous_patterns = {
                'python': ['os.system', 'subprocess', 'eval', 'exec', 'open(', '__import__', 'compile'],
                'bash': ['rm -rf', 'dd if=', 'mkfs', '> /dev/', 'curl', 'wget'],
                'php': ['exec(', 'system(', 'shell_exec(', 'passthru(', 'eval('],
            }
            
            patterns = dangerous_patterns.get(language, [])
            detected = [p for p in patterns if p in code]
            
            if detected:
                approved, reason = self.llm_helper.request_approval(
                    "code_execution",
                    {
                        "code": code,
                        "language": language,
                        "detected_patterns": detected
                    }
                )
                
                if not approved:
                    return {
                        "success": False,
                        "error": f"执行未获批准：{reason}",
                        "language": language
                    }
        
        # 3. 调用多语言执行器在隔离环境/沙盒中执行代码
        try:
            result = await self.lang_executor.execute_code(
                code=code,
                language=language,
                timeout=timeout,
                stdin=stdin,
                files=files
            )
            
            # 将执行器返回的状态码转换为布尔成功标志
            # 正式执行器以 status 枚举表示结果；测试替身或兼容执行器也可能直接返回
            # success 布尔值，两种形式都接受。
            success = (
                result.get('status') == ExecutionStatus.SUCCESS
                or result.get('success') is True
            )
            
            # 4. 长输出处理：截断头尾行，并将全量输出持久化保存到磁盘（离线安全），可选 LLM 摘要
            stdout = result.get('stdout', '')
            stderr = result.get('stderr', '')
            stdout, stdout_file = truncate_and_persist(stdout, "code_interpreter")
            stderr, stderr_file = truncate_and_persist(stderr, "code_interpreter")

            if Config.AUTO_SUMMARIZE_COMPLEX_OUTPUT and len(stdout) > MAX_OUTPUT_CHARS:
                stdout = self.llm_helper.summarize_output("code_interpreter", stdout)
            if Config.AUTO_SUMMARIZE_COMPLEX_OUTPUT and len(stderr) > MAX_OUTPUT_CHARS:
                stderr = self.llm_helper.summarize_output("code_interpreter", stderr)

            # 失败时至少返回一份本地可用的诊断，不强制请求 LLM。上游 Agent 可以先
            # 使用该字段决定是否重试，再按需调用更昂贵的模型分析能力。
            normalized_error = result.get('error')
            if not success and not normalized_error:
                normalized_error = stderr.strip() or f"进程退出码为 {result.get('returncode')}"

            error_analysis = None
            if not success:
                error_analysis = normalized_error

            return {
                "success": success,
                "status": result.get('status'),
                "language": result.get('language', language),
                "stdout": stdout,
                "stderr": stderr,
                "stdout_file": stdout_file,
                "stderr_file": stderr_file,
                "returncode": result.get('returncode'),
                "error": normalized_error,
                "compile_output": result.get('compile_output'),
                "phase": result.get('phase'),
                "execution_time": result.get('execution_time'),
                "sandbox": result.get('sandbox'),
                "verification": "passed" if Config.AUTO_VERIFY_CODE else "skipped",
                "error_analysis": error_analysis
            }
            
        except Exception as e:
            error_output = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return {
                "success": False,
                "error": error_output,
                "language": language
            }
    
    async def virtual_terminal(
        self,
        command: str,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """在虚拟终端中执行 Shell 命令。

        参数:
            command: 要执行的 Shell 命令
            timeout: 超时时间（秒）

        返回:
            包含执行成功标志、返回值、标准输出、标准错误及落盘文件路径的字典
        """
        # 1. 危险命令检测（rm -rf、格式化等高危指令拦截并触发事前审批）
        if Config.REQUIRE_APPROVAL_FOR_DANGEROUS_OPS:
            dangerous_commands = [
                'rm -rf', 'dd', 'mkfs', 'format',
                '> /dev/', 'chmod -R', 'chown -R'
            ]
            
            if any(dangerous in command for dangerous in dangerous_commands):
                approved, reason = self.llm_helper.request_approval(
                    "terminal_command",
                    {
                        "command": command,
                        "detected_patterns": [p for p in dangerous_commands if p in command]
                    }
                )
                
                if not approved:
                    return {
                        "success": False,
                        "error": f"命令执行未获批准：{reason}"
                    }
        
        # 2. 执行 Shell 命令（受限工作目录与超时控制）
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Config.WORKSPACE_DIR
            )
            
            stdout = result.stdout
            stderr = result.stderr

            # 3. 长输出处理：截断保留头尾行，并将全量输出持久化落盘到临时文件
            stdout, stdout_file = truncate_and_persist(stdout, "virtual_terminal")
            stderr, stderr_file = truncate_and_persist(stderr, "virtual_terminal")

            if Config.AUTO_SUMMARIZE_COMPLEX_OUTPUT:
                if len(stdout) > MAX_OUTPUT_CHARS:
                    stdout = self.llm_helper.summarize_output(
                        "virtual_terminal",
                        stdout
                    )
                if len(stderr) > MAX_OUTPUT_CHARS:
                    stderr = self.llm_helper.summarize_output(
                        "virtual_terminal",
                        stderr
                    )

            response = {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_file": stdout_file,
                "stderr_file": stderr_file,
                # 本地 stderr 是最可靠的第一手诊断；不因命令失败而自动花费模型调用。
                "error_analysis": (
                    None if result.returncode == 0
                    else stderr.strip() or f"命令退出码为 {result.returncode}"
                )
            }
            return response
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"命令执行超过 {timeout} 秒，已超时终止"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"命令执行失败：{str(e)}"
            }
