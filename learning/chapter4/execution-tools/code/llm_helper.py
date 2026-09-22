"""封装执行工具所需的 LLM 审批、总结、错误分析与语法校验。"""

import json
import datetime as dt
import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any
from openai import OpenAI
from config import Config


def _reasoning_safe_temperature(model, requested=1.0):
    """推理模型只接受 temperature=1；其他模型保留调用方请求值。"""
    m = str(model or "").lower().replace("/", "-")
    return 1 if ("kimi-k3" in m or "gpt-5" in m) else requested


def _parse_json_response(content):
    """从模型回复中提取 JSON，并兼容可选的 Markdown 代码围栏。"""
    text = (content or "").strip()
    if text.startswith("```"):
        # 去掉开头的 ``` / ```json 与结尾围栏。
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


class LLMHelper:
    """集中管理所有可能产生模型请求的辅助操作。"""
    
    def __init__(self):
        """初始化惰性客户端。

        构造对象时不会请求模型；只有审批、总结或非 Python 语法检查等方法第一次
        真正使用 LLM 时才创建客户端，因此基础文件和本地执行路径可离线运行。
        """
        self.client = None
        self.model = None
        self.provider = None

    def _record_receipt(self, purpose: str, request: dict, response, latency: float) -> None:
        """按需记录不含凭据的 Provider 请求回执，供正式实验审计。"""
        target = os.getenv("EXECUTION_LLM_RECEIPT_PATH")
        if not target:
            return
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        usage = getattr(response, "usage", None)
        choice = response.choices[0]
        row = {
            "purpose": purpose,
            "called_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "provider": self.provider,
            "request": request,
            "response": {
                "id": getattr(response, "id", None),
                "model": getattr(response, "model", None),
                "finish_reason": getattr(choice, "finish_reason", None),
                "content": choice.message.content,
            },
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            },
            "latency_seconds": round(latency, 3),
        }
        existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
        existing.append(row)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _ensure_client(self) -> None:
        """第一次需要模型时创建客户端；缺少 API Key 会抛出异常。"""
        if self.client is None:
            llm_config = Config.get_llm_config()
            # 当前支持的 Provider 都通过 OpenAI 兼容接口调用。
            self.client = OpenAI(
                api_key=llm_config["api_key"],
                base_url=llm_config.get("base_url")
            )
            self.model = llm_config["model"]
            self.provider = llm_config["provider"]
    
    def request_approval(
        self, 
        operation: str, 
        details: Dict[str, Any]
    ) -> tuple[bool, str]:
        """请求 LLM 审查危险操作，返回 `(是否批准, 原因)`。"""
        prompt = f"""You are a safety reviewer for an AI agent execution system.
Review the following operation and determine if it should be approved.

Operation: {operation}
Details: {json.dumps(details, indent=2)}

Analyze the operation for:
1. Potential data loss or destructive actions
2. Security risks
3. Resource consumption concerns
4. Compliance with best practices

Respond in JSON format:
{{
    "approved": true/false,
    "reason": "Brief explanation of your decision",
    "risk_level": "low/medium/high",
    "recommendations": ["List of recommendations if any"]
}}
"""
        
        try:
            self._ensure_client()
            request = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a cautious safety reviewer. Approve operations that are safe and reject risky ones."
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": _reasoning_safe_temperature(self.model, 0.1),
                "max_tokens": Config.MAX_TOKENS,
            }
            started = time.perf_counter()
            response = self.client.chat.completions.create(**request)
            self._record_receipt("dangerous_operation_review", request, response,
                                 time.perf_counter() - started)

            result = _parse_json_response(response.choices[0].message.content)
            return result["approved"], result["reason"]
            
        except Exception as e:
            # 审批链路异常时默认拒绝，遵循 fail-safe。
            return False, f"Approval check failed: {str(e)}"
    
    def summarize_output(
        self, 
        tool_name: str,
        output: str
    ) -> str:
        """使用 LLM 总结复杂工具输出；失败时返回截断后的原始内容。"""
        
        prompt = f"""Summarize the following output from the '{tool_name}' tool.
Focus on:
1. Key results or findings
2. Errors or warnings
3. Important patterns or insights
4. Actionable information

Output to summarize:
{output[:5000]}  # Limit input to avoid token limits

Provide a concise summary that captures the essential information."""
        
        try:
            self._ensure_client()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at summarizing technical output. Be concise and focus on actionable information."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=_reasoning_safe_temperature(self.model, 0.1),
                max_tokens=Config.MAX_TOKENS
            )

            summary = response.choices[0].message.content
            return f"[SUMMARIZED OUTPUT]\n{summary}\n\n[Original output length: {len(output)} characters]"

        except Exception as e:
            return f"[SUMMARIZATION FAILED: {str(e)}]\n\n{output[:Config.MAX_OUTPUT_LENGTH]}..."
    
    def analyze_error(
        self,
        tool_name: str,
        command: str,
        error_output: str
    ) -> str:
        """使用 LLM 分析错误输出并给出修复建议。"""
        prompt = f"""Analyze the following error from the '{tool_name}' tool:

Command/Code:
{command}

Error Output:
{error_output[:3000]}

Provide:
1. Root cause analysis
2. Suggested fixes
3. Prevention strategies

Be concise and practical."""
        
        try:
            self._ensure_client()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert debugger. Analyze errors and provide clear, actionable solutions."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=_reasoning_safe_temperature(self.model, 0.2),
                max_tokens=Config.MAX_TOKENS
            )

            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error analysis failed: {str(e)}"
    
    def verify_code_syntax(
        self,
        code: str,
        language: str = "python"
    ) -> tuple[bool, Optional[str]]:
        """验证代码语法，返回 `(是否合法, 错误信息)`。"""
        # Python 直接使用 compile() 做确定性的本地校验。
        if language == "python":
            try:
                compile(code, "<string>", "exec")
                return True, None
            except SyntaxError as e:
                return False, f"Syntax error at line {e.lineno}: {e.msg}"

        # JavaScript 使用 Node --check 做确定性校验，不执行代码，也不询问 LLM。
        if language in {"javascript", "js"}:
            try:
                process = subprocess.run(
                    ["node", "--check", "-"], input=code, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                return False, f"JavaScript linter unavailable: {exc}"
            if process.returncode == 0:
                return True, None
            return False, process.stderr.strip() or "JavaScript syntax check failed"
        
        # 其他语言暂时使用 LLM 做基础语法判断。
        prompt = f"""Check the following {language} code for syntax errors:

```{language}
{code}
```

Respond in JSON format:
{{
    "valid": true/false,
    "errors": ["List of syntax errors if any"],
    "warnings": ["List of warnings if any"]
}}
"""
        
        try:
            self._ensure_client()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a {language} syntax validator. Check code for syntax errors."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=_reasoning_safe_temperature(self.model, 0.1),
                max_tokens=Config.MAX_TOKENS
            )
            
            result = _parse_json_response(response.choices[0].message.content)
            if result["valid"]:
                return True, None
            else:
                return False, "; ".join(result["errors"])
                
        except Exception as e:
            # 校验服务本身失败时不伪装成“代码有错”，交给后续编译/执行阶段判断。
            return True, None
