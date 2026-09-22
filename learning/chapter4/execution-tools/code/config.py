"""执行工具 MCP Server 的配置管理。"""

import os
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# 导入当前目录 `.env` 中的环境变量；不会覆盖进程中已经存在的同名变量。
load_dotenv()


def _env_int(name: str, default: int) -> int:
    """读取整数环境变量；格式错误时打印警告并回退到默认值。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"警告：invalid {name}={raw!r}，不是整数，改用默认值 {default}",
              file=sys.stderr)
        return default


def _env_float(name: str, default: float) -> float:
    """读取浮点数环境变量；格式错误时打印警告并回退到默认值。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"警告：invalid {name}={raw!r}，不是数字，改用默认值 {default}",
              file=sys.stderr)
        return default


class Config:
    """集中保存 MCP Server、LLM、安全开关和工作区配置。"""
    
    # LLM Provider。只有审批、非 Python 校验或总结等路径真正需要模型。
    PROVIDER: str = os.getenv("PROVIDER", "kimi")
    
    # API Key 只从环境变量读取，不写入日志和返回结果。
    DASHSCOPE_API_KEY: Optional[str] = os.getenv("DASHSCOPE_API_KEY")
    SILICONFLOW_API_KEY: Optional[str] = os.getenv("SILICONFLOW_API_KEY")
    DOUBAO_API_KEY: Optional[str] = os.getenv("DOUBAO_API_KEY")
    KIMI_API_KEY: Optional[str] = os.getenv("KIMI_API_KEY")
    MOONSHOT_API_KEY: Optional[str] = os.getenv("MOONSHOT_API_KEY")
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    DASHSCOPE_BASE_URL: str = os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    # 可选模型名；未设置时使用各 Provider 的默认模型。
    MODEL: Optional[str] = os.getenv("MODEL")
    
    # 模型请求参数。
    TEMPERATURE: float = _env_float("TEMPERATURE", 0.7)
    MAX_TOKENS: int = _env_int("MAX_TOKENS", 4096)
    
    # 外部服务凭据。
    GOOGLE_CALENDAR_CREDENTIALS_FILE: str = os.getenv(
        "GOOGLE_CALENDAR_CREDENTIALS_FILE", 
        "credentials.json"
    )
    GITHUB_TOKEN: Optional[str] = os.getenv("GITHUB_TOKEN")
    
    # 安全开关。
    REQUIRE_APPROVAL_FOR_DANGEROUS_OPS: bool = (
        os.getenv("REQUIRE_APPROVAL_FOR_DANGEROUS_OPS", "true").lower() == "true"
    )
    AUTO_SUMMARIZE_COMPLEX_OUTPUT: bool = (
        os.getenv("AUTO_SUMMARIZE_COMPLEX_OUTPUT", "true").lower() == "true"
    )
    AUTO_VERIFY_CODE: bool = (
        os.getenv("AUTO_VERIFY_CODE", "true").lower() == "true"
    )
    MAX_OUTPUT_LENGTH: int = _env_int("MAX_OUTPUT_LENGTH", 1000)
    
    # 文件工具和终端的默认工作区。
    WORKSPACE_DIR: Path = Path(os.getenv("WORKSPACE_DIR", os.getcwd()))
    
    @classmethod
    def get_api_key(cls, provider: str) -> Optional[str]:
        """根据 Provider 名称读取对应 API Key。"""
        provider = provider.lower()
        provider = {"qwen": "dashscope", "bailian": "dashscope"}.get(provider, provider)
        if provider == "dashscope":
            return cls.DASHSCOPE_API_KEY
        elif provider == "siliconflow":
            return cls.SILICONFLOW_API_KEY
        elif provider == "doubao":
            return cls.DOUBAO_API_KEY
        elif provider in ["kimi", "moonshot"]:
            return cls.KIMI_API_KEY or cls.MOONSHOT_API_KEY
        elif provider == "openrouter":
            return cls.OPENROUTER_API_KEY
        return None
    
    @classmethod
    def effective_provider(cls) -> str:
        """解析实际 Provider；主 Provider 缺少 Key 时允许回退到 OpenRouter。"""
        provider = cls.PROVIDER.lower()
        provider = {"qwen": "dashscope", "bailian": "dashscope"}.get(provider, provider)
        if cls.get_api_key(provider):
            return provider
        if cls.OPENROUTER_API_KEY:
            return "openrouter"
        return provider

    @classmethod
    def validate(cls) -> None:
        """在真正使用 LLM 前验证 Provider 与 API Key。"""
        provider = cls.effective_provider()
        api_key = cls.get_api_key(provider)

        if not api_key:
            raise ValueError(
                f"Provider '{cls.PROVIDER.lower()}' 需要 API Key。"
                f"请设置 {cls.PROVIDER.upper()}_API_KEY 或 OPENROUTER_API_KEY。"
            )

    @classmethod
    def get_llm_config(cls) -> dict:
        """生成 LLMHelper 使用的 Provider 配置字典。"""
        provider = cls.effective_provider()
        api_key = cls.get_api_key(provider)

        if not api_key:
            raise ValueError(
                f"未找到 Provider '{cls.PROVIDER.lower()}' 的 API Key。"
                f"请设置 {cls.PROVIDER.upper()}_API_KEY 或 OPENROUTER_API_KEY。"
            )
        
        if provider == "dashscope":
            return {
                "provider": "dashscope",
                "api_key": api_key,
                "base_url": cls.DASHSCOPE_BASE_URL,
                "model": cls.MODEL or "qwen3.7-plus"
            }
        elif provider == "siliconflow":
            return {
                "provider": "siliconflow",
                "api_key": api_key,
                "base_url": "https://api.siliconflow.cn/v1",
                "model": cls.MODEL or "Qwen/Qwen3-235B-A22B-Thinking-2507"
            }
        elif provider == "doubao":
            return {
                "provider": "doubao",
                "api_key": api_key,
                "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                "model": cls.MODEL or "doubao-seed-1-6-thinking-250715"
            }
        elif provider in ["kimi", "moonshot"]:
            return {
                "provider": "kimi",
                "api_key": api_key,
                "base_url": "https://api.moonshot.cn/v1",
                "model": cls.MODEL or "kimi-k3"
            }
        elif provider == "openrouter":
            return {
                "provider": "openrouter",
                "api_key": api_key,
                "base_url": "https://openrouter.ai/api/v1",
                "model": cls.MODEL or "google/gemini-3.5-flash"
            }
        else:
            raise ValueError(
                f"不支持的 Provider：{provider}。可选值为 "
                "'dashscope'/'qwen'/'bailian'、'siliconflow'、'doubao'、"
                "'kimi'、'moonshot' 或 'openrouter'。"
            )


# 配置采用惰性验证：只有 LLMHelper 真正发起模型请求时才检查 API Key。
# 因此普通文件写入、代码运行和终端命令可以在无 Key 情况下离线使用。
