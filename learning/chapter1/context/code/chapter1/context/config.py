"""
上下文感知 Agent 的配置模块
"""

import os
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def _reasoning_safe_temperature(model, requested=1.0):
    """推理模型（如 Kimi K3, GPT-5 等）仅接受 temperature=1。
    对于这些模型返回 1；否则返回请求的数值，以保持非推理模型（Doubao、DeepSeek、早期 Moonshot 等）不受影响。"""
    m = str(model or "").lower().replace("/", "-")
    return 1 if ("kimi-k3" in m or "gpt-5" in m) else requested


# 提供商解析逻辑存放在共享的 agentbook 包中，以确保各章保持一致；详见 agentbook/providers.py。
# 这里的降级方案使得在未安装 agentbook 包的环境下也能直接运行本实验。
try:
    from agentbook.providers import (
        PROVIDERS,
        SUPPORTED_PROVIDERS,
        canonical_provider,
        canonical_provider as _canonical_provider,
        map_model_to_openrouter,
        resolve_backend,
        resolve_llm_backend,
    )
except ImportError:  # pragma: no cover - 仅在未安装 agentbook 包的环境下执行
    import sys as _sys

    _sys.path.insert(
        0, str(__import__("pathlib").Path(__file__).resolve().parents[2])
    )
    from agentbook.providers import (
        PROVIDERS,
        SUPPORTED_PROVIDERS,
        canonical_provider,
        canonical_provider as _canonical_provider,
        map_model_to_openrouter,
        resolve_backend,
        resolve_llm_backend,
    )


class Config:
    """Agent 配置类"""
    
    # 提供商配置
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "doubao").lower()
    
    # API 配置
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    DASHSCOPE_BASE_URL: str = os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    SILICONFLOW_API_KEY: str = os.getenv("SILICONFLOW_API_KEY", "")
    SILICONFLOW_BASE_URL: str = "https://api.siliconflow.cn/v1"
    
    ARK_API_KEY: str = os.getenv("ARK_API_KEY", "")
    ARK_BASE_URL: str = "https://ark.cn-beijing.volces.com/api/v3"
    
    MOONSHOT_API_KEY: str = os.getenv("MOONSHOT_API_KEY", "")
    MOONSHOT_BASE_URL: str = "https://api.moonshot.cn/v1"

    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
    )

    ZHIPU_API_KEY: str = os.getenv("ZHIPU_API_KEY", "")
    ZHIPU_BASE_URL: str = "https://open.bigmodel.cn/api/paas/v4"
    
    # 模型配置（默认根据提供商设定）
    MODEL_NAME: str = os.getenv("MODEL_NAME", "")  # 若未指定，将根据提供商自动设定
    MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.3"))
    MODEL_MAX_TOKENS: int = int(os.getenv("MODEL_MAX_TOKENS", "1000"))
    
    # Agent 配置
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "10"))
    ENABLE_REASONING: bool = os.getenv("ENABLE_REASONING", "true").lower() == "true"
    
    # 测试配置
    TEST_PDF_URL: str = os.getenv(
        "TEST_PDF_URL",
        "https://www.berkshirehathaway.com/qtrly/1stqtr23.pdf"
    )
    
    # 汇率配置（示例汇率——生产环境请使用实时 API）
    EXCHANGE_RATES = {
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
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE")
    LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    
    # 文件路径
    RESULTS_DIR: str = "results"
    TEST_PDFS_DIR: str = "fixtures/pdfs"
    
    @classmethod
    def get_api_key(cls, provider: str = None) -> str:
        """
        获取指定提供商的 API 密钥
        
        Args:
            provider: 提供商名称（默认为 LLM_PROVIDER）
            
        Returns:
            对应提供商的 API 密钥
        """
        provider = provider or cls.LLM_PROVIDER
        # 共享注册表掌握各提供商的密钥环境变量，因此随着提供商的增加此逻辑始终准确。
        try:
            return PROVIDERS[_canonical_provider(provider)].api_key()
        except KeyError:
            return ""
    
    @classmethod
    def get_default_model(cls, provider: str = None) -> str:
        """
        获取指定提供商的默认模型
        
        Args:
            provider: 提供商名称（默认为 LLM_PROVIDER）
            
        Returns:
            对应提供商的默认模型名称
        """
        provider = provider or cls.LLM_PROVIDER
        provider = provider.lower()
        
        if cls.MODEL_NAME:
            return cls.MODEL_NAME

        try:
            return PROVIDERS[_canonical_provider(provider)].default_model
        except KeyError:
            return ""
    
    @classmethod
    def validate(cls, provider: str = None) -> bool:
        """
        校验必要配置
        
        Args:
            provider: 待校验的提供商（默认为 LLM_PROVIDER）
        
        Returns:
            若配置有效返回 True
        """
        provider = provider or cls.LLM_PROVIDER
        # resolve_backend 已经兼顾了无需密钥的提供商（如 ollama）以及 OpenRouter 兜底，
        # 且其抛出的错误中明确指出了需要设置的具体变量——因此缺少密钥并不是唯一的信号。
        try:
            resolve_backend(provider)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            print("Please set it in .env file or as environment variable")
            return False
        
        return True
    
    @classmethod
    def create_directories(cls):
        """若目录不存在则创建必要目录"""
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
        os.makedirs(cls.TEST_PDFS_DIR, exist_ok=True)
    
    @classmethod
    def get_model_config(cls) -> dict:
        """
        以字典形式获取模型配置
        
        Returns:
            模型配置字典
        """
        return {
            "model": cls.MODEL_NAME,
            "temperature": _reasoning_safe_temperature(cls.MODEL_NAME, cls.MODEL_TEMPERATURE),
            "max_tokens": cls.MODEL_MAX_TOKENS
        }
    
    @classmethod
    def print_config(cls):
        """打印当前配置（隐藏敏感信息）"""
        provider = canonical_provider(cls.LLM_PROVIDER)
        api_key = cls.get_api_key(provider)
        print("\n" + "="*50)
        print("CONFIGURATION")
        print("="*50)
        print(f"Provider: {provider}")
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Temperature: {cls.MODEL_TEMPERATURE}")
        print(f"Max Tokens: {cls.MODEL_MAX_TOKENS}")
        print(f"Max Iterations: {cls.MAX_ITERATIONS}")
        print(f"API Key Set: {'Yes' if api_key else 'No'}")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print("="*50 + "\n")
