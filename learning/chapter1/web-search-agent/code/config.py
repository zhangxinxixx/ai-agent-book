"""
配置文件 - Kimi API 配置
"""

import os
from typing import Optional
from dotenv import find_dotenv, load_dotenv

load_dotenv()

from dotenv import load_dotenv

# 从工作目录向上查找最近的 .env，使得仓库根目录下的单一配置文件即可服务各章。
_dotenv_path = find_dotenv(usecwd=True)
load_dotenv(_dotenv_path or None)


# 提供商解析逻辑位于共享的 agentbook 包中，以确保各章保持一致；详见 agentbook/providers.py。
# 这里的降级方案使得在未安装 agentbook 包的环境下也能直接运行本实验。
try:
    from agentbook.providers import (
        canonical_provider,
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
        SUPPORTED_PROVIDERS,
        map_model_to_openrouter,
        resolve_backend,
        resolve_llm_backend,
    )


class Config:
    """配置类"""
    
    # Kimi API 配置
    MOONSHOT_API_KEY: str = os.getenv("MOONSHOT_API_KEY", "")
    # 向后兼容：如果没有 MOONSHOT_API_KEY，尝试使用 KIMI_API_KEY
    if not MOONSHOT_API_KEY:
        MOONSHOT_API_KEY = os.getenv("KIMI_API_KEY", "")
    
    KIMI_BASE_URL: str = "https://api.moonshot.cn/v1"
    
    # 模型配置
    DEFAULT_MODEL: str = "kimi-k3"  # 使用最新的 Kimi K3 模型

    # 搜索配置
    MAX_SEARCH_ITERATIONS: int = 5  # 最大搜索迭代次数（与 agent 默认值保持一致）
    # 这个超时同时作用于 Formula 工具调用和 chat completion。kimi-k3 以
    # reasoning_effort=max 运行，单次 completion 常需 1-3 分钟（validation/
    # 目录里保留的真实运行记录中有 161 秒、121 秒的调用），30 秒会让交互模式
    # 反复 "Request timed out"。默认值与 run_experiment_1_2.py 的 --timeout
    # 保持一致，确保 README 里的交互入口与验收脚本跑在同一配置下。
    SEARCH_TIMEOUT: float = float(os.getenv("SEARCH_TIMEOUT", "180"))
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def validate(cls) -> bool:
        """
        验证配置是否有效
        
        Returns:
            bool: 配置是否有效
        """
        if not cls.MOONSHOT_API_KEY:
            print("错误: 未设置 MOONSHOT_API_KEY 环境变量")
            print("请设置环境变量: export MOONSHOT_API_KEY='your-api-key'")
            print("(或者使用旧的环境变量名: export KIMI_API_KEY='your-api-key')")
            return False
        return True
    
    @classmethod
    def get_api_key(cls, api_key: Optional[str] = None) -> str:
        """
        获取 API Key
        
        Args:
            api_key: 可选的 API key，如果提供则使用，否则从环境变量获取
            
        Returns:
            API key
        """
        if api_key:
            return api_key
        return cls.MOONSHOT_API_KEY
