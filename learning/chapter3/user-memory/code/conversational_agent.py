"""前台对话智能体。

它负责当前会话的对话与长期记忆读取，但不直接修改长期记忆。
记忆增删改由 ``BackgroundMemoryProcessor`` 中的后台智能体完成。
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid
from openai import OpenAI
from config import Config, openrouter_model_id, PROVIDER_DEFAULT_MODELS
from conversation_history import ConversationHistory, ConversationTurn
from memory_manager import create_memory_manager, BaseMemoryManager, MemoryMode


def _reasoning_safe_temperature(model, requested=1.0):
    """推理模型强制使用 temperature=1，其他模型保留请求值。"""
    m = str(model or "").lower().replace("/", "-")
    return 1 if ("kimi-k3" in m or "gpt-5" in m) else requested

# 配置日志。
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ConversationConfig:
    """前台对话智能体的运行配置。"""
    enable_memory_context: bool = True  # 在上下文中读取记忆，但不直接更新。
    enable_conversation_history: bool = True
    max_memory_context: int = 10
    temperature: float = 0.7
    max_tokens: int = 4096


class ConversationalAgent:
    """只负责对话的前台智能体：读取长期记忆，维护当前会话，不写长期记忆。"""
    
    def __init__(self, 
                 user_id: str,
                 api_key: Optional[str] = None,
                 provider: Optional[str] = None,
                 model: Optional[str] = None,
                 config: Optional[ConversationConfig] = None,
                 memory_mode: MemoryMode = MemoryMode.NOTES,
                 verbose: bool = True):
        """初始化用户、模型客户端、当前会话和只读记忆管理器。"""
        self.user_id = user_id
        self.verbose = verbose
        self.config = config or ConversationConfig()
        self.memory_mode = memory_mode
        
        # 确定模型服务商。
        self.provider = (provider or Config.PROVIDER).lower()
        self.provider = {"qwen": "dashscope", "bailian": "dashscope"}.get(
            self.provider, self.provider
        )
        
        # 读取对应服务商的 API 密钥。
        api_key = api_key or Config.get_api_key(self.provider)

        # 主服务商缺少密钥但配置了 OPENROUTER_API_KEY 时，改走 OpenRouter。
        if not api_key and self.provider != "openrouter" and Config.OPENROUTER_API_KEY:
            model = openrouter_model_id(model or PROVIDER_DEFAULT_MODELS.get(self.provider))
            self.provider = "openrouter"
            api_key = Config.OPENROUTER_API_KEY

        if not api_key:
            raise ValueError(
                f"模型服务商 '{self.provider}' 需要 API 密钥。请配置对应密钥，"
                f"或配置 OPENROUTER_API_KEY 使用 OpenRouter 兜底。"
            )

        # 根据服务商初始化 OpenAI 兼容客户端。
        if self.provider == "dashscope":
            self.client = OpenAI(
                api_key=api_key,
                base_url=Config.DASHSCOPE_BASE_URL
            )
            self.model = model or PROVIDER_DEFAULT_MODELS["dashscope"]
        elif self.provider == "siliconflow":
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.siliconflow.cn/v1"
            )
            self.model = model or "Qwen/Qwen3-235B-A22B-Thinking-2507"
        elif self.provider == "doubao":
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://ark.cn-beijing.volces.com/api/v3"
            )
            self.model = model or os.getenv("ARK_MODEL", "doubao-seed-1-6-250615")
        elif self.provider == "kimi" or self.provider == "moonshot":
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.moonshot.cn/v1"
            )
            self.model = model or "kimi-k3"
        elif self.provider == "openrouter":
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            # 默认模型可被显式传入的兼容模型覆盖。
            self.model = model or "google/gemini-3.5-flash"
            # 兼容模型示例：google/gemini-3.5-flash、openai/gpt-5、anthropic/claude-sonnet-4。
        else:
            raise ValueError(f"不支持的模型服务商：{self.provider}。可用值：dashscope/qwen/bailian、siliconflow、doubao、kimi、moonshot、openrouter")
        
        # 只读访问长期记忆；同一个 user_id 决定读取哪一份持久化数据。
        self.memory_manager = create_memory_manager(user_id, memory_mode)
        
        # ConversationHistory 保存原始对话证据；它与提炼后的长期记忆不是同一个对象。
        self.conversation_history = ConversationHistory(user_id) if self.config.enable_conversation_history else None
        
        # 当前会话的原始消息只在 self.conversation 中参与连续对话。
        self.session_id = self._generate_session_id()
        self.conversation = []
        
        # 初始化系统提示词。
        self._init_system_prompt()
        
        logger.info(f"对话智能体初始化完成：用户={user_id}，服务商={self.provider}，模型={self.model}")
    
    def _generate_session_id(self) -> str:
        """生成唯一会话标识。"""
        return f"session-{uuid.uuid4().hex[:8]}"
    
    def _init_system_prompt(self):
        """初始化前台对话使用的系统提示词。"""
        system_content = """你是一名有帮助、能够提供个性化回答的助手。你可以读取从过去对话中提炼的用户信息，并据此给出符合上下文的回答。

你必须仔细分析当前上下文、用户问题和长期记忆，然后给出完整、具体的回答。
"""

        self.conversation = [
            {
                "role": "system",
                "content": system_content
            }
        ]
    
    def _get_memory_context(self) -> str:
        """把长期记忆和当前会话历史组装成临时上下文。"""
        if not self.config.enable_memory_context:
            return ""

        context_parts = []

        # 后台处理器使用另一个 MemoryManager 实例写磁盘，因此每轮请求前重新加载，
        # 才能让当前智能体看见刚完成的后台更新。
        self.memory_manager.load_memory()

        # 首先加入提炼后的长期记忆。
        memory_str = self.memory_manager.get_context_string()
        if memory_str:
            context_parts.append("=== 用户长期记忆 ===")
            context_parts.append(memory_str)
            context_parts.append("")
        
        # 原始 ConversationHistory 只注入当前 session_id 对应的轮次。
        # 旧会话的信息必须先被后台提炼成长期记忆，才能重新进入模型上下文；
        # 这样可避免把全部历史原文永久堆进每次请求。
        if self.conversation_history:
            session_turns = self.conversation_history.get_session_turns(
                self.session_id
            )
            
            if session_turns:
                context_parts.append("=== 当前会话历史 ===")
                context_parts.append(f"对话轮数：{len(session_turns)}")
                context_parts.append("")
                
                for turn in session_turns:
                    context_parts.append(f"[会话：{turn.session_id}，轮次：{turn.turn_number}，时间：{turn.timestamp}]")
                    context_parts.append(f"用户：{turn.user_message}")
                    context_parts.append(f"助手：{turn.assistant_message}")
                    context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_conversation_context(self) -> List[Dict[str, str]]:
        """返回不含系统提示词的当前会话副本，供后台处理。"""
        return [msg for msg in self.conversation[1:] if msg.get('role') != 'system']
    
    def chat(self, message: str) -> str:
        """临时注入记忆上下文，请求模型并持久化原始对话。"""
        # 仅在本轮 API 请求的用户消息后附加记忆上下文。
        memory_context = self._get_memory_context()
        
        if memory_context:
            full_message = f"{message}\n\n{memory_context}"
        else:
            full_message = message
        
        # 详细模式记录完整提示词，其中可能包含个人信息。
        if self.verbose:
            logger.info(f"用户请求：{message}")
            if memory_context:
                logger.info(f"已注入记忆上下文：{memory_context}")
            logger.info(f"发送给模型的完整提示词：{full_message}")
        
        # 只持久化用户原话。memory_context 仅在本轮 api_messages 中临时注入；
        # 若把 full_message 再保存进 conversation，会把历史嵌套复制到每一轮，
        # 使会话内 token 规模接近 O(N^2) 增长。
        self.conversation.append({"role": "user", "content": message})
        api_messages = self.conversation[:-1] + [{"role": "user", "content": full_message}]

        try:
            # 以流式模式请求模型。
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                temperature=_reasoning_safe_temperature(self.model, self.config.temperature),
                max_tokens=self.config.max_tokens,
                stream=True
            )
            
            # 汇总流式响应。
            assistant_message = ""
            if self.verbose:
                logger.info("正在接收流式回答……")
                
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    assistant_message += delta
                    # 实时输出收到的文本片段。
                    print(delta, end='', flush=True)
            
            print()  # 流式回答结束后换行。
            
            # 把最终助手回答加入当前会话。
            self.conversation.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            # 原始用户/助手对话先写 ConversationHistory，供后台记忆抽取使用。
            if self.conversation_history:
                self.conversation_history.add_turn(
                    session_id=self.session_id,
                    user_message=message,
                    assistant_message=assistant_message
                )
            
            if self.verbose:
                logger.info(f"用户：{message}")
                logger.info(f"助手：{assistant_message}")
            
            return assistant_message
            
        except Exception as e:
            error_msg = f"对话过程中发生错误：{str(e)}"
            logger.error(error_msg)
            return f"抱歉，对话过程中发生错误：{str(e)}"
    
    def reset_session(self):
        """开始新会话，但保留持久化长期记忆。"""
        self.session_id = self._generate_session_id()
        self._init_system_prompt()
        logger.info(f"已开始新会话：{self.session_id}")
    
    def get_session_id(self) -> str:
        """返回当前会话标识。"""
        return self.session_id
