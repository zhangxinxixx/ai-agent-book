"""后台记忆处理器。

它读取已经持久化的原始对话，让 ``UserMemoryAgent`` 决定哪些信息值得长期保存，
并通过 add/update/delete 工具写入 MemoryManager。它不参与前台回答生成。
"""

import json
import logging
import threading
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from openai import OpenAI
from config import Config, MemoryMode
from memory_manager import create_memory_manager, BaseMemoryManager
from conversation_history import ConversationHistory
from agent import UserMemoryAgent, UserMemoryConfig

# 配置日志。
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MemoryUpdate:
    """表示一次记忆新增、更新或删除决策。"""
    action: str  # 'add', 'update', 'delete', 'none'
    memory_id: Optional[str] = None
    content: Optional[str] = None
    reason: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class MemoryProcessorConfig:
    """后台记忆处理器配置。"""
    conversation_interval: int = 1  # Process after N conversation rounds (default: every round)
    min_conversation_turns: int = 1  # Minimum turns before processing
    context_window: int = 10  # Number of recent turns to analyze
    enable_auto_processing: bool = True
    temperature: float = 0.3  # Lower temperature for analysis
    output_operations: bool = True  # Output detailed memory operations


class BackgroundMemoryProcessor:
    """按对话轮次触发的后台处理器，与前台对话链解耦。"""
    
    def __init__(self,
                 user_id: str,
                 api_key: Optional[str] = None,
                 provider: Optional[str] = None,
                 model: Optional[str] = None,
                 config: Optional[MemoryProcessorConfig] = None,
                 memory_mode: MemoryMode = MemoryMode.NOTES,
                 verbose: bool = True):
        """
        Initialize the background memory processor
        
        Args:
            user_id: Unique user identifier
            api_key: API key (defaults to env based on provider)
            provider: LLM provider ('dashscope'/'qwen'/'bailian', 'siliconflow', 'doubao', 'kimi', 'moonshot')
            model: Model name (defaults to provider's default)
            config: Processor configuration
            memory_mode: Memory storage mode
            verbose: Enable verbose logging
        """
        self.user_id = user_id
        self.verbose = verbose
        self.config = config or MemoryProcessorConfig()
        self.memory_mode = memory_mode
        self.provider = provider
        self.model = model
        
        # 后台分析智能体拥有记忆写工具；前台 ConversationalAgent 没有这些写权限。
        agent_config = UserMemoryConfig(
            memory_mode=memory_mode,
            enable_memory_updates=True,  # Agent will use its tools to update memory
            enable_memory_search=True,  # Enable memory search tool
            enable_conversation_history=False,
            save_trajectory=False  # Don't save trajectory for background processing
        )
        self.analysis_agent = UserMemoryAgent(
            user_id=user_id,
            api_key=api_key,
            provider=provider,
            model=model,
            config=agent_config,
            verbose=self.verbose
        )
        
        # 初始化长期记忆与原始对话历史管理器。
        self.memory_manager = create_memory_manager(user_id, memory_mode)
        self.conversation_history = ConversationHistory(user_id)
        
        # 这些是当前进程的调度状态，不是持久化长期记忆。
        self.processing_thread = None
        self.stop_processing = False
        self.last_processed_timestamp = None
        self.processing_lock = threading.Lock()
        self.conversation_count = 0  # Track conversation rounds
        self.last_processed_count = 0  # Track last processed conversation count
        self.processed_turn_ids = set()  # Track which turns have been processed
        
        logger.info(f"后台记忆处理器初始化完成：用户={user_id}，服务商={provider or Config.PROVIDER}")
    
    def analyze_conversation(self, conversation_context: List[Dict[str, str]]) -> List[MemoryUpdate]:
        """
        Analyze conversation context and determine memory updates
        
        Args:
            conversation_context: List of conversation messages
            
        Returns:
            List of memory updates to apply
        """
        if len(conversation_context) < self.config.min_conversation_turns * 2:
            return []
        
        try:
            # 使用 UserMemoryAgent 分析对话。
            if self.verbose:
                logger.info("正在使用 UserMemoryAgent 分析对话……")
            
            # 把原始轮次整理成后台智能体可读取的消息格式。
            conversation_str = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in conversation_context
            ])
            
            # 创建记忆分析与更新任务。
            task = f"""Analyze this recent conversation and update my memory accordingly. 
Extract any important facts, preferences, or information that should be remembered.

Recent Conversation:
{conversation_str}

Please review this conversation and:
1. Add any new important information as memories
2. Update existing memories if there's new or changed information
3. Delete any memories that are no longer accurate

Focus on extracting factual information that would be useful for future conversations."""
            
            # 通过后台智能体的工具系统执行任务。
            result = self.analysis_agent.execute_task(task)
            
            if self.verbose:
                logger.info(f"记忆更新任务完成：success={result.get('success', False)}")
            
            # 后台智能体已经通过工具直接更新 MemoryManager，返回空列表不代表没有写入。
            return []
            
        except Exception as e:
            logger.error(f"分析对话失败：{e}")
            return []
    
    def apply_memory_updates(self, updates: List[MemoryUpdate]) -> Dict[str, Any]:
        """
        Apply memory updates to the memory manager
        
        Args:
            updates: List of memory updates to apply
            
        Returns:
            Summary of applied updates
        """
        results = {
            'added': 0,
            'updated': 0,
            'deleted': 0,
            'failed': 0,
            'details': []
        }
        
        for update in updates:
            try:
                if update.action == 'add' and update.content:
                    # 按不同记忆模式解析新增内容。
                    if self.memory_mode in [MemoryMode.NOTES, MemoryMode.ENHANCED_NOTES]:
                        memory_id = self.memory_manager.add_memory(
                            content=update.content,
                            session_id=f"background-{datetime.now().isoformat()}",
                            tags=update.tags
                        )
                    elif self.memory_mode == MemoryMode.JSON_CARDS:
                        # JSON 卡片模式需要解析结构化内容。
                        try:
                            if isinstance(update.content, str):
                                content_dict = json.loads(update.content)
                            else:
                                content_dict = update.content
                        except (json.JSONDecodeError, TypeError):
                            # JSON 解析失败时退回简单解析。
                            parts = str(update.content).split(':')
                            if len(parts) >= 2:
                                content_dict = {
                                    'category': 'personal',
                                    'subcategory': 'info',
                                    'key': parts[0].strip().replace(' ', '_').lower(),
                                    'value': ':'.join(parts[1:]).strip()
                                }
                            else:
                                content_dict = {
                                    'category': 'general',
                                    'subcategory': 'notes',
                                    'key': f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                    'value': update.content
                                }
                        
                        memory_id = self.memory_manager.add_memory(
                            content=content_dict,
                            session_id=f"background-{datetime.now().isoformat()}"
                        )
                    elif self.memory_mode == MemoryMode.ADVANCED_JSON_CARDS:
                        # 高级 JSON 卡片要求完整结构。
                        try:
                            if isinstance(update.content, str):
                                content_dict = json.loads(update.content)
                            else:
                                content_dict = update.content
                        except (json.JSONDecodeError, TypeError):
                            # 无法解析时跳过该操作。
                            results['failed'] += 1
                            continue
                        
                        # 从嵌套结构中提取卡片数据。
                        card_data = content_dict.get('card', {})
                        
                        memory_id = self.memory_manager.add_memory(
                            content=content_dict,
                            session_id=f"background-{datetime.now().isoformat()}",
                            backstory=update.reason or '',
                            person=card_data.get('person', 'User'),
                            relationship=card_data.get('relationship', 'primary account holder')
                        )
                    else:
                        memory_id = self.memory_manager.add_memory(
                            content=update.content,
                            session_id=f"background-{datetime.now().isoformat()}",
                            tags=update.tags
                        )
                    results['added'] += 1
                    
                    # 格式化展示内容。
                    if isinstance(update.content, dict):
                        # JSON 模式只显示摘要。
                        if self.memory_mode == MemoryMode.ADVANCED_JSON_CARDS:
                            card_key = update.content.get('card_key', 'unknown')
                            category = update.content.get('category', 'unknown')
                            display_content = f"{category}.{card_key}"
                        else:
                            display_content = json.dumps(update.content, ensure_ascii=False)[:100]
                    else:
                        display_content = str(update.content)[:50]
                    
                    results['details'].append(f"Added: {display_content}...")
                    
                    # 演示模式始终在终端显示实际写入操作。
                    print(f"  📝【新增记忆】{display_content}")
                    
                    if self.verbose:
                        logger.info(f"已新增记忆：{display_content}")
                
                elif update.action == 'update' and update.memory_id and update.content:
                    # 按不同记忆模式解析更新内容。
                    if self.memory_mode in [MemoryMode.NOTES, MemoryMode.ENHANCED_NOTES]:
                        success = self.memory_manager.update_memory(
                            memory_id=update.memory_id,
                            content=update.content,
                            session_id=f"background-{datetime.now().isoformat()}",
                            tags=update.tags
                        )
                    elif self.memory_mode == MemoryMode.JSON_CARDS:
                        # 解析 JSON 卡片更新内容。
                        try:
                            if isinstance(update.content, str):
                                content_dict = json.loads(update.content)
                            else:
                                content_dict = update.content
                        except (json.JSONDecodeError, TypeError):
                            # 简单值更新。
                            content_dict = {'value': update.content}
                        
                        success = self.memory_manager.update_memory(
                            memory_id=update.memory_id,
                            content=content_dict,
                            session_id=f"background-{datetime.now().isoformat()}"
                        )
                    elif self.memory_mode == MemoryMode.ADVANCED_JSON_CARDS:
                        # 高级 JSON 卡片要求完整结构。
                        try:
                            if isinstance(update.content, str):
                                content_dict = json.loads(update.content)
                            else:
                                content_dict = update.content
                        except (json.JSONDecodeError, TypeError):
                            # 无法解析时跳过该操作。
                            results['failed'] += 1
                            continue
                        
                        success = self.memory_manager.update_memory(
                            memory_id=update.memory_id,
                            content=content_dict,
                            session_id=f"background-{datetime.now().isoformat()}"
                        )
                    else:
                        success = self.memory_manager.update_memory(
                            memory_id=update.memory_id,
                            content=update.content,
                            session_id=f"background-{datetime.now().isoformat()}",
                            tags=update.tags
                        )
                    if success:
                        results['updated'] += 1
                        
                        # 格式化展示内容。
                        if isinstance(update.content, dict):
                            if self.memory_mode == MemoryMode.ADVANCED_JSON_CARDS:
                                card_key = update.content.get('card_key', 'unknown')
                                category = update.content.get('category', 'unknown')
                                display_content = f"{category}.{card_key}"
                            else:
                                display_content = json.dumps(update.content, ensure_ascii=False)[:100]
                        else:
                            display_content = str(update.content)[:50]
                        
                        results['details'].append(f"Updated {update.memory_id}: {display_content}...")
                        
                        # 演示模式始终显示更新操作。
                        print(f"  ✏️【更新记忆｜ID：{update.memory_id[:8] if len(update.memory_id) > 8 else update.memory_id}】{display_content}")
                    else:
                        results['failed'] += 1
                    
                    if self.verbose:
                        if isinstance(update.content, dict):
                            display_content = json.dumps(update.content, ensure_ascii=False)[:100]
                        else:
                            display_content = str(update.content)[:50]
                        logger.info(f"已更新记忆 {update.memory_id}：{display_content}")
                
                elif update.action == 'delete' and update.memory_id:
                    self.memory_manager.delete_memory(update.memory_id)
                    results['deleted'] += 1
                    results['details'].append(f"Deleted: {update.memory_id}")
                    
                    # 演示模式始终显示删除操作。
                    print(f"  🗑️【删除记忆｜ID：{update.memory_id}】")
                    
                    if self.verbose:
                        logger.info(f"已删除记忆：{update.memory_id}")
                
            except Exception as e:
                logger.error(f"执行记忆操作 {update.action} 失败：{e}")
                results['failed'] += 1
        
        return results
    
    def process_recent_conversations(self) -> Dict[str, Any]:
        """
        Process recent conversations and update memories
        
        Returns:
            Processing results with list of operations
        """
        with self.processing_lock:
            # 先推进处理计数。否则后面的提前返回会让 should_process() 永远为 True，
            # 后台线程将每秒重复触发同一批空处理。
            self.last_processed_count = self.conversation_count
            self.last_processed_timestamp = datetime.now()

            # 前台智能体通过另一份 ConversationHistory 实例写磁盘；这里必须重新加载，
            # 不能相信后台实例初始化时缓存的旧列表。
            self.conversation_history.load_history()

            # 读取最近的原始对话轮次。
            recent_turns = self.conversation_history.get_recent_turns(
                limit=self.config.context_window
            )

            if not recent_turns:
                return {
                    'message': 'No recent conversations to process',
                    'operations': [],
                    'summary': {'added': 0, 'updated': 0, 'deleted': 0}
                }

            # 进程内去重：同一 Processor 生命周期内不重复处理同一轮。
            unprocessed_turns = []
            for turn in recent_turns:
                # 为每一轮构造进程内去重标识。
                turn_id = f"{turn.session_id}_{turn.turn_number}_{turn.timestamp}"
                if turn_id not in self.processed_turn_ids:
                    unprocessed_turns.append(turn)
                    self.processed_turn_ids.add(turn_id)

            # 全部轮次都已处理时直接返回。
            if not unprocessed_turns:
                return {
                    'message': 'No new conversations to process',
                    'operations': [],
                    'summary': {'added': 0, 'updated': 0, 'deleted': 0}
                }
            
            # 转换为后台智能体需要的消息格式。
            conversation_context = []
            for turn in unprocessed_turns:
                conversation_context.append({
                    'role': 'user',
                    'content': turn.user_message
                })
                conversation_context.append({
                    'role': 'assistant',
                    'content': turn.assistant_message
                })
            
            # analysis_agent.execute_task() 会直接调用记忆工具完成写入，
            # analyze_conversation() 的空列表返回值不代表“没有发生记忆操作”。
            _ = self.analyze_conversation(conversation_context)
            
            # 从后台智能体读取工具调用轨迹，生成可观察的操作报告。
            tool_calls = getattr(self.analysis_agent, 'tool_calls', [])
            
            # 从工具调用中整理记忆操作列表。
            operations = []
            summary = {'added': 0, 'updated': 0, 'deleted': 0}
            
            for tool_call in tool_calls:
                if tool_call.tool_name == 'add_memory':
                    operations.append({
                        'action': 'add',
                        'content': tool_call.arguments.get('content'),
                        'result': tool_call.result
                    })
                    if tool_call.result and tool_call.result.get('success'):
                        summary['added'] += 1
                elif tool_call.tool_name == 'update_memory':
                    operations.append({
                        'action': 'update',
                        'memory_id': tool_call.arguments.get('memory_id'),
                        'content': tool_call.arguments.get('content'),
                        'result': tool_call.result
                    })
                    if tool_call.result and tool_call.result.get('success'):
                        summary['updated'] += 1
                elif tool_call.tool_name == 'delete_memory':
                    operations.append({
                        'action': 'delete',
                        'memory_id': tool_call.arguments.get('memory_id'),
                        'result': tool_call.result
                    })
                    if tool_call.result and tool_call.result.get('success'):
                        summary['deleted'] += 1
            
            # 清空本轮工具轨迹，避免混入下一次处理结果。
            self.analysis_agent.tool_calls = []
            
            # 组装最终处理结果。
            final_results = {
                'analyzed_turns': len(unprocessed_turns),
                'operations': operations,
                'summary': summary,
                'details': operations  # Operations are the details
            }

            return final_results
    
    def should_process(self) -> bool:
        """
        Check if memory processing should be triggered based on conversation count
        
        Returns:
            True if processing should occur
        """
        if self.conversation_count == 0:
            return False
        
        # 判断是否达到配置的对话轮次间隔。
        conversations_since_last = self.conversation_count - self.last_processed_count
        should_process = conversations_since_last >= self.config.conversation_interval
        
        # 调试日志用于观察触发计数。
        if should_process and self.verbose:
            logger.debug(f"是否应处理：当前轮数={self.conversation_count}，上次处理位置={self.last_processed_count}，间隔={self.config.conversation_interval}")
        
        return should_process
    
    def increment_conversation_count(self):
        """
        Increment the conversation counter
        """
        self.conversation_count += 1
        
        if self.verbose:
            logger.info(f"对话轮数：{self.conversation_count}，上次处理位置：{self.last_processed_count}")
    
    def _background_processing_loop(self):
        """
        Background loop for automatic memory processing based on conversation count
        """
        logger.info(f"正在启动后台记忆处理：每 {self.config.conversation_interval} 轮触发一次")
        
        while not self.stop_processing:
            try:
                # 每秒检查一次是否达到触发条件。
                time.sleep(1)
                
                if self.stop_processing:
                    break
                
                # 根据对话计数决定是否处理。
                if self.should_process():
                    if self.verbose:
                        logger.info(f"已触发后台处理：当前轮数={self.conversation_count}，上次处理位置={self.last_processed_count}")
                    
                    results = self.process_recent_conversations()
                    
                    if self.config.output_operations and results:
                        self._output_operations(results)
                    
                    if self.verbose:
                        logger.info(f"后台处理结果：{results.get('summary')}")
                        logger.info(f"已更新处理位置：{self.last_processed_count}")
                
            except Exception as e:
                logger.error(f"后台处理发生错误：{e}")
        
        logger.info("后台记忆处理已停止")
    
    def _output_operations(self, results: Dict[str, Any]):
        """
        Output memory operations in a formatted way
        
        Args:
            results: Processing results with operations
        """
        operations = results.get('operations', [])
        summary = results.get('summary', {})
        
        # 没有实际对话时不输出空操作日志。
        if results.get('message') in ['No recent conversations to process', 'No new conversations to process']:
            return
            
        if not operations:
            # 有对话但无需更新时，明确记录该结果。
            if results.get('analyzed_turns', 0) > 0:
                logger.info("📝 记忆操作：无（本轮不需要更新）")
            return
        
        logger.info(f"\n📝 记忆操作（共 {len(operations)} 条）：")
        logger.info("-" * 50)
        
        for i, op in enumerate(operations, 1):
            icon = {
                'add': '➕',
                'update': '📝',
                'delete': '🗑️'
            }.get(op['action'], '❓')
            
            logger.info(f"{i}. {icon} {op['action'].upper()}")
            if op.get('content'):
                logger.info(f"   内容：{op['content']}")
            if op.get('memory_id'):
                logger.info(f"   记忆 ID：{op['memory_id']}")
            if op.get('reason'):
                logger.info(f"   原因：{op['reason']}")
            if op.get('tags'):
                logger.info(f"   标签：{', '.join(op['tags'])}")
            logger.info("")
        
        logger.info(f"汇总：新增 {summary.get('added', 0)} 条，更新 {summary.get('updated', 0)} 条，删除 {summary.get('deleted', 0)} 条")
        logger.info("-" * 50)
    
    def start_background_processing(self):
        """启动后台记忆处理线程。"""
        if self.processing_thread and self.processing_thread.is_alive():
            logger.warning("后台记忆处理已经在运行")
            return
        
        self.stop_processing = False
        # 新进程启动时重置进程内去重集合。
        self.processed_turn_ids.clear()
        self.processing_thread = threading.Thread(
            target=self._background_processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        logger.info("后台记忆处理已启动")
    
    def stop_background_processing(self):
        """停止后台记忆处理线程。"""
        self.stop_processing = True
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        logger.info("后台记忆处理已停止")
    
    def process_conversation_batch(self, conversation_contexts: List[List[Dict[str, str]]]) -> List[Dict[str, Any]]:
        """
        Process multiple conversation contexts in batch
        
        Args:
            conversation_contexts: List of conversation contexts
            
        Returns:
            List of processing results
        """
        results = []
        
        for context in conversation_contexts:
            updates = self.analyze_conversation(context)
            
            operations = []
            for update in updates:
                operation = {
                    'action': update.action,
                    'content': update.content,
                }
                if update.memory_id:
                    operation['memory_id'] = update.memory_id
                operations.append(operation)
            
            if updates:
                apply_result = self.apply_memory_updates(updates)
                result = {
                    'operations': operations,
                    'summary': {
                        'added': apply_result['added'],
                        'updated': apply_result['updated'],
                        'deleted': apply_result['deleted']
                    }
                }
            else:
                result = {
                    'message': 'No updates needed',
                    'operations': [],
                    'summary': {'added': 0, 'updated': 0, 'deleted': 0}
                }
            
            results.append(result)
        
        return results
