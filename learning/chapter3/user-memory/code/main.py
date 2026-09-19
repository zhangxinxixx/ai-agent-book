#!/usr/bin/env python3
"""用户长期记忆实验主入口。

这套实现刻意拆成两条链路：
1. ``ConversationalAgent`` 负责低延迟对话，并只读长期记忆；
2. ``BackgroundMemoryProcessor`` 异步分析已经落盘的对话，再通过工具更新长期记忆。

因此“模型完成了一次回答”和“该轮信息已经写入长期记忆”是两个独立结果。
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Optional
from conversational_agent import ConversationalAgent, ConversationConfig
from background_memory_processor import BackgroundMemoryProcessor, MemoryProcessorConfig
from config import Config, MemoryMode

# 评估框架是可选依赖，只在 evaluation 模式中动态导入，避免普通交互启动时发生导入冲突。
EVALUATION_AVAILABLE = False
UserMemoryEvaluationFramework = None
TestCase = None

# 配置统一日志格式。
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """输出带分隔线的章节标题。"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_result(result: dict):
    """以适合实验观察的格式输出任务结果。"""
    if result.get('success'):
        print("\n✅ 任务执行成功！")
        if result.get('final_answer'):
            print("\n📝 最终回答：")
            print("-"*40)
            print(result['final_answer'])
    else:
        print("\n❌ 任务执行失败！")
        if result.get('error'):
            print(f"错误：{result['error']}")
    
    print(f"\n📊 执行统计：")
    print(f"  - 迭代次数：{result.get('iterations', 0)}")
    print(f"  - 工具调用次数：{len(result.get('tool_calls', []))}")
    
    if result.get('trajectory_file'):
        print(f"\n💾 执行轨迹已保存到：{result['trajectory_file']}")
    
    # 汇总每种工具的调用次数与成功状态。
    if result.get('tool_calls'):
        print(f"\n🔧 工具调用汇总：")
        tool_summary = {}
        for call in result['tool_calls']:
            tool_name = call.tool_name
            if tool_name not in tool_summary:
                tool_summary[tool_name] = {
                    'count': 0,
                    'success': 0,
                    'failed': 0
                }
            tool_summary[tool_name]['count'] += 1
            if call.error:
                tool_summary[tool_name]['failed'] += 1
            else:
                tool_summary[tool_name]['success'] += 1
        
        for tool_name, stats in tool_summary.items():
            print(f"  - {tool_name}：共 {stats['count']} 次 "
                  f"（成功 {stats['success']} 次，失败 {stats['failed']} 次）")
    
    # 展示长期记忆的前 500 个字符。
    if result.get('memory_state'):
        print(f"\n💭 长期记忆状态：")
        print("-"*40)
        memory_preview = result['memory_state'][:500]
        if len(result['memory_state']) > 500:
            memory_preview += "..."
        print(memory_preview)


def interactive_mode(user_id: str, memory_mode: MemoryMode = MemoryMode.NOTES, 
                    enable_background_processing: bool = True,
                    conversation_interval: int = 1,
                    provider: Optional[str] = None,
                    model: Optional[str] = None):
    """运行“前台对话 + 后台记忆处理”的分离式交互模式。"""
    print_section(f"交互模式 - 对话智能体（用户：{user_id}）")
    
    # 确定模型服务商并读取对应密钥。
    provider = (provider or Config.PROVIDER).lower()
    api_key = Config.get_api_key(provider)
    if not api_key:
        print(f"❌ 错误：请为模型服务商 '{provider}' 配置 API 密钥")
        if provider in ["kimi", "moonshot"]:
            print("   export MOONSHOT_API_KEY='your-api-key-here'")
        elif provider in ["dashscope", "qwen", "bailian"]:
            print("   export DASHSCOPE_API_KEY='your-api-key-here'")
        elif provider == "siliconflow":
            print("   export SILICONFLOW_API_KEY='your-api-key-here'")
        elif provider == "doubao":
            print("   export DOUBAO_API_KEY='your-api-key-here'")
        elif provider == "openrouter":
            print("   export OPENROUTER_API_KEY='your-api-key-here'")
        return
    
    # 前台智能体：读取长期记忆、维护当前会话、请求模型回答，但不直接写长期记忆。
    conv_config = ConversationConfig(
        enable_memory_context=True,
        enable_conversation_history=True
    )
    
    agent = ConversationalAgent(
        user_id=user_id,
        api_key=api_key,
        provider=provider,
        model=model,
        config=conv_config,
        memory_mode=memory_mode,
        verbose=True
    )
    
    # 后台处理器：读取已持久化的对话轮次，由独立的 UserMemoryAgent 决定增删改记忆。
    memory_processor = None
    if enable_background_processing:
        proc_config = MemoryProcessorConfig(
            conversation_interval=conversation_interval,
            min_conversation_turns=1,
            context_window=10,
            enable_auto_processing=True,
            output_operations=True
        )
        
        memory_processor = BackgroundMemoryProcessor(
            user_id=user_id,
            api_key=api_key,
            provider=provider,
            model=model,
            config=proc_config,
            memory_mode=memory_mode,
            verbose=True
        )
        
        memory_processor.start_background_processing()
        print(f"\n🧠 已启用后台记忆处理：每 {conversation_interval} 轮对话触发一次")
    
    print("\n✅ 对话智能体初始化完成")
    print(f"📦 记忆模式：{memory_mode.value}")
    print(f"🆔 会话标识：{agent.get_session_id()}")
    print(f"🔄 后台处理：{'开启' if enable_background_processing else '关闭'}")
    if enable_background_processing:
        print(f"📊 触发条件：每 {conversation_interval} 轮对话处理一次")
    print("\n可用命令：")
    print("  'memory'  - 显示当前长期记忆")
    print("  'process' - 手动触发记忆处理")
    print("  'save'    - 立即执行一次记忆处理")
    print("  'reset'   - 开始新的对话会话")
    print("  'quit'    - 停止后台线程并退出")
    print("  'exit'    - 停止后台线程并退出")
    print("\n也可以直接输入消息开始对话。")
    
    conversation_count = 0
    
    while True:
        try:
            print("\n" + "-"*60)
            user_input = input("【用户】> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                # 对话在 chat() 中已逐轮落盘；退出时只是不再主动触发新的记忆提炼。
                if memory_processor:
                    memory_processor.stop_background_processing()
                print("👋 已退出。原始对话已落盘，本次退出没有额外触发记忆提炼。")
                break
            
            elif user_input.lower() == 'save':
                # 立即处理最近尚未处理的对话。
                print("\n💾 正在处理并保存长期记忆……")
                if memory_processor:
                    results = memory_processor.process_recent_conversations()
                    print(f"✅ 记忆处理结果：{results}")
                else:
                    print("⚠️ 后台处理已关闭。原始对话已经保存，但该命令不会更新长期记忆。")
                continue
            
            elif user_input.lower() == 'memory':
                print("\n💭 当前长期记忆：")
                print("-"*40)
                # 后台处理器通过另一个管理器实例写磁盘，因此显示前必须重新加载。
                agent.memory_manager.load_memory()
                print(agent.memory_manager.get_context_string())
                
            elif user_input.lower() == 'process':
                if memory_processor:
                    print("\n🔄 正在手动触发记忆处理……")
                    results = memory_processor.process_recent_conversations()
                    
                    # 展示本轮实际执行的记忆操作。
                    operations = results.get('operations', [])
                    if operations:
                        print(f"\n📝 记忆操作（共 {len(operations)} 条）：")
                        for i, op in enumerate(operations, 1):
                            icon = {'add': '➕', 'update': '📝', 'delete': '🗑️'}.get(op['action'], '❓')
                            print(f"{i}. {icon} {op['action'].upper()}: {op.get('content', op.get('memory_id', 'N/A'))}")
                    else:
                        print("ℹ️ 本轮不需要更新长期记忆")
                    
                    summary = results.get('summary', {})
                    print(f"\n汇总：新增 {summary.get('added', 0)} 条，更新 {summary.get('updated', 0)} 条，删除 {summary.get('deleted', 0)} 条")
                else:
                    print("❌ 后台记忆处理未启用")
                    
            elif user_input.lower() == 'reset':
                agent.reset_session()
                print("✅ 已开始新的对话会话；长期记忆不会被删除")
                conversation_count = 0
                
            else:
                # 第 1 阶段：前台完成回答，并把原始 user/assistant 轮次写入 ConversationHistory。
                response = agent.chat(user_input)
                print(f"\n🤖 助手：{response}")
                conversation_count += 1
                
                # 第 2 阶段：只增加后台触发计数。真正的记忆写入由后台线程或 process/save 命令完成。
                if memory_processor:
                    memory_processor.increment_conversation_count()
                    
                    # 显示后台处理的触发状态。
                    if memory_processor.should_process():
                        print(f"\n【后台记忆处理】已达到 {conversation_interval} 轮触发条件")
                        # 给后台线程短暂时间完成本次处理。
                        time.sleep(2)
                    elif conversation_interval > 1:
                        conversations_until_process = conversation_interval - (conversation_count % conversation_interval)
                        if conversations_until_process < conversation_interval:
                            print(f"\n【后台记忆处理】再完成 {conversations_until_process} 轮对话后触发")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ 操作已中断。输入 'save' 执行记忆处理，或输入 'quit'/'exit' 停止后台线程并退出。")
        except Exception as e:
            print(f"\n❌ 错误：{str(e)}")
            logger.error(f"交互模式发生错误：{e}", exc_info=True)
    
    # 退出前停止后台线程。
    if memory_processor:
        memory_processor.stop_background_processing()


def demo_memory_system(memory_mode: MemoryMode = None, provider: Optional[str] = None, model: Optional[str] = None):
    """演示前台对话与后台记忆处理的分离式结构。"""
    print_section("演示：分离式记忆架构")
    
    # 确定模型服务商并读取密钥。
    provider = (provider or Config.PROVIDER).lower()
    api_key = Config.get_api_key(provider)
    if not api_key:
        print(f"❌ 请为模型服务商 '{provider}' 配置 API 密钥")
        if provider in ["dashscope", "qwen", "bailian"]:
            print("   export DASHSCOPE_API_KEY='your-api-key-here'")
        return
    
    # 创建演示专用用户。
    user_id = "demo_user"
    
    # 使用传入的记忆模式；未传入时交互选择。
    if memory_mode is None:
        memory_mode = select_memory_mode_interactive()
    
    # 初始化前台对话智能体。
    conv_config = ConversationConfig(
        enable_memory_context=True,
        enable_conversation_history=True
    )
    
    agent = ConversationalAgent(
        user_id=user_id,
        api_key=api_key,
        provider=provider,
        model=model,
        config=conv_config,
        memory_mode=memory_mode,
        verbose=True
    )
    
    # 演示模式使用每 2 轮触发一次的后台处理器。
    proc_config = MemoryProcessorConfig(
        conversation_interval=2,
        min_conversation_turns=1,
        output_operations=True
    )
    
    processor = BackgroundMemoryProcessor(
        user_id=user_id,
        api_key=api_key,
        provider=provider,
        model=model,
        config=proc_config,
        memory_mode=memory_mode,
        verbose=True
    )
    
    # 第一个会话：产生可供抽取的事实。
    print("\n📝 会话 1：产生原始对话")
    print("-"*40)
    
    messages = [
        "你好！我叫 Alice，在 TechCorp 担任产品经理。",
        "我习惯用 Python 写脚本，使用 VS Code，并且喜欢深色主题。",
        "我目前正在为公司开发一个新的移动应用项目。"
    ]
    
    for message in messages:
        print(f"\n👤 用户：{message}")
        response = agent.chat(message)
        print(f"🤖 助手：{response[:200]}..." if len(response) > 200 else f"🤖 助手：{response}")
        time.sleep(1)  # 两条消息之间短暂停顿。
    
    # 处理刚才产生的对话。
    print("\n\n🔄 正在从对话中提炼长期记忆……")
    print("-"*40)
    
    # 增加计数以满足处理条件。
    for _ in range(len(messages)):
        processor.increment_conversation_count()
    
    # 执行实际的记忆分析和工具调用。
    results = processor.process_recent_conversations()
    
    # 展示本轮记忆操作。
    operations = results.get('operations', [])
    if operations:
        print(f"\n📝 记忆操作（共 {len(operations)} 条）：")
        for i, op in enumerate(operations, 1):
            icon = {'add': '➕', 'update': '📝', 'delete': '🗑️'}.get(op['action'], '❓')
            print(f"{i}. {icon} {op['action'].upper()}: {op.get('content', op.get('memory_id', 'N/A'))}")
    else:
        print("ℹ️ 本轮不需要更新长期记忆")
    
    summary = results.get('summary', {})
    print(f"\n✅ 汇总：新增 {summary.get('added', 0)} 条，更新 {summary.get('updated', 0)} 条，删除 {summary.get('deleted', 0)} 条")
    
    # 新建会话，验证长期记忆是否跨会话保留。
    print("\n\n📝 会话 2：验证长期记忆持久化")
    print("-"*40)
    
    agent.reset_session()
    
    test_message = "你还记得我和我的工作吗？"
    print(f"\n👤 用户：{test_message}")
    response = agent.chat(test_message)
    print(f"🤖 助手：{response}")
    
    # 展示最终落盘的长期记忆。
    print("\n\n💭 最终长期记忆状态：")
    print("-"*40)
    print(agent.memory_manager.get_context_string())


def run_evaluation_mode(user_id: str, memory_mode: MemoryMode, verbose: bool = True, provider: Optional[str] = None, model: Optional[str] = None):
    """加载相邻评估框架并运行用户记忆测试案例。"""
    
    # 隔离加载评估框架，避免与当前目录的同名模块冲突。
    from pathlib import Path
    
    eval_framework_path = Path(__file__).parent.parent / "user-memory-evaluation"
    
    try:
        # 暂存当前已加载的同名模块。
        saved_modules = {}
        conflicting_modules = ['config', 'models', 'evaluator', 'framework']
        
        # 临时从 sys.modules 中移除冲突模块。
        for module_name in conflicting_modules:
            if module_name in sys.modules:
                saved_modules[module_name] = sys.modules[module_name]
                del sys.modules[module_name]
        
        # 临时把评估框架路径放到最高优先级。
        original_path = sys.path.copy()
        sys.path.insert(0, str(eval_framework_path))
        
        # 导入评估框架模块。
        import config as eval_config
        import models as eval_models  
        import evaluator as eval_evaluator
        import framework as eval_framework
        
        # 取得评估框架入口类。
        framework_class = eval_framework.UserMemoryEvaluationFramework
        
        # 恢复原始模块搜索路径。
        sys.path = original_path
        
        # 移除临时导入的评估模块，避免后续冲突。
        for module_name in conflicting_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]
        
        # 恢复原来暂存的同名模块。
        for module_name, module in saved_modules.items():
            sys.modules[module_name] = module
            
    except Exception as e:
        # 异常时也必须恢复模块环境。
        sys.path = original_path if 'original_path' in locals() else sys.path
        for module_name, module in saved_modules.items():
            sys.modules[module_name] = module
            
        print(f"❌ 错误：无法加载评估框架：{e}")
        print("请确认 user-memory-evaluation 已正确安装。")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print_section("评估模式 - 基于测试案例的评估")
    
    # 初始化评估框架。
    framework = framework_class()
    
    if not framework.test_suite:
        print("❌ 错误：没有加载到测试案例")
        sys.exit(1)
    
    print(f"\n✅ 已加载 {len(framework.test_suite.test_cases)} 个测试案例")
    
    # 确定模型服务商并读取密钥。
    provider = (provider or Config.PROVIDER).lower()
    api_key = Config.get_api_key(provider)
    if not api_key:
        print(f"❌ 错误：请为模型服务商 '{provider}' 配置 API 密钥")
        if provider in ["dashscope", "qwen", "bailian"]:
            print("   export DASHSCOPE_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # 先创建配置对象，再逐项设置评估所需选项。
    conv_config = ConversationConfig()
    conv_config.enable_memory_context = True
    conv_config.enable_conversation_history = True
    
    mem_config = MemoryProcessorConfig()
    mem_config.verbose = verbose
    
    # 使用一致的用户和记忆模式初始化前台智能体与后台处理器。
    agent = ConversationalAgent(
        user_id=user_id,
        api_key=api_key,
        provider=provider,
        model=model,
        config=conv_config,
        memory_mode=memory_mode,
        verbose=verbose
    )
    processor = BackgroundMemoryProcessor(
        user_id=user_id,
        api_key=api_key,
        provider=provider,
        model=model,
        config=mem_config,
        memory_mode=memory_mode,  # Pass memory_mode here!
        verbose=verbose
    )
    
    while True:
        print("\n" + "-"*60)
        print("可用操作：")
        print("1. 运行一个测试案例")
        print("2. 查看当前长期记忆")
        print("3. 清空记忆并重新开始")
        print("4. 退出评估模式")
        
        choice = input("\n请选择（1-4）：").strip()
        
        if choice == "1":
            # 先列出全部案例，再让用户选择。
            print("\n📋 可用测试案例：")
            framework.display_test_case_summary(show_full_titles=True, by_category=True)
            
            test_id = input("\n请输入要运行的测试案例 ID（输入 'cancel' 返回）：").strip()
            
            if test_id.lower() == 'cancel':
                continue
            
            test_case = framework.get_test_case(test_id)
            
            if not test_case:
                print(f"❌ 没有找到测试案例 '{test_id}'")
                continue
            
            print(f"\n{'='*60}")
            print(f"正在运行测试案例：{test_case.title}")
            print(f"类别：{test_case.category}")
            print("="*60)
            
            # 测试前清空记忆和对话状态，避免案例之间互相污染。
            print("\n🧹 正在清空测试前的全部记忆和对话状态……")
            
            # 1. 同时清空前台智能体和后台处理器的记忆管理器。
            if hasattr(agent.memory_manager, 'clear_all_memories'):
                agent.memory_manager.clear_all_memories()
                # 验证前台记忆确实已清空。
                memory_check = agent.memory_manager.get_context_string()
                if "No previous memory" not in memory_check:
                    print(f"  ⚠️ 警告：对话智能体的记忆可能没有完全清空")
                else:
                    print(f"  ✅ 对话智能体的记忆已清空")
            
            if hasattr(processor.memory_manager, 'clear_all_memories'):
                processor.memory_manager.clear_all_memories()
                # 验证后台记忆确实已清空。
                memory_check = processor.memory_manager.get_context_string()
                if "No previous memory" not in memory_check:
                    print(f"  ⚠️ 警告：后台处理器的记忆可能没有完全清空")
                else:
                    print(f"  ✅ 后台处理器的记忆已清空")
            
            # 2. 完全清空原始对话历史。
            if agent.conversation_history:
                agent.conversation_history.conversations = []
                agent.conversation_history.save_history()
                print(f"  ✅ 已清空用户 {user_id} 的对话历史")
            
            if processor.conversation_history:
                processor.conversation_history.conversations = []
                processor.conversation_history.save_history()
                print(f"  ✅ 已清空后台处理器的对话历史")
            
            # 3. 重置前台智能体的当前会话状态。
            agent.conversation = []
            agent._init_system_prompt()
            
            # 4. 重置工具调用计数。
            if hasattr(agent, 'tool_call_counts'):
                agent.tool_call_counts = {}
            
            print(f"  ✅ 所有记忆和状态均已清空，可以开始测试")
            
            # 处理测试案例中的历史对话。
            print(f"\n📚 正在处理 {len(test_case.conversation_histories)} 组对话历史……")
            
            # 从测试案例构造对话上下文。
            conversation_contexts = []
            
            for i, history in enumerate(test_case.conversation_histories, 1):
                print(f"\n对话 {i}/{len(test_case.conversation_histories)}：{history.conversation_id}")
                
                # 为当前历史记录构造消息列表。
                conversation = []
                
                # 按角色转换每条消息。
                for msg in history.messages:
                    if msg.role.value == "user":
                        conversation.append({"role": "user", "content": msg.content})
                    elif msg.role.value == "assistant":
                        conversation.append({"role": "assistant", "content": msg.content})
                
                conversation_contexts.append(conversation)
                
                # 同时写入对话历史，供后台处理器读取。
                if agent.conversation_history and hasattr(agent.conversation_history, 'add_turn'):
                    # 将用户与助手消息按轮次配对。
                    user_msg = None
                    for msg in history.messages:
                        if msg.role.value == "user":
                            user_msg = msg.content
                        elif msg.role.value == "assistant" and user_msg:
                            agent.conversation_history.add_turn(
                                session_id=f"eval_{history.conversation_id}",
                                user_message=user_msg,
                                assistant_message=msg.content
                            )
                            user_msg = None
            
            # 让后台处理器分析所有对话。
            if conversation_contexts:
                print(f"\n💾 正在为全部对话提炼长期记忆……")
                try:
                    results = processor.process_conversation_batch(conversation_contexts)
                    
                    # 汇总所有批次的记忆操作。
                    total_added = sum(r.get('summary', {}).get('added', 0) for r in results)
                    total_updated = sum(r.get('summary', {}).get('updated', 0) for r in results)
                    total_deleted = sum(r.get('summary', {}).get('deleted', 0) for r in results)
                    
                    print(f"  ✅ 记忆处理完成：")
                    print(f"     - 新增：{total_added} 条")
                    print(f"     - 更新：{total_updated} 条")
                    print(f"     - 删除：{total_deleted} 条")
                except Exception as e:
                    print(f"  ⚠️ 记忆处理失败：{e}")
            
            # 关键隔离：清空原始对话以模拟新会话。
            # 评估目标是结构化长期记忆，而不是直接读取原始聊天记录。
            if agent.conversation_history:
                # 暂存当前历史，评估结束后恢复。
                saved_conversations = agent.conversation_history.conversations if hasattr(agent.conversation_history, 'conversations') else []
                # 清空历史列表，模拟全新会话。
                agent.conversation_history.conversations = []
                print("\n🔄 已清空原始对话历史，开始新的会话")
                print("   （智能体接下来只能使用结构化长期记忆）")
            
            # 重置前台智能体当前消息。
            agent.conversation = []
            agent._init_system_prompt()
            
            # 后台处理器和前台智能体持有不同管理器实例，必须从磁盘重新加载，
            # 才能让前台看到后台刚写入的长期记忆。
            agent.memory_manager.load_memory()
            
            # 展示当前可用的长期记忆。
            memory_context = agent.memory_manager.get_context_string()
            if memory_context:
                print("\n💾 当前可用的长期记忆：")
                print("-"*40)
                print(memory_context[:500] + "..." if len(memory_context) > 500 else memory_context)
                print("-"*40)
            else:
                print("\n⚠️ 没有可用的结构化长期记忆")
            
            # 在没有原始历史的条件下回答测试问题。
            print(f"\n{'='*60}")
            print("用户问题：")
            print("-"*60)
            print(test_case.user_question)
            print("="*60)
            
            # 此时前台智能体只能使用结构化长期记忆。
            print("\n🤔 正在生成回答……")
            response = agent.chat(test_case.user_question)
            
            print("\n📝 智能体回答：")
            print("-"*60)
            print(response)
            print("-"*60)
            
            # 评估完成后恢复之前暂存的原始历史。
            if agent.conversation_history and 'saved_conversations' in locals():
                agent.conversation_history.conversations = saved_conversations
            
            # 提交回答并执行评估。
            print("\n⚖️ 正在评估回答……")
            result = framework.submit_and_evaluate(test_id, response)
            
            if result:
                # 展示评估结果。
                is_passed = result.passed if result.passed is not None else result.reward >= 0.6
                status = "✅ 通过" if is_passed else "❌ 未通过"
                
                print(f"\n{'='*60}")
                print("评估结果：")
                print("-"*60)
                print(f"状态：{status}")
                print(f"奖励得分：{result.reward:.3f}/1.000")
                
                if result.reasoning:
                    print(f"\n评估理由：")
                    print(result.reasoning)
                
                if result.suggestions:
                    print(f"\n改进建议：")
                    print(result.suggestions)
                print("="*60)
            else:
                print("❌ 评估执行失败")
            
            # 为下一测试案例清空当前历史。
            agent.conversation_history = []
            
        elif choice == "2":
            # 查看当前长期记忆。
            print("\n📄 当前长期记忆状态：")
            print("-"*60)
            print(processor.memory_manager.get_context_string())
                
        elif choice == "3":
            # 经用户确认后清空记忆。
            if input("\n⚠️ 确定清空全部记忆吗？请输入 yes 确认：").lower() == "yes":
                # 清空前台与后台持有的长期记忆。
                if hasattr(agent.memory_manager, 'clear_all_memories'):
                    agent.memory_manager.clear_all_memories()
                if hasattr(processor.memory_manager, 'clear_all_memories'):
                    processor.memory_manager.clear_all_memories()
                
                # 清空原始对话历史。
                if agent.conversation_history:
                    agent.conversation_history.conversations = []
                    agent.conversation_history.save_history()
                
                # 重置当前会话。
                agent.conversation = []
                agent._init_system_prompt()
                
                print("✅ 长期记忆和对话历史已清空")
                
        elif choice == "4":
            print("\n正在退出评估模式……")
            break
        else:
            print(f"❌ 无效选择：{choice}")


def select_mode_interactive() -> str:
    """让用户交互选择评估、对话或演示模式。"""
    print("\n" + "="*60)
    print("  🚀 选择运行模式")
    print("="*60)
    
    print("\n1. 评估模式")
    print("   - 运行 user-memory-evaluation 中的测试案例")
    print("   - 使用预设场景检查记忆系统")
    print("   - 输出评分和反馈")
    
    print("\n2. 交互模式")
    print("   - 与智能体实时对话")
    print("   - 后台自动处理长期记忆")
    print("   - 命令：memory、process、save、reset、quit/exit")
    
    print("\n3. 演示模式")
    print("   - 快速演示记忆系统")
    print("   - 展示对话如何被提炼成记忆")
    print("   - 验证记忆能否跨会话保留")
    
    print("\n" + "-"*60)
    
    while True:
        try:
            choice = input("\n请选择模式（1-3）：").strip()
            
            if choice == '1':
                print("✅ 已选择：评估模式")
                return "evaluation"
            elif choice == '2':
                print("✅ 已选择：交互模式")
                return "interactive"
            elif choice == '3':
                print("✅ 已选择：演示模式")
                return "demo"
            else:
                print("❌ 选择无效，请输入 1、2 或 3。")
        except KeyboardInterrupt:
            print("\n\n⚠️ 用户已取消操作")
            sys.exit(0)
        except Exception as e:
            print(f"❌ 错误：{e}")


def select_memory_mode_interactive() -> MemoryMode:
    """让用户交互选择长期记忆的存储模式。"""
    print("\n" + "="*60)
    print("  📝 选择记忆模式")
    print("="*60)
    
    print("\n1. 简单笔记")
    print("   - 保存简单事实和偏好")
    print("   - 每条记忆是一行事实")
    print("   - 示例：'用户邮箱：john@example.com'")
    
    print("\n2. 增强笔记")
    print("   - 保存带上下文的完整信息")
    print("   - 每条记忆可以是一段完整描述")
    print("   - 示例：'用户在 TechCorp 担任高级工程师，")
    print("             从事机器学习工作 3 年……'")
    
    print("\n3. JSON 卡片")
    print("   - 分层结构化记忆")
    print("   - 格式：类别 → 子类别 → 键 → 值")
    print("   - 示例：personal.contact.email → 'john@example.com'")
    
    print("\n4. 高级 JSON 卡片")
    print("   - 带元数据的完整记忆卡片")
    print("   - 每张卡片包含背景、人物和关系")
    print("   - 避免不同人物或场景之间的信息混淆")
    print("   - 示例：区分孩子与年长父母的医疗信息")
    
    print("\n" + "-"*60)
    
    while True:
        try:
            choice = input("\n请选择模式（1-4）：").strip()
            
            if choice == '1':
                print("✅ 已选择：简单笔记模式")
                return MemoryMode.NOTES
            elif choice == '2':
                print("✅ 已选择：增强笔记模式")
                return MemoryMode.ENHANCED_NOTES
            elif choice == '3':
                print("✅ 已选择：JSON 卡片模式")
                return MemoryMode.JSON_CARDS
            elif choice == '4':
                print("✅ 已选择：高级 JSON 卡片模式")
                return MemoryMode.ADVANCED_JSON_CARDS
            else:
                print("❌ 选择无效，请输入 1、2、3 或 4。")
        except KeyboardInterrupt:
            print("\n\n⚠️ 用户已取消操作")
            sys.exit(0)
        except Exception as e:
            print(f"❌ 错误：{e}")


def main():
    """解析 CLI，创建目录，再把执行分派到 interactive/demo/evaluation。"""
    parser = argparse.ArgumentParser(
        description="用户长期记忆实验：分离当前会话与跨会话持久记忆，并观察后台记忆抽取。"
    )
    
    parser.add_argument(
        "--mode",
        choices=["interactive", "demo", "evaluation"],
        default=None,
        help="运行模式；不指定时交互选择"
    )
    
    parser.add_argument(
        "--user",
        type=str,
        default="default_user",
        help="记忆所属用户 ID（默认：default_user）"
    )
    
    
    # BooleanOptionalAction 同时生成 --background-processing 和
    # --no-background-processing。不要使用 type=bool：字符串 "False"
    # 在 Python 中仍会被转换为 True，容易让实验条件与预期不一致。
    parser.add_argument(
        "--background-processing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用/禁用后台记忆处理（默认：启用）"
    )
    
    parser.add_argument(
        "--conversation-interval",
        type=int,
        default=1,
        help="每完成 N 轮对话触发一次记忆处理（默认：1）"
    )
    
    parser.add_argument(
        "--memory-mode",
        choices=["notes", "enhanced_notes", "json_cards", "advanced_json_cards"],
        help="长期记忆存储模式；不指定时交互选择"
    )
    
    parser.add_argument(
        "--provider",
        choices=["dashscope", "qwen", "bailian", "siliconflow", "doubao", "kimi", "moonshot", "openrouter"],
        default=None,
        help="模型服务商（默认读取 PROVIDER，未配置时为 kimi）"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型名称（默认使用对应服务商的默认模型）"
    )
    
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="关闭详细日志（默认开启）"
    )
    
    args = parser.parse_args()
    
    # --no-verbose 用于关闭默认开启的详细日志。
    verbose = not args.no_verbose
    
    # 命令行服务商优先于环境变量。
    provider = args.provider or Config.PROVIDER
    
    # 校验所选服务商的必要配置。
    if not Config.validate(provider):
        sys.exit(1)
    
    # 创建记忆、对话和日志目录。
    Config.create_directories()
    
    # 未通过命令行指定时，进入交互选择。
    execution_mode = args.mode
    if execution_mode is None:
        # 让用户选择执行模式。
        execution_mode = select_mode_interactive()
    
    # 确定记忆存储模式。
    if args.memory_mode:
        # 命令行已经指定模式。
        mode_map = {
            "notes": MemoryMode.NOTES,
            "enhanced_notes": MemoryMode.ENHANCED_NOTES,
            "json_cards": MemoryMode.JSON_CARDS,
            "advanced_json_cards": MemoryMode.ADVANCED_JSON_CARDS
        }
        memory_mode = mode_map[args.memory_mode]
    else:
        # 交互选择记忆模式。
        memory_mode = select_memory_mode_interactive()
    
    print("\n" + "🧠"*40)
    print("  用户长期记忆系统 - 前台与后台分离架构")
    print("🧠"*40)
    
    if execution_mode == "demo":
        demo_memory_system(memory_mode, provider, args.model)
    
    elif execution_mode == "evaluation":
        run_evaluation_mode(args.user, memory_mode, verbose, provider, args.model)
    
    elif execution_mode == "interactive":
        interactive_mode(
            user_id=args.user,
            memory_mode=memory_mode,
            enable_background_processing=args.background_processing,
            conversation_interval=args.conversation_interval,
            provider=provider,
            model=args.model
        )
    
    else:
        # 防御性分支：argparse 和交互选择正常情况下不会进入这里。
        print(f"❌ 未知运行模式：{execution_mode}")
        sys.exit(1)
    
    print("\n👋 用户长期记忆实验已结束。")


if __name__ == "__main__":
    main()
