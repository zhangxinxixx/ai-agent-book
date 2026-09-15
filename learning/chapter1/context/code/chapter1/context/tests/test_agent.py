#!/usr/bin/env python3
"""
上下文感知 Agent 测试脚本
验证安装和基本功能
"""

import sys
from agent import ContextAwareAgent, ContextMode, ToolRegistry
import unittest
from unittest.mock import MagicMock, patch


class TestToolRegistry(unittest.TestCase):
    """测试工具注册表功能"""
    
    def test_calculator(self):
        """测试计算器工具"""
        tools = ToolRegistry()
        
        # 基本算术运算
        result = tools.calculate("2 + 2")
        self.assertEqual(result["result"], 4)
        
        # 复杂表达式
        result = tools.calculate("(10 * 5) + (20 / 4)")
        self.assertEqual(result["result"], 55.0)
        
        # 带数学函数
        result = tools.calculate("sqrt(16) + abs(-5)")
        self.assertEqual(result["result"], 9.0)
        
    def test_currency_converter(self):
        """测试货币转换工具"""
        tools = ToolRegistry()
        
        # 美元转欧元
        result = tools.convert_currency(100, "USD", "EUR")
        self.assertIn("converted_amount", result)
        self.assertIn("exchange_rate", result)
        self.assertGreater(result["converted_amount"], 0)
        
        # 货币符号归一化 (US$, S$, A$, C$, $)
        result_us = tools.convert_currency(100, "US$", "EUR")
        self.assertEqual(result_us["from_currency"], "USD")
        self.assertEqual(result_us["converted_amount"], 92.0)

        result_s = tools.convert_currency(100, "S$", "USD")
        self.assertEqual(result_s["from_currency"], "SGD")
        self.assertIn("converted_amount", result_s)

        result_a = tools.convert_currency(100, "A$", "USD")
        self.assertEqual(result_a["from_currency"], "AUD")
        self.assertIn("converted_amount", result_a)

        result_c = tools.convert_currency(100, "C$", "USD")
        self.assertEqual(result_c["from_currency"], "CAD")
        self.assertIn("converted_amount", result_c)
        # 无效货币
        result = tools.convert_currency(100, "XXX", "YYY")
        self.assertIn("error", result)
        result_invalid_s = tools.convert_currency(100, "S$INVALID", "USD")
        self.assertIn("error", result_invalid_s)

    def test_convert_currency_string_and_formatted_amounts(self):
        """
        验证 convert_currency 是否支持字符串和格式化数值金额。
        
        LLM 工具调用经常以字符串形式传递数值参数（例如 "100"、"$1,000.00"）。
        此前传入字符串会在浮点除法时触发 TypeError。该测试通过断言数字字符串和格式化货币字符串能够正确转换，防止回归缺陷。
        """
        tools = ToolRegistry()
        result_str = tools.convert_currency("100", "USD", "EUR")
        self.assertEqual(result_str["converted_amount"], 92.0)
        self.assertEqual(result_str["original_amount"], 100.0)

        result_formatted = tools.convert_currency("$1,000.00", "USD", "EUR")
        self.assertEqual(result_formatted["converted_amount"], 920.0)
        self.assertEqual(result_formatted["original_amount"], 1000.0)

        result_us_dollar = tools.convert_currency("US$100", "USD", "EUR")
        self.assertEqual(result_us_dollar["converted_amount"], 92.0)
        self.assertEqual(result_us_dollar["original_amount"], 100.0)

        result_currency_code = tools.convert_currency("USD$1,000", "USD$", "EUR")
        self.assertEqual(result_currency_code["converted_amount"], 920.0)
        self.assertEqual(result_currency_code["original_amount"], 1000.0)

        result_comma_large = tools.convert_currency("1,234,567.89", "USD", "EUR")
        self.assertEqual(result_comma_large["original_amount"], 1234567.89)

        result_euro_sym = tools.convert_currency("€ 500.25", "EUR", "USD")
        self.assertIn("converted_amount", result_euro_sym)

        result_invalid_str = tools.convert_currency("invalid_str", "USD", "EUR")
        self.assertIn("error", result_invalid_str)
    
    def test_pdf_parser_structure(self):
        """测试 PDF 解析器结构（不依赖实际 PDF）"""
        tools = ToolRegistry()
        
        # 使用无效 URL 进行测试（应优雅处理错误）
        result = tools.parse_pdf("http://invalid-url-for-testing.com/test.pdf")
        self.assertIn("error", result)


class TestContextModes(unittest.TestCase):
    """测试不同上下文模式"""
    
    @patch.dict('os.environ', {'SILICONFLOW_API_KEY': 'test_key'})
    def setUp(self):
        """设置测试夹具"""
        self.api_key = "test_key"
    
    def test_context_mode_initialization(self):
        """测试 Agent 在不同上下文模式下的初始化"""
        for mode in ContextMode:
            agent = ContextAwareAgent(self.api_key, mode)
            self.assertEqual(agent.context_mode, mode)
            self.assertEqual(agent.trajectory.context_mode, mode)
    
    def test_context_building(self):
        """测试不同模式下的上下文构建"""
        # 全上下文模式
        agent = ContextAwareAgent(self.api_key, ContextMode.FULL)
        agent.trajectory.reasoning_steps = ["Step 1", "Step 2"]
        agent.trajectory.tool_calls.append(
            MagicMock(tool_name="test", arguments={}, result={"test": "result"})
        )
        
        context = agent._build_context()
        self.assertIn("Previous Reasoning Steps", context)
        self.assertIn("Tool Call History", context)
        
        # 无推理模式
        agent_no_reasoning = ContextAwareAgent(self.api_key, ContextMode.NO_REASONING)
        agent_no_reasoning.trajectory.reasoning_steps = ["Step 1"]
        context = agent_no_reasoning._build_context()
        self.assertNotIn("Previous Reasoning Steps", context)
        
        # 无历史模式
        agent_no_history = ContextAwareAgent(self.api_key, ContextMode.NO_HISTORY)
        agent_no_history.trajectory.tool_calls.append(
            MagicMock(tool_name="test", arguments={}, result={"test": "result"})
        )
        context = agent_no_history._build_context()
        self.assertEqual(context, "")


class TestAblationScenarios(unittest.TestCase):
    """测试消融实验场景"""
    
    def test_tool_execution(self):
        """测试工具执行"""
        agent = ContextAwareAgent("test_key", ContextMode.FULL)
        
        # 测试计算器执行
        result = agent._execute_tool("calculate", {"expression": "2 + 2"})
        self.assertEqual(result["result"], 4)
        
        # 测试未知工具
        result = agent._execute_tool("unknown_tool", {})
        self.assertIn("error", result)
    
    def test_trajectory_reset(self):
        """测试轨迹重置"""
        agent = ContextAwareAgent("test_key", ContextMode.FULL)
        
        # 向轨迹中添加部分数据
        agent.trajectory.reasoning_steps.append("Test step")
        agent.trajectory.tool_calls.append(
            MagicMock(tool_name="test", arguments={})
        )
        
        # 重置
        agent.reset()
        
        # 检查是否已清空
        self.assertEqual(len(agent.trajectory.reasoning_steps), 0)
        self.assertEqual(len(agent.trajectory.tool_calls), 0)
        self.assertEqual(agent.trajectory.context_mode, ContextMode.FULL)

    def test_provider_error_evidence_redacts_credential_fragments(self):
        agent = ContextAwareAgent("test_key", ContextMode.FULL)
        leaked = "Bearer sk-test-secret-token and <ak-provider-token>"
        agent.client.chat.completions.create = MagicMock(
            side_effect=RuntimeError(leaked)
        )

        result = agent.execute_task("smoke", max_iterations=1)
        recorded = agent.trajectory.api_turns[-1]["error"]["message"]

        self.assertNotIn("sk-test-secret-token", recorded)
        self.assertNotIn("ak-provider-token", recorded)
        self.assertIn("[redacted]", recorded)
        self.assertNotIn("sk-test-secret-token", result["error"])


def run_integration_test():
    """运行简单的集成测试"""
    print("\n" + "="*60)
    print("INTEGRATION TEST")
    print("="*60)
    
    # 检查 API Key 是否可用
    import os
    api_key = os.getenv("SILICONFLOW_API_KEY")
    
    if not api_key:
        print("⚠️ Skipping integration test (no API key found)")
        print("Set SILICONFLOW_API_KEY to run integration tests")
        return False
    
    print("✅ API key found, running integration test...")
    
    try:
        # 创建 Agent
        agent = ContextAwareAgent(api_key, ContextMode.FULL)
        
        # 不需要外部 PDF 的简单任务
        simple_task = "Calculate: What is 15% of $2500? Then convert the result to EUR."
        
        print(f"\nTest task: {simple_task}")
        print("Running...")
        
        # 带超时执行
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Integration test timed out")
        
        # 设置 30 秒超时
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        try:
            result = agent.execute_task(simple_task, max_iterations=3)
            signal.alarm(0)  # 取消定时器
            
            print("\n✅ Integration test completed!")
            print(f"Success: {result.get('success', False)}")
            print(f"Tool calls: {len(result['trajectory'].tool_calls)}")
            
            if result.get('final_answer'):
                print(f"Answer preview: {result['final_answer'][:100]}...")
            
            return True
            
        except TimeoutError:
            print("❌ Integration test timed out")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        return False


def main():
    """主测试运行器"""
    print("\n" + "="*60)
    print("CONTEXT-AWARE AGENT TEST SUITE")
    print("="*60)
    
    # 运行单元测试
    print("\n📋 Running unit tests...")
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试用例
    suite.addTests(loader.loadTestsFromTestCase(TestToolRegistry))
    suite.addTests(loader.loadTestsFromTestCase(TestContextModes))
    suite.addTests(loader.loadTestsFromTestCase(TestAblationScenarios))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 汇总
    print("\n" + "="*60)
    print("UNIT TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All unit tests passed!")
    else:
        print("❌ Some tests failed")
        sys.exit(1)
    
    # 如条件允许则运行集成测试
    print("\n" + "="*60)
    integration_success = run_integration_test()
    
    # 最终汇总
    print("\n" + "="*60)
    print("FINAL TEST SUMMARY")
    print("="*60)
    
    if result.wasSuccessful():
        print("✅ Unit tests: PASSED")
    else:
        print("❌ Unit tests: FAILED")
    
    if integration_success:
        print("✅ Integration test: PASSED")
    else:
        print("⚠️ Integration test: SKIPPED or FAILED")
    
    print("\n🎉 Testing complete!")
    print("="*60 + "\n")
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
