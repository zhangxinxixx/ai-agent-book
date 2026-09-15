"""Terminal output should explain the current agent stage in Chinese."""

import logging
from unittest.mock import MagicMock

from agent import ContextAwareAgent, ContextMode


def test_agent_runtime_logs_label_iteration_request_and_terminal_response(caplog, capsys):
    agent = ContextAwareAgent("test-key", ContextMode.FULL)
    message = MagicMock()
    message.tool_calls = None
    message.content = "任务已完成。"
    message.reasoning_content = None
    message.model_extra = {}
    message.dict.return_value = {"role": "assistant", "content": "任务已完成。"}
    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    response.model_dump.return_value = {"id": "test-response", "choices": []}
    agent.client = MagicMock()
    agent.client.chat.completions.create.return_value = response

    caplog.set_level(logging.INFO, logger="agent")
    result = agent.execute_task("smoke", max_iterations=1)

    assert result["completed"] is True
    assert "【执行轮次】第 1/1 轮" in caplog.text
    assert "【模型请求】" in caplog.text
    assert "【模型响应】" in caplog.text
    assert "【终止条件】" in caplog.text
    verbose_output = capsys.readouterr().out
    assert "【请求载荷｜第 1 轮】" in verbose_output
    assert "【模型响应｜第 1 轮】" in verbose_output


def test_http_status_logs_are_labeled_for_terminal_readers(caplog):
    caplog.set_level(logging.INFO, logger="httpx")

    logging.getLogger("httpx").info("HTTP Request: POST https://example.test/v1/chat HTTP/1.1 200 OK")

    assert "【HTTP 通信】" in caplog.text
