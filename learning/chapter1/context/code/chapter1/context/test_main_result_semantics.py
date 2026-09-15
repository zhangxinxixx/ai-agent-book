import logging
import sys
from types import SimpleNamespace

import pytest

import main as context_main
from main import _completed, get_sample_tasks, main, print_task_result


def test_completed_field_is_authoritative_over_legacy_success_alias():
    assert _completed({"completed": False, "success": True}) is False
    assert _completed({"completed": True, "success": False}) is True


def test_completed_falls_back_for_old_result_artifacts():
    assert _completed({"success": True}) is True
    assert _completed({"success": False}) is False


def test_task_result_printout_labels_each_terminal_section(capsys):
    result = {
        "completed": True,
        "iterations": 2,
        "trajectory": SimpleNamespace(tool_calls=[]),
        "final_answer": "任务已完成。",
        "error": None,
    }

    print_task_result(result, context_mode="full")

    output = capsys.readouterr().out
    assert "【任务执行结果】" in output
    assert "【上下文模式】" in output
    assert "【终止响应】" in output
    assert "【执行轮次】" in output
    assert "【工具调用】" in output
    assert "【最终回答】" in output


def test_samples_present_chinese_learning_text_without_changing_task_structure():
    samples = get_sample_tasks()

    assert len(samples) == 5
    assert samples[0]["name"] == "📊 货币换算任务"
    assert "将 1000 USD" in samples[0]["task"]
    assert "Q1: $2,500,000 USD" in samples[2]["task"]
    assert "file://" in samples[1]["task"] or "https://" in samples[1]["task"]


def test_plural_modes_with_a_name_switches_mode_without_executing_a_task(
    monkeypatch, capsys
):
    """常见输入 ``modes no_history`` 应被兼容，不能意外消耗一次模型调用。"""

    created_modes = []

    class FakeAgent:
        def __init__(self, _api_key, context_mode, **_kwargs):
            created_modes.append(context_mode)
            self.conversation_history = []
            self.trajectory = SimpleNamespace(tool_calls=[])

        def execute_task(self, _task):
            raise AssertionError("模式切换不能执行任务或请求模型")

    commands = iter(["modes no_history", "quit"])
    monkeypatch.setattr(context_main, "ContextAwareAgent", FakeAgent)
    monkeypatch.setattr(context_main, "ensure_sample_pdfs", lambda: True)
    monkeypatch.setattr(context_main, "get_sample_tasks", lambda: [])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))

    context_main.interactive_mode("test-key", provider="moonshot", model="kimi-k3")

    output = capsys.readouterr().out
    assert "【命令兼容】" in output
    assert "【上下文模式已切换】no_history" in output
    assert created_modes == [context_main.ContextMode.FULL, context_main.ContextMode.NO_HISTORY]


def test_cli_accepts_kimi_alias_and_resolves_the_moonshot_provider(
    monkeypatch, caplog
):
    """The documented ``kimi`` CLI alias must reach Moonshot resolution."""

    monkeypatch.setenv("MOONSHOT_API_KEY", "")
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--mode", "single", "--task", "smoke", "--provider", "kimi"],
    )
    caplog.set_level(logging.ERROR)

    with pytest.raises(SystemExit) as exited:
        main()

    assert exited.value.code == 1
    assert "No API key found for provider 'moonshot'" in caplog.text
