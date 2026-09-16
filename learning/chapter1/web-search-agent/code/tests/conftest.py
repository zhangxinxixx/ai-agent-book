"""网络搜索 Agent 测试套件的共享 pytest 夹具。"""

import json
import socket
from types import SimpleNamespace

import pytest

PROVIDER_ENV_VARS = (
    "MOONSHOT_API_KEY",
    "KIMI_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_MODEL",
)


@pytest.fixture(autouse=True)
def isolate_provider_environment(monkeypatch):
    """确保所有测试中均不包含开发者凭据和提供商覆盖项。"""
    for variable in PROVIDER_ENV_VARS:
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch):
    """若单元测试意外尝试建立网络连接则快速失败报错。"""

    def deny_network(*args, **kwargs):
        raise AssertionError("Unit tests must not access the external network")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket, "getaddrinfo", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)


@pytest.fixture
def make_tool_call():
    """为 mock 的模型响应构建符合 SDK 形状的最小工具调用对象。"""

    def factory(
        *,
        name="web_search",
        arguments=None,
        call_id="call-1",
    ):
        payload = arguments if arguments is not None else {"query": "example"}
        return SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(
                name=name,
                arguments=json.dumps(payload, ensure_ascii=False),
            ),
        )

    return factory


@pytest.fixture
def make_choice():
    """为确定性 Agent 测试构建符合 SDK 形状的最小聊天选项对象。"""

    def factory(
        *,
        finish_reason="stop",
        content="",
        reasoning_content=None,
        tool_calls=None,
    ):
        return SimpleNamespace(
            finish_reason=finish_reason,
            message=SimpleNamespace(
                content=content,
                reasoning_content=reasoning_content,
                tool_calls=list(tool_calls or []),
            ),
        )

    return factory
