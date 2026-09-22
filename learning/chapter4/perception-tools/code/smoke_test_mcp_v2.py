#!/usr/bin/env python3
"""离线 MCP v2 冒烟测试：启动 stdio 通信、拉取工具列表（tools/list）并调用单个工具（call_tool）。

本脚本演示了 MCP（Model Context Protocol，模型上下文协议）客户端与服务端的完整交互流程：
1. 【启动通信】：通过 stdio（标准输入输出）启动 MCP 服务端进程，建立双向 JSON-RPC 通信管道。
2. 【协议握手】：客户端与服务端协商协议版本（例如 2026-07-28）。
3. 【工具发现（tools/list）】：客户端查询服务端提供了哪些可用工具及其 JSON Schema。
4. 【工具调用（tools/call）】：客户端传入结构化参数发起调用，服务端执行并返回结构化结果。
"""

from __future__ import annotations

import asyncio
import os
import sys
from importlib.metadata import version
from pathlib import Path

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client

# 当前脚本所在目录与 MCP 服务端入口脚本（src/main.py）
HERE = Path(__file__).resolve().parent
SERVER = HERE / "src" / "main.py"
# 实验 4-2 约定的 MCP 协议版本号
PROTOCOL_VERSION = "2026-07-28"


async def smoke_test() -> None:
    """运行 MCP 客户端与服务端的端到端冒烟测试。"""
    # 1. 检查 Python 环境中的 MCP SDK 版本（本实验要求 MCP SDK 2.x）
    sdk_version = version("mcp")
    if sdk_version.split(".", 1)[0] != "2":
        raise RuntimeError(f"实验 4-2 要求 mcp>=2,<3；当前检测到的版本为：{sdk_version}")

    # 2. 配置通过子进程 stdio 方式启动的服务端参数
    #    相当于在后台运行：python src/main.py，通过标准输入输出作为传输层
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER)],
        env=os.environ.copy(),
    )

    # 3. 创建并进入 MCP Client 上下文管理器
    #    在此阶段，客户端与服务端会自动建立连接并完成协议握手（Initialize Handshake）
    async with Client(stdio_client(parameters)) as client:
        # 校验握手协商后的协议版本是否符合预期
        if client.protocol_version != PROTOCOL_VERSION:
            raise RuntimeError(
                f"预期的协议版本为 {PROTOCOL_VERSION}，但服务端协商返回了 {client.protocol_version}"
            )

        # 4. 【tools/list 工具发现】：
        #    客户端向服务端发送 tools/list 请求，获取服务端暴露的所有感知工具元数据
        listed = await client.list_tools()
        names = {tool.name for tool in listed.tools}
        if "file_reader" not in names:
            raise RuntimeError("tools/list 返回的工具列表中缺少基础工具 file_reader")

        # 5. 【tools/call 工具调用】：
        #    客户端向服务端发送 tools/call 请求，指定工具名称 "file_reader" 并提供参数字典
        #    服务端解析参数、读取本地文件并返回响应结构
        result = await client.call_tool(
            "file_reader",
            arguments={"file_path": str(HERE / "requirements.txt"), "max_length": 2_000},
        )
        if result.is_error:
            raise RuntimeError(f"tools/call 调用失败：{result.content!r}")

        # 6. 获取服务端信息并输出冒烟测试成功摘要
        server_name = client.server_info.name if client.server_info else None
        print(
            f"MCP 冒烟测试通过：sdk 版本={sdk_version}, "
            f"协议版本={client.protocol_version}, 服务端名称={server_name}, 成功发现工具数={len(names)}"
        )


if __name__ == "__main__":
    asyncio.run(smoke_test())
