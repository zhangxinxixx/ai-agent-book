"""将执行工具注册并暴露为标准输入输出 MCP Server。"""

import asyncio
import json
from typing import Any
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

from config import Config
from llm_helper import LLMHelper
from file_tools import FileTools
from execution_tools import ExecutionTools
from external_tools import ExternalTools
from extended_tools import ExtendedTools


# 创建 MCP Server 实例。
server = Server("execution-tools")

# 共享同一批工具实例；MCP 层只负责说明书、参数接收和调用路由。
llm_helper = LLMHelper()
file_tools = FileTools(llm_helper)
execution_tools = ExecutionTools(llm_helper)
external_tools = ExternalTools(llm_helper)
extended_tools = ExtendedTools()


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """返回给 MCP Client/Agent 的工具说明书，而不是搜索或执行结果。"""
    return [
        types.Tool(
            name="file_write",
            description="向文件写入内容，并在写入前自动执行语法校验",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径（相对工作区或绝对路径）"
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的内容"
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "是否允许覆盖已存在文件",
                        "default": False
                    }
                },
                "required": ["path", "content"]
            }
        ),
        types.Tool(
            name="file_edit",
            description="通过搜索和替换编辑已有文件",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径"
                    },
                    "search": {
                        "type": "string",
                        "description": "要搜索的文本"
                    },
                    "replace": {
                        "type": "string",
                        "description": "替换后的文本"
                    }
                },
                "required": ["path", "search", "replace"]
            }
        ),
        types.Tool(
            name="code_interpreter",
            description="执行多语言代码并返回结构化结果；Python 优先使用 Docker，不可用时会明确降级为本地进程",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "要执行的代码"
                    },
                    "language": {
                        "type": "string",
                        "description": "编程语言（python、javascript、typescript、go、java、cpp、rust、php、bash）",
                        "default": "python"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "执行超时秒数",
                        "default": 30.0
                    },
                    "stdin": {
                        "type": "string",
                        "description": "可选的程序标准输入"
                    },
                    "files": {
                        "type": "object",
                        "description": "可选附加文件（文件名到内容的映射）",
                        "additionalProperties": {"type": "string"}
                    }
                },
                "required": ["code"]
            }
        ),
        types.Tool(
            name="virtual_terminal",
            description="执行 Shell 命令，并返回退出码、标准输出和标准错误",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的 Shell 命令"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "超时秒数",
                        "default": 30
                    }
                },
                "required": ["command"]
            }
        ),
        types.Tool(
            name="google_calendar_add",
            description="向 Google Calendar 添加真实日历事件",
            inputSchema={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "事件标题"
                    },
                    "start_time": {
                        "type": "string",
                        "description": "开始时间（ISO 8601，例如 2024-01-01T10:00:00）"
                    },
                    "end_time": {
                        "type": "string",
                        "description": "结束时间（ISO 8601）"
                    },
                    "description": {
                        "type": "string",
                        "description": "事件描述"
                    },
                    "location": {
                        "type": "string",
                        "description": "事件地点"
                    }
                },
                "required": ["summary", "start_time", "end_time"]
            }
        ),
        types.Tool(
            name="github_create_pr",
            description="创建 GitHub Pull Request",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_name": {
                        "type": "string",
                        "description": "仓库名（owner/repo 格式）"
                    },
                    "title": {
                        "type": "string",
                        "description": "PR 标题"
                    },
                    "body": {
                        "type": "string",
                        "description": "PR 描述"
                    },
                    "head_branch": {
                        "type": "string",
                        "description": "源分支"
                    },
                    "base_branch": {
                        "type": "string",
                        "description": "目标分支",
                        "default": "main"
                    }
                },
                "required": ["repo_name", "title", "body", "head_branch"]
            }
        ),
        types.Tool(
            name="excel_create_with_formula_and_screenshot",
            description="创建 XLSX、写入公式，并使用 LibreOffice 渲染真实截图",
            inputSchema={
                "type": "object",
                "properties": {
                    "output_path": {"type": "string"},
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "item": {"type": "string"},
                                "quantity": {"type": "number"},
                                "unit_price": {"type": "number"},
                            },
                            "required": ["item", "quantity", "unit_price"],
                        },
                    },
                },
                "required": ["output_path", "rows"],
            }
        ),
        types.Tool(
            name="webhook_post",
            description="向真实 HTTPS Webhook 地址发送 JSON POST 请求",
            inputSchema={"type": "object", "properties": {
                "url": {"type": "string"}, "payload": {"type": "object"}},
                "required": ["url", "payload"]}
        ),
        types.Tool(
            name="browser_navigate",
            description="使用真实无头 Chromium 打开网页、提取内容并保存截图",
            inputSchema={"type": "object", "properties": {
                "url": {"type": "string"}, "screenshot_path": {"type": "string"}},
                "required": ["url", "screenshot_path"]}
        ),
        types.Tool(
            name="virtual_desktop_execute",
            description="通过 X11 键盘事件操作有界面 Chromium 桌面并保留截图",
            inputSchema={"type": "object", "properties": {
                "url": {"type": "string"},
                "screenshot_path": {"type": "string"},
                "expected_title": {"type": ["string", "null"]}},
                "required": ["url", "screenshot_path"]}
        ),
        types.Tool(
            name="virtual_mobile_execute",
            description="通过 ADB 操作正在运行的 AndroidWorld 模拟器并保留截图",
            inputSchema={"type": "object", "properties": {
                "container_name": {"type": "string"},
                "screenshot_path": {"type": "string"}},
                "required": ["container_name", "screenshot_path"]}
        ),
        types.Tool(
            name="environment_capabilities",
            description="检查真实 Computer Use 容器和 Android 设备是否可用",
            inputSchema={"type": "object", "properties": {}}
        )
    ]


@server.call_tool()
async def handle_call_tool(
    name: str,
    arguments: dict[str, Any] | None
) -> list[types.TextContent]:
    """按工具名路由调用，并把结果统一序列化成 MCP 文本内容。"""
    if arguments is None:
        arguments = {}
    
    try:
        # 这是工具真正被路由执行的位置。
        if name == "file_write":
            result = await file_tools.write_file(
                path=arguments["path"],
                content=arguments["content"],
                overwrite=arguments.get("overwrite", False)
            )
        elif name == "file_edit":
            result = await file_tools.edit_file(
                path=arguments["path"],
                search=arguments["search"],
                replace=arguments["replace"]
            )
        elif name == "code_interpreter":
            result = await execution_tools.code_interpreter(
                code=arguments["code"],
                language=arguments.get("language") or "python",
                timeout=arguments.get("timeout", 30.0),
                stdin=arguments.get("stdin"),
                files=arguments.get("files")
            )
        elif name == "virtual_terminal":
            result = await execution_tools.virtual_terminal(
                command=arguments["command"],
                timeout=arguments.get("timeout", 30)
            )
        elif name == "google_calendar_add":
            result = await external_tools.google_calendar_add(
                summary=arguments["summary"],
                start_time=arguments["start_time"],
                end_time=arguments["end_time"],
                description=arguments.get("description"),
                location=arguments.get("location")
            )
        elif name == "github_create_pr":
            result = await external_tools.github_create_pr(
                repo_name=arguments["repo_name"],
                title=arguments["title"],
                body=arguments["body"],
                head_branch=arguments["head_branch"],
                base_branch=arguments.get("base_branch", "main")
            )
        elif name == "excel_create_with_formula_and_screenshot":
            result = await extended_tools.excel_create_with_formula_and_screenshot(
                arguments["output_path"], arguments["rows"])
        elif name == "webhook_post":
            result = await extended_tools.webhook_post(arguments["url"], arguments["payload"])
        elif name == "browser_navigate":
            result = await extended_tools.browser_navigate(
                arguments["url"], arguments["screenshot_path"])
        elif name == "virtual_desktop_execute":
            result = await extended_tools.virtual_desktop_execute(
                arguments["url"], arguments["screenshot_path"], arguments.get("expected_title"))
        elif name == "virtual_mobile_execute":
            result = await extended_tools.virtual_mobile_execute(
                arguments["container_name"], arguments["screenshot_path"])
        elif name == "environment_capabilities":
            result = await extended_tools.environment_capabilities()
        else:
            raise ValueError(f"未知工具：{name}")
        
        # MCP TextContent 中承载 JSON 字符串，供上游 Agent 作为 Observation 使用。
        return [
            types.TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )
        ]
        
    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": f"工具执行失败：{str(e)}"
                }, indent=2)
            )
        ]


async def main():
    """通过标准输入输出启动 MCP Server 并进入消息循环。"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="execution-tools",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
