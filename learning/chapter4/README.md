# Chapter 4：Tools、Tool Calling 与 MCP

本章学习 Agent 为什么需要 Tools，以及 Function Calling / Tool Calling 和 MCP 各自解决什么问题。学习材料来自本地最新的实验 4-2 与 4-4 代码快照。

章节正文：[第四章：工具](第四章-工具.md)

## 学习目标

- 理解 LLM 负责判断，Tool 负责产生真实动作。
- 理解 Tool Calling 是“生成结构化调用请求”，不是模型直接执行函数。
- 理解 MCP Host、Client、Server 以及 `tools/list`、`tools/call`。
- 跑通感知工具和执行工具实验。
- 能说清一次完整的“判断 → 调用工具 → 获取结果 → 回答”流程。

## 项目目录

| 实验 | 重点 | 入口 |
| --- | --- | --- |
| 4-2 Perception Tools + MCP | 工具发现、读取外部信息、把结果转成 Observation | [进入 4-2](perception-tools/) |
| 4-4 Execution Tools + MCP | Guard、安全执行、环境副作用和结果加工 | [进入 4-4](execution-tools/) |

每个实验目录都包含：

- `code/`：本次学习使用的源码快照。
- `学习笔记.md`：机制与关键概念。
- `实验逻辑流程.md`：架构、时序和状态变化。
- `实验总结.md`：本轮结果、收获与证据边界。
- `运行手册.md`：项目原有运行说明。
- `assets/`：静态架构图。

## 一次完整的 Tool Calling 流程

```text
1. 用户提出任务
2. LLM 根据工具 Schema 判断是否需要工具
3. LLM 返回 tool_call：工具名 + 参数
4. Agent Harness 校验参数、权限和风险
5. Harness 通过 MCP Client 或本地函数调用 Tool
6. Tool 访问环境并返回结构化 Tool Result
7. Harness 把结果作为 Observation 追加到对话
8. LLM 再次判断：继续调用工具，或生成最终回答
```

MCP 统一第 5、6 步的工具发现和调用协议；它本身不负责第 2、8 步的 Agent 决策。

## 本轮验证结论

- 4-2：MCP 冒烟测试通过，SDK `2.2.0`，协议 `2026-07-28`，发现 127 个工具，并成功调用 `file_reader`。
- 4-4：严格离线 demo 运行成功；文件写入、语法拦截、代码执行、Shell 校验、长输出持久化和危险命令拒绝均得到实际结果；自动测试为 `63 passed`。

本轮没有调用在线模型或外部服务，也没有把固定 demo 说成 LLM 自主选择工具。历史 `validation` 没有发布，避免把旧实验结果误当成当前源码的实时结果。

## 打卡简述

> 一次完整的 Tool Calling 中，模型先根据用户问题和工具 Schema 判断是否需要工具，并生成包含工具名和参数的结构化调用请求。Harness 收到请求后先校验参数、路径、权限和危险操作，再通过本地函数或 MCP Server 执行工具。工具把 `success`、输出和错误等结果返回给 Harness，Harness 将其作为 Observation 交回模型。模型根据真实结果决定继续调用工具，还是组织最终回答。模型负责判断，工具负责执行，Guard 负责执行前控制风险，Verifier 负责执行后核验证据。

打卡截图建议使用实际运行 `smoke_test_mcp_v2.py` 或 `cli.py demo` 的终端结果；本目录不使用伪造截图代替个人实跑证据。

