# Chapter 1 · Web Search Agent 学习目录

本目录对应上游实验：[bojieli/ai-agent-book · chapter1/web-search-agent](https://github.com/bojieli/ai-agent-book/tree/main/chapter1/web-search-agent)。

| 部分 | 内容 |
| --- | --- |
| [`code/`](code/) | 当前本地版本的搜索 Agent 源码、测试和依赖，保留中文备注与中文运行提示 |
| [运行手册](运行手册.md) | 环境准备、离线演示、真实搜索、正式实验和故障排查 |
| [实验逻辑流程](实验逻辑流程.md) | Kimi、Formula Tools、Fiber 与本地 Agent Loop 的完整调用顺序 |

## 运行入口

```powershell
cd code
python -m pip install -r requirements.txt
python main.py --provider offline-demo
python -m pytest -q
```

真实搜索需要自行配置 Moonshot API Key，并可能产生模型与联网工具费用。不要把 API Key、`.env`、原始响应或含敏感查询的日志提交到 GitHub。
