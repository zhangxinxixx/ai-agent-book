# Chapter 1 · Context 学习目录

本目录对应上游实验：[bojieli/ai-agent-book · chapter1/context](https://github.com/bojieli/ai-agent-book/tree/main/chapter1/context)。

| 部分 | 内容 |
| --- | --- |
| [`code/`](code/) | 当前本地版本的实验源码、测试、依赖和示例文件，保留中文备注与中文运行提示 |
| [学习笔记](学习笔记.md) | 核心概念、五种上下文模式、代码阅读顺序和动手路线 |
| [实验逻辑流程](实验逻辑流程.md) | Agent 主循环、上下文消融、证据采集和结果判定流程 |
| [实验总结](实验总结.md) | 离线测试、真实模型实验结果、限制和下一步 |

## 运行入口

```powershell
cd code
python -m pip install -r requirements.txt
python main.py --help
python -m pytest -q
```

真实模型实验需要自行设置对应 provider 的环境变量。不要把 API Key、`.env`、授权头或含密钥的日志提交到 GitHub。

