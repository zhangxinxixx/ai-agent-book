# 实验 2-10：Context Compression 上下文压缩

本目录对应上游实验：[bojieli/ai-agent-book · chapter2/context-compression](https://github.com/bojieli/ai-agent-book/tree/main/chapter2/context-compression)。

| 部分 | 内容 |
| --- | --- |
| [`code/`](code/) | 当前本地版本的实验源码、测试、依赖与中文学习脚本 |
| [运行手册](运行手册.md) | 环境准备、运行顺序、参数、输出解释和故障排查 |
| [实验逻辑流程](实验逻辑流程.md) | 入口调用链、消息状态变化、六种策略分支和 Mermaid 时序图 |
| [实验总结](实验总结.md) | 本轮实测结果、指标限制、结论和下一轮计划 |

## 最小运行顺序

```powershell
cd code
python -m pip install -r requirements.txt
python -m pytest -q -p no:cacheprovider
python .\my_agent_test.py
python .\experiment.py --help
```

真实模型实验需要自行设置 Provider Key。不要提交 `.env`、API Key、授权头、缓存或未经脱敏的真实调用日志。
