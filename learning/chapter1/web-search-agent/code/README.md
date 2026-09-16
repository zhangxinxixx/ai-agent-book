# Web Search Agent 实验代码副本

本目录保存本地 `chapter1/web-search-agent` 的学习版本，包含中文备注、中文运行提示和测试。

目录说明：

- 当前目录：搜索 Agent 源码、实验脚本与测试。
- `agentbook/`：实验依赖的共享 provider 解析模块。

运行：

```powershell
cd code
python -m pip install -r requirements.txt
python main.py --provider offline-demo
python -m pytest -q
```

来源：[bojieli/ai-agent-book](https://github.com/bojieli/ai-agent-book)，上游代码使用 Apache License 2.0；许可证见 [`LICENSE.upstream`](LICENSE.upstream)。本目录中的修改版代码继续保留上游许可与来源说明。

不要提交 `.env`、API Key、授权头、缓存、真实查询内容或 Provider 原始响应。
