# Context 实验代码副本

本目录保存本地 `chapter1/context` 的学习版本，包含中文备注、中文运行提示、测试和示例文件。

代码目录保持上游相对结构：

- `chapter1/context/`：实验源码与测试。
- `agentbook/`：实验依赖的共享 provider 解析模块。

运行：

```powershell
cd chapter1/context
python -m pip install -r requirements.txt
python -m pytest -q
python main.py --help
```

来源：[bojieli/ai-agent-book](https://github.com/bojieli/ai-agent-book)，上游代码使用 Apache License 2.0；许可证见 [`LICENSE.upstream`](LICENSE.upstream)。本目录中的修改版代码继续保留上游许可与来源说明。

不要提交 `.env`、API Key、授权头、缓存或真实模型调用日志。
