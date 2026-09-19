# User Memory 代码快照

这里保存实验 3-1 / 3-2 的可公开代码快照。

快速开始：

```powershell
python -m pip install -r requirements.txt
Copy-Item .\env.example .\.env
python .\main.py --help
python .\main.py --mode interactive --memory-mode notes
```

完整说明见上一级的[运行手册](../运行手册.md)。

## 未包含内容

`data/`、`logs/`、`results/`、`validation/`、`.env` 和缓存文件没有复制到这个公开目录。这些内容可能包含运行产物、真实对话、用户记忆或密钥。

为避免把看似真实的金融或联系信息放进公开仓库，示例中的账号、路由号和联系地址使用 `TEST_*` 或 `example.com` 占位符；这不会改变相关解析逻辑。

上游项目：[bojieli/ai-agent-book](https://github.com/bojieli/ai-agent-book/tree/main/chapter3/user-memory)。原始许可证见 [LICENSE.upstream](./LICENSE.upstream)。
