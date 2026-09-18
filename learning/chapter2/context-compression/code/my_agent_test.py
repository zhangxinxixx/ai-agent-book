from dataclasses import dataclass
from typing import List


# =========================
# 1. Message 数据结构
# =========================

@dataclass
class Message:
    role: str
    content: str


# =========================
# 2. Context Manager
# =========================

class ContextManager:

    def __init__(
        self,
        max_chars: int = 1500,
        keep_recent: int = 2
    ):
        """
        max_chars:
            Context 最大字符数。

        keep_recent:
            压缩时，最近多少条消息保留全文。
        """

        # TODO 1:
        # 保存 max_chars
        self.max_chars = max_chars

        # TODO 2:
        # 保存 keep_recent
        self.keep_recent = keep_recent
        # TODO 3:
        # 创建一个空的 messages 列表
        self.messages: List[Message] = []


    def add_message(
        self,
        role: str,
        content: str
    ):
        """
        添加一条消息。

        示例：
        add_message("user", "修复登录Bug")
        """

        # TODO:
        # 创建 Message
        # 然后追加到 self.messages
        message = Message(role=role, content=content)
        self.messages.append(message)


    def context_size(self) -> int:
        """
        返回当前所有消息 content 的字符总数。
        """

        # TODO:
        # 遍历所有 Message
        # 对 message.content 求 len()
        # 最后求和
        return sum(len(message.content) for message in self.messages)


    def should_compress(self) -> bool:
        """
        判断当前 Context 是否超过 max_chars。
        """

        # TODO:
        # 使用 context_size()
        return self.context_size() > self.max_chars


    def summarize_old_messages(
        self,
        messages: List[Message]
    ) -> str:
        """
        第一版先不用 GPT。

        模拟摘要：
        每条消息只保留前 80 个字符。
        """

        summary_parts = []

        # TODO:
        # 遍历 messages
        #
        # 每条生成：
        # role: content前80字符
        #
        # 放进 summary_parts
        for message in messages:
            summary_parts.append(f"{message.role}: {message.content[:80]}")
        # TODO:
        # 使用 "\n".join(...)
        # 返回最终 summary
        return "\n".join(summary_parts)


    def compress(self):
        """
        压缩 Context：

        1. 如果不需要压缩，直接返回
        2. 最近 keep_recent 条保留全文
        3. 更老的消息生成 summary
        4. 重建 self.messages
        """

        # TODO 1:
        # 判断 should_compress()
        # 如果 False，直接 return
        if not self.should_compress():
            return


        # TODO 2:
        # 切出 recent_messages
        #
        # 提示：
        # self.messages[-self.keep_recent:]
        recent_messages = self.messages[-self.keep_recent:]

        # TODO 3:
        # 切出 old_messages
        #
        # 提示：
        # self.messages[:-self.keep_recent]
        old_messages = self.messages[:-self.keep_recent]

        # TODO 4:
        # 调用 summarize_old_messages()
        summary = self.summarize_old_messages(old_messages)

        # TODO 5:
        # 创建一条 summary Message
        #
        # role 可以先用 "system"
        #
        # content:
        # "Previous work summary:\n" + summary
        # summary_message = Message(role = 'summary', content = "Previous work summary:\n" + summary)
        summary_message = Message(
            role="summary",
            content=summary
        )

        # TODO 6:
        # 重新构造 self.messages：
        #
        # [summary_message] + recent_messages
        self.messages = [summary_message] + recent_messages

    def print_context(self):
        """
        打印当前 Context。
        """

        print("\n" + "=" * 60)
        print("当前上下文")
        print("=" * 60)

        # TODO:
        # 遍历 self.messages
        #
        # 打印：
        # 序号
        # role
        # content
        for idx, message in enumerate(self.messages, start=1):
            print(f"{idx}. {message.role}: {message.content}")
        print(
            f"\n【上下文大小】"
            f"{self.context_size()} 字符"
        )


# =========================
# 3. 模拟 Codex 工作过程
# =========================

def main():

    manager = ContextManager(
        max_chars=2000,
        keep_recent=2
    )

    # 初始任务
    manager.add_message(
        "system",
        "You are a coding agent."
    )

    manager.add_message(
        "user",
        "修复登录接口 500 错误"
    )

    # 模拟 Codex 工作 10 轮
    for i in range(1, 11):

        manager.add_message(
            "assistant",
            f"Step {i}: inspect file_{i}.py"
        )

        manager.add_message(
            "tool",
            (
                f"Result from file_{i}.py: "
                + "This is a very long tool result. " * 15
            )
        )

        size = manager.context_size()

        print(
            f"【执行轮次】第 {i} 轮："
            f"当前上下文 {size} 字符"
        )

        # TODO:
        # 如果 should_compress()
        #
        # 打印：
        # >>> Context too large, compressing...
        #
        # 然后调用 compress()
        if manager.should_compress():
            print(
                f"【触发压缩】压缩前："
                f"{manager.context_size()} 字符"
            )

            manager.compress()

            print(
                f"【压缩完成】压缩后："
                f"{manager.context_size()} 字符"
            )


    print("\n【最终上下文】")
    for idx, message in enumerate(manager.messages, start=1):
        if message.role == "summary":
            print(f"{idx}. 【旧轨迹摘要】\n{message.content}")
        else:
            print(f"{idx}. {message.role}: {message.content}")


if __name__ == "__main__":
    main()

