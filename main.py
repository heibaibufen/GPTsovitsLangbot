from pkg.plugin.context import register, handler, llm_func, BasePlugin, APIHost, EventContext
from pkg.plugin.events import *  # 导入事件类
from pkg.platform.types import *
import gen_voice


# 注册插件
@register(name="GPT-sovits-Langbot", description="语音合成9插件", version="0.1", author="heibai")
class MyPlugin(BasePlugin):

    # 插件加载时触发
    def __init__(self, host: APIHost):
        pass

    # 异步初始化
    async def initialize(self):
        pass

    # 当收到个人消息时触发
    @handler(PersonNormalMessageReceived)
    async def person_normal_message_received(self, ctx: EventContext):
        msg = ctx.event.text_message  # 这里的 event 即为 PersonNormalMessageReceived 的对象

        ref_wav_path = "E:/vits/voice/xunlian/daheita/daheita_voice/1f1232b352ae14d0.wav"
        prompt_text = "要打发这场战斗，猜猜看接下来你还要再来几杯？"
        prompt_language = "中文"
        text = clean_markdown(ctx.event.response_text)
        text_language = "中文"
        res_path = gen_voice.tts(ref_wav_path, prompt_text, prompt_language, text, text_language)
        try:
            res_path = await gen_voice.tts(ref_wav_path, prompt_text, prompt_language, text, text_language)
            if not res_path:  # 文本过长或生成失败
                return
            # 构建消息链
            message_elements = []
            message_elements.append(Plain(text))
            # 使用Voice消息发送
            with open(res_path, "rb") as f:
                base64_audio = base64.b64encode(f.read()).decode()
            message_elements.append(Voice(base64=base64_audio))

            # 构建消息链并发送
            if message_elements:
                msg_chain = MessageChain(message_elements)
                await ctx.reply(msg_chain)

        except Exception as e:
            print(f"生成语音失败: {e}")
            return
        if msg == "hello":  # 如果消息为hello

            # 输出调试信息
            self.ap.logger.debug("hello, {}".format(ctx.event.sender_id))

            # 回复消息 "hello, <发送者id>!"
            ctx.add_return("reply", ["hello, {}!".format(ctx.event.sender_id)])

            # 阻止该事件默认行为（向接口获取回复）
            ctx.prevent_default()

    # 当收到群消息时触发
    @handler(GroupNormalMessageReceived)
    async def group_normal_message_received(self, ctx: EventContext):
        msg = ctx.event.text_message  # 这里的 event 即为 GroupNormalMessageReceived 的对象
        if msg == "hello":  # 如果消息为hello

            # 输出调试信息
            self.ap.logger.debug("hello, {}".format(ctx.event.sender_id))

            # 回复消息 "hello, everyone!"
            ctx.add_return("reply", ["hello, everyone!"])

            # 阻止该事件默认行为（向接口获取回复）
            ctx.prevent_default()

    # 插件卸载时触发
    def __del__(self):
        pass
