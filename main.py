# -*- coding: utf-8 -*-
"""
GPT-sovits语音合成插件
版本: 0.2
作者: heibai
"""
import os
from dataclasses import dataclass
from typing import Dict, List, AsyncGenerator

# 导入框架相关模块
from pkg.plugin.context import register, handler, BasePlugin, APIHost, EventContext
from pkg.plugin.events import NormalMessageResponded
from pkg.platform.types import MessageChain, Plain, Voice
from pkg.command import entities
from pkg.command.operator import CommandOperator, operator_class

# 导入自定义模块
from .gen_voice import get_model_list, tts, change_sovits_weights, change_gpt_weights
from .text_cleaner import clean_markdown
from gradio_client import file

# --------------------------
# 配置类定义
# --------------------------
@dataclass
class VoiceConfig:
    """语音合成配置参数类"""
    # 默认参考音频路径
    ref_wav_path: str = "E:/vits/voice/xunlian/feixiao/feixiao_voice/0a7e5bbd8f963f3b.wav_0000000000_0000162240.wav"
    # 提示文本
    prompt_text: str = "炎老，我说的这些正是联盟内部对景元将军的非议。"
    # 提示语言（中文/英文/日文）
    prompt_language: str = "中文"
    # 输出文本语言
    text_language: str = "中文"
    # SoVITS模型路径
    sovits_path: str = "GPT_SoVITS/pretrained_models/s2Gv3.pth"
    # GPT模型路径
    gpt_path: str = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
    # 语音开关状态
    voice_switch: bool = True
    # 是否返回文本
    return_text: bool = False

# --------------------------
# 常量定义
# --------------------------
HELP_TEXT = """!gsl 帮助 - 显示帮助
!gsl 开启/关闭 - 语音开关
!gsl 文本开启/关闭 - 文本返回开关
!gsl 状态 - 查看当前配置
!gsl 参考音频 <路径> - 设置参考音频
!gsl 提示文本 <文本> - 设置提示文本
!gsl 提示语言 <语言> - 设置提示语言
!gsl 文本语言 <语言> - 设置输出语言
!gsl 模型列表 <sovits/gpt> - 查看可用模型
!gsl sovits模型 <序号/路径> - 切换SoVITS模型
!gsl gpt模型 <序号/路径> - 切换GPT模型"""

class GlobalConfigManager:
    _instance = None
    user_configs: Dict[int, VoiceConfig] = {}
    global_config = VoiceConfig()
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            print("✅ 创建全局配置管理器")
        return cls._instance

# --------------------------
# 命令处理器
# --------------------------
@operator_class(name="gsl", help="语音合成插件命令", privilege=1)
class GSLOperator(CommandOperator):
    """处理语音合成相关命令的运算符类"""
    
    def __init__(self, host: APIHost):
        super().__init__(host)
        self.config_mgr = GlobalConfigManager()

        print(f"🔗 命令处理器连接全局配置: {id(self.config_mgr)}")

    def _get_config(self, user_id: int) -> VoiceConfig:
        print(f"🔍 获取用户 {user_id} 配置")
        """获取用户配置（如果不存在则创建默认配置）"""
        if user_id not in self.config_mgr.user_configs:
            self.config_mgr.user_configs[user_id] = VoiceConfig(
                ref_wav_path=self.config_mgr.global_config.ref_wav_path,
                prompt_text=self.config_mgr.global_config.prompt_text,
                prompt_language=self.config_mgr.global_config.prompt_language,
                text_language=self.config_mgr.global_config.text_language,
                sovits_path=self.config_mgr.global_config.sovits_path,
                gpt_path=self.config_mgr.global_config.gpt_path,
                voice_switch=self.config_mgr.global_config.voice_switch,
                return_text=self.config_mgr.global_config.return_text
            )
            print(f"📝 为用户 {user_id} 创建新配置")
        return self.config_mgr.user_configs[user_id]
    
    async def _list_models(self, model_type: str,user_id: int) -> str:
        """获取模型列表"""
        models = get_model_list()
        config = self._get_config(user_id)
        current_sov = config.sovits_path
        current_gpt = config.gpt_path
        
        result = []
        if model_type == "sovits":
            result.append("📢 可用SoVITS模型:")
            for idx, path in enumerate(models["sovits"], 1):
                status = " (当前)" if path == current_sov else ""
                result.append(f"{idx}. {path}{status}")
        elif model_type == "gpt":
            result.append("🧠 可用GPT模型:")
            for idx, path in enumerate(models["gpt"], 1):
                status = " (当前)" if path == current_gpt else ""
                result.append(f"{idx}. {path}{status}")
        else:
            return "❌ 无效模型类型"
        return "\n".join(result)
    
    async def execute(self, context: entities.ExecuteContext) -> AsyncGenerator[entities.CommandReturn, None]:
        """命令执行入口"""
        user_id = int(context.query.sender_id)
        params = context.crt_params
        command = params[0] if params else ""
        args = params[1:] if len(params) > 1 else []
        
        try:
            result = await self._handle_command(user_id, command, args)
        except Exception as e:
            result = f"命令执行错误: {str(e)}"

        yield entities.CommandReturn(text=result)

    async def _handle_command(self, user_id: int, cmd: str, args: List[str]) -> str:
        """具体命令处理逻辑"""
        config = self._get_config(user_id)
        print(f"🔧 处理用户{user_id}命令{cmd} | 当前配置:", vars(config))
        # 语音开关命令
        if cmd in ["开启", "on"]:
            config.voice_switch = True
            return "✅ 语音功能已开启"
            
        elif cmd in ["关闭", "off"]:
            config.voice_switch = False
            config.return_text = True  # 新增：关闭语音时强制开启文本
            return "⛔ 语音功能已关闭，已自动开启文本返回"
            
        # 文本返回开关
        elif cmd == "文本开启":
            config.return_text = True
            return "📝 文本返回已开启"
            
        elif cmd == "文本关闭":
            if not config.voice_switch:
                return "❌ 语音已关闭时不能关闭文本返回"
            config.return_text = False
            return "📝 文本返回已关闭"
            
        # 状态查询
        elif cmd == "状态":
            return (
                "⚙️ 当前配置状态：\n"
                f"🔊 语音开关: {'🟢 开启' if config.voice_switch else '🔴 关闭'}\n"
                f"📃 文本返回: {'🟢 开启' if config.return_text else '🔴 关闭'}\n"
                f"🎵 参考音频: {config.ref_wav_path}\n"
                f"📌 提示文本: {config.prompt_text}\n"
                f"🌐 提示语言: {config.prompt_language}\n"
                f"🌍 输出语言: {config.text_language}\n"
                f"🤖 Sovits模型: {config.sovits_path}\n"
                f"🧠 GPT模型: {config.gpt_path}"
            )
            
        # 参考音频设置
        elif cmd == "参考音频":
            if not args:
                return "❌ 请输入音频路径"
            config.ref_wav_path = " ".join(args)
            return f"🎵 参考音频已设置为: {config.ref_wav_path}"
            
        # 提示文本设置
        elif cmd == "提示文本":
            if not args:
                return "❌ 请输入提示文本"
            config.prompt_text = " ".join(args)
            return f"📝 提示文本已设置为: {config.prompt_text}"
            
        # 语言设置
        elif cmd == "提示语言":
            if not args or args[0] not in ["中文", "英文", "日文"]:
                return "❌ 支持语言: 中文/英文/日文"
            config.prompt_language = args[0]
            return f"🌐 提示语言已设置为: {args[0]}"
            
        elif cmd == "文本语言":
            if not args or args[0] not in ["中文", "英文", "日文"]:
                return "❌ 支持语言: 中文/英文/日文"
            config.text_language = args[0]
            return f"🌍 输出语言已设置为: {args[0]}"
        elif cmd == "模型列表":
            if not args or args[0] not in ["sovits", "gpt"]:
                return "❌ 请指定模型类型: sovits 或 gpt"
            return await self._list_models(args[0],user_id)            
        elif cmd == "sovits模型":
            if not args:
                return "❌ 请输入模型序号或路径"
            
            models = get_model_list()["sovits"]
            # 尝试按序号选择
            if args[0].isdigit():
                index = int(args[0]) - 1
                if 0 <= index < len(models):
                    selected = models[index]
                else:
                    return f"❌ 无效序号，可用范围1-{len(models)}"
            else:
                selected = " ".join(args)
                if selected not in models:
                    return f"❌ 模型不存在，请使用!gsl 模型列表 sovits 查看可用模型"
            
            try:
                change_sovits_weights(selected)
                config.sovits_path = selected
                return f"🤖 SoVITS模型已切换为: {selected}"
            except Exception as e:
                return f"❌ 模型加载失败: {str(e)}"
                
        elif cmd == "gpt模型":
            if not args:
                return "❌ 请输入模型序号或路径"
            
            models = get_model_list()["gpt"]
            # 尝试按序号选择
            if args[0].isdigit():
                index = int(args[0]) - 1
                if 0 <= index < len(models):
                    selected = models[index]
                else:
                    return f"❌ 无效序号，可用范围1-{len(models)}"
            else:
                selected = " ".join(args)
                if selected not in models:
                    return f"❌ 模型不存在，请使用!gsl 模型列表 gpt 查看可用模型"
            
            try:
                change_gpt_weights(selected)
                config.gpt_path = selected
                return f"🤖 gpt模型已切换为: {selected}"
            except Exception as e:
                return f"❌ 模型加载失败: {str(e)}"
                
        # 帮助命令
        elif cmd == "帮助":
            return HELP_TEXT
            
        return "❌ 未知命令，输入!gsl 帮助 查看可用命令"

# --------------------------
# 插件主类
# --------------------------
@register(name="GPT-sovits-Langbot", description="语音合成插件", version="0.2", author="heibai")
class GSLPlugin(BasePlugin):
    """语音合成插件主类"""
    
    def __init__(self, host: APIHost):
        super().__init__(host)
        self.config_mgr = GlobalConfigManager()
        change_sovits_weights(self.config_mgr.global_config.sovits_path)
        change_gpt_weights(self.config_mgr.global_config.gpt_path)
        print(f"🔗 插件主类连接全局配置: {id(self.config_mgr)}")


    @handler(NormalMessageResponded)
    async def handle_response(self, ctx: EventContext):
        """处理消息响应，生成语音"""
        user_id = ctx.event.sender_id
        print(f"🔍 处理用户 {user_id} 的响应")
        config = self.config_mgr.user_configs.get(user_id)
        print(f"🔍 当前生效配置:", vars(config))
        if not config.voice_switch:
            # 强制保持文本返回开启
            if not config.return_text:
                config.return_text = True
            return  # 直接返回，不执行后续语音生成逻辑
        try:
            # 清理文本并生成语音
            text = clean_markdown(ctx.event.response_text)
            res_path = tts(
                file(config.ref_wav_path),
                config.prompt_text,
                config.prompt_language,
                text,
                config.text_language
            )

            if res_path:
                # 构建消息元素
                elements = [Voice(path=res_path)]
                if config.return_text:  # 根据配置添加文本
                    elements.insert(0, Plain(text))

                # 发送语音消息
                await ctx.reply(MessageChain(elements))
                os.remove(res_path)  # 清理临时文件

                # 如果不需要文本则阻止默认响应
                if not config.return_text:
                    ctx.prevent_default()
        except Exception as e:
            print(f"❌ 语音生成失败: {str(e)}")
    def __del__(self):
        """插件卸载时的清理操作"""
        pass