# GPT-SoVITS-Langbot 语音合成插件

![版本](https://img.shields.io/badge/版本-0.2-blue)

## 📌 核心功能

- **多用户独立配置**：每个用户拥有独立的语音合成配置
- **智能语音合成**：基于 GPT-SoVITS 的高质量语音生成
- **实时控制**：支持运行时动态调整所有参数
- **多语言支持**：中文/英文/日文语音合成
- **模型管理**：支持动态切换 SoVITS/GPT 模型

## 🚀 快速开始
### 环境安装
1. **安装GPT-SoVITS v3**：
   - 克隆官方仓库：`git clone https://github.com/RVC-Boss/GPT-SoVITS.git`
   - 安装依赖：`pip install -r requirements.txt`
   - 详细步骤请参考 [官方文档](https://github.com/RVC-Boss/GPT-SoVITS)

2. **准备训练数据**：
   - 具体数据准备要求请参考官方文档或网络教程

3. **训练模型**：
   - 训练方法请查阅相关教程：[GPT-SoVITS官方文档](https://github.com/RVC-Boss/GPT-SoVITS)

4. **启动推理服务**：
   - 确保推理服务正常运行并监听`http://localhost:9872`

5. **安装插件**：
   - 将插件放入Langbot的`plugins/`目录
   - 重启Langbot服务
   - 使用`!gsl 状态`验证插件是否加载成功

### 基本使用
1. 发送消息自动触发语音合成
2. 使用 `!gsl 帮助` 查看所有命令

## ⚙️ 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ref_wav_path` | str | [见代码] | 参考音频路径 |
| `prompt_text` | str | "参考音频文本" | 提示文本 |
| `prompt_language` | str | "中文" | 提示语言 (中文/英文/日文) |
| `text_language` | str | "中文" | 输出语言 (中文/英文/日文) |
| `sovits_path` | str | [见代码] | SoVITS 模型路径 |
| `gpt_path` | str | [见代码] | GPT 模型路径 |
| `voice_switch` | bool | True | 语音开关 |
| `return_text` | bool | False | 是否返回文本 |

## 🎛️ 完整命令列表

### 基础控制
!gsl 开启/关闭 - 开关语音功能
!gsl 文本开启/关闭 - 开关文本返回
!gsl 状态 - 查看当前配置
### 参数配置
!gsl 参考音频 <路径> - 设置参考音频
!gsl 提示文本 <文本> - 设置提示文本
!gsl 提示语言 <语言> - 设置提示语言
!gsl 文本语言 <语言> - 设置输出语言
### 模型管理
!gsl 模型列表 <sovits/gpt> - 查看可用模型
!gsl sovits模型 <序号/路径> - 切换SoVITS模型
!gsl gpt模型 <序号/路径> - 切换GPT模型

## 💻 使用示例

```bash
# 查看当前状态
!gsl 状态

# 切换为英文输出
!gsl 文本语言 英文

# 查看可用GPT模型
!gsl 模型列表 gpt

# 切换GPT模型(通过序号)
!gsl gpt模型 2

# 关闭语音只返回文本
!gsl 关闭
```

## ⚠️ 注意事项

1. 模型路径需使用绝对路径
2. 首次使用会自动创建用户配置
3. 关闭语音时会强制开启文本返回
4. 临时语音文件会自动清理

## 📜 版本历史

- v0.2 (当前)
  - 新增多用户支持
  - 完善模型管理功能
  - 优化错误处理

- v0.1
  - 初始版本发布

## 📧 联系作者

- 作者：heibai
- 问题反馈：创建 Issue