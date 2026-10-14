# Script2Gif - 本地 APNG

一个离线 Windows 桌面工具：输入完整剧本，自动拆分后使用本地 Wan2.1 T2V-1.3B
逐个生成动态镜头，导出多份 APNG 和原始 PNG 序列帧。不需要分镜确认、API Key 或 FFmpeg。

## 功能

- 中文或英文剧本输入，直接拆分并批量生成
- 分辨率选择：512x288 或 832x480
- 每镜头播放时长输入：0.5–5 秒，默认 1.7 秒
- 模型目录和推理环境选择；随机种子内部固定为 42
- 独立进程推理、阶段进度和取消
- APNG 动画预览、播放暂停、另存为和打开输出目录
- 镜头结果列表、整体进度、失败记录；取消保留已完成结果

拆分规则：优先按空行段落拆分，没有多个段落时按行拆分，单行文本按中英文句末标点拆分。
这是离线规则拆分，不包含大模型分镜分析。所有镜头串行生成，使用同一分辨率和时长。

## 启动

已准备好本地环境时，双击 `run_local_apng.bat`，或在项目目录运行：

```powershell
.\.venv-wan\Scripts\python.exe main.py
```

新设备先按照 [环境与模型准备说明](scripts/LOCAL_APNG.md) 安装推理依赖、
下载或导入模型，并安装桌面依赖：

```powershell
.\.venv-wan\Scripts\python.exe -m pip install -e .
```

目前已验证 Python 3.12、Windows、16 GB 显存 NVIDIA GPU。
其他硬件尚未验证；模型文件约 27 GB，模型和推理环境需要独立准备。
现有打包脚本尚未完成本地模型运行时的分发验证。

## 文件位置

- `models/wan2.1-t2v-1.3b/`：默认模型目录
- `outputs/desktop/批次目录/`：原始剧本、批次记录、编号 APNG、原始帧和运行报告
- `scripts/local_apng.py`：模型下载、硬件检测与离线生成入口

模型和生成结果不提交到 Git。复制完整模型目录到另一台电脑后，
可在界面中选择对应目录，配合兼容的推理环境离线运行。

## 验证

```powershell
.\.venv-wan\Scripts\python.exe -m pytest
.\.venv-wan\Scripts\python.exe scripts/verify_desktop_apng.py --render-only
```

移除 `--render-only` 会执行真实模型生成验证。截图保存到 `outputs/desktop-qa/`。
