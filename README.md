# BiliCourseAI

BiliCourseAI 是一个面向 Bilibili 课程视频的 AI 辅助学习报告 CLI 原型。

它只分析视频本体：标题、分 P、字幕、时间轴、关键截图和视觉理解；不处理弹幕、评论、相关推荐。

当前项目仍处于原型阶段，适合本地研究、课程笔记生成和交互式节点展开。

## 功能概览

- Bilibili 视频/分 P 元数据抓取
- Bilibili 字幕抓取与基础句读整理
- LLM 生成课程知识树骨架
- 交互式展开某个知识节点
- 自动截取辅助理解图，并调用视觉模型分析课件、公式、板书或图表
- 输出本地 `report.json` 与可浏览的 `report.html`
- 本地 `serve` 模式支持网页内展开/重做

## 环境要求

- Python 3.11 或更高版本
- Windows 10/11、Linux 或 macOS
- 可访问 Bilibili 与所配置的 OpenAI-compatible LLM 服务
- Bilibili 手机 App，用于扫码登录获取字幕访问凭据

项目依赖 `imageio-ffmpeg` 下载/定位 ffmpeg，可在多数机器上自动工作。

## 安装

### Windows PowerShell

```powershell
git clone https://github.com/wangcy7889/BiliCourseAI.git
cd BiliCourseAI

python -m pip install -U pip
python -m pip install -e .

bilicourse --help
```

### Linux / macOS

```bash
git clone https://github.com/wangcy7889/BiliCourseAI.git
cd BiliCourseAI

python -m pip install -U pip
python -m pip install -e .

bilicourse --help
```

## 本地数据目录

默认情况下：

- 如果在项目根目录运行，数据写入当前项目下的 `data/` 和 `config/`
- 如果从其他目录运行，数据写入用户目录下的 `.bilicourseai/`

可以用环境变量覆盖：

```powershell
$env:BILICOURSE_HOME = "$HOME\BiliCourseAI-Work"
```

```bash
export BILICOURSE_HOME="$HOME/bilicourseai-work"
```

也可以分别指定：

- `BILICOURSE_DATA_DIR`
- `BILICOURSE_CONFIG_DIR`

真实配置和报告数据默认不会被 git 跟踪。

## 配置 LLM

BiliCourseAI 使用 OpenAI-compatible API。文本模型和视觉模型可以共用同一个 base URL/API key，也可以分别配置。

```bash
bilicourse config llm \
  --base-url "https://your-openai-compatible-endpoint/v1" \
  --api-key "YOUR_API_KEY" \
  --text-model "SDU-AI/DeepSeek-V4-Flash" \
  --vision-model "Ali-dashscope/Qwen3.5-Plus" \
  --disable-thinking
```

如果文本模型和视觉模型来自不同服务：

```bash
bilicourse config llm \
  --text-base-url "https://text-provider.example.com/v1" \
  --text-api-key "YOUR_TEXT_API_KEY" \
  --text-model "SDU-AI/DeepSeek-V4-Flash" \
  --vision-base-url "https://vision-provider.example.com/v1" \
  --vision-api-key "YOUR_VISION_API_KEY" \
  --vision-model "Ali-dashscope/Qwen3.5-Plus" \
  --disable-thinking
```

也可以使用环境变量：

```bash
export BILICOURSE_BASE_URL="https://your-openai-compatible-endpoint/v1"
export BILICOURSE_API_KEY="YOUR_API_KEY"
export BILICOURSE_TEXT_MODEL="SDU-AI/DeepSeek-V4-Flash"
export BILICOURSE_VISION_MODEL="Ali-dashscope/Qwen3.5-Plus"
export BILICOURSE_ENABLE_THINKING="false"
```

双端点环境变量为：

```bash
export BILICOURSE_TEXT_BASE_URL="https://text-provider.example.com/v1"
export BILICOURSE_TEXT_API_KEY="YOUR_TEXT_API_KEY"
export BILICOURSE_VISION_BASE_URL="https://vision-provider.example.com/v1"
export BILICOURSE_VISION_API_KEY="YOUR_VISION_API_KEY"
```

配置文件示例见：

```text
config/llm_settings.example.json
```

不要提交真实 API key。

## 登录 Bilibili

字幕接口经常需要登录态。推荐扫码登录：

```bash
bilicourse auth qr
```

然后检查状态：

```bash
bilicourse auth status --validate
```

也可以手动设置 cookie：

```bash
bilicourse auth set
```

配置文件示例见：

```text
config/bilibili_credentials.example.json
```

不要提交真实 Bilibili cookie。

## 从零生成一个报告

### 1. 先抓元数据和字幕

`probe` 不调用 LLM，适合先确认视频是否有字幕。

```bash
bilicourse probe BVxxxxxxxxxx
```

也可以传完整 URL：

```bash
bilicourse probe "https://www.bilibili.com/video/BVxxxxxxxxxx"
```

### 2. 生成知识树骨架

```bash
bilicourse outline BVxxxxxxxxxx \
  --outline-window-seconds 720 \
  --outline-overlap-seconds 75 \
  --llm-request-delay 2.2
```

如果只想先生成某一个分 P，适合分 P 很多的大合集：

```bash
bilicourse outline BVxxxxxxxxxx --part-page 8
```

此模式只会抓取指定分 P 的字幕，不会逐个初始化其他分 P。如果同一个报告目录里已经有其他分 P，已有内容会被保留。

输出通常位于：

```text
data/reports/视频标题__BV号/report.json
data/reports/视频标题__BV号/report.html
```

`outline` 阶段只生成可展开的课程树骨架，不做完整图片分析。

### 3. 打开交互服务

```bash
bilicourse serve "data/reports/视频标题__BV号"
```

也可以用更短的写法，只要能唯一匹配 `data/reports` 下的报告目录即可：

```bash
bilicourse serve BVxxxxxxxxxx
bilicourse serve "标题关键词"
bilicourse serve "data/reports/视频标题__BV号/report.json"
```

浏览器打开：

```text
http://127.0.0.1:8765/
```

如果端口被占用：

```bash
bilicourse serve "data/reports/视频标题__BV号" --port 8770
```

网页里的按钮会调用本地服务：

- `展开为笔记`：把骨架叶子扩展成学习笔记
- `展开划分`：把骨架分支继续拆成子节点
- `重做笔记`：重做当前叶子
- `重做划分`：重做当前分支划分

重做只基于当前节点状态，不依赖旧历史结果。

## 命令行展开节点

如果不使用本地网页服务，也可以手动展开指定节点：

```bash
bilicourse expand "data/reports/视频标题__BV号/report.json" \
  --block-id p1-n1 \
  --max-visual-requests 4 \
  --llm-request-delay 2.2
```

展开后会重写同目录下的 `report.json` 和 `report.html`。

## 视频类型建议

### 普通课程视频

```bash
bilicourse probe BVxxxxxxxxxx
bilicourse outline BVxxxxxxxxxx --outline-window-seconds 720 --outline-overlap-seconds 75
bilicourse serve BVxxxxxxxxxx
```

策略：

- 多 P 视频先尊重分 P
- 较短分 P 优先视作可展开叶子
- 单 P 或长 P 会根据字幕构建知识树
- 内部软窗口只用于控制上下文长度，不作为用户可见节点
- 过长 leaf 会被质量门改成 branch，提示继续展开

### 多 P 大合集的单 P 快速初始化

当一个视频合集有很多分 P，而你只想先分析其中某一 P 时，使用：

```bash
bilicourse outline BVxxxxxxxxxx --part-page 42
```

这个模式会先获取视频基本信息和分 P 列表，用来定位 P42 的 `cid`；随后只抓取 P42 的字幕、只对 P42 生成 outline。它不会逐个抓取其他分 P 字幕，所以适合先从大合集里挑一个分 P 初始化。

增量行为：

- 第一次运行 `--part-page 7`：生成只包含 P7 的报告。
- 之后运行 `--part-page 8`：只抓取并生成 P8，同时保留已有的 P7。
- 不带 `--part-page`：初始化/更新完整视频报告，会逐个处理所有分 P。

### 逐题讲解或合集视频

如果一个分 P 基本对应一道题，优先使用题目树模式：

```bash
bilicourse outline BVxxxxxxxxxx --part-tree-mode question
bilicourse serve BVxxxxxxxxxx
```

## 常用参数

`--outline-window-seconds 720`

控制 LLM 每次处理的字幕软窗口，默认约 12 分钟。它只用于上下文控制，不是最终课程节点。

`--outline-overlap-seconds 75`

相邻软窗口之间额外保留的重叠上下文，用来减少切断完整知识点的概率。

`--part-page N`

只抓取并处理第 N 个分 P。适合多 P 大合集的快速初始化，也适合调试指定分 P；已有报告中的其他分 P 会保留。

`--part-tree-mode question`

按分 P 标题构建题目树，适合逐题讲解合集。

`--max-visual-requests 4`

展开叶子节点时最多保留几张辅助理解图。`serve` 默认值为 4，可以按需要调大。图片会存到报告目录下的 `frames/`，HTML 使用相对路径引用。

`--llm-request-delay 2.2`

每次 LLM 请求之间的等待秒数。遇到 RPM 限制时可以调大。

`--disable-thinking`

对支持该参数的模型传入 `enable_thinking=false`，通常能更快、更省。

## 输出结构

典型报告目录：

```text
data/reports/视频标题__BV号/
  report.json
  report.html
  frames/
```

`report.html` 可以直接打开；如果需要展开或重做节点，请使用 `bilicourse serve`。

图片路径应保持为相对路径，例如：

```html
<img src="frames/example.jpg">
```

## 排错

### 卡在 `Mode: outline skeleton`

先确认视频是否很长、分 P 很多或 LLM 响应很慢。

```bash
bilicourse probe BVxxxxxxxxxx
```

可以只测试某个分 P：

```bash
bilicourse outline BVxxxxxxxxxx --part-page 1 --max-outline-windows 1
```

也可以增大请求间隔：

```bash
bilicourse outline BVxxxxxxxxxx --llm-request-delay 3.5
```

### 没有字幕

当前版本主要依赖 Bilibili 字幕。没有字幕时，程序只能拿到标题、分 P 等元数据，无法可靠理解视频内容。

为了保持轻量化，项目暂不默认接入本地 ASR。

### 图片加载失败

通常说明 `report.json` 引用的图片文件已经不存在。重新展开或重做对应节点即可重新生成。

### 查看更多命令

```bash
bilicourse --help
bilicourse outline --help
bilicourse expand --help
bilicourse serve --help
```

## 开发

```bash
python -m pip install -e .
python - <<'PY'
from pathlib import Path
import py_compile
for path in Path("src/bilicourseai").rglob("*.py"):
    py_compile.compile(str(path), doraise=True)
PY
```

主要目录：

```text
src/bilicourseai/           Python 源码
src/bilicourseai/ai/        LLM 编排层：骨架、展开、分段、文本增强
src/bilicourseai/outline/   知识树骨架相关提示词、窗口和节点归一化
src/bilicourseai/visual/    截图候选选择、视觉分析和视觉流水线
src/bilicourseai/reports/   report.json/report.html 写入、合并和选择
src/bilicourseai/templates/ HTML 报告模板
config/*.example.json       示例配置
```

分层约定：`ai/` 只负责调用模型和编排流程；`outline/`、`visual/`、`reports/` 放对应领域的实现细节。CLI 和本地服务优先依赖这些包的公开出口，避免重新引入旧的平面模块。
