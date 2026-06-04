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

BiliCourseAI 使用 OpenAI-compatible API，只需要提供 base URL、API key 和模型名。

```bash
bilicourse config llm \
  --base-url "https://your-openai-compatible-endpoint/v1" \
  --api-key "YOUR_API_KEY" \
  --text-model "Ali-dashscope/Qwen3.5-Plus" \
  --vision-model "Ali-dashscope/Qwen3.5-Plus" \
  --disable-thinking
```

也可以使用环境变量：

```bash
export BILICOURSE_BASE_URL="https://your-openai-compatible-endpoint/v1"
export BILICOURSE_API_KEY="YOUR_API_KEY"
export BILICOURSE_TEXT_MODEL="Ali-dashscope/Qwen3.5-Plus"
export BILICOURSE_VISION_MODEL="Ali-dashscope/Qwen3.5-Plus"
export BILICOURSE_ENABLE_THINKING="false"
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
  --max-visual-requests 2 \
  --llm-request-delay 2.2
```

展开后会重写同目录下的 `report.json` 和 `report.html`。

## 视频类型建议

### 普通课程视频

```bash
bilicourse probe BVxxxxxxxxxx
bilicourse outline BVxxxxxxxxxx --outline-window-seconds 720 --outline-overlap-seconds 75
bilicourse serve "data/reports/视频标题__BV号"
```

策略：

- 多 P 视频先尊重分 P
- 较短分 P 优先视作可展开叶子
- 单 P 或长 P 会根据字幕构建知识树
- 内部软窗口只用于控制上下文长度，不作为用户可见节点
- 过长 leaf 会被质量门改成 branch，提示继续展开

### 逐题讲解或合集视频

如果一个分 P 基本对应一道题，优先使用题目树模式：

```bash
bilicourse outline BVxxxxxxxxxx --part-tree-mode question
bilicourse serve "data/reports/视频标题__BV号"
```

## 常用参数

`--outline-window-seconds 720`

控制 LLM 每次处理的字幕软窗口，默认约 12 分钟。它只用于上下文控制，不是最终课程节点。

`--outline-overlap-seconds 75`

相邻软窗口之间额外保留的重叠上下文，用来减少切断完整知识点的概率。

`--part-page N`

只处理第 N 个分 P，适合调试长视频。

`--part-tree-mode question`

按分 P 标题构建题目树，适合逐题讲解合集。

`--max-visual-requests 2`

展开叶子节点时最多保留几张辅助理解图。图片会存到报告目录下的 `frames/`，HTML 使用相对路径引用。

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
python -m py_compile src/bilicourseai/*.py
```

主要目录：

```text
src/bilicourseai/          Python 源码
src/bilicourseai/templates HTML 报告模板
config/*.example.json      示例配置
```

## 开源前注意

发布前请确认不会提交：

- `config/*.json` 里的真实 API key 或 Bilibili cookie
- `data/` 下的报告、截图、字幕和分析结果
- `.conda/`、`.venv/` 等本地环境
- `_refs/` 下的第三方参考项目副本

`.gitignore` 已默认忽略这些路径。

## License

尚未选择许可证。正式开源前请添加 `LICENSE` 文件。
