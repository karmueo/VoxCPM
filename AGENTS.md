# Repository Guidelines（仓库贡献指南）

## 项目结构与模块组织

- `src/voxcpm/`：Python 包源码。`core.py` 提供公共 API，`cli.py` 实现命令行入口，`model/` 与 `modules/` 实现语音合成，`training/` 与 `timestamps/` 提供训练和时间戳功能。
- `tests/`：自动化回归测试；`scripts/`：微调脚本和手动推理工具。
- `conf/`：按模型版本组织 YAML 训练配置，包括 `voxcpm_v1`、`voxcpm_v1.5` 和 `voxcpm_v2`。
- `app.py` 与 `lora_ft_webui.py`：Gradio 界面；`i18n.py`：界面翻译。`examples/` 存放示例音频与训练元数据，`assets/` 存放文档图片，`docker/` 存放部署资源。

## 构建、测试与开发命令

使用 Python 3.10–3.12，在仓库根目录执行以下命令。

- `python -m pip install -e '.[dev]'`：以可编辑模式安装项目及测试、代码风格工具。
- `python -m pytest tests/`：运行自动化测试；可指定测试文件路径，仅验证相关模块。
- `python -m black src tests scripts`：格式化 Python 代码。
- `python -m flake8 --max-line-length=120 src tests scripts`：检查代码风格；仓库未提供专用 Flake8 配置。
- `python app.py --port 8808`：启动本地网页演示；推理需要模型权重和适用硬件。
- `voxcpm --help`：查看命令行子命令与参数。
- 先执行 `python -m pip install build`，再执行 `python -m build`：生成 wheel 和源码分发包，与发布流程一致。

## 代码风格与命名约定

使用四个空格缩进；函数与变量采用 `snake_case`，类名采用 `PascalCase`，常量采用 `UPPER_CASE`。Black 的目标版本为 Python 3.10，行宽上限为 120 字符。类型注解和文档字符串遵循相邻代码的约定。模型公共辅助函数放入 `model/utils.py`，界面翻译集中维护于 `i18n.py`。

## 测试规范

使用 pytest，测试文件命名为 `test_*.py`，测试函数命名为 `test_*`。行为变更应补充回归测试，优先使用临时文件和模拟对象，避免不必要的模型下载。仓库未配置覆盖率门槛，可使用 `pytest-cov` 按需测量。`scripts/test_voxcpm_*_infer.py` 属于需要模型检查点的手动推理验证脚本。验证推理变更时，记录模型版本与运行设备。

## 提交与拉取请求规范

提交标题应简洁，使用动作描述，并优先沿用历史提交中的 `feat:`、`fix:`、`refactor:` 或 `docs:` 前缀。每次提交聚焦一个主题。拉取请求应说明问题与变更后的行为，关联相关议题，并列出验证命令和结果。界面变更附截图，音频变更提供可复现的配置。修改通用使用说明时，同步更新中英文 README。

## 配置与模型检查点

下载的模型、数据集、适配器和生成结果应放入已被 Git 忽略的 `models/`、`data/`、`lora/`、`output/` 或 `pretrained_models/` 目录。加载 PyTorch 检查点时，保留显式的 `weights_only=True` 参数。
