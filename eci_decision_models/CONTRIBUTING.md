# 贡献与开发规范

本子项目位于父仓库 `MRBalance_replay/eci_decision_models/`，但拥有独立的 Python 工程配置。父项目原有文件只读，本项目不得修改它们。

## 工具链

以下命令在 `eci_decision_models/` 子项目目录内运行：

| 目的 | 命令 |
| --- | --- |
| 安装依赖 | `uv sync` |
| Lint | `uv run ruff check .` |
| 格式化 | `uv run ruff format .` |
| 类型检查 | `uv run mypy .` |
| 测试 | `uv run pytest` |
| 构建发行包 | `uv build` |

pre-commit 从父仓库根目录显式运行，配置中的 `files: ^eci_decision_models/` 会把检查范围限制在本子项目。首次运行前先在子项目执行 `uv sync`，让 `uv run --project` 能找到本子项目的开发依赖：

```bash
# 在 MRBalance_replay/ 根目录运行
uv run --project eci_decision_models pre-commit run \
  --config eci_decision_models/.pre-commit-config.yaml --all-files
```

**不要执行 `pre-commit install`。** 本子项目不是独立 Git 仓库，安装会写入父仓库 `.git/hooks/`，可能覆盖父项目已有的钩子。钩子只包含快速自动修复项，不运行测试或慢检查。

## 命名规范

| 对象 | 规则 | 示例 |
| --- | --- | --- |
| 模块/文件 | snake_case | `esc_data.py` |
| 函数/方法/变量 | snake_case | `load_esc_items` |
| 类 | PascalCase | `ChoicePrediction` |
| 常量 | UPPER_SNAKE_CASE | `DEFAULT_TIMEOUT_SECONDS` |
| 私有成员 | 前缀单下划线 | `_client` |
| 布尔值 | is_/has_/can_ 前缀 | `is_valid` |
| 测试 | `test_*.py`，函数 `test_*` | `test_metrics.py` |

## 代码风格

- 使用 ruff 进行 lint 和格式化，行宽 100，双引号。
- mypy 启用 strict；公共函数、数据边界和模型适配层应有明确类型标注。
- 测试使用 pytest，文件名和测试函数都以 `test_` 开头。
- 不提交 `.env`、API key、虚拟环境、缓存和实验原始大输出。
- 父项目的 `reference/` 是第三方参考实现；不要修改、复制或纳入本项目构建。
- 本项目不启用 Conventional Commits；提交说明保持简洁、说明意图即可。

## 数据与实验纪律

- ESC 测试数据只读引用，不复制、不改写。
- 当前 ESC test 永不用于训练或调参；后续 Laya 微调必须使用独立训练集。
- 原始响应、标准化预测、指标和元数据分开保存，边界见 [experiments/README.md](experiments/README.md)。
- 任何实验都必须记录数据指纹、prompt 版本、选项顺序、模型标识和运行命令；不得静默丢弃失败样本。
- 密钥只从本机 `.env` 或环境变量读取，不写入日志、元数据或版本控制。

## 提交与决策

在父仓库根目录提交，只暂存 `eci_decision_models/` 内的文件，不触碰父项目原有改动。

重要架构决策写入 `docs/adr/`，命名格式为 `NNNN-简短标题.md`；格式说明见 [docs/adr/README.md](docs/adr/README.md)。
