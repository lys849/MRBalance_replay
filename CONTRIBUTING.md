# 贡献与开发规范

## 工具链

| 目的 | 命令 |
| --- | --- |
| 安装依赖 | `uv sync` |
| Lint | `uv run ruff check .` |
| 格式化 | `uv run ruff format .` |
| 类型检查 | `uv run mypy .` |
| 全部 pre-commit 钩子 | `uv run pre-commit run --all-files` |
| 测试 | `uv run pytest` |
| 构建发行包 | `uv build` |

提交前至少运行 `ruff check`、`mypy` 和 `pytest`。pre-commit 只挂快速自动修复钩子，不运行测试或慢检查。

## 命名规范

| 对象 | 规则 | 示例 |
| --- | --- | --- |
| 模块/文件 | snake_case | `user_service.py` |
| 函数/方法/变量 | snake_case | `get_user_by_id` |
| 类 | PascalCase | `UserService` |
| 常量 | UPPER_SNAKE_CASE | `MAX_RETRIES` |
| 私有成员 | 前缀单下划线 | `_cache` |
| 布尔值 | is_/has_/can_ 前缀 | `is_valid` |
| 测试 | `test_*.py`，函数 `test_*` | `test_user_service.py` |

## 代码风格

- 使用 ruff 进行 lint 和格式化，行宽 100，双引号。
- mypy 启用 strict，公共函数和模块边界应有明确类型标注。
- 测试使用 pytest，文件名与测试函数均以 `test_` 开头。
- 不提交 `.env`、API key、虚拟环境与缓存；`outputs/` 下的原始运行产物不入库，只有归档到 `experiments/<run-id>/` 的实验结果入库（见「版本控制原则」）。
- `paper_report/` 是独立子项目，`reference/` 是第三方参考实现；不要修改或纳入本项目构建。

## 版本控制原则

**隔离。** 并行任务使用 `git worktree`，路径为 `.slim/worktrees/<lane>`，分支为 `omos/<lane>`（从 `master` 创建）。一个 lane 只负责一个关注点，不要在主 checkout 上做 lane 的工作。

**小步提交。** 一个提交只做一件事；代码、测试、文档尽量分开。每个提交前运行 `uv run ruff check .`、`uv run mypy .`、`uv run pytest`，保证任一提交都能单独回退、单独验证。

**提交信息。** 沿用仓库现有风格，如 `feat(runner): ...`、`fix(debate): ...`、`docs: ...`、`test: ...`，用中文说明意图；不要求严格的 Conventional Commits。

**实验产物。** 区分原始运行产物与归档实验结果：

- `outputs/` 下的原始 `response.jsonl` 与日志不入库（已由 `.gitignore` 忽略）。
- 需要留存的实验归档到 `experiments/<run-id>/`，其中 `meta.json`、`metrics.json`、`metrics.md`、`response.jsonl.gz` 入库。
- 每次实验使用唯一 `run-id`；`meta.json` 作为 manifest，记录 commit、配置、数据指纹与指标，保证结果可追溯、可复现。

**禁止提交。** API key / `.env`、`reference/`、虚拟环境，以及各类缓存。

## 提交与决策

未启用 Conventional Commits，提交信息保持简洁、说明意图即可。

重要架构决策写入 `docs/adr/`，命名格式为 `NNNN-简短标题.md`，并在 `docs/adr/README.md` 中了解模板。
