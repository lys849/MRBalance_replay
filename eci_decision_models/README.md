# eci-decision-models

在 Event Causality Identification（ECI）上研究 **Jev** 与 **Laya** 两种 System One 决策模型的应用。主任务是把事件对判为 `cause`、`caused by` 或 `none` 三分类，首轮数据为 ESC。本项目比较的是两个决策模型的零样本表现，不是 MRBalance 多智能体辩论框架的复刻。

## 项目定位

- **主任务**：给定句子与两个事件，在统一的三选项 `Choice` 下预测方向性因果关系。
- **对照口径**：`cause` 与 `caused by` 合并为正类、`none` 为负类，派生二分类指标，仅用于和 MRBalance 原二分类指标对照。
- **首轮模型**：Jev（OpenRouter 托管）与 Laya（`convaiinnovations/laya-typed-decisions`，421M，零样本）。
- **首轮数据**：仅 ESC 测试集；CTB 后续扩展。
- **首轮基线**：不加入随机、多数类或生成式 LLM 等额外基线，只比较 Jev 与 Laya。
- **后续方向**：选项顺序鲁棒性、CTB、Laya 微调。微调必须先取得独立训练集；当前 ESC test 永不用于训练或调参。

详细的研究问题、输入语义、指标定义、阶段计划与解释边界见 [docs/application_plan.md](docs/application_plan.md)。

## 与父项目的边界

本目录是 [MRBalance_replay](../) 内的独立 Python 子项目：

- 拥有自己的 `pyproject.toml`、`uv` 环境、工具链配置和包名 `eci_decision_models`。
- 不创建独立 Git 仓库；父仓库仍是 Git 根。提交时只暂存 `eci_decision_models/` 内的文件。
- 父项目原有文件一律视为只读，本项目不修改父项目的代码、配置或文档。
- ESC 数据只读引用 `../reference/MRBalance/data/test/ESC_intra_data.json`，不复制、不改写。
- 父项目的 `reference/` 是第三方参考实现；本项目不修改它，也不把它纳入构建。

注意：`.pre-commit-config.yaml` 是为“从父仓库根目录显式调用”设计的，且没有安装 Git 钩子，避免覆盖父项目已有的钩子。命令见下文。

## 当前状态：Phase 1 已完成；Laya 非 ESC 探针 READY；Laya-only ESC 探索性评测已运行

Phase 1 离线实现、remediation、联合验证与 Oracle Gate 1 复审均已完成，门禁状态为 **READY**。Laya 的 6 例真实非 ESC 探针已通过隔离 runner 执行；Oracle Gate 2 评审提出的三项 must-fix（稳定 `record_id`、冻结状态枚举、文档覆盖表述）已修复，v8 工件已生成，Oracle 复审结论为 Laya Gate 2 **READY**。Jev 适配器已按 `typesafe-sdk==0.7.5` 离线实现并通过 fake-only 测试，但真实 Jev 非 ESC 探针尚未运行（当前无 `OPENROUTER_API_KEY`）。**Phase 2 整体门禁仍为 OPEN**。

按用户指示，已使用 Laya 对 ESC 测试集完成一次 **Laya-only 零样本探索性全量评测**：1600/1600 预测有效，Accuracy 0.2444、Macro-F1 0.2090，混淆矩阵与全部指标见 [`experiments/20261104T105741Z-laya-esc-full1600/`](experiments/20261104T105741Z-laya-esc-full1600/)。该 run 只覆盖 Laya 一侧，**不是 Jev/Laya 对比，也不关闭 Phase 2/3 门禁**；机器可读结果在 `metrics.json`，完整运行记录在 `meta.json`，逐样本预测在 `predictions.jsonl`。

正式 Laya 探针工件位于 `artifacts/probes/20261104T095247Z-laya-non-esc-v8/`：通过 spawn 子进程与父进程硬墙钟截止的隔离 runner 运行，6/6 返回完整三选项概率，运行环境与项目 `uv.lock` 一致，使用固定 revision、MPS 和单状态 float32 推理。v8 与 v7 的 `record_id` 集合、choice、三选项概率、request hash 和 response hash 全部一致；v7/v8 与 v6 的预测和概率也一致。`record_id` 为 `probe:<canonical sha256>`，不含 `run_id`/位置，跨 run 稳定；逐样本 `status` 严格限定冻结的 5 值枚举。v2（独立临时环境）与 v5（锁环境）的 choice 全部一致；`negation` 的一个 4 位概率因 torch/transformers 版本不同从 `0.3698` 变为 `0.3699`，因此不声称跨环境比特级复现。探针结果只证明接口与概率可用，不作为分类效果或校准质量结论。

Jev 适配器位于 `src/eci_decision_models/jev.py`，使用可选依赖 `typesafe-sdk==0.7.5`；客户端固定为 `TypeSafeClient(api_key=..., base_url=https://openrouter.ai/api, compression=False)`，并从 `raw_http_response.json()` 读取原始响应。离线测试全部使用 fake，不访问网络、不需要 key；真实 Jev 探针尚未接线/运行。

```bash
# 默认 dry-run，不加载模型
uv run python -m eci_decision_models.probe

# 同时安装 Laya 与 Jev 可选依赖
uv sync --extra jev --extra laya

# 显式授权真实 Laya；仅固定6例非ESC探针，run-id 不得复用
uv run python -m eci_decision_models.probe \
  --authorize-real-laya --download-weights --device auto \
  --run-id <new-laya-probe-run-id> --max-requests 6 --max-retries 0

# Laya-only ESC dry-run（默认，不加载模型）
uv run python -m eci_decision_models.esc_run

# Laya-only ESC 正式 run；Phase 2 整体未关闭时只能标记为探索性结果
uv run python -m eci_decision_models.esc_run \
  --authorize-real-laya --download-weights --device mps \
  --max-requests 1600 --max-retries 0 --max-wall-seconds 7200 \
  --run-id <new-laya-esc-run-id>
```

版本化金标位于 `artifacts/gold_choice_reconstructed/v1/`：1600 行、500 条 causal、零 hard fail；方向来源和批准的 evaluation fallback 均逐行标记。可用以下命令只读重建（不会调用模型）：

```bash
uv run python -m eci_decision_models.gold \
  --esc-data ../reference/MRBalance/data/test/ESC_intra_data.json \
  --xml-root data/EventStoryLine \
  --xml-version 46edefee5e82e0917b823abe0a18bf8c7770f15c \
  --output-dir artifacts/gold_choice_reconstructed/v1
```

依赖已通过 `uv sync` 安装，`uv.lock` 已生成。需要重建环境时，在子项目目录运行：

```bash
uv sync
```

## 常用命令

以下命令在子项目目录内运行：

| 目的 | 命令 |
| --- | --- |
| 安装依赖并生成锁文件 | `uv sync` |
| Lint | `uv run ruff check .` |
| 检查格式 | `uv run ruff format --check .` |
| 类型检查 | `uv run mypy .` |
| 测试 | `uv run pytest` |
| 构建发行包 | `uv build` |

pre-commit 必须从父仓库根目录显式指定子项目配置运行，确保钩子只检查本子项目。首次运行前先在子项目执行 `uv sync`：

```bash
# 在 MRBalance_replay/ 根目录运行
uv run --project eci_decision_models pre-commit run \
  --config eci_decision_models/.pre-commit-config.yaml --all-files
```

不要在本子项目执行 `pre-commit install`：父仓库已有钩子，安装会写入父 `.git/hooks/`，可能覆盖原有配置。

## 数据与环境变量

数据路径、模型标识等非秘密配置写在 `.env.example`。复制为 `.env` 后填写本机密钥：

```bash
cp .env.example .env
# 只在本机填写 OPENROUTER_API_KEY
```

`.env` 已被忽略；任何密钥、令牌或请求头都不得写入代码、日志或实验工件。

## 文档索引

- [docs/application_plan.md](docs/application_plan.md)：应用研究计划与指标口径。
- [docs/adr/README.md](docs/adr/README.md)：架构决策记录格式与索引。
- [experiments/README.md](experiments/README.md)：未来实验工件边界。
- [CONTRIBUTING.md](CONTRIBUTING.md)：开发命令、命名规范与协作约定。
