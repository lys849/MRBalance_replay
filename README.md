# MRBalance 近似复现与 Laya 应用探索

课程项目公开代码快照。MRBalance 使用 DeepSeek，Laya 使用 convaiinnovations/laya-typed-decisions。完整方法差异与结果见 docs/experiment_results.md 和 docs/paper_alignment_audit.md。历史 run 名 paper 表示随机角色与旧 importance 近似。

## 快速验证

安装 Python 3.10+ 和 uv 后，在本目录执行。

```sh
uv sync --frozen
uv run ruff check src tests scripts
uv run mypy src tests
uv run pytest
python3 scripts/verify_tables.py
cd eci_decision_models
uv sync --frozen
uv run ruff check src tests
uv run mypy src tests
uv run pytest
```

verify_tables.py 用归档 P/R 验算 MRBalance F1，用五折混淆矩阵重算 Laya Macro-F1、折均值和样本标准差。MRBalance 原始预测未随此快照公开，脚本验证的是聚合表的算术一致性。归档中的 commit 为原实验代码身份。meta 中本机目录替换为 ${PROJECT_ROOT}、${HF_CACHE} 和 ${USER_HOME}。

## 数据和模型准备

运行模型实验前，按下方原项目说明获取 MRBalance 配套数据。EventStoryLine v0.9 原始语料请从 https://github.com/tommasoc80/EventStoryLine 获取，放到 eci_decision_models/data/EventStoryLine，并保留 annotated_data/v0.9 和 evaluation_format/full_corpus/v0.9。使用方自行核对上游使用条款。测试中的 EventStoryLine 小型 XML 为合成 fixture。真实语料测试按环境变量启用，默认跳过。ZIP 版本也跳过依赖 .git 的元数据测试。

Laya 训练使用 uv sync --extra laya 安装可选依赖。运行 python -m eci_decision_models.finetune_data --help 生成 intra pool 和五折数据，再执行 experiments/pilot-fold0/run_folds.sh。fold0 以同样的 laya-train 参数使用 fold0 的 train/test 文件。模型权重从 Hugging Face 下载，需具备对应设备和存储。各折训练四轮、soft-ce、seed 209，完整配置见 meta 与脚本。在线 MRBalance 调用需要自行配置 .env。

本快照提供程序、合成测试和聚合实验指标。下载训练权重和重跑在线实验需要网络及相应计算资源。项目历史运行说明保留如下。

---

# MRBalance Replay

本项目是课程项目，目标是复现并改进 **MRBalance: A Framework for Enhancing Event Causality Identification in Multi-Agent Debates via Role Assignment**（Knowledge-Based Systems 2025）。项目以官方参考实现为对照，逐步重建可配置、可测试、可复现的 MRBalance 流水线，并探索角色分配、提示词与辩论终止策略的改进。

当前仓库**全链路已实现并可运行**：配置、数据读取、LLM 客户端、多智能体辩论（含单 agent 基线）、评测与 CLI runner 均已落地并有测试覆盖；ESC/CTB 全量复现与角色/prompt 消融已完成，结果见 [docs/experiment_results.md](docs/experiment_results.md) 与 `experiments/`。

## 目录结构

```text
.
├── src/mrbalance_replay/
│   ├── __init__.py          # 包与版本号
│   ├── agents/              # 多智能体角色与辩论编排（占位）
│   ├── prompts/             # 角色提示词与辩论提示词（占位）
│   └── evaluation/          # 指标与结果评估（占位）
├── tests/
│   └── test_smoke.py        # 冒烟测试：导入包并检查版本号
├── docs/adr/                # 架构决策记录
├── paper_report/            # 论文阅读报告子项目，独立维护
├── reference/MRBalance/     # 官方参考代码浅克隆，本地存在但被 .gitignore 排除
├── pyproject.toml           # 项目元数据、依赖、ruff/mypy/pytest 配置
├── uv.lock                  # uv 锁文件
├── .env.example             # 环境变量样例
├── .pre-commit-config.yaml  # 快速格式化/检查钩子
├── .editorconfig
├── CONTRIBUTING.md
└── README.md
```

## 快速开始

要求：Python `>=3.10`，已安装 [uv](https://docs.astral.sh/uv/)。

```bash
# 1. 安装依赖并创建 .venv（同时安装 editable 包）
uv sync

# 2. 准备本地环境变量
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY
# 默认端点 https://api.deepseek.com，模型 deepseek-flash，思考强度 none

# 3. 运行测试与检查
uv run pytest
uv run ruff check .
uv run mypy .
```

`.env` 已被忽略，不会进入版本库。代码中应通过 `python-dotenv` / 环境变量读取凭据，禁止硬编码。

### 关于思考强度（DeepSeek V4）

DeepSeek 自 V4 起默认开启思考（`reasoning_effort=high`），2025 年的 `deepseek-chat` / `deepseek-reasoner` 已于 2026-07-24 下线。复现默认使用 `deepseek-flash` + `reasoning_effort=none`，原因是：

- 论文 baseline 为非推理模型（gpt-4o-mini），关闭思考才具备可比性；
- 辩论框架为 4 agents × 3 rounds，思考 token 会成倍放大成本与延迟；
- 官方答案依赖 `(X)` 选项标记；长思维链更容易干扰选项识别。

改进阶段可用 `MRBALANCE_REASONING_EFFORT=low|high` 做对照实验。

### 运行方式

配置好 `.env`（至少 `OPENAI_API_KEY`）后，使用内置 runner 执行单通道复现：

```bash
# 全量运行 ESC，默认输出到 outputs/esc/response.jsonl
uv run python -m mrbalance_replay

# 与上面等价；也可以直接运行 runner 模块
uv run python -m mrbalance_replay.runner --help

# 小样本冒烟、限制并发
uv run python -m mrbalance_replay --limit 5 --concurrency 2

# 运行结束后计算并打印 P/R/F1、正负类召回与准确率
uv run python -m mrbalance_replay --evaluate

# 更换端点模型 / 思考强度 / 标签解析模式
uv run python -m mrbalance_replay \
  --model deepseek-flash --reasoning-effort none --label-mode consensus
```

常用参数：`--input`（默认 `reference/MRBalance/data/test/ESC_intra_data.json`）、`--output-dir`（默认 `outputs/esc`）、`--limit`、`--concurrency`、`--label-mode {consensus,naive,fixed}`、`--evaluate`、`--seed`、`--shuffle-roles-each-round`、`--selection {listwise,importance}`。

**标签口径。** `gpt_pred_bi_causal_label` 由 `--label-mode` 决定，默认 `consensus`：

- `consensus`：由辩论共识字母映射（A/B → 1，C → 0，无共识 → `null`）。这是与官方 `Muty_agent_debate.py` 端到端等价的标签口径。
- `naive`：把 `[^a-z]` 分词后首个 a/b/c 应用到原始散文回复，对应 ECI.py 基线的天真解析，不代表官方辩论流程。
- `fixed`：在原始散文上优先取最后一个括号选项，并识别 “No causal relationship” 文案。

输出同时保留 `consensus_answer`、`label_consensus`、`label_prose_naive`、`label_prose_fixed`，可离线比较三种口径。

**断点续跑。** 结果逐条追加到 `<output-dir>/response.jsonl`，每条样本记录原字段以及上述标签字段、`gpt_response`、`n_calls`、`prompt_tokens`、`completion_tokens`。重新运行时会跳过已成功记录（按记录内容指纹，兼容 CTB 中重复的 `item_id`）；带 `error` 的失败记录默认会重试，可用 `--no-retry-failed` 关闭。删除输出文件即可从头重跑。失败样本也会写入一行并带 `error` 字段，不会中断其余样本。

**并发与复现。** 每个样本独立创建 `DebateGraph`（图有状态，绝不跨样本复用），由 `--concurrency` 指定的线程池并行执行；写入侧由互斥锁保护并逐条 flush。每个样本用 `--seed` 与记录内容指纹派生独立随机数生成器，因此同一 seed 在并发下仍可复现标签序列。并发提高吞吐，但请结合端点限流调整。

**成本估算。** 官方四角色、三轮且无提前共识时，每个样本最多 4 + 4 + 2 + 1 = **11 次 LLM 调用**（10 次智能体调用 + 1 次 listwise 排序）。若使用 `--selection importance`，第三轮由本地 importance 分数选人，不再调用 ranker，完整流程最多为 **10 次**。若提前达成共识，调用次数会进一步减少。建议先用 `--limit` 小规模验证，再估算全量费用。

## 论文设置与官方代码的差异（可切换开关）

论文 5.3 节与官方仓库默认实现有两处不一致。为做对照实验，`DebateGraph` 与 runner 提供以下开关，**默认值均复刻官方代码**：

| 开关 | 默认值 | 官方行为 | 打开后的行为 |
|---|---|---|---|
| `shuffle_roles_each_round` / `--shuffle-roles-each-round` | `False` | 每个 agent 在全部轮次固定同一角色（官方默认 4 个 `knowledge graphs expert`） | 每轮激活前从 `agent_roles` 去重集合（含 `"none"`）为每个 agent 随机抽角色，使用图内 `rng`，可复现 |
| `selection` / `--selection` | `listwise` | 第三轮前调用 listwise LLM ranker 选 top-2 | `importance`：不调用 ranker，按上一轮 active agent 的 importance 分数选 top-2 agent id |

Python 用法：

```python
from mrbalance_replay.agents import DebateGraph

graph = DebateGraph(
    agents=4,
    rounds=3,
    agent_roles=("knowledge graphs expert", "causal analysis expert", "NLPexpert"),
    shuffle_roles_each_round=True,  # 论文设置：每轮随机分配角色
    selection="importance",  # 论文设置：importance 选 top-2（近似实现）
)
```

命令行用法：

```bash
uv run python -m mrbalance_replay.runner \
  --shuffle-roles-each-round \
  --selection importance \
  --limit 5 --output-dir outputs/esc_paper_setting
```

**importance 选员的近似方式与局限。** 论文未公开 importance 的具体计算与并列处理细节，这里的 `selection="importance"` 是近似实现：

- 在进入下一轮前，取上一轮 active agent 的答案多数票作为基准答案；
- 使用与 `backward` 完全相同的边权回传公式，在截至上一轮的子图上计算 importance；
- 按 importance 从高到低选两个 agent id，importance 相同时按 agent id 升序打破并列；
- 由于上一轮是该子图的最后一层，与多数答案一致的 agent 会共享相同的基准 importance（`1/n`），因此该近似的区分度有限，不等价于论文未披露的评分机制；
- 该模式不调用 ranker，并按实际 agent id 匹配被选中的节点（不受官方 position/id 缺陷开关影响）；
- 这是**论文描述的近似实现，不是官方代码行为**；官方复现仍应使用默认 `selection="listwise"`。

### 单 agent baseline

用于复现论文 Table 3 的 **“w/o All”** 消融（单 LLM 直接推理，不做多智能体辩论）。论文中 GPT-4o-mini 的结果为 **P 41.7 / R 95.4 / F1 58.1**；本项目的默认模型是 `deepseek-flash`，因此数字不能直接与论文对齐，只用于同代码库下的对照实验。

```bash
# 单 agent zero-shot：system 使用 causal analysis expert 角色，user 为原问题
uv run python -m mrbalance_replay --mode single_zeroshot --output-dir outputs/esc_single_zeroshot

# 单 agent zero-shot-COT：在 user 问题后追加 Let's think step by step.
uv run python -m mrbalance_replay --mode single_cot --output-dir outputs/esc_single_cot
```

两种模式每个样本恰好 1 次 LLM 调用，输出字段、断点续跑、并发和评测与辩论模式一致。`--agents`、`--rounds`、`--selection`、`--shuffle-roles-each-round` 在单 agent 模式下被忽略。官方 `ECI.py` 的 zero-shot-COT 分支是空实现，因此 `single_cot` 属于本项目的补充实现，而非官方行为复刻。

## 参考代码说明

`reference/MRBalance/` 是官方仓库的浅克隆，仅用于本地阅读和逐项复现：

```bash
[ -d reference/MRBalance/.git ] || git clone --depth 1 https://github.com/18370343199/MRBalance.git reference/MRBalance
```

该目录具有以下约束：

- 官方实现采用 **GPL-3.0**；本项目不将其纳入构建或版本库，`.gitignore` 已忽略 `reference/`。
- 官方 README 声明环境为 Python 3.9 + 最新版 `openai`，但代码实际依赖 `backoff`、`tenacity`、`pandas`、`httpx`。
- 官方代码存在路径、字段、配置和流程问题，不能直接作为可复用库导入；复现时应在 `src/mrbalance_replay/` 中重写适配层，而不是修改参考目录。

## 官方代码已知问题

以下问题基于 `reference/MRBalance` 当前浅克隆版本，作为复现与修复清单：

1. **数据路径与仓库结构不符。** 多个入口脚本的参数默认值和 `choices` 使用 `../data/ESC_intra_data.json`、`../data/CTB_intra_data.json`，但实际数据位于 `data/test/ESC_intra_data.json` 与 `data/test/CTB_intra_data.json`。从 `CausalDebate/` 目录直接运行时，默认配置会找不到文件。
2. **API 凭据硬编码。** `CausalDebate/util.py` 直接写入 `api_key="you API key"`、`base_url="base url"`；`ECI.py`、`Muty_no_consensus.py`、`one_round.py` 中也有硬编码的空 key/base URL。应统一改为从环境变量或 `.env` 读取。
3. **评分脚本与数据字段不匹配。** `ECI_compute_score.py` 读取 `item["labels"]` 和 `item["type"]`，但 `Muty_agent_debate.py` 输出的 JSONL 保留的是 `bi_causal_label`，且没有 `type` 字段，直接评分会触发 `KeyError`。此外 `ECI.py` 按 CSV 读取输入并写入 `labels`/`type`，与 JSONL 测试数据也不兼容，需要先做数据格式适配。
4. **`Muty_no_consensus.py` 与 `one_round.py` 存在实现缺陷。**
   - `Muty_no_consensus.py` 中硬编码 `item_buckets = item_buckets[1271:]`，会跳过前 1271 条样本；输出文件以 `a+` 打开，重复运行会追加重复记录。
   - 两者都会在模型名不属于 `gpt-3.5-turbo-1106` / `gpt-3.5-turbo-0125` / `gpt-4o-mini` 时访问未定义的 `client2`；一旦更换模型就会失败。
   - `one_round.py` 通过“最长上下文”筛选智能体输出，并依赖跨文件的隐式导入，流程脆弱，不能直接当作可信基线。
5. **默认角色配置未体现论文的角色分工。** `Muty_agent_debate.py`、`Muty_no_consensus.py` 和 `one_round.py` 的默认 `ROLES` 都是 4 个重复的 `knowledge graphs expert`；而 `prompt_roles.py` 已定义知识图谱、因果分析、NLP 三类专家角色。复现实验时需要按论文把角色分配到不同 agent，而不是只改变提示词调优类型。

## 复现路线图

- [x] **P0 项目骨架**：uv 依赖、src 布局、规范、冒烟测试、ADR 目录。
- [x] **P1 配置与数据层**：统一 `.env` 配置；实现 ESC/CTB JSONL 读取、事件对拼接与字段规范化；修正 `bi_causal_label` 等字段口径。
- [x] **P2 API 客户端**：封装 OpenAI-compatible 客户端，统一超时、重试、退避、并发与 token 统计；移除 `client2` 式分支。
- [x] **P3 基线复现**：实现三角色多智能体辩论、轮次控制与共识判定；对齐官方 prompt 与评估指标（P/R/F1、正/负类准确率）。
- [x] **P4 改进实验**：角色平衡、提示词消融、共识/终止策略、随机种子与成本控制；每组实验以 `experiments/<run-id>/` 归档配置与指标。
- [ ] **P5 结果与报告**：整理复现结果、失败案例与消融结论，接入课程报告。[`docs/experiment_results.md`](docs/experiment_results.md) 已汇总 9 组全量实验。

## 开发规范

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest
uv run pre-commit run --all-files   # 首次使用前安装钩子：uv run pre-commit install
```

详细命名与协作约定见 [CONTRIBUTING.md](CONTRIBUTING.md)；架构决策模板见 [docs/adr/README.md](docs/adr/README.md)。
