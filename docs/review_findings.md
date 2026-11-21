# 代码审查记录与处理

> 审查对象：master @ `81ebeee`（含 runner、辩论引擎、评测、离线分析）
> 审查方式：独立 @oracle 审查 + 自查交叉核对
> 记录日期：2026-10-24

## 阻塞项（已修复，`a517bca`）

### B1 标签口径错误 —— 曾有结论作废

**问题**：官方 `Muty_agent_debate.py:115` 存入 `gpt_response` 的是**共识字母**（`"A"/"B"/"C"`），
`:117-125` 的 `[^a-z]` 分词作用于单个字母是**正确**的。该分词器的缺陷只影响把自由散文当输入的场景
（`ECI.py` 单智能体基线）。

我们的 runner 存的是**原始长回复（散文）**，却把同一分词器套上去，于是解释文本里的冠词 "a"
被误读成选项 A，人为制造出大量假阳性。

**作废的结论**：
- ❌「官方端到端标签解析有 bug」
- ❌「修正解析器可提升 15–24 个 F1 点（ESC 58.0→73.0，CTB 42.4→66.7）」
- ❌ 上述对比所依据的 `official` 模式指标

这些数字只描述"把 ECI.py 基线的分词器用在散文上"的效果，与 MRBalance 辩论流程无关。

**修复**：
- 输出新增 `consensus_answer`、`label_consensus`、`label_prose_naive`、`label_prose_fixed`
- `--label-mode {consensus,naive,fixed}`，**默认 `consensus`**（与官方端到端等价）
- `parsing.extract_final_label` 的 `official` 更名为 `naive`，并在 docstring 说明其适用场景
- `analysis/label_modes.py` 改为三模式对比，并标注回退行数

**影响**：主实验结论不受影响（此前跑出的 `gpt_pred_bi_causal_label` 用的是 `fixed`，
与 `consensus` 在试点上结果一致：CTB 60.0/75.0/66.7 两者相同）。

## 重要项

| 编号 | 问题 | 处理 |
|---|---|---|
| I1 | 失败样本因指纹匹配被永久跳过，续跑无法只重试失败 | 已修（`bb1ec35`）：`error` 行不参与续跑去重，新增 `--retry-failed` |
| I2 | 并发线程共用全局 `random`，`--seed` 不能保证重跑一致 | 已修（`4d25ef2`）：每样本派生 `Random(seed ^ record_key[:8])` 注入 graph/neuron/ranker |
| I3 | 共识答案为 `""` 时会回退到最后一条原始回复，可能得出与官方不同的标签 | 已修（`bb1ec35`）：`None`/`""` 直接返回 `None` 并在 docstring 说明差异 |
| I4 | 试点取数据头部（非随机）且 temperature=0.2 非确定，外推不可靠 | 已记录：试点仅作可行性证据，正式结论用全量测试集；如需区间估计应多 seed 重复 |

## 建议项

| 编号 | 问题 | 处理 |
|---|---|---|
| S1 | gold 为字符串 `"0"/"1"` 时整批静默进 `skipped`，指标全 0 | 已修（`d081174`）：自动转换，非法值告警一次 |
| S2 | `llm.py` 把 `None` 内容转 `""`，导致官方"None→随机字母"回退失效 | 已修：空回复用注入 RNG 随机选 A/B/C |
| S3 | 内容完全相同的重复输入行会被去重跳过 | 已记录：对完全重复样本只计一次，报告中注明 |
| 可选 | 退避无 jitter，并发会同步重试 | 未做（成本低优先度，后续可加） |

## 保真度核对结论（未发现问题）

- 共识阈值 `> floor(2/3*N)`、round0/1 触发点、round≥2 的 `len(idxs) > floor(2/3*len(idx_mask))` 与参考一致
- listwise top-2 剪枝与 `LLMLP.py:132-135` 等价；官方 `idx in idx_mask` 的 position-vs-agent-id 缺陷**保留为默认行为**，另有 `fix_top2_activation` 开关（2026-10-25 已实测其影响，见下节）
- prompt 风格仅第一轮生效（`force_style=False`），与参考一致
- importance 回传公式与 `LLMLP.py:158-179` 相同，唯一偏差是无匹配节点时告警并置 0（官方此路径会 `ZeroDivisionError`）
- 指标口径与论文一致：`R` = 正类召回，`F1` 由 P 与 `R_pos` 计算，`Acc` = 总体准确率，`R_neg` = 负类召回

## 追加测量：round-3 top-2 激活缺陷（2026-10-25）

### 缺陷语义

官方 `LLMLP.py:132-138` 在每轮先 `shuffle(loop_indices)`，随后用 `if idx in idx_mask` 比较
**打乱后的位置**与 ranker 返回的 **agent id**。两次 shuffle 相互独立，因此该条件等价于
"从 4 个 agent 中随机取 2 个"：ranker 调用照常计费，但它的选择结果对谁上场没有影响。

修复实现改为比较 `node_idx % agents`，默认关闭以保持官方行为；2026-10-25 起可通过
`--fix-top2-activation` 启用，单测见 `tests/test_debate.py`、`tests/test_runner.py`。

### 测量设计

归档响应只保留最终共识与一条原始回复，没有逐轮 transcript，无法直接回放。采用
**配对分叉复放**（`scripts/top2_branch_replay.py`）：

1. 每条样本先跑一次，记录第 0/1 轮与 ranker 的 9 次响应；
2. 用同一 seed 新建两个图，重放这 9 次响应，使第 3 轮入口的节点状态、边权与 RNG 完全一致；
3. 两分支只差第 3 轮选人规则（`idx in idx_mask` vs `node_idx % agents in idx_mask`）；
4. buggy 分支必须逐字复现冻结运行，否则该样本作废。

自然设置下第 3 轮触达率仅 0.4–2.5%，故测量时用 `--force-rounds` 关闭早停，让每条样本都进入
第 3 轮。该设置只用于估计缺陷的条件效应，不代表线上轨迹。

### 结果

| 样本集 | n | 选中的 agent 对变化 | 答案字母变化 | 二值标签变化 | 复放一致 |
|---|---:|---:|---:|---:|---:|
| `official-esc` 归档第 3 轮样本 | 19 | 16（84%） | 0 | 0 | 19/19 |
| `imp-role-balanced` 归档第 3 轮样本 | 31 | 24（77%） | 0 | 0 | 31/31 |

- 选人变化率与理论值 5/6 一致，说明修复确实改变了"谁上场"；
- role-balanced 的 24 次变化中，23 次只替换 2 人中的 1 人（overlap=1），仅 1 次完全不重叠；
- 两臂答案分布完全相同（A 11 / B 8 / C 12）。

### 结论

缺陷**行为上真实存在，结果上近乎无效**：

1. 第 3 轮所有 agent 看到的是同一组第 1 轮回复，角色只改 system prompt，答案主要由共享上下文决定；
2. 最终答案是 2 人众数，替换 2 票中的 1 票很难改变多数；两分支的 agent 也高度趋同；
3. 因此在当前流水线中，team member optimization 近似 no-op，与 top-2 缺陷本身无关。

处理决定：

- 默认保持官方行为（保真复刻），**不为该缺陷重跑 9 组主实验**；
- `--fix-top2-activation` 保留，供"listwise + 异构角色"或未来引入 per-agent 记忆时复测；
- 该缺陷不构成对主实验结论的威胁。

### 附带发现：第 3 轮归属不可复现

两次自然复跑中，归档第 3 轮样本只有 2/19、0/19（official）和 5/31（role-balanced）再次进入
第 3 轮。聚合指标仍可复现（前 200 条标签一致率 93.5%，调用分布同为 76.5% 停在第 0 轮），
但"哪条样本进入第 3 轮"对采样极其敏感。报告应区分"聚合可复现"与"逐样本轨迹可复现"。

## 安全卫生

- 仓库内无 `.env`（仅 `.env.example`）；`.gitignore` 覆盖 `.env`、`.env.*`、`reference/`、`outputs/`
- 提交历史中从未包含密钥或 `reference/` 内容
