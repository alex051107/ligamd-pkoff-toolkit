# LiGaMD experimental pKoff 双主线改进计划

```text
DOCUMENT_STATUS       = PROPOSED_FOR_REVIEW
EXECUTION_AUTHORITY   = NONE_FROM_THIS_DOCUMENT
TARGET                = ASSAY_DERIVED_EXPERIMENTAL_PKOFF
PHYSICAL_KOFF         = NOT_ESTIMATED
```

这份计划想解决一个很具体的问题。当前 N31 上，Static20 加 Dynamic10 的
Ridge 和 Random Forest 点估计略好，但区间跨零；常见 tabular family、两种
typed representation 和一次固定 output-head LoRA 也已经检查过。继续追加
模型名字，会增加挑选自由度，却没有回答 Dynamic10 为什么不稳定。

下一步同时推进两条线。第一条线准备真正可用的新 exact-ligand groups。第二
条线保留已有 P512 frame 的顺序，做一次带 shuffled control 的低容量开发
实验。两条线共享 endpoint、P512、exact-three replicas 和 experimental
pKoff 的定义，不共享科学结论。

## 当前目标

本轮立即允许的工作只有两项 label-blind 工程。

1. 建立一张候选准入表，逐项记录 label authority、exact-ligand identity、
   三条 endpoint-passing replicas、P512/Current30 readiness 和 development
   exposure。
2. 从现有 P512 source-frame identities 序列化
   `31 systems x 3 replicas x 512 frames x 11 channels`。不重跑 MD，不重判
   endpoint，不重新采样，也不读取 pKoff、fold 或模型误差。

本轮不授权训练 GRU、读取新组预测误差、重新拟合 N31、更新 bundled models
或运行 LoRA。

## 主线 A  准备新 ligand groups

### A1  一张候选准入表

候选来源是私有协作者 roster。选择顺序不得使用 `koff` 大小、当前预测误差
或某一候选是否容易降低 MAE。每个系统只保留一个最终状态。

```text
READY_FOR_LOCKED_INDEPENDENT_EVALUATION
READY_ONLY_FOR_DEVELOPMENT_SENSITIVITY
BLOCKED_LABEL
BLOCKED_IDENTITY
BLOCKED_ENDPOINT
BLOCKED_FEATURE
OUT_OF_SCOPE_NEW_FAMILY
```

同家族的新 ligand panel 和新 protein-family transfer panel 分开。一个系统
即使有三条完成的模拟，也不能跳过 assay ligand、protein construct、label
source 和 exact chemical identity 的核对。

### A2  只为已通过前置门的候选生成 features

复用公开路线的固定合同。

```text
endpoint       = endpoint-v2 / simple two-metric episode boundary
sampler        = P512
replicas       = exactly three endpoint-passing routes
system output  = Static20, Dynamic10, Combined30
```

少于三条合格路线、identity 未闭合或 feature 非有限时，系统保持 blocked。
不得改用两条 replica、从更多 attempts 中根据结果补挑三条，或放宽 endpoint。

### A3  冻结一次 independent-group comparison

在读取该批模型误差前固定下面的设计。

```text
EVALUATION_DESIGN               = FROZEN_N31_TO_NEW_GROUPS_ONCE
TRAINING_COHORT                 = FROZEN_N31_ONLY
NEW_GROUP_LABELS_ENTER_TRAINING = NO
PREPROCESSING_AND_MODEL         = FROZEN_N31_FULL_FIT_AUTHORITY
PRIMARY_PAIR                    = COMBINED30_P512_RIDGE_MINUS_STATIC20_RIDGE
DUMMY                           = FROZEN_N31_TRAINING_LABEL_MEDIAN
```

新组标签不能进入 scaling、feature reduction、hyperparameter choice、model fit
或样本选择。若已有 full-fit bundle 的训练来源无法闭合，只能在读取新组误差
前依据现有 N31 合同各 fit 一次并冻结。

由于协作者 roster 已经带有实验值，这项工作称为
`LOCKED_INDEPENDENT_GROUP_EVALUATION`。若一个 group 已参与 endpoint、feature、
sampler 或 model development，则降级为 `DEVELOPMENT_SENSITIVITY`。

把新 groups 并入 expanded cohort 后重新做 grouped cross-validation 可以成为
未来的模型更新，但不能与这次 frozen-N31-to-new-groups comparison 混称。

### A4  样本量与停止点

计划不预设 5、10 或 15 个 group。先确定最小有意义的 paired MAE difference、
现有 group-loss difference 方差、alpha、power、family composition 和 admission
attrition，再做 precision 或 power planning。若合格 groups 不足，只报告
case-level predictions 和准入缺口，不用不稳定的 bootstrap 选模型，也不补换
更容易的候选。

## 主线 B  检查 frame order 是否被压缩掉

### B1  Label-blind sequence serializer

每条 replica 只读取现有 P512 source identities 和对应的 11 个物理通道。

```text
one replica  = 512 real frames x 11 channels
one system   = 3 replicas x 512 frames x 11 channels
one label    = one system-level experimental pKoff
```

要求每条 route 的 source index 严格递增，shape 和数值有限。`source_index` 与
normalized progress 只进入审计 metadata，首版 encoder 不使用。三条 replicas
仍不是三条监督样本。

只做一个 synthetic ordering test 加一个 93-route shape/finite/monotonic check。
不重新读取 NetCDF 做第二套 endpoint 或 P512 parity。

### B2  条件式的一次 GRU development smoke

只有 serializer 与训练合同通过审查且用户再次明确批准，才运行下面三臂。

```text
S = fold-local Static20 Ridge
O = S + ordered-sequence GRU residual
H = S + frame-order-shuffled GRU residual
```

`O minus H` 是唯一 primary contrast。它检验同一批 frame 的真实顺序是否比
错误顺序更有信息。`O minus S` 是 secondary qualification，检验 ordered
temporal block 是否超过 Static20。它们都不估计 physical kinetics。

模型固定为一个小网络。

```text
input size        = 11
hidden size       = 8
layers            = 1
direction         = unidirectional
dropout           = 0
replica encoder   = shared weights
replica pooling   = mean of three final hidden states
head              = Linear(8, 1) residual correction
optimizer         = AdamW
learning rate     = 1e-3
weight decay      = 1e-4
training steps    = 200 full-batch updates
seed              = one frozen seed
hyperparameter tuning = none
```

Outer split 使用 exact-ligand LOGO。每个 outer fold 只能在 outer-training groups
内构造 cross-fitted Static20 residual targets。Outer-test group 不进入 scaler、
Static20 fit、residual target、GRU fit 或任何选择。Ordered 与 shuffled arms
使用同一初始化、fold、scaler、optimizer 和训练步数；shuffle 按 replica 固定，
11 个 channels 作为整行一起移动。

预先冻结的统计合同如下。

```text
PRIMARY_METRIC   = EXACT_LIGAND_GROUP_EQUAL_MAE
PRIMARY_DELTA    = O_MINUS_H
SECONDARY_DELTA  = O_MINUS_S
BOOTSTRAP        = PAIRED_GROUP_BOOTSTRAP_5000_SEED_20260816

SUPPORTED     = DELTA < 0 AND CI_UPPER < 0 AND IMPROVED_GROUPS >= 14_OF_27
NOT_SUPPORTED = DELTA >= 0
INCONCLUSIVE  = OTHER_FINITE_ENGINEERING_VALID_RESULT
```

科学解释前只做一个合并的工程有效性检查。

```text
ENGINEERING_VALID =
  ALL_27_OUTER_FOLDS_COMPLETE
  AND ALL_PREDICTIONS_AND_LOSSES_FINITE
  AND ORDERED_AND_SHUFFLED_BUDGETS_IDENTICAL
  AND PARAMETER_UPDATE_NORM > 1e-12
  AND BOTH_RESIDUAL_CORRECTION_STANDARD_DEVIATIONS > 1e-12
```

若不满足，结果只能是 `ENGINEERING_INVALID`。此时停止并报告工程失败，不调
learning rate、不换 seed，也不能解释成没有顺序信号。

实验完成一次后停止。Primary 不支持时关闭当前 order-sensitive branch；
Primary 支持而 secondary 不支持时，只报告 development-set order sensitivity；
两者都支持时冻结候选并等待 independent groups。任何结果都不触发 TCN、
Transformer、第二个 GRU、第二个 seed 或 architecture tournament。

本实验允许的最高表述是，N31 development panel 上按路径进度选择的 P512
frame 顺序是否含有与 experimental pKoff residual 相关的可学习信息。它不能
升级成 unbiased time dynamics、physical kinetics 或 independent-group
generalization。

## Deep Learning 与 LoRA 的后续边界

```text
LABEL_BLIND_SEQUENCE_ENGINEERING = ALLOWED_NOW
ONE_FROZEN_TEMPORAL_SMOKE        = ALLOWED_ONLY_AFTER_EXPLICIT_APPROVAL
SUPERVISED_DECISION_CHANGE       = BLOCKED_UNTIL_INDEPENDENT_GROUPS
```

LoRA 不只用于语言模型。它需要一个有意义、来源和训练数据可审计的 pretrained
encoder。当前固定的 rank-1 output-head LoRA 已经完成且表现不利，不再搜索
rank、module、seed 或 optimizer。

未来只允许一条 reopening sequence。

1. 选择一个 license、checkpoint lineage 和 training data 可审计的 pretrained
   protein 或 ligand encoder。
2. 建立 exact input-overlap ledger。
3. 冻结 encoder，只运行一个 linear 或 Ridge probe。
4. 在 independent groups 上确认 frozen embedding 的增量。
5. 只有前四步通过，才做一次同 checkpoint、head、fold 和训练预算下的 fixed
   encoder LoRA 与 frozen control 比较。

若 frozen probe 没有增量，LoRA 不启动。Static encoder LoRA 即使降低绝对
pKoff error，也不能单独证明 LiGaMD trajectory 有价值。当前没有适合直接
低秩适配的 pretrained temporal trajectory backbone；为了 LoRA 从零预训练
一个大 backbone 属于本阶段的过度设计。

## 现在执行、条件式执行、删除

| Component | Status | Reason |
| --- | --- | --- |
| Candidate admission ledger | Run now | 直接处理当前新组准入 blocker |
| P512 sequence serializer | Run now | Label-blind、复用现有 frames、回答后续表示问题 |
| Frozen N31 to new groups comparison | Hold | 当前合格新 groups 为零 |
| One S/O/H GRU smoke | Hold for explicit approval | 只允许一次 development-only test |
| PLS in locked evaluation | Delete | Development-exposed post-hoc family，不增加主问题清晰度 |
| BiCoA A/B secondary | Hold | 仅限 descriptor-valid 且 input-overlap-safe groups |
| More tabular families | Delete | 当前表示上已覆盖主要低容量函数族 |
| TCN/Transformer/model zoo | Delete | 小样本下增加选择自由度，不回答顺序 null |
| LoRA rank/module/seed sweep | Delete | 当前 output-head implementation 已关闭 |

## 最小验证预算

| Batch | Focused check | Independent review | Full model rerun | Hash |
| --- | --- | --- | --- | --- |
| Admission ledger | one schema/enum/duplicate/label-blind check | none until cohort freeze | no | no |
| Sequence serializer | one synthetic ordering test plus one 93-route contract check | one combined review at runnable milestone | no | one only if a deterministic tensor is transferred |
| GRU smoke | one leakage/coverage/budget/result-contract check | one combined review after the single run | exactly one authorized run | no routine hash |
| Locked new-group evaluation | one freeze/admission/prediction check | one combined review before label-error readback | one frozen application | one digest for the frozen cohort/model authority |

相同代码、输入和风险的检查不重复。修复后只重跑受影响的 category 一次。

## 计划的成功与失败

计划成功不等于 MAE 必须下降。成功是候选准入、sequence representation、比较
设计和停止规则都被执行到可解释的状态，并且不因结果不好而追加模型或改样本。

计划失败包括根据标签或误差选候选、让新组标签进入训练、将 P512 说成均匀
时间、把 development result 说成 external validation，或在一次结果不理想后
继续搜索模型、seed、threshold 和 architecture。
