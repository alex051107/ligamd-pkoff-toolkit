# 给 ChatGPT Pro 的关键审查提示词

把下面完整代码块粘贴给已连接本 GitHub 仓库或当前 pull request 的 ChatGPT Pro。它要求审查者先读取公开材料，再明确区分可核对的结论、私有材料不足和真正的方法漏洞。

```text
你是一名独立的分子模拟、统计学习和研究可重复性审查者。请对这个 LiGaMD 到实验 pKoff predictor 项目做一次批判性审查。你的任务是寻找会推翻当前结论、扭曲 MAE 或造成不当模型选择的地方，不是帮助项目为已有结果辩护。

你只能把 GitHub 仓库和本 pull request 中公开可见的内容当作可独立核对的证据。不要假设你看到私有 pKoff workbook、逐系统 assay 条件、ligand identity crosswalk、原始轨迹、topology、cluster log 或逐行 OOF predictions。公开材料不能证明的事项请写 NOT_EVALUABLE_FROM_PUBLIC_REPO，并说明最少还需要哪一类聚合或脱敏证据。

先确认你能找到唯一标记：pKoff workflow critical-review handoff, 28 August 2026。若找不到，说明你没有审查到当前材料，请停止并说明缺少哪个文件或分支。

请按下面顺序阅读：
1. docs/reviews/critical-handoff-2026-08-28/README.md
2. docs/reviews/critical-handoff-2026-08-28/n34-aggregate-evidence-v1.tsv
3. docs/reviews/critical-handoff-2026-08-28/admission-audit-aggregate-v1.tsv
4. docs/reviews/critical-handoff-2026-08-28/1eby-cohort-receipt-v1.md
5. docs/reviews/model-improvement-plan-2026-08-16/README.md
6. docs/reviews/model-improvement-plan-2026-08-16/current-evidence.md
7. docs/p512-sequence-v2-interior-shuffle.md
8. docs/endpoint-and-sampler.md
9. docs/method-evidence.md
10. docs/model-development-roadmap.md
11. MODEL_CARD.md
12. README.md
13. ligamd_pkoff/resources/models/experimental_n31_registry_v1/model_scoreboard.tsv
14. ligamd_pkoff/resources/models/experimental_n31_registry_v1/paired_dynamic_increment.tsv
15. ligamd_pkoff/resources/contracts/p512_sequence_v2.json

项目目前报告的私有聚合状态如下。你可以审查其逻辑和所需证据，但不能假装已独立重算：
- 34 个历史 development-exposed systems，30 个 exact-ligand groups。
- Static20 + Ridge 的 group-equal MAE 为 0.8162；Combined30 + Ridge 为 0.7762。
- 定义 delta = Combined30 MAE - Static20 MAE，得到 -0.0400 pKoff；5,000 次 paired group bootstrap interval 为 [-0.1196, +0.0146]；14/30 groups 的误差下降。
- 因区间包含 0，项目没有把 Combined30 说成稳定改进、最终模型或外部验证。
- 37 个历史完成系统经过无标签检查，35 个具有 Current30 特征链；其中两个没有凑齐三条 complete-exit passes，另一个因重复模拟资产和 assay-ligand mismatch 不进入独立 supervised row。
- 新 1EBY 的初始、冻结 18-replica cohort 有一条 complete-exit pass，不能通过 exact-three gate，因而没有进入特征、预测或 MAE。另有六条同条件 production 的提交时间早于该 18 条 cohort 的 first endpoint selection readout；它们是同一系统的单独 cohort，不是新的 ligand group，且尚未进入特征或模型。请以 cohort receipt 审查时间边界是否足够。

请逐项审查以下论断。每项严格使用 VERIFIED、WRONG、DATA_INSUFFICIENT 或 NOT_EVALUABLE_FROM_PUBLIC_REPO 之一开头，并给出简短理由及 GitHub path/line range 或外部一手来源。不要输出隐藏思维过程。

C1. 目前 N34 只支持“Combined30 的点估计较低但稳定增量未建立”，不支持最终模型选择、外部泛化或物理 koff 估计。
C2. paired group bootstrap interval 跨过 0 时，不能用 -0.0400 的点估计宣称 Dynamic10 已可靠降低 MAE。
C3. Dynamic10 没有稳定增量，不能推出原始 trajectory 完全无信息；它只否定当前十个摘要在当前开发比较中已被证明有稳定增益。
C4. exact-ligand leave-one-group-out 与 fold-local feature preprocessing 处理了部分标签泄漏风险，但未自动消除 protein family、assay、construct、chemical-state 或 production-protocol shift。
C5. 以三条 endpoint-passing replicas 才构建一个系统行，可能引入选择效应；当前工作把 endpoint 设为 label-blind contract 是否足以处理这个问题。
C6. 1EBY cohort receipt 所记录的时间顺序和禁止混池规则，是否足以排除“看见首批 endpoint 结果后才启动六条”的特定 rescue 风险。它不能把第二 cohort 变成新的 ligand group。若证据仍不足，请给出最小修订，而不是建议无限增加 replicas。
C7. P512 的内部顺序测试应固定首帧和 endpoint、只乱序 510 个内部帧。若 ordered 与 matched shuffled 有差异，它能说明什么；若没有差异，又不能说明什么。
C8. Static20 是回答“轨迹摘要是否在起始结构和化学描述符之外增加预测信息”的必要对照，不是为了人为抬高 Combined30 的门槛。
C9. 当前最可能导致结论错误的环节是数据量、表征压缩、标签/模拟不匹配、endpoint selection，还是其他因素。请不要只给一个猜测；按现有证据能否区分来判断。
C10. 继续在 N34 上搜索新的 tabular model、seed、optimizer、feature window 或 endpoint 阈值，能否提供独立确认。

然后提出一个最小修订方案。对下面每项输出 KEEP、REVISE 或 DELETE。REVISE 必须给出一条能直接实施的替代方案；若新增一个方法，必须同时删除或替换一个现有组件，不能扩张成 model zoo。

S1. 保持固定的 label-blind endpoint、P512、exact-three 和 exact-ligand LOGO 合同。
S2. 将 1EBY 的新增六条作为与初始 18 条分开的 cohort，不混合选择。
S3. 对新且 development-unexposed 的 exact-ligand groups 只运行一次已冻结的 Static20 Ridge 对 Combined30 Ridge 比较。
S4. 一次 fixed ordered-versus-interior-shuffled sequence feasibility probe，使用相同 P512 帧、同一初始化和相同训练预算。
S5. 在 N34 上增加更多 tabular family、seed、architecture 或 learning-rate rescue。
S6. 在系统进入 supervised comparison 前公开一个只含状态计数的 admission audit，按 production、endpoint、feature、identity、label、exposure 分层，不公开个人标签或轨迹。

最后回答五个强制问题：
1. 哪一个当前结论是证据最充分的，哪一个最脆弱？不要打分，只说明原因。
2. 当前材料是否足以把 performance limitation 主要归因于 sample size？如果不足，请明确哪些解释仍不可区分。
3. 对 1EBY 的 cohort 设计，什么事实会构成真正的 post-hoc rescue，什么事实仍可被视为预先存在的独立 cohort？
4. 若只能补一份不泄露私有数据的聚合证据表，你要求它包含哪些列？
5. 哪个现有步骤应立即停止或合并，以避免 over-engineering？

按以下格式作答，总长不超过 1,500 个中文词：

## 逐项结论
每项 C1-C10 一行。

## 流程取舍
每项 S1-S6 一行。

## 必须修复的问题
最多五项，按对结论影响排序。每项必须说明影响的结论和最小修复。

## 最终判断
严格写这八行：
OVERALL_DECISION = KEEP | REVISE | REJECT
CURRENT_CLAIM_CEILING = ...
PRIMARY_UNRESOLVED_CAUSES = ...
NEW_COHORT_RULE = ...
ONE_ALLOWED_NEXT_COMPARISON = ...
STOP = ...
PUBLIC_EVIDENCE_GAP = ...
PHYSICAL_KOFF_CLAIM = NO

限制：
- 不要把 development cohort 的较低 MAE 写成 external validation。
- 不要把 LiGaMD 帧或 P512 排序写成物理时间或 physical kinetics。
- 不要要求上传原始轨迹、private labels 或完整 identity mapping。
- 不要建议 AutoML、ensemble、model zoo、seed sweep、架构 tournament 或阈值救援。
- 提到代码或公开证据时给出可定位的 path 和行号。
- 外部事实只引用论文、官方文档或作者发布的代码，并给 DOI、版本或 commit。
```

审查回复回来后，先核对它引用的仓库位置和外部来源，再决定是否修改流程。审查意见本身不是新的实验结果。
