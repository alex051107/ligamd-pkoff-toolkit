# 给 GitHub-connected ChatGPT Pro 的审查指令

你是一名独立的分子模拟与小样本机器学习方法审查者。你只能依据当前 GitHub
仓库及这个 pull request 中公开的内容作答。不要假设你能看到私有 workbook、
逐系统标签、ligand identity crosswalk、原始轨迹或 OOF rows。

请先确认你读取的分支或 pull request 中能搜索到唯一标记
`LIGAMD_MODEL_PLAN_REVIEW_20260816`。若找不到，说明你仍在默认分支的旧版本，
不要继续审查新计划。

## 先读什么

请按以下顺序阅读。

1. `docs/reviews/model-improvement-plan-2026-08-16/README.md`
2. `docs/reviews/model-improvement-plan-2026-08-16/current-evidence.md`
3. `docs/reviews/model-improvement-plan-2026-08-16/proposed-plan-zh.md`
4. `README.md`
5. `MODEL_CARD.md`
6. `docs/endpoint-and-sampler.md`
7. `docs/method-evidence.md`
8. `docs/model-development-roadmap.md`
9. `docs/code-map.md`
10. `DATA_NOTICE.md`
11. `ligamd_pkoff/resources/models/experimental_n31_registry_v1/model_scoreboard.tsv`
12. `ligamd_pkoff/resources/models/experimental_n31_registry_v1/paired_dynamic_increment.tsv`
13. `ligamd_pkoff/resources/contracts/p512_sequence_v1.json`

如果审查某段代码，请给出 repository path 和行号。不要因为某个 private-derived
aggregate 无法在公开仓库重算，就把它改写成已验证；请标记
`NOT_EVALUABLE_FROM_PUBLIC_REPO`。

## 你要回答的问题

### A  上下文是否足够

判断这个 GitHub packet 是否足以审查下面五件事。每项返回 `SUFFICIENT`、
`PARTIAL` 或 `INSUFFICIENT`，并说明缺的最小材料。

1. 工具链和 predictor 的职责边界。
2. 当前 Dynamic10 证据的强弱。
3. candidate admission 与 locked evaluation 的防泄漏设计。
4. ordered-versus-shuffled GRU 的 11-channel、fold-local scaling、固定
   permutation、科学问题和停止规则。
5. Deep Learning 与 LoRA 的 reopening 条件。

### B  核对关键判断

每项返回 `SUPPORTED`、`NOT_SUPPORTED` 或
`NOT_EVALUABLE_FROM_PUBLIC_REPO`，给出一至三句依据。

1. 当前证据不能把 Dynamic10 称为稳定增量，也不能把它称为完全无信息。
2. 在 N31 上继续追加常见 tabular families，主要增加后验选择，不能提供独立确认。
3. 协作者的 `Finished` 状态不能直接替代 label、identity、endpoint、feature 和
   exposure admission。
4. `FROZEN_N31_TO_NEW_GROUPS_ONCE` 能把 independent confirmation 与未来
   expanded-cohort refit 分开。
5. `O minus H` 检查 frame order；`O minus S` 才检查 ordered temporal block
   是否超过 Static20。
6. Outer-training 内 cross-fitted Static20 residual target 是必要的泄漏防护。
7. P512 不是均匀物理时间，因此 GRU 正结果不能解释为 physical kinetics。
8. 当前 rank-1 output-head LoRA 应关闭；未来 encoder LoRA 必须先有 frozen
   probe、overlap ledger 和 independent-group evidence。

### C  Ponytail-style 计划删减

对下列组件返回 `KEEP`、`REVISE` 或 `DELETE`。`REVISE` 必须给一句可直接
替换的方案。建议增加一个方法时，必须删除或替换一个现有组件，不能扩大清单。

1. 一张只新增 `TRAJECTORY_PROTOCOL_COMPATIBILITY` gate 的 candidate admission ledger。
2. 同家族 new-ligand 与新家族 transfer panel 分开。
3. N31-trained Ridge pair 作为 locked evaluation 的唯一 primary pair。
4. Label-blind P512 sequence serializer。
5. 一次 S/O/H residual GRU smoke。
6. 一张合并的 `ENGINEERING_VALIDITY_RECEIPT` 加一个三态 scientific rule；不保留
   独立 parameter-update-norm gate。
7. 单 seed、固定预算、无 rescue。
8. BiCoA A/B 仅在 Ridge primary 完整报告后另立 overlap-safe secondary。
9. Frozen encoder probe 后才允许一次 encoder LoRA。
10. 删除更多 tabular families、TCN/Transformer tournament 和 LoRA sweeps。

### D  必须作出的决策

1. 当前最值得立即执行的是 candidate ledger、sequence serializer，还是两者并行？
2. N31 上的一次 S/O/H smoke 是否值得做？它可以改变哪一条结论，不能改变哪一条？
3. 当前计划里最像 over-engineering 的一项是什么？请删除或合并，不要只批评。
4. 当前唯一新增的 trajectory-protocol gate 是否足以阻止 acquisition shift 被误写
   成 Dynamic10 或 family-transfer 失败？不要再增加第二个 admission 状态机。
5. 若 private workbook 不能公开，公开 packet 还缺什么最小的 aggregate evidence
   才能审查计划？不要要求上传逐系统 labels、ligand identities 或 raw trajectories。
6. LoRA 最合理的未来对象是 static encoder、temporal encoder，还是暂不开放？
   只能选一个，并说明 matched frozen control。

## 输出格式

按 A、B、C、D 的顺序回答。最后严格给出下面八行。

```text
CONTEXT_SUFFICIENCY = SUFFICIENT | PARTIAL | INSUFFICIENT
OVERALL_PLAN_DECISION = KEEP | REVISE | REJECT
RUN_NOW = ...
HOLD = ...
DELETE = ...
SINGLE_PRIMARY_CONTRAST = ...
STOP_RULE = ...
CLAIM_CEILING = ...
```

限制如下。

- 不写隐藏思维过程。
- 不建议 model zoo、AutoML、ensemble 或 rank/seed/module sweep。
- 不把 N31 development results 称为 external validation。
- 不把 trajectory input 称为 physical `koff` estimator。
- 不把 GitHub 中没有的 private evidence 补造成事实。
- 外部技术事实只引用论文、官方代码或作者 checkpoint，并给 DOI、version 或 commit。
- 总输出控制在 1,800 个中文词以内。
