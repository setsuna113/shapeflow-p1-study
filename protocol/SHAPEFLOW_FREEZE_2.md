# ShapeFlow 实验计划(执行冻结版 Freeze-2 · 2026-08-02)

本文档自包含、独立成立,是本轮唯一的执行依据。Freeze-1 不被本文废止:它是一个**已完成的研究**,其结论与产物保持有效,只是其 H 边界结论的适用范围由 §1.2 重新界定。冻结后任何改动必须以 amendment 形式追加(见 §12),不得改写正文。

---

## 1. 本轮的由来

### 1.1 Freeze-1 留下的问题

Freeze-1 已执行完毕:560 cell offered,520 committed,全部评分。其结论是一个**工作量结论**——P1 显著降低 GPU-busy——而质量**明确未建立**(`reports/P1_FINDINGS.md` §8:工作负载从未通过它自己预注册的 evidence-recall 门,12.2% 的 P0 准确率也无法分辨臂间差异)。

因此 broker 的前提没有被证明。broker 的全部意义是"在质量合格的前提下换取资源形状",而"质量合格"这一项 Freeze-1 没有交付。本轮只做这一项,止于**是否允许进入 broker 研究**的入口,不碰 queue、admission、2-swap、SLO 或在线 broker 收益。

### 1.2 一个比 selector 更前置的缺陷:实验单位

冻结合同 `contracts/docs/HC_MECHANISM_v1.md:15-21` 规定 H 边界:

> **Unit.** One **gather batch** = the sibling tool-call batch of a single assistant turn.
> **P0.** One summarization request per page.
> **P1.** **One whole-batch selector request** covering every page in the batch.

但 `configs/variants.yaml` 中每一个 H variant 都是 `scope: per_page`,而 `strategies/page_h.py` **扇出两次**:先按 sibling(`:92-104`),再按 sibling 内的 page(`_groups_for`)。所以 Freeze-1 的 H 臂对每个 batch 发出 N 次 selector 请求,**每次各自带一份 512 token 预算**。

三点必须写在案上:

1. 合同文档(`e6018ec`,2026-07-28)写于实现(`ae1cea9`/`c684aa2`,2026-07-24)之后四天,且自称"pins the mechanism as implemented"。它 pin 的不是实现。**不存在任何 declared deviation**,`OFFER_INTERFACE_v1.md` 的 C1–C10 也没有一条约束 selector 请求数。这是 drift,不是决定。
2. 代码库中**没有任何 scope 值能表达该合同**:sibling 循环位于 scope 分支之上。修复不是改配置。
3. **原子发布没有坏**,C 边界也完全合规。被违反的是**决策/请求单位**,以及由此而来的**预算作用域**。

已测量的后果(1,808 个已存 H checkpoint):每 batch 恰好 1 个 sibling,平均 **9.07 页**(5–33)。故 per-page 实现下 H batch 的**实际预算上限约为 9×512 ≈ 4,644**,而非 512。

**结论**:Freeze-1 的 H 数字描述的是一个 per-page 机制。它们作为"LLM 不遵守隐藏的 rendered-token 预算,而 CPU exact packing 可以"依然成立,**但不构成 whole-batch P1 的证据**。本轮 H 臂全部重建,命名为 `P1*`,重新 mint execution binding。

### 1.3 已测量的事实(不再重新推导)

| 事实 | 来源 |
|---|---|
| 1,808 H + 715 C checkpoint 在盘,覆盖 183 个 task_id;每臂都存(`runner.py:726`),在 strategy 运行前捕获,故与处理无关 | `/storage/nvme/shapeflow-data/*/checkpoints` |
| 每 batch 恰好 1 个 sibling;平均 9.07 页;P0 共需 16,399 次 page summary | checkpoint 普查 |
| H 处 P0 每 batch 约 9.07×418 ≈ **3,790 decode token**;C 处 `COMPRESSOR_P0` completion p50 = **3,174** | provider ledger |
| prefill 占 H token 成本约九成且两臂几乎相同(3,379 vs 2,977 prompt p50);**节省全在 decode** | provider ledger |
| `max_model_len: 32768`,`selector_max_completion_tokens: 768` | `configs/stack.yaml:48`,`configs/week1.yaml:282` |
| `selected_ids` 的 `maxItems: 64` 是**事后校验器而非解码上限**:超长列表使样本失败,且 token 已计费 | `schemas/selector_output.schema.json:35` |
| 未消耗 task 约 **350**(b1_confirm∪b2 余 180,FV 150,预留 20);b1_select 的 80 已全部消耗 | `bench/bcplus/splits.py:35` |

---

## 2. 三项 operator 决定(本轮不再讨论)

**D1 — 预算保持 512,作用于整个 batch。** whole-batch P1 因此需要相对 P0 压缩 **7.4×**(H)/ **6.2×**(C)。这一 confound 被**明确接受**:质量若守住,结论很强(form 与 amount 两项同时改变都安全);质量若下降,本设计**不声称**能区分是 form 还是 amount 造成的。该限制必须出现在每一份门报告与最终结论中。

**D2 — `top_k` 保持 5,二元短答案准确率从主端点撤下。** 竞争力门以 0.1488 对 0.40 失败,且 `reports/gates/COMPETENCE_PILOT.md` 已证明该配置下没有任何一次运行可能通过该门。主质量端点改为**高分辨率且无需 judge** 的一族(§6.3),官方 grader 降为 secondary sentinel。

**D3 — task 预算 350**,各阶段按实有规模缩编、阶段间 task-disjoint,并按各自的 n 如实报告降低后的功效。

---

## 3. 顺序(固定)

```
构造合法 P1*  →  敲定唯一 selector  →  单次 P1 对最终报告的因果影响
              →  routeability 与 oracle 上界  →  mixed / all-P1 饱和
              →  Broker Entry Gate
```

前一环不成立,后一环不启动。

---

## 4. Phase 0:普查(零 GPU、零 task)

三个数,都不能估算,且都决定后续成本:

1. **页面字节重建率。** checkpoint 只以 `raw_content_id` 命名页面,字节随运行丢弃。经
   `occurrence_id →(运行自己的 retrieval trace)→ docid →(CorpusStore)→ 全文 →(apply_shared_budget)→ 截断文本`
   重建,并**要求 sha256 等于记录的 id**。不匹配记为 `PAGE_BYTES_UNRECOVERABLE` 并计数,**绝不猜测、绝不以 snippet 顶替**。
   该检查是本阶段的全部要点:把貌似合理的字节换给 selector,它照样发布、照样有publication rate,而它选的是另一份文档。
2. **whole-batch prompt token 分布** ⇒ 精确的上下文窗口溢出比例。`SharedContentBudget.derive` 是按**单页**对窗口定尺的,九页视图没有任何窗口保证。
3. **每视图候选数** ⇒ `maxItems: 64` 是否成为约束。

产物:`reports/gates/FREEZE2_CENSUS.json`。命令:`shapeflow freeze2-census`。

**窗口溢出的预先裁定**:装不下窗口的 batch **声明为 P1-ineligible 并回退 P0**,该比例作为 `ρ_mech` 的具名分量报告。不采用 prefilter——prefilter 本身是一个未经检验的选择阶段,会 confound 全部臂。shootout 在**窗口可行层**上进行,使各臂看到逐字节相同的 batch;不可行层单独报告(这本身是一等发现:**LLM selector 族有 CPU packer 没有的窗口资格上限**)。仅当可行层 < 60% 时,才启用统一 top-*m*-per-page prefilter,并以 `S0-LEX-PACK-FULL`(无 prefilter、CPU、免费)界定其代价。

---

## 5. Phase 1:构造合法 P1*

### 5.1 `scope: whole_batch`

新增 scope 值,**不重命名 `per_tool_call`**。两者仅在"每 batch 一个 sibling"时重合;重命名会使将来的双 sibling batch 在 whole-batch 之名下静默发出两次请求、两份预算——即以修好的名字复原该缺陷。多 sibling batch 一律**拒绝**(`WHOLE_BATCH_MULTI_SIBLING`),走既有 P1 失败路径:组件试验记录失败,E2E 整批回退 P0。

`ProsePageStrategy` 在实现 whole-batch 形态前**拒绝**该 scope,理由同上。

### 5.2 `budget_pack_v1`:确定性精确装箱

新增 aggregator,**不改动 `coverage_budget_v1`**(H03/H_HIER_COVERAGE/C02/C03 跑在其上,必须保持可复现)。

- **取消 force 分支。** `coverage_budget_v1:225` 会把矛盾对与最小来源数强行装入并超预算,`preflight.py:392` 随即拒绝**整批**——这正是 H 处 100% 拒绝的机制。改为**成对原子准入 + 回滚**:`preflight` 拒绝的是**单边**矛盾,不是**被丢弃**的矛盾;两边同弃合法且免费。
- **每次准入判定都用精确成本**(`view.coster()`,与 preflight 同一 renderer 与冻结 tokenizer)。所声称的保证**只有**一条:*返回对象逐字节等于 coster 最后一次度量的对象,且该次度量 ≤ 512*。不声称近似比——成本函数既非单调也非次模(共享 `SOURCE:` 头可使加入一个 span 更便宜;bridge 复纳可使其跳变)。**与最优的差距在 Phase 3 测量,不用断言掩盖。**
- 每 batch ≤ 64 次精确 `view.cost`(schema 上限所限),与候选数无关。**拒绝**增量式加性 token 缓存:那会成为 renderer 之外的第二个"成本"定义。
- 把 `parse_selection` 移出 CPU selector 的准入循环。此后**每个候选 selector 只输出排序**,共享同一装箱器,使 S0/S1/S2/S3 之间恰好只差一件事。

### 5.3 离线 shootout harness

`shapeflow replay-selectors`:驱动**生产 strategy 对象**重放已存 checkpoint,不重跑 agent。C 边界无需任何重建。新增 op class `PAGE_P1_SELECTOR_BATCH`,否则 whole-batch selector 的工作量会与 Freeze-1 的 per-page `PAGE_P1_SELECTOR_LOCAL` 混计。以 `fork_key` 内容寻址、一次写入,断点续跑即"文件已存在"。

### 5.4 候选 selector

| ID | 设计 |
|---|---|
| `S0-LEX-PACK` | whole-batch BM25 排序 + 精确装箱 |
| `S1-HYBRID-PACK` | lexical + TF-IDF 余弦 + 来源多样性/MMR + 精确装箱(用已声明的 scikit-learn;`torch` 不在图依赖内) |
| `S2-LLM-RANK-PACK` | 一次 whole-batch LLM 请求,只输出排序;CPU 精确装箱 |
| `S3-ANCHOR-FILL` | LLM 输出 must-keep anchor 与排序;CPU 填满余量(填充器必须位于 **selector** 内,`preflight._provenance_errors:466` 禁止发布不在 selection 内的 span) |
| `CTRL-SHORT-PROSE` | whole-batch 有界短 prose,同 512 预算 |

### 5.5 GQ0(机械合法性,硬门,离线)

预算/ID/lineage/原子性违反 **= 0**;strict-valid 非空发布率 task-cluster 95% LCB **≥ 95%**;长 batch 层 LCB ≥ 90%;telemetry 完整性 ≥ 99%;partial publish = 0;P0 replay parity、crash/resume、engine epoch binding 通过。**不过 GQ0 者直接淘汰,不看质量。**

### 5.6 GQ1(局部质量 shootout,无 judge)

同一 checkpoint 上配对比较 P0、S0、S1、S2、S3、CTRL-SHORT-PROSE。全部经 span → occurrence → docid 计算,三值制(绝不二值):

1. **evidence 文档保留率**——分母是**本 batch 所offer 的**,不是 benchmark 拥有的。这把 reducer 与 retriever 分开,正是 0.1488 那次失败所要求的。分母为空 ⇒ `NO_EVIDENCE_OFFERED`。
2. **来源覆盖**——发布/offer 的相异 occurrence 数。**完全不需要答案键**,可在线报告。它回答 512 token 的 whole-batch 装箱是否坍缩到单页,这是 7.4× 压缩的首要风险。
3. **hard negative 干扰**,以及 **negative displacement**:同一 batch 内发布了 negative 而丢弃了本 batch offer 的 evidence 文档的次数。batch 内配对事件,无需跨臂归一。
4. **答案串保留**——与 P0 在同 batch 上配对(McNemar)。对"7.4× 压缩是否毁掉答案"最直接的无 judge 回答。
5. **矛盾保留**——已在 `SelectionOutcome` 上。
6. **引用正确性**在 H 处按构造为 1.0,是回归检查而非端点。

**防火墙切分即设计**:`campaign/replay.py` 为处理侧,只产出以 span/occurrence id 为键的结构化记录;`bench/bcplus/selection_quality.py` 为评测侧,`EVALUATOR_ONLY = True` 且必须登记进 `FREEZE_1.evaluator_modules`。连接键(occurrence_id)对处理侧可见,标签不可见——这正是"用这些指标挑 selector"合法的理由。

### 5.7 冠军规则(先于数据写死)

1. 淘汰全部 GQ0 或硬质量失败者;
2. 在质量最优者的**预冻结 2 个百分点** indifference band 内;
3. 取 **all-offered** 工作量最省者(含失败、重试、回退与 selector 花费);
4. 仍并列取实现最简、CPU+LLM 成本最低者;
5. **只冻结一个 `S*`**,不得有第二个进入 Phase 2。

至多允许**一次**修复,且只在 development split 上;修复后 mint 新版本并在 confirm split 复测。不得在同一 confirm split 上反复挑选。

**若 `CTRL-SHORT-PROSE` 在质量与工作量上同时占优,则 kill pointer-specific 主张**,结论改为"短形式表示,而非 span-ID 指针"。这是一个**真实可能**的结局:按 Freeze-1 数字它已经领先。

---

## 6. Phase 2:单次 P1* 对最终报告的因果影响

### 6.1 2C 先行(frozen continuation,无需图重放)

逐字恢复 `campaign/fork.py`(`git show 3ed684e^`)。它是纯规划与校验,当初被删的理由(没有 backend)正是恢复它的理由:`run_production_forks` 在 `backend is None` 时 fail-closed,它就是那个在 backend 存在前拒绝花钱的东西。

新增 `FrozenContinuationBackend`:同一 `CCheckpoint` 产出 P0 note 与 P1* note,经 `ContinuationEnvelope.substitute` 放入同一个冻结 supervisor 槽位,**只重跑 final writer**。`slot_for` 返回 `None` 的边界在**任何臂被 offer 之前**排除,ITT 分母不受影响。

**零上游三重独立强制**:结构(search seam 抛 `ZeroUpstreamViolation`)、计数(ledger op-class 直方图:恰好一次 compressor + 一次 writer,`PAGE_* == 0`)、信封(`anchor_engine_epoch` / `anchor_today_str` 逐臂复核——两字段已存在但**当前无人检查**)。

**先决缺陷**:重载后的 envelope 只存 note 摘要而非文本(`continuation.py:279`),`substitute` 无法跨进程运行。修法是 `runner._store_continuation` 另存 note 文本向量并按 `content_sha256` 校验;digest 仍只承诺摘要,完整性性质不变。

**工作量必须双基准报告**——`POST_BOUNDARY_INCREMENTAL` 与 `TOTAL_WITH_SHARED_UPSTREAM`,面向决策的**比例一律用后者**。仅用前者正是其自身 docstring 警告的"同一 fixture 报 90% 还是 56%"的伪造。

### 6.2 2H(单边界总效应)

确定性重放至边界(图无持久 checkpointer),retrieval 由记录的 `retrieval_trace` 应答,miss 即致命。约 30 行的边界闸门 wrapper:仅在计划 digest 处委派给 `PageSelectionStrategy`,其余一律 vendor。

此处**不是**零上游,而是**同一上游**:要求 `first_trajectory_divergence(...) is None`。发散的配对**作废并如实报告**,绝不修补、绝不替换。

### 6.3 端点

ΔQ = Q(P1*) − Q(P0)。通过需同时:质量单侧 LCB ≥ −ε_Q;task 级 hard-incident 超额 UCB ≤ 3 个百分点;all-offered 工作量节省 LCB ≥ 15%。**平均质量绝不抵消 hard incident。**

---

## 7. Phase 3:机会有多大(零 GPU)

四个互不相同的上界,绝不合并为一个数:`ρ_mech`(机械合法,**含窗口溢出排除**)、`ρ_oracle`(事后 clairvoyant,不可部署)、`ρ_xfit`(仅用决策前可观测特征的 task-grouped cross-fit;不得用 gold,不得用 queue/KV)、`p*_E2E`(取自 Phase 4 剂量曲线)。

主 routeability 按 task 加权 `ρ = (1/N) Σ_i (1/m_i) Σ_j R_ij`;event 加权另报为系统负载指标。

**经济上界** `S_oracle = Σ R_ij·max(ΔW_ij,0) / Σ W_0,ij`。其 95% UCB < 15% 即 kill——clairvoyant 都没有空间。同处测量**装箱器自身的损失**:贪心相对于 offered 视图上穷举/ILP 装箱的差距,与 selector 的**排序**损失分开报告。

---

## 8. Phase 4:mixed 与 all-P1 饱和

**8.1 固定 C-note 组合**(把累积效应与轨迹分化分开):配对每个可归因 C checkpoint,固定 root messages / brief / note 顺序,只变 q_C ∈ {0,.25,.5,.75,1},每个中间剂量至少两套互补 mask。报告五个 cell 均值与联立 CI,**以及** κ_C = μ(1) − 2μ(.5) + μ(0),不拟合直线。

**8.2 自然 E2E 剂量**,先 H-only,p_H ∈ {0,.5,1}。需要 `design/point_assignment.py`(`configs/prereg.yaml:213` 已预注册为 `phased_low_discrepancy`、`structural_coordinate` 索引,尚未实现)。以**结构坐标**索引,绝不用运行计数器——重试或续跑会使标签依赖于执行历史。赋值表在开跑前物化并 hash,新增 canary `decision_point_assignment_matches_table`;嵌套赋值使 25% 的 P1 集是 50% 的子集。主分析为**assigned-policy ITT**,实际暴露只作 mediator 描述。

---

## 9. 决策门与 kill/pivot 总表

| 结果 | 裁决 |
|---|---|
| 无 selector 通过 GQ0 | **kill** 当前 P1* constructor,不做质量实验 |
| CPU 或 short prose 在质量与工作量上同时占优 | **kill** LLM/pointer-specific 主张,保留短形式 |
| P1* 局部质量全域不安全 | **kill** 该 seam |
| `S_oracle` UCB < 15% | **kill** routable-P1 |
| 单次 P1* 即导致报告 NI 失败 | **kill** 该边界 |
| mixed 安全、all-P1 有害 | 保留 **bounded exposure**,kill all-P1 |
| all-P1 安全且始终最省 | **静态 all-P1**,form broker 无必要 |
| P0 始终最好 | **kill P1** |
| H、C 各自安全但 H+C 有害 | 只保留单 seam |
| oracle 空间大但 `ρ_xfit` 弱 | 异质性存在但不可预测,kill 动态路由主张 |
| 存在非平凡安全区、`ρ_xfit` 可靠、聚合节省 ≥ 15% | **进入独立的 broker 协议** |

门 PASS 自动推进;门 FAIL **立即停止**,产出 measured-vs-threshold、诊断与预先写好的计价选项,等待人工。无自动迭代,无自动 pivot。

### Broker Entry Gate

须同时满足:(1) 唯一冻结的 `S*` 在真实 whole-batch treatment 上机械合法;(2) 报告级质量 NI、hard incident、all-offered 工作量全部通过;(3) `ρ_xfit` 非零且 unsafe UCB ≤ 5%;(4) 剂量曲线证明并非 P0 或 P1 的静态全域支配;(5) P0/P1 的 prefill/decode/KV 形状确实不同;(6) 若要保留通用 agentic 主张,C 或第二个真实 seam 需重复通过。

通过后**只**获得:*"存在质量合格、可预测适用范围非零、资源形状不同的替代 form。"* **仍未证明 broker。**

---

## 10. 规模、顺序与成本

| 阶段 | task | GPU-h |
|---|---:|---:|
| 0 普查 | 0 | **0** |
| 1a CPU shootout(1,808 H + 715 C) | 0 | **0** |
| 1b LLM shootout | 0 | ~9 |
| 1c P0 参照 | 0 | ~8 |
| 2C frozen continuation | 70 | ~6 |
| 2H 单边界 H → 报告 | 80 | ~12 |
| 3 routeability + oracle 上界 | 0 | **0** |
| 4 饱和 | 150 | ~12.5 |
| 5 entry gate | 20 | ~2.8 |
| **合计** | **320 / 350** | **~50 GPU-h(双 lane 约 25 h wall)** |

**关键路径**:普查 → whole_batch + 装箱器 → 1a → 1b → **2C 先于 2H** → 3 → 4 → 5。

标定量**实测而非建模**(对 `reports/BCPLUS_campaign1.json` 的两参数 prefill/decode 拟合把 `H_PROSE_CONTROL` 预测偏差 2 倍,因为 `interval_union_seconds` 是重叠区间的并集而各臂并发结构不同):P0 265.4 s/cell,H_MARKDOWN_ID 280.0,C_ID 230.3,H_PROSE_CONTROL 112.0,H_CPU_CONTROL 87.1。**每个 LLM 阶段先跑 50 单位试点再占用 lane。**

Phase 0/1/3 **不消耗任何新 task**——它们重放已花掉的 183 个 task_id。2C(70)、2H(80)、4(150)、5(20)互不相交,取自 b1_confirm∪b2(180)与 FV(150),余 30 预留。

---

## 11. 统计与纪律

独立 n 是 **task 数**;多边界与多 seed 只降噪,不增加 n。阶段间 hierarchical gatekeeping,阶段内 Holm 或 task 级 max-T bootstrap。strata、位置与额外交互标 exploratory。gold/qrels 只进评测侧;selector 不得见 P0 输出或 truth packet。失败、超时、重试、回退与 final writer 全部计入 all-offered work。telemetry 缺失 ⇒ `NOT_ESTIMABLE`,不得 complete-case 删除。**Freeze-1 的 80-task 结果只用于设计与功效,绝不与 P1* 确认合并。**

---

## 12. 修订规则

本文冻结。此后改动以 amendment 追加于文末:编号、日期、动因、改动内容、影响的实验/门;正文不改写。任何 amendment 若发生在相应数据已被查看之后,受影响的结论自动降级为 exploratory。

---

## Amendment 1 — selector 拆成两个 CPU packer 夹一次 LLM 排序(2026-08-02)

**动因。** Phase 0 普查(`reports/gates/FREEZE2_CENSUS.json`)测出两道**互相独立**的 selector 障碍,当前的 direct full-view LLM selector 在**两端同时**不成立:

| 障碍 | 发生阶段 | 证据 |
|---|---|---|
| **输入上限** | LLM 调用之前,whole-batch prompt 装不进 context | 61.9% 的 batch 超过 32,000;prompt 中位数 37,094 |
| **输出上限** | LLM 选完 ID 之后,renderer 展开超过 512 | Freeze-1 rejected output 中位约 4.1×512 |

在现定义下,full-view LLM selector 的 event-weighted 机械可达率上界只有 `1 − 1011/1632 = 38.1%`,且尚未扣除 schema error、timeout 与 512-output failure。

**裁定:这是 "direct full-view LLM 方案失败",不是 P1 失败。** 本轮不因此 kill P1。

**改动内容。** H 边界的 P1* 结构改为:

```text
whole-batch evidence
→ CPU PromptPacker        (解决 32k 输入上限)
→ ≤ context limit 的 source-balanced view
→ 一次 LLM,只做语义排序
→ CPU EvidencePacker      (解决 512 输出上限)
→ ≤512-token P1 payload
→ preflight
```

LLM 不再负责猜任何 token budget;两个 budget 各由一个确定性 CPU packer 保证。这仍然是**每个 gather batch 恰好一次 selector 请求**,§5.1 的 whole_batch 单位不变。

**PromptPacker 合同(确定性、非生成式)。**
- 用精确模型 tokenizer 计数;
- 先扣除 system、instructions、schema、最大 completion 与安全余量;
- **每个非空 page 至少保留一个 evidence-bearing span**;
- 保留 title、heading、table header 等必要上下文;
- 剩余空间按 query relevance、source diversity、contradiction potential 填充;
- 记录全部 dropped span、page coverage 与 token accounting;
- 不得使用 gold、P0 summary 或任何 evaluator label。

**合同澄清(§5.1 与 `HC_MECHANISM_v1.md:20` 的 "covering every page")。** 判定为:**每个 page 至少有真实的 evidence representation**,而非必须包含每个 raw candidate span。理由:后一种读法在 32k 模型上使 one-call whole-batch LLM selector 结构性失败,只剩换长上下文模型或改 P1 合同两条路;而 §5.1 的"整批一次请求、整批原子发布"在前一种读法下完全成立。超过 32k 就静默截尾**明确禁止**——它系统性丢掉 batch 后部页面,不能称为 covering whole batch。

**§5.4 的候选 selector 表作废,改为四个匹配臂加一个 diagnostic:**

| Arm | 输入 view | 排序者 | 最终 packer | 回答什么 |
|---|---|---|---|---|
| `CPU-FULL` | 全部 candidates | CPU | exact ≤512 | CPU 产品候选与上界 |
| `CPU-PROMPTVIEW` | PromptPacker 后 | CPU | exact ≤512 | 单独测 prompt pruning 的损失 |
| `LLM-PROMPTVIEW` | 同一 PromptView | LLM | exact ≤512 | 同等可见证据下 LLM 是否更聪明 |
| `SHORTPROSE-PROMPTVIEW` | 同一 PromptView | LLM prose | completion ≤512 | pointer 是否优于普通短输出 |
| `LLM-FULL` | 全部 candidates | LLM | exact ≤512 | **diagnostic only**:prompt ≤32k 才运行,否则记 `PROMPT_INFEASIBLE`;留在 all-offered 分母;不得晋级 |

两个关键对比必须分开报告:

- `CPU-FULL − CPU-PROMPTVIEW` = **为了塞进 LLM 而删证据的代价**;
- `LLM-PROMPTVIEW − CPU-PROMPTVIEW` = **同等可见证据下 LLM 的增量价值**。

缺了前者,LLM 落败时无法分辨是排序差还是它只看到被截断的候选。

**`LLM-PROMPTVIEW` 的三道预注册门。**
1. **输入合法性**:≥95% 的 whole batch 能构造 context-compliant PromptView;长 batch 层 ≥90%;每个 page 有真实 evidence representation。
2. **输出合法性**:budget / ID / lineage / atomicity 零违规;最终 payload ≤512。
3. **增量价值**:同一 PromptView 下相对 `CPU-PROMPTVIEW` 有可测的质量提升,且覆盖其自身额外的 prefill/decode 成本。

**kill 规则。**
- PromptPacker 无法在多数 batch 上同时满足 context limit 与每页 evidence coverage ⇒ kill one-call LLM selector;
- 装得下但 LLM 不胜 CPU ⇒ 最终 selector 采用 CPU;
- LLM 质量更好但全口径成本不划算 ⇒ 仍采用 CPU;
- 仅部分 batch LLM 胜 ⇒ 冻结 hybrid policy,但归入后续 quality qualifier,**不得事后挑样本**。

**明确不采用的"修复"。** 换 64k/128k 模型(可作独立 variant,但改变模型、成本与 resource shape,不是免费修复);退回 per-page LLM(违反 whole-batch P1 定义,是另一种 P1);多次 hierarchical LLM call(违反"一次 selector 请求",且可能吃掉 P1 的 work saving,只能作独立 treatment);先用另一个 LLM 总结再选择(重新引入生成式 compressor,可能退化回 P0);超过 32k 静默截尾。

**影响的实验/门。** §5.4、§5.5、§5.7 的候选集与冠军规则按本 amendment 执行;§9 增补一行:若 `CPU-FULL` 最终胜出,P1 与 dynamic shape 仍然成立,死掉的只是"LLM selector 必不可少"这一条子主张。

**降级声明。** 本 amendment 在 Phase 0 普查数据已被查看之后作出。普查是 P0-only 的描述性测量、不含任何臂间比较,但按 §12 的规则,由它触发的结构改动使**受影响结论的确认性下降**:`LLM-PROMPTVIEW` 相关的门在本轮按 confirmatory 执行,而"为什么淘汰 direct full-view LLM"这一判断本身是 data-dependent 的,记为 exploratory。
