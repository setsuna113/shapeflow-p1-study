# ShapeFlow 实验计划(执行冻结版 Freeze-1 · 2026-07-27)

本文档自包含、独立成立,是唯一的执行依据。冻结后任何改动必须以 amendment 形式追加(见 §12),不得改写正文。

---

## 1. 背景与术语

**系统**:Open Deep Research(ODR)风格的 deep research agent,部署在 vLLM 上。Supervisor 派出多个 researcher,researcher 以 ReAct 循环搜索网页;证据压缩发生在两个边界:

- **H 边界(网页侧)**:同一 assistant turn 内的多个页面抓取以 asyncio.gather 汇合成一个 **gather batch**,对 batch 做证据压缩;
- **C 边界(close 侧)**:researcher 结束时(ResearchComplete / max tool calls / 无 tool call)将全历史压缩为提交给 supervisor 的 note。

**两种压缩 form**:

- **P0(prose)**:H 边界每页一个 summarize 请求生成自由文本摘要;C 边界 compressor prefill 全历史后自由 decode 长 prose。decode 逐 token 串行,是主要 GPU 瓶颈。
- **P1(span-ID selection)**:网页按 content_hash 快照并切成带原文位置/来源的小片段;模型只输出片段 ID,CPU 校验 ID、合并片段、稳定排序并渲染成 evidence payload。**H 边界为 whole-batch selector**:整个 gather batch 一次 selector 请求,结果整批原子发布(batch 内禁止 P0/P1 混排;失败整批回退 P0)。**C 边界为 close 后独立 selector 请求**(方案 A 形态)。P1 缩短的是自由 decode;selector 仍需 prefill 候选证据,语义判断(相关性、矛盾、gap)仍是 NLP 问题。

**Broker**:位于 vLLM 原生调度器入场处的仲裁层。每项待发工作以 **offer**(替代执行计划集)形式到达:P0-plan = N 个 summary 请求描述;P1-plan = 1 个 batch-selector 请求 + CPU render。broker 在每个 scheduler tick 联合决定:每项工作用哪个 form、本轮准入哪些工作。质量是**硬资格门**(P1 不合格只保留 P0),不是可交易权重;求解预算 2ms,超时/估值失效一律 fail-closed 降级(P0-only 或 solo-greedy)。

**决策单位**:HCheckpoint = 一个 gather batch;CCheckpoint = 一次 researcher close。page 级指标只作为 HCheckpoint 内部子指标。

**contract risk(合同风险)向量**:对一次决策,P1 相对 P0 在以下维度的风险——证据保留(答案要点是否保住)、干扰精度(是否引入误导性内容)、矛盾/来源/预算机械合规。

**论文定位**:systems 论文。贡献是 broker(form+admission 联合在线决策及其 vLLM 集成),不声称 span-ID selection 本身有新颖性。

## 2. 论文声明与叙事链

- **C1(异质性与可预测性)**:在 H/C 边界上,P1 的 contract risk 在工作项之间显著异质,可由 broker 决策时刻可观测的特征预测,且 contract risk 对最终任务质量风险具有经校准的预测力。
- **C2a(headroom 可实现)**:risk-controlled predictor 门控的 P0/P1 混合,在预注册 ε/δ 界内的工作节省显著超过任何静态策略,并捕获 local-clairvoyant 参照的大部分。
- **C2b(联合性)**:form 与 admission 的联合决策优于最强的顺序解耦方案(先定 form 再准入;先准入再定 form)。
- **C3(系统收益)**:native vLLM serving(continuous batching + APC)下,broker 在 ITT 质量 ε 界内提升 SLO 下最大可持续到达率,并改善能耗与尾延迟。

叙事链:**安全异质性存在 → 可预测 → 门控优于静态 → joint 优于 sequential → serving 中兑现**。前一环不成立,后一环不启动。

## 3. 数据、工作负载与切分

### 3.1 BrowseComp-Plus(主底座)

830 个多约束短答案 query + 约 10 万网页固定语料,带人工核验的 evidence/gold 文档与 hard negatives。

**工作负载命名与偏离声明**:我们的设定统一称 **"BC+-corpus full-document ShapeFlow workload"**——检索命中后返回完整文档(平均约 5k words),供 H 边界压缩。这偏离官方 leaderboard 的 snippet 型 search tool 设定,论文中如实声明,不与官方榜单数字并排比较。

**冻结项**:retriever 与索引 commit、top-k、切片规则、full-document fetch 规则、token-aware 共享截断、P0/P1 完全相同的 frozen input view 与 overflow policy。

**切分**(task-ID manifest + sha256 已冻结,seed=20260727):

| 层 | 规模 | 用途 |
|---|---|---|
| BCP-internal-dev / iteration-development(ITD) | 380 | B1-select 80 · B1-confirm 170 · B2 110(三者任务两两不交) |
| BCP-internal-dev / final-validation(FV) | 150 | FV-A 75:B3 首次门评估;FV-B 75:唯一一次迭代后的复测保留,此前零接触 |
| BCP-sealed-confirmatory | 300 | 仅 S1,此前零接触,任何调参不得染指 |

**泄漏防火墙**:qrels/gold/negative 标注只进 evaluator 进程;agent、selector、broker 特征、predictor 训练特征侧物理隔离,CI 加静态检查。

### 3.2 DeepResearchGym(C 边界补充,dev 消耗池)

仅使用 **FineWeb 索引**(注册即用;ClueWeb22 不在本计划内,不注册、不签其许可)。约 600 题用于 C 边界 profiling;所有 API 响应全量缓存并记版本 hash。**退路**:若服务器到 DRGym API 不可达,C 边界 profiling 退回 BC+(答案可推出探针),report 构念由 G-A 兜底;不自建语料。

### 3.3 DeepResearch Bench(仅 guardrail)

100 题(50 中 / 50 英)+ RACE/FACT 评测,只在 G-A 一次性使用;evaluator 以仓库 commit pin(判官模型 ID、prompt 以 commit 内配置为准,全部入 manifest)。

### 3.4 通用冻结

所有臂同一模型 checkpoint(agent 各角色同底座)、同 prompt 版本、同索引;seed 固定;质量探针为固定小模型、版本 pin;检索一律走冻结后端,live web 只出现在 G-A。

## 4. 预注册(prereg.yaml,Phase 0 出口一次冻结)

在任何 treatment 输出产生之前,以下条目全部数值化、写入 `prereg.yaml` 并 hash 入库。允许使用的唯一数据是 P0-only 试点(§5.3)。

1. **Endpoints**:primary quality = BC+-workload E2E accuracy(ITT);co-secondary = evidence recall、W 各分量、DRGym report 指标。primary work = W。
2. **W(token-work)**:`W = α·uncached_prefill + β·cached_prefill + γ·decode`;α/β/γ 由本硬件 P0-only/synthetic microbench 实测各 token 类资源单价标定,冻结后不得因 P1 数据调整;各分量永远全量上报,防指标挑选。
3. **Work 记账范围**:selector/summary、fallback、受轨迹影响的全部后续 agent calls、final answer call、失败请求已花 tokens 全部计入;predictor/renderer 的 CPU 开销单列;shadow/双跑消耗计入实验 ledger、标记 measurement overhead、不计入被评估策略的理论 work。
4. **Qualification 规则**:`qualified = evidence_loss ≤ τ_e ∧ interference_risk ≤ τ_i ∧ 矛盾/来源/预算机械检查全过`,τ 值冻结。
5. **δ**(任务级质量事故率上限)与 **δ_gate**(P(unsafe | broker 选 P1) 的单侧置信上界阈值)。
6. **ε 设定程序**:由功效试点(§5.3)估配对不一致率、serving 随机性、探针噪声,计算 1pt 非劣所需任务数;300 题可支撑则主 ε=1pt(敏感性 {0,2}),不可支撑则主 ε=2pt(敏感性 {1}),并在论文声明精度极限。treatment 前定死,此后不动。
7. **最小意义节省**:门控相对静态的 W 节省阈值(默认 15%,冻结时可调),G1/G3 引用。
8. **S1 数值**:latency SLO、deadline、completion fraction 下限、backlog slope 容忍、warm-up/measurement/drain 窗长、instability 判定规则、λ 网格与"先括后聚"程序。
9. **统计口径**:决策级配对差 + task 聚类稳健标准误;任务级配对 bootstrap;S1 为 TraceBlock 级配对 bootstrap;质量一律"均值差 + 事故率"双报;非劣检验单侧。
10. **迭代规则**:每个门允许至多一次修复迭代;迭代后的复测只能用 FV-B(或对 S1 而言不存在复测——sealed 只跑一次);否则结果只能标 exploratory。

## 5. Phase 0:契约、回归门与试点(系统轨起点)

### 5.1 契约(先于一切)

**(a) H/C 机制契约**:pin 当前实现——H 为 whole-batch selector + 整批原子发布(禁止 batch 内 `[P1(A), P0(B)]` 混排;失败整批回退 P0);C 为 close 后独立 selector(方案 A)。契约带版本号,任何变更走 amendment。

**(b) Offer 接口契约(六步语义)**:

1. upstream 提交 alternative-plan descriptors(P0-plan / P1-plan),**不启动任何 form**;
2. scheduler tick 对 queue/KV/APC 状态取 **versioned snapshot**;
3. broker 原子选择本轮 admitted jobs 及各自 form;
4. 只 materialize 被选 form;
5. 选 P0 时,N 个子请求进入 native vLLM 队列,由原生 continuous batching **独立调度——禁止 gang scheduling**;
6. snapshot version 失效 → 整项重试,绝不部分 materialize;重试超限 → fail-closed 降级(P0-only)。

原子性保护的是 **form 选择 + job admission + materialization**,不是子请求共同执行。B5 模拟、S2 测试与真实 extension 引用同一份契约文本。

**(c) 可观测性声明**:broker 决策时刻可观测的特征与引擎遥测(队列、KV、APC 状态)成文列举;凡不可观测的量,禁止进入成本模型与 predictor 特征。B1-confirm 的特征清单回填本契约。

### 5.2 回归门(全部通过才开科学轨)

1. **Native parity**:actual-graph 下 P0 与 pinned ODR 行为一致;禁止强制串行改变图结构。
2. **测量纪律**:工作量只允许需求向量与 trace 层 interval-union busy time;**Σ request-latency 冒充 GPU work 禁用**,CI 静态检查。
3. **上下文与容量**:P0/P1 相同 frozen input view、token-aware 共享截断与 overflow policy;长页 p99 容量 gate,不得静默丢长页。
4. **发布 gate**:每边界 P1 publication / fallback / strict-valid 率阈值;canary 中 P1 必须真实发布非零 span。
5. **Schema/config 单一 registry** 生成 + equality 测试,防枚举漂移。
6. **四引擎 soak** + 故障注入 + crash/restart-resume;lane/epoch 正确性证明。
7. **缓存卫生**:APC 臂内复用允许、跨臂禁止(block salt / engine epoch);R0(URL/快照身份)、R1(exact 复用)语义测试。
8. **ITT ledger**:所有 work item、失败、fallback、timeout 全留痕;任何分析不得静默剔除。
9. **泄漏防火墙**(§3.1)生效验证。
10. **Canary 矩阵**:{短页, 长页, 空结果, 重复 URL, tool 异常, timeout} × {H batch, C close} × {P0, P1, CPU-selector, SHORT_PROSE} × native 并发+APC × 4 GPU × restart,全组合跑通且遥测齐全。

### 5.3 试点(P0-only,其数据允许用于 §4 冻结)

- **Competence gate**:ITD 内 100 题试点,P0 accuracy 与 evidence recall 高于预注册下限;不过则先换模型/检索配置再回到冻结,否则 ε 非劣无意义、predictor 无从学起。
- **功效试点**:约 150 题 × ≥3 seed 重复 P0-only,估配对不一致率、serving 随机性、探针噪声,产出 §4.6 的 ε 决定与 S1 block 数核算。
- **W microbench**:标定 α/β/γ。
- **四 lane 同构性**:四条单引擎 lane 在相同 TraceBlock 上统计不可区分;否则 lane 作为 blocking 因子进模型。

## 6. 科学轨

### B1-select(P1 变体筛选,结果隔离)

- **数据**:ITD 子集 A(80 题)跑 P0 收割 H/C checkpoints。
- **设计**:相同 checkpoint 上配对比较 P1 变体(chunker 粒度 × 输出自由度 × 聚合方式)与两个归因 control——CPU 词法选择器(LLM selector 是否必要;若 CPU 在某分层安全,可升级为 broker 第三 form)、SHORT_PROSE(收益是否只是"少写字")。
- **产出**:champion form。**本段结果只用于选择,不得作为效应量引用**(winner's curse 隔离)。

### B1-confirm(C1 主证据 + predictor 训练集)

- **数据**:ITD 子集 B(170 题,与 A 不交)+ DRGym FineWeb 约 600 题(C 边界)。
- **单位**:HCheckpoint / CCheckpoint;P0 对照物是该 checkpoint 的完整替代计划(N 个 summary vs 1 个 selector)。
- **标签(contract-risk 向量)**:
  - 证据保留:对 atomize 后的答案要点(机器 atomize + 人工审计,审计封顶 10% 抽样),固定小模型探针仅凭渲染 payload 判断要点是否保住(P0 摘要同探针配对);
  - 干扰精度:hard-negative 页是否引入误导性断言(探针判定;负例身份 ≠ 自动有害,须实测);
  - 矛盾/来源/预算合规:机械检查。
- **特征记录**(全部须在 5.1(c) 可观测集合内):页/batch token 长、chunk 数与长度分布、检索分与名次、query-页面词面重叠、表格/列表密度、语言、轨迹位置、researcher 轮次、selector prefill 大小、facet 估计。
- **续跑校准(local→E2E link)**:约 60 个分叉点,来自约 60 个**不同任务**;同 checkpoint 分叉 P0/P1 后,**双分支下游一律强制 P0**,隔离单次决策的 downstream 效应;跑到任务终点取 E2E 质量差;task 聚类推断。检验 contract 标签对 E2E 风险的预测力,并提供 fixed-checkpoint replay 参照(供 B2 的反馈问题)。
- **交付**:risk 分布与方差分解(batch 内/任务内/任务间)、特征偏依赖、(特征, contract-risk, Δcost) 数据集、观测接口需求清单。
- **规模**:约 8k 次 checkpoint 级配对(每臂一次 prefill + 短 decode,便宜)+ 60 对续跑。

### B2(二维策略剂量-响应,ITT)

- **策略**:π(p_H, p_C)——每个**合格**决策点(机械合同可满足)以概率 p 分配 P1;p 是 assignment probability,不是实际占比。
- **网格**:主网格 **(p_H, p_C) ∈ {0,50,100}²**,9 cell × ITD 的 110 题(同批任务跑每个策略);可识别 H 主效应、C 主效应、H×C 交互。仅当 H 三点曲线呈非单调/阈值形时,追加 {25,75}×p_C=0 两 cell,标 exploratory。
- **分析**:assigned policy 的 ITT 为主;realized exposure(实际 P1 决策数/占比)另报,不当作固定可比样本。曲线称 **policy response**;非线性不得直接解释为"反馈放大"(替代解释:难度混合、accuracy 阈值性、决策数差异),反馈问题由续跑 replay 对照回答。
- **副产品**:on-policy checkpoints(供 B3 校准)与 offer 流记录(供 B5 作分布素材;**禁止跨策略时间戳重放**——offer 的出现时间与内容是 policy-dependent 的)。

### B3(risk-controlled predictor 与 headroom)

**三段数据流(防 P0 分布 → broker 分布偏移)**:

1. **训练**:B1 数据集,task-grouped 切分;
2. **校准**:从 B2 的 on-policy checkpoints 抽样 **shadow-run 另一个 form(不发布反事实)**取 contract 标签,**封顶 800**;GPU 消耗入 ledger、标 measurement overhead;operating point 在含 on-policy 标签的校准集上选定,使 **δ_gate(P(unsafe|选 P1) 单侧 UCB)达标**;
3. **评估**:FV-A(75 题)E2E,策略五臂——always-P0 / always-P1 / 最优静态(取 B2)/ **predictor 门控** / **local-clairvoyant 门控**。

**local-clairvoyant 定义与限制**:决策点双跑取事后 contract 标签的门控,**离线分析性质**;它是"局部资格的事后知识"参照,不是 sequential E2E 上限,不进入任何 timed serving。

**报告**:AUC、分层校准(页长/语言/域)、shift 失效率、运营点曲线(false-qualify UCB ↔ 放弃的节省)、均值与事故率双报。predictor 推理必须装进 broker 2ms 预算(线性/小 GBM,特征 ingest 时 CPU 预计算)。

### B4(资源需求向量与成本标定;与 B2 并行)

- **预测对象(逐 plan)**:uncached / cached prefill tokens、decode tokens、KV block-seconds、输出长度分布、cache-hit 概率、conditional service 分布。
- **时间维**:只在 trace/block 层以 interval-union busy time、makespan、joules 回归校准需求→时间映射;可加 request latency 禁用。
- **敏感性**:向估值注入实测误差,检验 greedy+2-swap 装载结果的稳定性。

### B5(fixed-offer jointness microbenchmark;CPU-only)

- **输入**:合成 + 由 B2 记录的 offer **分布**重构的 fixed offer streams(明确声明:不是任何真实策略轨迹的反事实),扫负载与异质性;成本用 B4 标定值;调度语义严格遵守 §5.1(b)。
- **对比**:joint greedy+2-swap vs form-first→cost-aware admission vs admission-first→form vs 静态。
- **声明边界**:B5 只能证明"给定 offer 流时 joint packing 有无 headroom",是**必要条件筛除门**;C2b 的确证证据在 S1(⑥ vs ④/⑤)。

## 7. 系统轨

### S1(sealed confirmatory serving;headline)

- **拓扑**:四条独立同构的单引擎 lane(每 lane 一张 GPU 一个 vLLM engine),并行承载不同 TraceBlock。estimand = **单 vLLM engine 上的 broker serving**;无 cross-GPU routing / migration / global admission。
- **任务**:BCP-sealed-confirmatory(300,首次接触);arrival trace 覆盖泊松与 burst 两种形态。
- **臂(7)**:① P0+FCFS ② P1+FCFS ③ 最优静态混合+FCFS ④ predictor 定 form → cost-aware admission(顺序 A)⑤ admission 先选 → 入选内定 form(顺序 B)⑥ ShapeFlow joint ⑦ P0+broker admission(admission-only 归因)。**无 clairvoyant 臂**(在线取真标签需双跑,扰动队列/APC/时序,测量摧毁被测对象)。
- **Block 设计**:每关键 cell **6–8 个短 block(每 block 30–50 arrivals)**;arm 顺序随机化;每 block 独立 engine epoch / cache namespace;warm-up / measurement / drain 三窗分离;censoring 依 prereg。
- **λ 程序("先括后聚")**:每臂 3 个 λ 点短 block 探测括住 λ*(SLO 下最大可持续到达率),再在括点集中投重复 block;全臂完整重复只在主 λ regime + 1 个饱和点;λ 细扫仅限 headline 对(① vs ⑥)。
- **指标**:λ*(primary)、joules/任务、p50/p95、backlog 增长、completion fraction、drain makespan、KV 压力/preemption、APC hit、solver 开销与降级频率。**质量以全部 arrivals 为分母(ITT)**:timeout / drop / restart / fallback / 未完成 / admission 后取消 / P1 发布失败全部计入。
- **推断**:TraceBlock 级配对 bootstrap。
- **归因**:⑥ vs ④/⑤ = 联合性(C2b);⑥ vs ②/③ = form 收益;⑦ = admission 单独收益。

### S2(系统 microbenchmark)

- **总关键路径**:snapshot → 特征 → predict → solve → **atomic commit** 的端到端延迟分布(2ms 是 solver 预算,总路径另立预算);与 §5.1(b) 六步语义的一致性测试。
- **Solver**:64/128/256 offer 池的达标率、2-swap 改进率、到点降级频率与降级解质量。
- **鲁棒性**:stale snapshot / 估值缺失注入 → fail-closed 行为逐条验证;R0/R1 命中率;APC hit/miss 与 eviction 遥测的描述统计(cache 跨时间外部性明确标注为 future work,不做因果声明)。

## 8. G-A(live paired descriptive guardrail)

DRB 100 题,champion 系统 vs P0-only,**同时间窗交错**运行 live 检索(经自有栈),报告 RACE/FACT 差值与置信区间。**描述性,不称 confirmatory**(live web 不可控性如实声明)。这是全计划唯一的 live web 环节。

## 9. 决策门与 kill/pivot 总表

| 门 | 位置 | 通过判据 | 不过时 |
|---|---|---|---|
| **G0** | Phase 0 出口 | §5 全部契约、回归门、试点完成;§4 prereg 冻结 | 科学轨不开工 |
| **G0'** | 系统轨 | §5.1(b) 六步语义实现 + S2 一致性测试通过 | S1 不开工 |
| **G1** | B1-confirm 后 | (i) qualified 份额结合 Δcost 分布,折算的潜在节省 ≥ 最小意义节省;(ii) 特征关联显著(筛查 AUC ≥ 0.65);(iii) local→E2E link 显著 | 全域不安全 → kill P1;全域安全 → 交 G2 按节省分流;link 不显著 → 标签重设计一次(复测用 FV-B) |
| **G2** | B2 后 | 主 ε 下存在非平凡可行段。曲线全平时分流:P1 的 W 节省显著为正 → always-P1 静态化(pivot admission-only 论文);节省不显著 → **kill P1** | 任意 p>0 即崩 → kill |
| **G3** | B3 后 | ε 界内:(i) clairvoyant−最优静态 绝对节省 CI>0 且 ≥ 最小意义节省;(ii) predictor−最优静态 绝对收益 CI>0;(iii) capture ratio 点估计 ≥60%(CI 一并报,参考项);(iv) 运营点 δ_gate UCB 达标 | 特征迭代一次(复测用 FV-B);仍不过 → pivot 静态配置论文或 kill |
| **G4** | B4 内 | 估值噪声注入下 greedy 装载稳定 | 修成本模型,B5/S1 顺延 |
| **G5** | B5 后 | 必要条件:对 joint **最有利**的 offer 流族上,joint 相对最强顺序方案有稳定优势(在 B4 噪声包络内) | 放弃联合性主张(pivot:risk-controlled form 门控 + 标准准入)或 kill broker 核心 |
| **SLO** | S1 | ITT 质量 ε 界内,λ* 显著提升;C2b 由 ⑥ vs ④/⑤ 确证 | 如实报告,论文主张相应收缩 |

## 10. 依赖与执行顺序

```
科学轨:  [G0] → B1-select → B1-confirm →(G1)→ B2 →(G2)→ B3 →(G3)→ B5 →(G5)──┐
系统轨:  契约实现(§5.1)→ 回归门 → offer/state-machine → S2 →(G0')────────────┤
                                                          全门齐 → S1 →(SLO)→ G-A
```

**立即可启动(不等 prereg 数值化)**:offer/state-machine 接口、schema/ledger/回归 gates、BC+ adapter 与 split manifest、P0 competence 与功效试点、B1 checkpoint harness、B4 microbench 仪表、四 lane 环境。

**B1 treatment 开跑前置**:qualification τ、δ/δ_gate、W 权重、ε 主点、最小意义节省、S1 数值全部数值化;§5.1 契约冻结;切分层级冻结。

## 11. 规模量级与伸缩旋钮

| 实验 | 量级(默认) |
|---|---|
| B1 | 约 8k 次 checkpoint 配对 + 60 对续跑 |
| B2 | 9 cell × 110 题 ≈ 1000 次 E2E(条件扩展 +220) |
| B3 | 5 策略 × 75 题 ≈ 375 次 E2E(FV-B 复测最多 +375)+ shadow ≤800 |
| B4 | 复用 B1/B2 物料 + microbench |
| B5 | CPU-only |
| S1 | 7 臂 ×"先括后聚"≈ 2.5–4k 任务执行 |
| G-A | 200 次 |

**可伸缩(预算紧张时按此序裁,比例不动)**:B2 每 cell 任务数 110→80;B2 条件扩展取消;S1 臂④⑤⑦仅主 λ regime;atomization 审计维持 10% 封顶;G-A 整体可弃。
**不可伸缩**:S1 的多 block 重复、prereg 与切分纪律、ITT ledger、task-disjoint 数据流、qrels 防火墙。

## 12. 修订规则

本文冻结。此后改动以 amendment 追加于文末:编号、日期、动因、改动内容、影响的实验/门;正文不改写。任何 amendment 若发生在相应数据已被查看之后,受影响的结论自动降级为 exploratory。
