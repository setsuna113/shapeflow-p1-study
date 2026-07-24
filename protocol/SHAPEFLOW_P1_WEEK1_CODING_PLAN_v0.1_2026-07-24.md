# ShapeFlow P1 Week-1 Coding Plan v0.1

> 日期：2026-07-24  
> 状态：第一版可执行草案；供 coding agent 实施  
> 目标机器：`sjtu/LiuYancheng` 上租用单张 RTX 4090，可连续使用约 168 小时  
> 建议新仓库：`C:\Users\yl998\Documents\programming\deepresearch\shapeflow-p1-study`  
> 本计划只验证 P1；不修改 ShapeFlow proposal，不实现 broker，不修改 vLLM scheduler，不复用正在运行或已封存的 drbat/drbo run。

---

## 0. Coding agent 的总任务

Coding agent 必须完成以下闭环，而不是只搭一个空 harness：

1. 新建独立 Git 仓库 `shapeflow-p1-study`；
2. pin Open Deep Research、模型、vLLM、Python 依赖和 prompt/schema；
3. 实现真实 Tavily 网页采集与冻结重放；
4. 实现 P0、`WEBPAGE_P1`、`CLOSE_P1` 及规定的 P1 设计变体；
5. 实现两个可复放的 intervention checkpoint；
6. 实现 selector、chunker、aggregator、renderer、preflight 与结构正确性检查；
7. 实现逐请求、逐阶段的 GPU/CPU/work/trajectory 遥测；
8. 实现 TruthPacket、盲评 judge、质量指标和人工审计队列；
9. 实现“筛选 → mini end-to-end → policy freeze → honest holdout”的自动状态机；
10. 实现断点续跑、预算上限、GPU watchdog、API 重试、secret redaction；
11. 本文冻结的 auto-launch protocol/预算/阈值和运行时凭据注入完成后，所有测试和 smoke gate 通过即自动启动 Week-1 实验；
12. 自动生成可审计的最终报告，明确裁决：
    - `WEBPAGE_P1`：KEEP / CONDITIONAL / MECHANISM_ONLY / KILL_STRUCTURAL / KILL_HARM / KILL_NO_HEADROOM / NOT_ESTABLISHED；
    - `C_VISIBLE`：同一套裁决；
    - `C_REGISTRY`：作为独立扩展裁决，禁止冒充 compressor-only；
    - H×C interaction：positive / neutral / negative / unresolved；
    - champion P1：具体设计、效应大小、适用 envelope、覆盖率和失败边界。

“自动启动”不等于绕过安全门。任何缺 secret、环境不匹配、P0 parity 失败、数据未冻结、schema 不闭合或 smoke 失败都必须 fail closed，并生成 `reports/BLOCKED.md`，禁止静默使用替代模型、替代数据或默认参数继续。

用户在 2026-07-24 已明确要求“建完自动开始跑”；该指令就是 v0.1 的启动授权。Coding agent 将本文的固定预算和 decision thresholds 写成 hash-locked `launch_approval.json`，通过全部 hard gates 后无需再次询问或等待，直接启动 treatment 长跑。只有安全/完整性 gate 失败才停，并立即生成 `BLOCKED` 报告。

---

## 1. 研究问题与结论边界

### 1.1 必须回答的五个问题

**RQ1 — WEBPAGE-P1 是否有用？**  
对 Tavily 返回的 `raw_content`，以短 evidence-ID selection 替代默认 `summarize_webpage`，是否在最终报告质量不劣的前提下减少全流程工作？

**RQ2 — CLOSE-P1 / Compressor-P1 是否有用？**  
在每个 researcher 关闭后，以 evidence-ID selection/digest 替代 `compress_research` 长 prose，是否在 supervisor trajectory 与最终报告质量不劣的前提下减少全流程工作？

**RQ3 — 两个节点能否组合？**  
WEBPAGE-P1 会改变 researcher 后续轨迹和 CLOSE-P1 的输入，必须测 H×C interaction，禁止把两者收益简单相加。

**RQ4 — 哪种 P1 有用？**  
必须比较 chunking、selector scope、aggregation、output freedom、close mode；不能把某一个 prompt 的失败写成“P1 无用”，也不能把某一个成功变体写成“所有 P1 有用”。

**RQ5 — 什么时候有用？**  
只能使用 decision-time 可见的 pre-treatment 特征，输出一个在 untouched holdout 上验证过的 eligibility envelope。

### 1.2 不在本周范围内

本周禁止实现或声称：

- gateway/scheduler/broker policy；
- vLLM scheduler fork、KV eviction/pinning、cache-aware scheduling；
- 多租户准入、deadline/fairness、2 ms solver；
- Registry R1 的产品实现；
- 第二 workflow；
- Parrot/JITServe 等 related-work baseline；
- EuroSys headline；
- population-level “P1 永远安全”；
- live web 上的严格因果结论。

Week-1 是 P1 的设计与 kill/keep study。它可以支撑“继续/收缩/停止 P1”，不能直接替代最终论文主实验。

### 1.3 科学与工程裁决分开

最终必须区分：

- `KEEP`：质量门通过，完整 work 明确下降；
- `CONDITIONAL`：只在预冻结 envelope 内通过，并报告覆盖率；
- `MECHANISM_ONLY`：隔离层成立但 operational 层未成立，deployment/proposal NO-GO；
- `KILL_STRUCTURAL/HARM/NO_HEADROOM`：分别表示确定性结构失败、质量伤害、或 work-saving 上界低于最小意义；
- `NOT_ESTABLISHED`：置信区间仍跨越最小有意义边界。

工程/提案决策中，`NOT_ESTABLISHED` 按 `NO-GO` 处理；科学报告中不得把它改写成“证明 P1 无效”。

---

## 2. 已核实的 ODR 控制流事实

### 2.1 固定上游版本

使用现有 drbat 已 pin 的 Open Deep Research：

```text
repository: https://github.com/langchain-ai/open_deep_research
commit: 408da442a661ea5e40a6163329f82e3f22628949
```

新仓库以 Git submodule 固定该 commit。vendor 目录只读；所有改动由：

- 自有 adapter；
- 明确的 strategy interface；
- `patches/odr_p1_hooks.patch`

实现。run manifest 必须记录 submodule commit 与 patch SHA256。

### 2.2 WEBPAGE-P1 的准确节点

`HTML-P1` 只保留为历史口头简称。代码、schema、图表和最终裁决必须统一命名为：

```text
WEBPAGE_P1 / PAGE_RAW_CONTENT
```

原因：Tavily `include_raw_content` 返回的是清洗/解析后的 markdown 或 text，不是浏览器原始 HTML bytes。

Hook 位于：

```text
Tavily search result
  -> URL-level result record
  -> raw_content exists?
  -> P0 summarize_webpage OR P1 page selector
  -> formatted tool observation
  -> Researcher next ReAct turn
```

严格语义：

- 只有 `raw_content` 非空的 result 才是 H-P1 eligible；
- `raw_content` 缺失时保持 vendor 的 `result.content` 路径；
- P0/P1 必须从相同 frozen bytes、相同截断规则开始；
- P0 timeout/error 返回 raw content 的现有行为必须保留并进入 all-offered 成本；
- 一页不是一次 tool call，也不是一个 GPU batch；
- 同内容不同 URL occurrence 不得丢失 citation lineage。

`WEBPAGE_P1` 的 primary 输入是共同的 `SourceView`。P0 与 P1 必须看到相同 bytes，并遵循 pinned ODR 的相同 `max_content_length`/截断规则。若 P1 看完整页而 P0 只看前缀，该变体必须改名 `FULL_PAGE_P1`，不得混入 primary 因果比较。

真正的 `DOM_HTML_P1` 只能作为 secondary：

- 另行抓取并冻结 HTTP response bytes、MIME、status、redirect chain 和 fetch time；
- 明确网页许可、失败和动态渲染规则；
- P0/P1 都从同一个 frozen DOM/HTML object 开始；
- 不允许把 Tavily markdown 当 DOM，也不允许 browser live drift 混进 primary；
- 一周预算不足时优先砍掉该 secondary，不得牺牲 `WEBPAGE_P1` holdout。

### 2.3 Compressor-P1 的准确节点

代码命名：

```text
RESEARCHER_CLOSE
```

Hook 位于：

```text
Researcher ReAct
  -> one of:
       ResearchComplete
       max_react_tool_calls
       no tool call/native-search exit
  -> P0 compress_research OR P1 close selector
  -> researcher handoff
  -> supervisor ToolMessage
  -> next supervisor decision and final writer notes
```

严格语义：

- 主实验单位是一个 researcher handoff；
- 三种退出路径全部进入同一 treatment policy；
- 自然调用 `ResearchComplete` 的子集不能代表全部 close；
- P1 输出会同时影响 supervisor 控制回路和 final writer；
- 不能只比较 immediate compressor latency；
- P1-only `ResearchCompleteWithSelection(...)` 是改变 tool schema/停止策略的独立 extension，不能替代其他退出路径的 fallback selector，也不能归因成 reducer-only。

必须拆开两个不同的 treatment：

```text
C_VISIBLE:
  selector 只能读取原 P0 compressor 可见的 researcher_messages 与 tool outputs

C_REGISTRY:
  selector 还可回读 Registry/SourceStore 中未出现在上述可见输入里的 raw page spans
```

`C_VISIBLE` 才是 compressor-only 的主实验。`C_REGISTRY` 同时改变 reducer 和 evidence visibility，只能作为 registry-assisted extension 单独报告。实现上：

- `VisibleCompressorView` 是由 pinned ODR compressor 的实际输入 bytes 派生的不可变 view；
- provenance edge 可以回指 source object，但不得借 provenance 偷渡 hidden text 给 `C_VISIBLE`；
- H+C 时，C 只能看到 H 已实际发布进 tool output 的内容；
- `C_REGISTRY` 的结果不得被用于支持“只替换 compressor 即可”的结论。

### 2.4 主 end-to-end 的四个逻辑臂

```text
Arm P0          : WEBPAGE=off, C_VISIBLE=off
Arm H           : WEBPAGE=on,  C_VISIBLE=off
Arm C_VISIBLE   : WEBPAGE=off, C_VISIBLE=on
Arm H+C_VISIBLE : WEBPAGE=on,  C_VISIBLE=on
```

`C_REGISTRY` 进入单独的 extension block；若预算允许，可再测 `H+C_REGISTRY`，但不得替代上述 2×2。只有经 screening/mini-ITT 存活的节点才进入 holdout。若 H 或 C 被明确 kill，holdout 自动缩为 P0 对存活 policy；禁止为了凑 2×2 继续烧 GPU。

---

## 3. Secret 与外部 API 规则

### 3.1 绝不持久化用户在聊天中提供的 key

仓库、plan、patch、shell history、日志、SQLite、异常栈、测试 fixture 和 report 中都不得出现真实 key。

正式 Linux service 优先通过 systemd encrypted credential 或 repo 外 `0600` credential file 读取：

```text
TAVILY_API_KEY_FILE
DEEPSEEK_API_KEY_FILE
```

本地开发 fallback 才允许读取：

```text
TAVILY_API_KEY
DEEPSEEK_API_KEY
```

可选配置：

```text
DEEPSEEK_BASE_URL
DEEPSEEK_JUDGE_MODEL
```

用户已在父任务中提供本次运行所需 credentials，并明确授权自动开跑。Coding agent 必须通过平台/运行时 secret injection 将其送入 provider service，绝不把明文复制到 plan、repo、普通 `.env`、命令行、日志或 report。预先轮换不再是本次 launch gate；实验完成后仍建议撤销/轮换这组已出现在聊天记录中的 credentials。

只有 Tavily/DeepSeek provider service 读取相应 credential。环境变量 allowlist 本身不是完整安全边界；有管理员权限时使用三个 Unix identity/unit：

```text
shapeflow-api      # 唯一持有 API credentials；本地 Unix socket provider
shapeflow-runner   # coordinator/ODR；无 credential path 权限
shapeflow-infer    # vLLM；无 credential path 权限
```

provider socket 使用 filesystem ACL + peer-credential 校验，只接受固定 schema，内部执行 rate limit、budget reservation、redacted logging；不提供任意 URL/header/body 转发。凭据由 provider unit 的 `LoadCredentialEncrypted=` 或 root-owned `0400` file 注入。vLLM、ODR、chunker 和分析进程使用显式环境变量 allowlist。

若无管理员权限，auto-launch fallback 是独立 user-systemd provider unit：secret 只通过 systemd credential/anonymous FD 进入 provider 内存，coordinator 在启动任何 child 前删除自身 credential reference 并 scrub environment，provider 不执行网页内容或模型生成代码。该模式明确记录 `SECRET_ISOLATION=PROCESS_ONLY`，不能声称 UID-grade isolation，但不阻断本次授权运行。若连这种不落盘/不入 argv 的注入都不可用，才 `BLOCKED_NO_SECURE_SECRET_INJECTION`。

### 3.2 `.env` 与日志

必须提供 `.env.example`，只含变量名和占位符。`.gitignore` 至少包含：

```gitignore
.env
.env.*
!.env.example
data/
runs/
logs/
secrets/
*.sqlite
*.sqlite-wal
*.sqlite-shm
```

`SecretRedactor` 必须在所有 logging handler 与 exception renderer 前执行，覆盖：

- `tvly-...`
- `sk-...`
- `Authorization: Bearer ...`
- `api_key=...`
- query string 中的 token/key

测试必须注入 fake secret 并证明 stdout、stderr、JSONL、SQLite 与 report 全部没有 secret substring。

还必须：

- 提交 `.gitleaks.toml`，pre-commit/CI/launch gate 均运行 secret scan；
- logging 采用字段 allowlist，不先 dump 完整 request/header/environment 再做过滤；
- 禁止 `set -x`、HTTP debug trace、URL query credential；
- TLS verification 不得关闭；
- 所有本地服务只监听 `127.0.0.1`；
- 网页内容一律视为不可信数据，selector 无 shell、tool 或额外网络权限。

### 3.3 Tavily 只用于 acquisition 与 live-validity

正式 treatment run 禁止 live Tavily。

官方 Python SDK 当前支持：

```python
client.search(
    query=...,
    search_depth="advanced",
    include_raw_content="markdown",
    include_answer=False,
    include_usage=True,
    max_results=...,
)
```

参考：

- https://docs.tavily.com/sdk/python/reference
- https://docs.tavily.com/documentation/best-practices/best-practices-extract

必须记录 `request_id`、`response_time`、`failed_results`、usage、原始 response hash。不得只保存 `content` 字段。

### 3.4 DeepSeek 只做 evaluator/truth-packet 辅助

DeepSeek 不得用于：

- P0/P1 treatment 推理；
- selector；
- researcher/supervisor/final writer；
- 自动改 prompt；
- 根据结果生成新 treatment。

它只允许用于：

- treatment 前 arm-independent `AcquisitionSpec` decomposition（若用户 task 缺失）；
- TruthPacket candidate generation；
- blind quality judge；
- judge disagreement explanation；
- human audit queue 的候选排序。

使用 OpenAI-compatible `/chat/completions`，JSON mode 必须：

- `response_format={"type":"json_object"}`；
- prompt 明确要求 JSON 并给出 schema/example；
- 对 empty content、length truncation、invalid JSON 做有限重试；
- 每次响应都做本地 JSON Schema validation；
- 失败后标记 `JUDGE_UNAVAILABLE`，禁止用 0 或均值插补。

参考：

- https://api-docs.deepseek.com/guides/json_mode/
- https://api-docs.deepseek.com/api/create-chat-completion

模型名不得在代码中写死。启动前解析 `DEEPSEEK_JUDGE_MODEL`，写入 freeze manifest 后不再改变。

### 3.5 API 预算和 circuit breaker

配置中必须存在：

```yaml
api_budget:
  tavily_max_requests: 500
  tavily_max_credits: 1000
  deepseek_max_requests: 2000
  deepseek_max_input_tokens: 20000000
  deepseek_max_output_tokens: 3000000
  deepseek_max_usd: 10.00
  max_remote_calls: 2500
campaign_budget:
  max_gpu_hours: 150
  max_wall_hours: 168
  max_disk_bytes: 8589934592
  min_free_disk_bytes: 12884901888
retry_policy:
  max_retries_per_request: 3
  max_consecutive_failures: 5
```

以上就是本次 auto-launch v0.1 的授权硬上限，不是占位符。provider 在第一次 treatment 前读取实际 balance/usage；effective budget 只能按确定性公式收紧为 `min(上限, 可用余额减安全 reserve)`，并在 outcome 不可见时写入 effective protocol hash。若 effective Tavily budget 不够 96-task acquisition，按预先随机 task order 缩小 registry 后再 split；不得超上限或事后提高。少于 24 个可用 tasks 时仍完成 code/GPU smoke 和 component characterization，但最终写 `PARTIAL_INSUFFICIENT_API_BUDGET`。

预算检查必须是 dispatch 前的 admission control：

1. SQLite transaction 先按最坏情况 reserve credits/tokens/USD/GPU time/disk；
2. reserve 失败则 work item 进入 `BLOCKED_BUDGET`，不得先调用再检查；
3. 响应后按实际 usage settle，多余 reservation 释放；
4. timeout-after-send 按最坏成本保留 reservation，标成 `FAILED_UNKNOWN`；
5. retry 创建新 attempt，所有失败和可能重复计费都进入 cost ledger。

达到任一预算上限或连续失败阈值后：

- acquisition/judge 阶段停止；
- 已完成数据保留；
- 写 `reports/BLOCKED_API_BUDGET.md`；
- 不自动提升预算；
- GPU 不进入依赖未完整数据的阶段。

---

## 4. 新仓库与目录结构

### 4.1 初始化

Coding agent 执行：

```bash
mkdir -p shapeflow-p1-study
cd shapeflow-p1-study
git init -b main
git submodule add https://github.com/langchain-ai/open_deep_research vendor/open_deep_research
git -C vendor/open_deep_research checkout 408da442a661ea5e40a6163329f82e3f22628949
```

禁止自动创建 GitHub remote、push 或开 PR；除非用户另行授权。

代码仓库与实验工作目录分开：

```text
authoring repo:
  C:\Users\yl998\Documents\programming\deepresearch\shapeflow-p1-study

server private bare remote:
  sjtu:/storage/nvme/shapeflow-p1-study.git

server experiment checkout:
  /storage/nvme/shapeflow-p1-study
```

Coding agent 在本地完成并提交每个 gate 后，可创建上述**私有服务器 bare remote**并普通 push；这不是 GitHub/publication。若任一路径已存在且 identity 不匹配，必须停下生成 `BLOCKED_EXISTING_PATH.md`，禁止删除、覆盖或 force-push。服务器只从明确 commit checkout；dirty tree 禁止 launch。

`protocol/launch_approval.json` 至少包含：

```text
protocol_sha
budget_sha
decision_thresholds_sha
approval_mode: USER_EXPLICIT_AUTO_LAUNCH
approval_source_date: 2026-07-24
approved_at_utc
```

Coding agent 只把本文已固定的值 materialize/hash，不能自行改值；这不是再次请求审批。审批文件与 commit SHA 匹配后，server checkout 自动执行 `scripts/bootstrap_and_run.sh`。原始 snapshots、object store、SQLite 和 logs 永不进入 Git；最终只回传 redacted reports、Parquet exports 和 hash manifests。

### 4.2 目标目录树

```text
shapeflow-p1-study/
  AGENTS.md
  README.md
  pyproject.toml
  uv.lock
  .gitignore
  .gitleaks.toml
  .env.example
  protocol/
    study_v1.yaml
    arms_v1.yaml
    budget_v1.yaml
    split_manifest.json
    launch_approval.json
    freezes/                             # tracked; outcome-free design locks
  vendor/
    open_deep_research/                 # pinned submodule; read-only
  patches/
    odr_p1_hooks.patch
  configs/
    stack.yaml
    acquisition.yaml
    task_source.yaml
    variants.yaml
    decision.yaml
    week1.yaml
    judge.yaml
  schemas/
    task.schema.json
    snapshot.schema.json
    source_occurrence.schema.json
    evidence_span.schema.json
    query_attempt.schema.json
    checkpoint.schema.json
    variant.schema.json
    selector_output.schema.json
    truth_packet.schema.json
    request_event.schema.json
    run_result.schema.json
    freeze_record.schema.json
  src/shapeflow_p1/
    __init__.py
    cli.py
    config.py
    canonical.py
    hashing.py
    secrets.py
    logging.py
    doctor.py
    ops/
      preflight.py
      watchdog.py
      gpu_lease.py
      service_status.py
    providers/
      external_call_ledger.py
      retry.py
      rate_limit.py
    acquire/
      tavily_client.py
      source_pool.py
      snapshot_store.py
      frozen_search.py
    odr/
      adapter.py
      hooks.py
      checkpoints.py
      close_reason.py
      p0_parity.py
    evidence/
      identity.py
      normalizer.py
      chunkers.py
      manifest.py
      lineage.py
    p1/
      contracts.py
      prompts.py
      selectors.py
      aggregators.py
      renderer.py
      preflight.py
      fused_close.py
    runtime/
      openai_proxy.py
      vllm_process.py
      request_tags.py
      nvml_sampler.py
      cpu_accounting.py
      cache_control.py
    experiment/
      coordinator.py
      state_machine.py
      randomization.py
      block_design.py
      promotion.py
      budget.py
      ledger.py
      heartbeat.py
    evaluation/
      truth_packet.py
      truth_builder.py
      judge_client.py
      atomizer.py
      citation_eval.py
      trajectory_eval.py
      human_audit.py
    analysis/
      paired.py
      screening.py
      factorial.py
      heterogeneity.py
      bootstrap.py
      figures.py
      report.py
  tests/
    unit/
    property/
    integration/
    fixtures/
  scripts/
    bootstrap_server.sh
    bootstrap_and_run.sh
    resume_week1.sh
    status.sh
    stop_safely.sh
  systemd/
    shapeflow-api-provider.service.template
    shapeflow-steward.service.template
    shapeflow-evaluator.service.template
    shapeflow-vllm.service.template
    shapeflow-p1-week1.service.template
    shapeflow-holdout-gate.service.template
  data/                                 # gitignored
    tasks/
    acquisition/
    snapshots/
    source_pools/
    truth_packets/
    checkpoints/
  runs/                                 # gitignored
    ledger.sqlite
    events/
    outputs/
    metrics/
  reports/
    README.md
  manifests/
  exports/
  object_store/                         # gitignored; zstd content-addressed blobs
  logs/                                 # gitignored
```

### 4.3 Python 与依赖

优先使用 `uv`，Python 3.12。`pyproject.toml` 必须 pin 直接依赖，`uv.lock` 必须提交。

建议直接依赖：

```text
typer
pydantic
PyYAML
orjson
jsonschema
httpx
tenacity
tavily-python
openai
numpy
scipy
pandas
statsmodels
scikit-learn
matplotlib
seaborn
psutil
pynvml
beautifulsoup4
markdown-it-py
hypothesis
pytest
pytest-asyncio
```

不要为了方便引入会在线调用模型的 agent framework。Treatment 路径只使用 pinned ODR 与本地 vLLM。

---

## 5. 环境与 stack freeze

### 5.1 禁止静默替换

`doctor` 必须解析并记录：

- repo commit/tag、`dirty=false`；
- OS、kernel、container/image digest；
- GPU name、UUID、VRAM；
- NVIDIA driver；
- CUDA runtime；
- Python、PyTorch；
- vLLM version、git SHA；
- model、tokenizer revision；
- quantization、KV dtype；
- attention backend；
- max model length；
- APC、chunked prefill；
- chat template hash；
- ODR commit、patch hash；
- prompt bundle hash；
- config/schema hashes；
- Tavily SDK、OpenAI SDK 与 judge model string；
- requested/returned judge model 与 provider fingerprint；
- provider pricing table snapshot、来源和检索日期；
- UTC timezone、locale、`PYTHONHASHSEED`；
- acquisition object root/Merkle root；
- analysis package hash。

任何与 `configs/stack.yaml` 不一致的字段都必须停止，不允许自动升级/降级。

### 5.2 2026-07-24 只读服务器快照

当前已只读核实：

```text
host: LiuYancheng
GPU: 4 × RTX 4090；核实时均无 compute load
local model: /storage/nvme/reme/models/Qwen3-14B-AWQ
existing read-only vLLM: /storage/nvme/drbat/.venv, version 0.24.0
/storage/nvme free: 约 24 GB
SSH execution identity: root
systemd: 245，system-level units 可用（user manager 当前 offline）
```

这是启动前线索，不是永久保证。`doctor` 必须重新核实，并遵守：

- 默认只租一张 GPU；
- 利用 root 只完成一次性 service-user/unit/ACL 安装；实际 provider/runner/infer/steward/evaluator 全部降权运行，禁止以 root 跑模型或处理网页；
- 以 GPU UUID 而不是 index 冻结设备；
- 通过跨进程 `flock` lease 占用；连续检查显存、利用率和 foreign PID；
- 旧项目曾使用 GPU 0/1，当前空闲不代表可永久抢占；
- 发现外部任务只等待/暂停，绝不 kill 或 `pkill`；
- 新 repo 建自己的 controller venv；不得向 `/storage/nvme/drbat/.venv` 安装包；
- 现有 vLLM 环境只能在版本/依赖 hash 通过后作为只读 executable；
- raw blobs 用 zstd content-addressed store，campaign 默认上限 8 GB；
- free space 低于 12 GB 时停止 admission 并安全暂停；
- confirmatory block OOM 后不得静默降低 context/concurrency；原配置整 block 重跑，改配置只能记为 diagnostic。

### 5.3 默认候选 stack

若服务器已经有当前项目使用的冻结模型，可将下列值作为候选：

```yaml
model:
  repo: Qwen/Qwen3-14B-AWQ
  revision: 31c69efc29464b6bb0aee1398b5a7b50a99340c3
  enable_thinking: false
engine:
  max_model_len: 16384
  gpu_memory_utilization: 0.90
  enable_prefix_caching: true
  enable_chunked_prefill: true
sampling:
  temperature: 0.3
  n: 1
```

Coding agent 必须先验证本机实际 artifact；若 revision 不存在，停止并报告，禁止默默换成 latest。

### 5.4 因果隔离模式与 operational 模式

现有审计已经表明 co-batching 可能改变模型输出，因此分成两层：

**Primary quality/causal mode**

- `max_num_seqs=1`；
- telemetry gateway `max_upstream_inflight=1`，所有 sibling requests 在 gateway 排队但不计入 service work；
- APC off；
- 一次只运行一个 task attempt；
- task×seed×arm 使用完整区组、AB/BA/period 平衡；
- shared-path seed 由稳定 logical node ID 派生；
- 用于质量 NI、trajectory、H/C main effect 与 interaction。

**Secondary operational mode**

- native APC on；
- continuous batching/chunked prefill 按冻结部署配置；
- 所有 arm 使用相同到达 trace、并发和 cache reset/warmup protocol；
- 单独报告 throughput、TTFT/TPOT、cached tokens、能耗与端到端 wall time；
- 只有在此层仍有 operational saving，才能声称部署收益；若只在隔离层成立，裁决为 `MECHANISM_ONLY / PROPOSAL_NO_GO`。

operational 的独立实验单位不是 request/task，而是 `TraceBlock`：

```text
TraceBlock {
  trace_block_id,
  4 distinct holdout tasks,
  4 fixed simultaneous arrivals (primary concurrency=4),
  exact request/task mapping,
  arm replay order,
  cache namespace/reset record
}
```

- 同一 TraceBlock 对 P0/H/C/H+C 逐臂 replay，arm order 用 Williams/Latin-square 平衡；
- 最少 8 个 independent TraceBlocks（32 distinct tasks），目标 12；同一 task 不跨 block；
- mini operational pilot 估计 paired `log(makespan_P1/makespan_P0)` SD，计算 10% saving、80% power 所需 blocks；
- 若所需 blocks >12 或实际完成 <8，写 `operational_power_shortfall=true`，不得 deployment KEEP；
- block endpoints：makespan、throughput、NVML joules、p95 request latency、failure count；质量仍按 task 输出评估；
- inference 对 paired TraceBlock ratios 做 one-sided 95% cluster/bootstrap CI，不能对单个 aggregate trace 或 request 当独立样本。

operational guards v0.1：

```text
block makespan saving LCB95 >= 10%
throughput improvement LCB95 >= 10%
p95 latency ratio UCB95 <= 1.10
GPU joule ratio UCB95 <= 1.00
terminal-failure risk increase UCB95 <= 2pp
critical-harm risk increase UCB95 <= 3pp
```

本周不修改 vLLM scheduler。通过 request proxy 注入稳定 request ID，读取 OpenAI usage、启动时发现并冻结实际 `/metrics` 名称，并用 NVML 旁路采样。不得按其他 vLLM 版本猜 metric 名称。

### 5.5 APC 专项规则

本周不研究 cache policy，但 P1 separate selector 的成本依赖真实 prefix reuse：

- operational 主臂统一开启 native APC；
- 首选给每个 `(protocol, block, arm)` 注入独立 `cache_salt` namespace，同时同 arm 内保持一致；
- `doctor` 必须用“同 salt 第二次命中、不同 salt 不命中”的 paired probe 验证 vLLM 0.24 实际支持，并冻结 request field/metric evidence；
- 若当前栈不支持可验证 salt，fallback 是每个 arm/block 使用新 vLLM process + 相同 warmup；restart/warmup 成本单列，measurement window 从 readiness/sentinel 通过后开始；
- 禁止依赖未验证的 reset endpoint；
- branch 内自然产生的 prefix reuse 保留；
- 独立 sub-study 比较：
  - dedicated selector prompt；
  - prefix-preserving selector prompt；
  - fused close；
- holdout 的 P0/H/C/H+C 使用相同 cache-reset 协议；
- 若增加 warm operational sensitivity，必须使用独立 namespace/进程和相同 task sequence；
- 不得把“刚执行完 researcher”当作 APC hit，必须记录实际 cached tokens。

若 salt isolation与fresh-process fallback都无法证明跨 arm 无 carry-over，operational block hard-stop 为 `BLOCKED_APC_ISOLATION`；isolated APC-off causal study仍继续，但最终不得给 deployment KEEP。

---

## 6. Tavily acquisition 与冻结检索

### 6.1 为什么不能让每个 arm live search

P1 会改变 query 和 trajectory。相同 query 必须得到相同 source universe；不同 arm 不能因为网页更新、排名漂移或 API 偶发缺失获得不同世界。

Week-1 使用两层设计：

1. **主因果实验**：Tavily 在 acquisition window 构建 task-local frozen source pool；所有 treatment 通过 deterministic local search 查询相同 source pool；
2. **外部效度**：主实验完成后，用小规模 live Tavily 运行，只作描述性 sanity。

### 6.2 Task-local source pool

每个 `TaskSpec` 在任何 source fetch 前必须含不可变：

```text
AcquisitionSpec {
  original_question,
  authored_facets[],
  fixed_queries[],
  conflict_probe?,
  negative_or_gap_probe?,
  table_list_numeric_probe?,
  authoring_method,
  authoring_model_prompt_fingerprint?,
  content_sha256
}
```

首选人工/用户随 task 提供。若缺失，允许在 treatment 前用冻结 prompt/model/schema 做一次 arm-independent decomposition，保存 provider fingerprint，并把整个 AcquisitionSpec 封存后再调用 Tavily；这类 task 标记 `MACHINE_DECOMPOSED`. TruthPacket 可以在 acquisition 后细化 truth facets，但绝不能回写 AcquisitionSpec 或补抓对某 arm 有利的 sources。

对每个 task，acquisition queries 至少覆盖：

- 原始 research question；
- 每个 required facet 的固定 query；
- conflict/contradiction probe；
- negative-evidence/no-answer probe；
- table/list/date/numeric probe（若 task 类型需要）。

这些 query 在 task registry 冻结时生成，不能根据 P0/P1 输出新增。

Tavily acquisition 参数必须显式配置；禁止 `auto_parameters=True`。建议：

```yaml
tavily:
  search_depth: advanced
  include_raw_content: markdown
  include_answer: false
  include_usage: true
  max_results_per_query: 8
  timeout_seconds: 60
```

每个 response 原样存：

```text
request_id
query_id
query_text
request parameters
response_time
usage
failed_results
rank
title
url
content
score
raw_content
published_date
fetch timestamp
raw response SHA256
```

### 6.3 Snapshot identity 与 occurrence

```text
CanonicalSnapshot {
  content_hash,
  raw_content_format,
  raw_content_bytes,
  normalization_version
}

SourceOccurrence {
  occurrence_id,
  task_id,
  query_id,
  url,
  title,
  rank,
  score,
  published_date,
  content_hash
}
```

`content_hash` 收敛 bytes；`occurrence_id` 保留 URL/query/citation lineage。相同内容的两个来源不能只剩一条 citation。

同时派生两个不同 view：

```text
AuditOccurrenceGraph:
  保存所有 query/rank/URL occurrences，仅用于 provenance/evaluation

VendorVisibleOccurrenceView:
  精确复现 pinned ODR 的 URL 去重、first-occurrence order 和可见 metadata
```

primary P0/WEBPAGE selector 只能读取 `VendorVisibleOccurrenceView`；完整 audit graph 不能给 P1 当额外 source-diversity/rank signal。若研究完整 occurrence metadata，另列 `MULTI_OCCURRENCE_META_EXT`，不得归入 primary WEBPAGE-P1。

### 6.4 FrozenSearchService

GPU run 时：

- 拒绝任何 outbound Tavily request；
- 对 researcher query 使用 task-local BM25/index；
- tie-break 固定；
- top-k 固定；
- 返回 Tavily-like `title/url/content/raw_content` records；
- 所有结果由 snapshot hash 可追溯；
- 相同 query + source pool 必须 byte-identical；
- 查询本身可因 treatment 分叉，这是 end-to-end effect 的一部分。

测试必须在禁网环境中完成一次完整 P0 task。

代码中拆成三个不可混用的 backend：

```text
TavilyCaptureBackend        # 只在 acquisition/live-validity 调外网
ExactSnapshotReplayBackend # checkpoint fork；query-key miss 立即 invalid
FrozenTaskCorpusBackend     # E2E；在 task-local source pool 上确定性检索
```

三个身份不可合并：

```text
query_snapshot_id = hash(exact query + every Tavily parameter + API/schema version)
content_id        = hash(raw_content bytes)
occurrence_id     = hash(query_snapshot_id + rank + URL occurrence)
```

checkpoint replay miss 不能 fallback live search。URL 不是 immutable content identity；canonical content dedup 也不得吞掉不同 query/rank/URL occurrence。

---

## 7. Task registry 与数据拆分

### 7.1 Task source precedence

Coding agent 按以下顺序选择任务：

1. 用户提供且与 Tavily/web citation contract 兼容的 sealed task registry；
2. 在任何 P1 output 前新建并封存的 Tavily-compatible research task pool；
3. 若以上都不存在，构建一个明确标为 `FORMATIVE_MACHINE_AUTHORED` 的 task pool。

机器生成 fallback 不得被写成论文 confirmatory corpus。

旧 drbat probes 绑定其 frozen BM25/MCP corpus、`document_id` citation contract，且原配置 `search_api=none`。它们只能连同原 corpus adapter 用于 regression/smoke；禁止直接接 Tavily source pool，也禁止进入本实验 SCREEN/POWER_PILOT/HOLDOUT。drbo calibration prompts 和旧 formative tasks 同样不得进入新 holdout。

### 7.2 必须覆盖的任务 strata

至少覆盖：

- low / medium / high evidence volume；
- 1–2 / 3–4 / 5+ facets；
- 单来源事实；
- 多来源综合；
- source conflict；
- negative evidence / unresolved；
- citation dense；
- table/list heavy；
- high redundancy；
- raw_content missing / extraction failure；
- natural ResearchComplete / max calls / no-tool close。

### 7.3 Split

按 `topic + source cluster` 分组后再拆分，禁止同主题改写或共享大量 source 的任务跨 split：

```text
SCREEN
POWER_PILOT
HOLDOUT
JUDGE_CALIBRATION
LIVE_VALIDITY
```

HOLDOUT 在 policy freeze 前不得被 treatment runner 打开。单 checkout/单 UID/单 SQLite 的“约定不读”不算隔离，使用 steward/gate/evaluator 三段：

```text
shapeflow-steward:
  acquisition/truth build for every split
  owns encrypted holdout task+corpus package and all TruthPackets

shapeflow-runner:
  before freeze sees DESIGN only
  after gate sees holdout task+frozen corpus only
  never sees any TruthPacket

shapeflow-evaluator:
  after treatment completion reads frozen outputs + TruthPackets
  never controls treatment/promotion
```

`shapeflow-holdout-gate.service` 使用 runner 不可读的 key/ACL。它只在 tracked holdout freeze commit/tag/hash、ledger phase 和 approval都匹配时，将**无 evaluator labels**的 holdout task+corpus materialize 到 runner-readable directory；TruthPacket 始终留在 evaluator UID。提前读取、gate 重复 release 或 hash mismatch 都是 fatal incident。若无法建立该 UID/ACL/key-release boundary，honest holdout 标记 `BLOCKED_NO_HOLDOUT_ISOLATION`，不得声称 confirmatory。

### 7.4 样本量不硬编码

前 48–72 小时估计：

- paired work difference 的 SD；
- task/source cluster ICC；
- quality discordance；
- failure rate；
- 每 task-arm GPU hours。

然后按 `configs/decision.yaml` 中冻结的最小有意义效应与 power 计算剩余样本量。重复 seed 不能冒充新的独立 task。

为让 coding agent 能预建 manifests，v0.1 的**容量规划目标**是 96 tasks：

```text
DESIGN/SCREEN: 32
HONEST_HOLDOUT: 48
RESERVE: 16
```

RESERVE 在任何 outcome 前随机排序，只能替换 arm 开跑前即判定不可用的任务。这个 96 是规划上限，不是假定机器必跑完的样本承诺；实测 mini-E2E p90 block-hours 决定最终可行 n。

holdout 样本用 mini 的方差而不是 effect estimate 规划：

```text
n_quality[j] = ((z_0.95 + z_0.80) * sd_paired_quality[j] / NI_margin[j])^2
               for every continuous co-primary quality guard j
n_binary[k]  = paired-transition Monte Carlo for 80% power
               for every binary qualified/harm guard k
n_work       = ((z_0.95 + z_0.80) * sd_log_work_ratio / abs(log(0.90)))^2

n_required = ceil(1.15 * max(all n_quality[j], all n_binary[k], n_work))
n_feasible = floor(56 GPU-hours / observed_p90_isolated_complete_block_hours)
n_available = 48  # RESERVE 只替换，不增加 independent n
n_plan      = min(n_available, n_feasible)

n_operational_trace_required
  = ceil(((z_0.95 + z_0.80) * sd_log_trace_ratio / abs(log(0.90)))^2)
n_operational_trace_feasible
  = floor(24 GPU-hours / observed_p90_traceblock_hours)
n_operational_trace_plan
  = min(12, n_operational_trace_feasible)
```

若 `n_plan < n_required`，写 `confirmatory_power_shortfall=true`。若 operational plan 少于 8 或小于其 required，写 `operational_power_shortfall=true`。仍跑最大可行完整 blocks，但不得把 16 个 reserve 既作 replacement 又作新增 n，也不得放宽 margin 来制造结论。

---

## 8. Evidence IR 与 chunking

### 8.1 Evidence identity

```text
EvidenceSpan {
  span_id,
  content_hash,
  source_occurrence_ids[],
  char_start,
  char_end,
  token_start,
  token_end,
  text_sha256,
  kind,
  heading_path[],
  facet_ids[],
  token_len,
  chunker_version
}
```

`span_id` 必须是 canonical fields 的 SHA256 派生 ID。任何同 ID 不同 bytes、悬空 occurrence、越界 offset 都是 fatal。

`C_VISIBLE` 不得用 raw-source `EvidenceSpan` 假装代表 compressor 输入。它使用独立 namespace：

```text
VisibleMessageSpan {
  visible_span_id,
  message_id,
  message_role,
  byte_start,
  byte_end,
  exact_text_sha256,
  kind: TOOL_EVIDENCE | MODEL_DERIVED_CONTEXT | USER_CONTEXT,
  visible_compressor_view_hash
}
```

规则：

- selector context 是 pinned P0 compressor 实际可见的 lossless `researcher_messages` 全部 bytes；
- selectable spans 也只来自这些 bytes；
- tool output 中的模型生成 webpage summary 只能按其**可见 summary bytes**寻址，不得反向映射后偷渡 raw page text；
- AI reasoning 可作为 `MODEL_DERIVED_CONTEXT` 被保留，但永远不升级成 source evidence/citation；
- primary quality evidence metrics只计 `TOOL_EVIDENCE`；
- canonical raw-source `EvidenceSpan` 只对 WEBPAGE-P1 与 `C_REGISTRY` 可见。

### 8.2 非证据状态

“没有找到证据”不能伪造成 source span：

```text
QueryAttempt {
  attempt_id,
  task_id,
  researcher_id,
  query_text,
  result_occurrence_ids[],
  status: SUCCESS | EMPTY | FAILED | TIMEOUT,
  gap_facets[]
}
```

P1-TYPED/P1-BRIDGE 可以引用 `attempt_id` 表示 gap/negative evidence；P1-ID 不具备该表达能力，这正是需要实验的差异。

### 8.3 Chunker

实现三个 deterministic chunker：

1. `fixed_token_v1`
   - 固定 token window；
   - overlap 由 config 控制；
   - 作为控制，不作为默认 winner。
2. `paragraph_sentence_v1`
   - 先 paragraph；
   - 超长 paragraph 再 sentence/token split；
   - 保留 offsets。
3. `markdown_structure_v1`
   - heading、paragraph、list、blockquote、code、table；
   - table header 必须随 row；
   - list item 与父 heading 关联；
   - 用 heading metadata 代替大 overlap；
   - 超 cap 时确定性细分。

Chunk size 与 output budget 使用小型 response-surface 配置，不做随意全排列。所有 chunker 必须能从 snapshot 精确重建 span text。

跨 chunker 的主预算是 deterministic renderer 最终 materialize 给下游的 token 数，而不是“最多 N 个 ID”。否则粗 chunk 会天然携带更多文字，比较不公平。ID cap 只作为结构 guard。Markdown chunker 还必须：

- table row 总是携带对应 table header；
- list item 携带必要 lead-in；
- heading breadcrumb 放在 metadata/renderer 中；
- 不用大 overlap 人为复制 evidence。

---

## 9. P1 selector、output contract 与 aggregation

### 9.1 Selector 必须使用 treatment 模型

P1 selector 使用与 P0 compressor 相同的本地目标模型/量化/engine。DeepSeek 不得参与 selector。

Selector prompt 输入：

- task question；
- researcher subquestion/research topic bytes that are already visible to P0；
- candidate span IDs 与 exact text；
- primary 中仅 `VendorVisibleOccurrenceView` 的 source/citation metadata；完整 audit occurrence graph 只供 evaluator；
- query attempts；
- selected-token budget；
- output schema；
- 明确禁止生成输入集合外的事实/ID。

每次 selector call 记录完整 rendered-prompt hash、candidate-set hash 与 response token IDs。

Treatment 路径只允许使用：

```text
TreatmentConstraintView {
  task question/subquestion bytes already visible to P0,
  VendorVisibleOccurrenceView,
  runtime QueryAttempts already present in P0-visible messages,
  optional deployment pins only if the exact same bytes are present in P0 task prompt
}
```

`AcquisitionSpec` 只供 steward 抓 source，primary treatment 永远不可读，避免给 P1 而不给 P0 的额外规划信息。它与 evaluator-only TruthPacket 都使用不同 UID、目录、schema 和 ID namespace。P1 可在自身 structured output 内声明临时 facet labels，但这些是 treatment output，不是 gold；`critical truth item`、gold contradiction pair 和 TruthPacket labels 永不进入 selector/aggregator/preflight。

### 9.2 Output contracts

#### P1-ID

```json
{
  "selected_ids": ["..."]
}
```

不允许自由 prose。

#### P1-TYPED

```json
{
  "selections": [
    {
      "span_id": "...",
      "facet_ids": ["..."],
      "role": "support|contradict|background"
    }
  ],
  "gaps": [
    {
      "facet_id": "...",
      "query_attempt_ids": ["..."]
    }
  ]
}
```

所有字段机械可验证。

在 `C_VISIBLE` 中，`span_id` 的 schema discriminator 必须接受 `visible_span_id`，并保留 `kind`；只有 `TOOL_EVIDENCE` 可以产生 citation/source binding，`MODEL_DERIVED_CONTEXT` 只用于保留计划、gap 或 connective context。

#### P1-BRIDGE

在 P1-TYPED 基础上允许：

```json
{
  "bridges": [
    {
      "text": "...",
      "evidence_ids": ["..."]
    }
  ]
}
```

硬规则：

- 全局与 per-bridge token cap；
- 每条 bridge 至少一个 evidence ID；
- bridge 不能引入输入 spans 不含的实体/数值；
- bridge 成本完整计入 selector decode；
- unbounded bridge 等同 P0，schema/preflight 必须拒绝。

### 9.3 Selector scope

实现：

- `per_page`；
- `per_tool_call`；
- `hierarchical`：per-page recall-first shortlist + cross-source ID reducer。

Hierarchical 的第二次 LLM work 单列，禁止隐藏在 aggregator 中。

### 9.4 Aggregator

实现：

1. `stable_union_v1`
   - exact ID validation；
   - 同 source 相邻 span 合并；
   - stable facet/source/offset order；
   - byte duplicate 可合并内容，但 occurrence/citation edges 保留。
2. `coverage_budget_v1`
   - selector 自声明 facet 的 coverage 或 explicit gap；
   - source-diversity minimum；
   - 若 selector 在可见输入内标出 support/contradict candidates，则成对保留；
   - 只允许预先由 task author 给出的 deployment pin；默认无 gold critical pin；
   - global token budget；
   - deterministic tie-break。
3. `global_rerank_v1`
   - 仅对 hierarchical shortlist 运行；
   - 仍只输出 IDs/typed schema；
   - 自身 prefill/decode 入账。

### 9.5 Renderer

Renderer 只做确定性 CPU 序列化，不做语义改写。输出至少包含：

```text
facet
role
source title/url
span ID
exact span text
gap/query attempt markers
```

P0/P1 下游 prompt 必须分别保存 exact bytes 与 token count。

### 9.6 Preflight

在 publish 前检查：

- 所有 ID 属于 candidate namespace；
- offsets/text hash 匹配；
- citation occurrence 闭合；
- selector 自声明 facets 均有 selection 或 explicit gap（只检查 schema/ID，不判断 gold coverage）；
- selector 声明的可见 conflict candidates 保留闭合；
- visible QueryAttempt 对应的 negative/gap marker；
- selected-token budget；
- rendered downstream prompt budget；
- bridge token cap；
- output schema。

Truth critical recall、真实 contradiction-pair recall 和 required evidence coverage 全部在 publish 后由隔离 evaluator 计算，绝不能成为 treatment preflight oracle。

失败：

- component trial：记录 failure，不自动把失败样本删掉；
- end-to-end：按 policy fallback 到 P0，P1 已花成本仍计入原 job；
- publish 后不得伪装成无成本 P0。

---

## 10. 第一轮 variant registry

`configs/variants.yaml` 至少定义以下候选；每个 variant 有 canonical JSON 与 SHA256：

| ID | Node | Chunk/scope | Contract | Aggregation | Close/prompt |
|---|---|---|---|---|---|
| P0 | H/C | vendor | prose | vendor | vendor |
| H00-CPU | H | markdown/per-page | CPU lexical IDs | MMR/stable union | no LLM selector |
| H00-PROSE | H | markdown/per-page | short prose | token-matched | separate |
| H01 | H | fixed-token/per-page | ID | stable union | separate |
| H02 | H | markdown/per-page | ID | stable union | separate |
| H03 | H | markdown/per-page | TYPED | coverage budget | separate |
| H04 | H | markdown/hierarchical | TYPED | global rerank | separate |
| H05 | H | markdown/hierarchical | BRIDGE | coverage budget | separate |
| C01 | C | manifest/per-researcher | ID | stable union | dedicated selector |
| C02 | C | manifest/per-researcher | TYPED | coverage budget | dedicated selector |
| C03 | C | manifest/per-researcher | BRIDGE | coverage budget | dedicated selector |
| C04 | C | manifest/per-researcher | TYPED | coverage budget | prefix-preserving selector |
| C05-FUSED-EXT | C_FUSED_EXT | visible view | TYPED | coverage budget | new P1-only close tool + fallback |
| C00-CPU | C_VISIBLE | visible view | CPU lexical IDs | MMR/stable union | no LLM selector |
| C00-PROSE | C_VISIBLE | visible view | short prose | token-matched | separate |
| C06-REG | C_REGISTRY | registry spans | TYPED | coverage budget | registry-assisted |

说明：

- C01–C04 的 selector 输入是完整 `VisibleCompressorView`，输出 namespace 是 `VisibleMessageSpan`；AI reasoning 只能标成 `MODEL_DERIVED_CONTEXT`，不能自动成为 evidence；
- C06-REG 才能读取 hidden registry spans，必须单独估计 `visible-only × registry-assisted`；
- `SHORT_PROSE` 区分 pointer/ID 机制与“单纯少 decode”；`CPU_LEXICAL` 区分 LLM selection 与确定性 lexical selection；
- TruthPacket oracle 只用于 frozen-state ceiling，不进入产品候选或最终 end-to-end；
- pinned `ResearchComplete` 是空 schema，C05-FUSED-EXT 不能通过普通 close-after hook 实现；它必须使用独立的 P1-only tool/graph variant（例如 `ResearchCompleteWithSelection`），保留三种退出路径的 dedicated-selector fallback，并报告为“停止策略 + reducer”联合扩展，不得归因成 compressor-only；
- H/C 内部 budget 档位由 block design 分配，不再复制成几十个手写 variant；
- screening 用 balanced incomplete block/D-optimal assignment，而不是每个 checkpoint 跑全部候选；
- 预指定重点交互：
  - node × output contract；
  - chunker × selected-token budget；
  - selector scope × conflict；
  - close/prompt × actual APC cached tokens。
  - visible-only × registry-assisted。

---

## 11. ODR patch 与 checkpoint fork

### 11.1 Strategy interface

在自有代码定义：

```python
class PageTransformStrategy(Protocol):
    async def transform_tool_batch(
        self,
        *,
        task_ctx: TaskContext,
        assistant_turn: FrozenAssistantTurn,
        captured_tool_calls: Sequence[CapturedToolCall],
    ) -> Sequence[ToolMessage]: ...


class ResearchCloseStrategy(Protocol):
    async def close_researcher(
        self,
        *,
        task_ctx: TaskContext,
        researcher_state: FrozenResearcherState,
        evidence_manifest: EvidenceManifest,
        close_reason: CloseReason,
    ) -> ResearcherHandoff: ...
```

P0 strategy 调用 vendor 原逻辑。P1 strategy 调用 selector/aggregator/renderer。WEBPAGE reducer 的局部单位是每个 search tool call 的完整 result set；但 checkpoint/fork/publish 单位是一次 assistant turn 中**全部 sibling tool calls 的 batch**。内部可对每个 search result set 做 per-page map/per-tool reduce，非 search sibling 原样执行；最后严格按 pinned ODR join order 一次性发布所有 ToolMessages，不能提前 fork 单页或单 search-call。

### 11.2 Patch 允许修改的位置

Patch 只能：

- 注入 `PageTransformStrategy`；
- 注入 `ResearchCloseStrategy`；
- 捕获完整 assistant-turn tool-call batch 与 pinned join order；
- 捕获 close reason；
- 发 checkpoint/event；
- 注入 logical date；
- 传递 task/researcher/query/request IDs；
- 将模型 base URL 指向本地 telemetry proxy。

实现方式：

- submodule 保持只读；
- bootstrap 用 `git archive`/等价只读 materialization 复制到 `.build/open_deep_research-patched/` 后应用 patch；
- 用 `ContextVar` 传递 task/attempt/node/form/variant/boundary ID，禁止模块全局可变 current-run；
- hook 至少覆盖 search backend、page boundary capture/reduce、researcher close capture/reduce、supervisor publish；
- pinned `compress_research` 会修改 message list，checkpoint 前必须 lossless clone；第一个 fork 不得污染后续 fork；
- hooks-off golden test 使用 mock LLM/search 对 request hash、tool bytes 与 publish sequence 做 parity。

P0 下禁止修改：

- prompt bytes；
- tool schemas；
- join order；
- retry count；
- token limits；
- supervisor/researcher control flow；
- result formatting；
- fallback。

`C05-FUSED-EXT` 不走上述 reducer-only patch contract。它在独立 P1 graph adapter 中增加新 tool schema，拥有单独 arm ID、checkpoint schema、P0 comparator 和报告列；绝不修改/替换原 `ResearchComplete`。只有 C_VISIBLE dedicated selector 已证明可行后才运行该 extension。

### 11.3 P0 parity gate

CPU fixture 与小模型 fixture 上，hook-off P0 必须与 vendor：

- 请求序列；
- rendered prompt bytes；
- tool message bytes；
- close reason；
- final result；
- retry/fallback

一致。若本地模型存在非确定性，只要求 request envelope 与 application bytes 一致，不要求随机输出 bitwise 相同。

P0 parity 失败禁止任何 GPU screening。

### 11.4 Checkpoints

实现两个不可变 checkpoint：

```text
H_CHECKPOINT:
  task/assistant-turn IDs
  lossless assistant AIMessage and ordered sibling tool calls
  every search call's complete vendor-visible result set
  all SourceView hashes plus non-search sibling outputs
  researcher state immediately before publishing the whole ToolMessage batch
  RNG/sampling envelope

C_CHECKPOINT:
  task/researcher IDs
  full researcher message envelope
  evidence manifest
  query attempts
  close reason
  RNG/sampling envelope
```

Checkpoint 文件 content-addressed。Forked-state component trial 的所有 variant 必须从同一 checkpoint hash 启动。

checkpoint 是 reducer 之前的完整状态，不是只存 prompt。fork job key：

```text
hash(protocol_sha, boundary_id, variant_id, seed, prompt_renderer_version)
```

父对象 immutable；每个 fork 只写新的 content-addressed objects。

---

## 12. Telemetry 与完整 work accounting

### 12.1 每个 model request 的 op class

至少：

```text
PAGE_P0_SUMMARY
PAGE_P1_SELECTOR_LOCAL
PAGE_P1_SELECTOR_GLOBAL
RESEARCHER_REACT
COMPRESSOR_P0
COMPRESSOR_P1_SELECTOR
SUPERVISOR_CONTINUE
FINAL_WRITER
JUDGE_ATOMIZE
JUDGE_TRUTH
JUDGE_REPORT
```

Judge 不计入 treatment GPU work，但单独报告 API token/cost。

### 12.2 Request event

每条 request：

```text
run_id, attempt_id, task_id, arm_id, variant_id
node, op_class, researcher_id, query_id, checkpoint_hash
proxy_ingress_ts, upstream_dispatch_ts, upstream_response_end_ts, proxy_response_end_ts
engine_enqueue_ts?, first_token_ts?, engine_end_ts?
prompt_tokens, cached_prompt_tokens?, completion_tokens
finish_reason, retry_ordinal, terminal_status
prompt_sha256, output_sha256
GPU clock/temp/power/memory snapshots
CPU process time, RSS delta
```

带 `?` 的字段只有在 doctor 证明 vLLM 0.24 当前栈能可靠导出时才填；非流式响应不能伪造 first-token time，Prometheus histogram 不能反推逐请求 timestamp/cached tokens。`reports/ENVIRONMENT_MANIFEST.json` 记录 telemetry capability matrix 和每个字段的 source。

### 12.3 GPU/CPU 账

报告以下互不替代的量：

- prompt/completion tokens；
- cached prompt tokens；
- proxy service wall time；TTFT/TPOT 仅在可靠可观测时报告；
- stage critical-path time；
- vLLM throughput/queue metrics；
- NVML utilization、power、energy；
- process CPU-seconds；
- chunking/aggregation/render CPU-seconds；
- peak RAM；
- Tavily/DeepSeek calls 与 tokens；
- retry/fallback/restart cost。

禁止把 CPU 与 API 成本用一个未经批准的权重折成“GPU equivalent”。

本周不以不可观测的“逐请求 engine-busy seconds”作为 primary：

- isolated mode 强制单 upstream in-flight；`W_isolated` = 所有 treatment-model calls 的 `upstream_response_end - upstream_dispatch_ts` 之和，gateway queue wait 另报且不重复计入 work；
- 同时以 task-window NVML joules、token work 和 E2E wall 作 guards；
- operational mode 不强行把并发 GPU 时间归到单 request，primary 是固定 arrival trace 下完整 block makespan、throughput、GPU joules 和 tail latency；
- gateway 必须断言 upstream intervals 不重叠；一旦重叠，该 task/block work metric invalid；
- 若以后增加 vLLM request-level trace patch，它是 instrumentation extension，必须 parity-test；不能用 ingress wall time冒充 engine internal metric。

所有外部调用使用独立事务状态机：

```text
INTENT
  -> BUDGET_RESERVED
  -> SENT
  -> RESPONSE_STORED
  -> VALIDATED
  -> COMMITTED
```

- `SENT` 后 timeout 进入 `FAILED_UNKNOWN`，不能当作“没调用”；
- 429 尊重 `Retry-After`，5xx capped exponential backoff + full jitter；
- 400/401/402/422 fail fast；
- 不允许失败时静默切换 provider/model；
- DeepSeek 保存 requested/returned model、usage、provider request ID 与 `system_fingerprint`（若提供）；
- 一个 frozen stage 内 fingerprint/model 漂移则暂停并新建 provider epoch，不能直接合并。

### 12.4 Work balance

每个 task 至少输出：

```text
W_H_selector
+ ΔW_researcher_after_H
+ W_C_selector
- W_P0_page_summary_avoided
- W_P0_compressor_avoided
+ ΔW_supervisor
+ ΔW_final_writer
+ W_retry/fallback
+ CPU critical-path work
```

即时 selector savings 和 end-to-end total effect 必须分开。

---

## 13. TruthPacket 与质量评测

### 13.1 P0 不是 gold

所有 arm 针对同一个 frozen TruthPacket 评价，不能用 P0 summary/raw_notes 作为唯一真值。

```text
TruthPacket {
  task_id,
  required_facets[],
  atomic_evidence[],
  accepted_support_edges[],
  contradiction_pairs[],
  negative_evidence[],
  known_gaps[],
  critical_items[],
  citation_occurrences[],
  authoring_method,
  verifier_status,
  content_sha256
}
```

对每个 C checkpoint，evaluator 另建：

```text
VisibleTruthProjection {
  checkpoint_hash,
  visible_compressor_view_hash,
  projected_atoms[] {
    truth_atom_id,
    supporting_visible_span_ids[],
    projection_status: EXPLICITLY_VISIBLE | NOT_VISIBLE | AMBIGUOUS
  },
  projection_model_prompt_hash,
  audit_status,
  content_sha256
}
```

projection 只在 evaluator UID 中使用。它判断原 compressor 输入的 exact visible bytes 是否已明确承载某 truth atom；不把 raw truth bytes回写给 treatment。C direct-selector recall 的分母只包含 `EXPLICITLY_VISIBLE` atoms，回答“已经进入 compressor 输入的信息保留了多少”；`NOT_VISIBLE/AMBIGUOUS` 不惩罚 C reducer。完整 TruthPacket coverage 只用于 end-to-end final report，避免把上游网页摘要的信息损失归罪给 C。

### 13.2 自动 TruthPacket builder

由于 Week-1 需无人值守启动，可用 DeepSeek 产生 candidate packet，但必须：

1. 只读 frozen source pool；
2. atomizer 先产生候选 atomic items；
3. verifier 对每个 item 绑定 exact span IDs；
4. 任何无 exact span 的事实不得进入 accepted truth；
5. contradiction 两边均有 source span；
6. negative/gap 绑定 query attempt，而不是模型臆测；
7. JSON Schema validation；
8. 保存 prompt、model、response hash；
9. 标记 `MACHINE_CANDIDATE_PENDING_HUMAN_AUDIT`。

没有人类审核的 Week-1 质量结果必须在报告中写 `provisional`，不得伪装为论文级 gold。

随机 15–20% truth packets、所有 critical miss、judge disagreement 与 near-margin cases 自动进入 blinded human audit queue；无人值守阶段不伪造审核完成状态。

自动实验只能产出 `PROVISIONAL_*` 科学裁决，直到人工 audit gate 完成。将 `PROVISIONAL_KEEP/CONDITIONAL/KILL_HARM` 升级为不带前缀的最终裁决，至少要求：

- 随机分层审核样本达到预冻结比例；
- critical atoms/source edges 零错误；
- non-critical accepted atoms/source edges 审核正确率达到预冻结门（v0.1 建议 ≥95%）；
- judge/automatic metric 的关键 harm 漏检为零；
- 所有 audit corrections 以新 TruthPacket version 重算所有 arm，不只改 P1。

若 audit 未完成，报告仍能可靠交付结构错误、work/time/energy 和机器质量结果，但 proposal 决策必须标记 `PENDING_HUMAN_AUDIT / NO-GO-YET`。`KILL_NO_HEADROOM` 可基于纯 work 上界独立报告，但不得顺带声称质量已验证。

### 13.3 Selector-level metrics

- H：weighted raw-evidence recall；C_VISIBLE：weighted visible-projected-atom recall；
- macro facet recall；
- selected-token precision/efficiency；
- critical-item miss；
- contradiction-pair recall；
- negative/gap retention；
- invalid/out-of-set ID；
- citation-lineage closure；
- budget/preflight failure。

### 13.4 Supervisor trajectory

不得要求 P1 复现 P0 文本或完全相同轨迹。评价有限、可解释统计：

- waves；
- ConductResearch fan-out；
- generated query set；
- query novelty/redundancy；
- required facet follow-up；
- premature ResearchComplete；
- max-call/no-tool exit；
- key evidence first-discovery；
- final evidence coverage；
- repeated/failed retrieval。

### 13.5 Final report metrics

- atomic-claim factual correctness；
- claim grounding；
- required-facet coverage；
- critical evidence coverage；
- citation correctness；
- citation-claim association；
- citation coverage；
- contradiction handling；
- negative/gap honesty；
- overall blind pairwise preference；
- terminal failure/empty/truncated report。

实现必须生成 arm-blind `ReportClaim` ledger；不得只让 judge 输出一个总分：

```text
ReportClaim {
  claim_id, exact_text, factual_or_verifiable,
  linked_citation_occurrence_ids[],
  matched_truth_atom_ids[],
  support_status: SUPPORTED | CONTRADICTED | UNSUPPORTED | UNVERIFIABLE,
  critical_harm_flags[]
}
```

matching protocol：

1. 固定 atomizer 把 report 切成 atomic claims；
2. exact entity/number/date/source-edge rules先匹配；
3. 剩余 claim↔truth atom 由冻结的 arm-blind JSON judge判断 entail/contradict/unrelated；
4. `covered(atom)` 只有在 claim 语义覆盖且至少一个 linked citation occurrence 的 accepted source span 支持该 claim 时成立；
5. judge 不确定/invalid 不按成功插补，进入人工队列；
6. 全部 mapping、分母和 corrections versioned；同一 TruthPacket version 对所有 arm 重算。

每 task 的确定定义：

```text
weighted_required_atom_recall
  = sum(weight(atom) for required atom if covered)
    / sum(weight(atom) for all required atoms)

critical_atom_safety
  = fraction of critical atoms that are covered,
    or are TruthPacket-known-unresolved and explicitly reported unresolved

grounded_claim_precision
  = supported factual/verifiable claims / all factual/verifiable claims
  # denominator=0 => 0 for a nontrivial research task

citation_correctness
  = citations whose occurrence actually supports at least one adjacent claim
    / all emitted citations

citation_association
  = supported claim-citation edges / all emitted claim-citation edges

citation_completeness
  = supported factual claims with >=1 citation / all supported factual claims

required_facet_coverage
  = required truth facets with >=1 covered required atom
    or correct explicit unresolved marker
    / all required truth facets

contradiction_handling
  = truth contradiction pairs with both sides represented and uncertainty handled
    / truth contradiction pairs  # denominator=0 => NA
```

`critical_harm=1` 若出现任一：unsupported/contradicted critical factual claim、错误关键数值/实体、一边倒掩盖已知关键 contradiction、把 search gap 写成事实不存在、或 citation 明确反驳所绑 claim。

```text
qualified_report=1 iff
  nonempty and not truncated/terminal-failed
  weighted_required_atom_recall >= 0.85
  critical_atom_safety == 1
  grounded_claim_precision >= 0.95
  citation_correctness >= 0.95
  citation_association >= 0.90
  required_facet_coverage >= 0.90
  critical_harm == 0
```

`quality_pass` 就是 `qualified_report`；binary harm endpoint 是 `critical_harm`。pairwise preference、style 和 trajectory只作 secondary，不能替代上述可计算 endpoints。

### 13.6 Judge protocol

- arm/variant/path 全部盲化；
- A/B 顺序随机；
- order swap；
- judge 看共同 frozen sources，不看 arm 自己的 raw_notes；
- judge model/prompt/schema freeze；
- empty/invalid response 重试后仍失败则 `JUDGE_UNAVAILABLE`；
- 不把同一 judge 的两次调用当独立 rater；
- 生成随机审计样本、所有 disagreement、所有 critical miss、所有近 margin case 的人工队列。

---

## 14. 实验状态机与一周预算

### 14.1 状态

```text
NEW
  -> DOCTOR_PASSED
  -> ACQUISITION_COMPLETE
  -> SNAPSHOTS_FROZEN
  -> P0_PARITY_PASSED
  -> GPU_SMOKE_PASSED
  -> SCREEN_RUNNING
  -> SCREEN_COMPLETE
       -> MINI_ITT_RUNNING
       -> MINI_ITT_COMPLETE
       -> POLICY_FROZEN
       -> HOLDOUT_RUNNING
       -> HOLDOUT_COMPLETE
       -> LIVE_VALIDITY_OPTIONAL
     OR
       -> CHARACTERIZE_NO_GO_RUNNING
       -> CHARACTERIZE_NO_GO_COMPLETE
  -> REPORT_COMPLETE
```

另有 terminal：

```text
BLOCKED
COMPLETE_NO_GO
BUDGET_EXHAUSTED
ABORTED_SAFELY
```

所有 transition 由 SQLite transaction + immutable freeze record 驱动。

若 screening 早期显示所有设计都失败，只能 transition 到 `CHARACTERIZE_NO_GO_RUNNING`：继续运行预注册的 best-case、CPU/short-prose controls 与失败 strata，估计“即使最佳条件也无 headroom”或明确 harm boundary；完成后进入 `CHARACTERIZE_NO_GO_COMPLETE -> REPORT_COMPLETE -> COMPLETE_NO_GO`。`KILLED_ALL_P1` 不作为另一个冲突 terminal。

### 14.2 时间预算

```text
0–8h     doctor / acquisition / parity / smoke
8–40h    forked-state component screening
40–68h   mini end-to-end ITT + variance/power pilot
68–72h   automatic policy/envelope freeze
72–128h  untouched isolated holdout
128–152h independent operational TraceBlocks
152–160h preregistered rerun reserve / APC sensitivity / optional live validity
160–168h frozen analysis / integrity verification / final reports
```

按 144–150 个有效 GPU-hours 规划，保留其余时间给 engine restart、API acquisition、analysis 与故障。

### 14.3 Phase 0：doctor、acquisition、smoke

必须依次通过：

1. secret presence（只检查存在，不打印值）；
2. exact stack；
3. Tavily 2-query smoke；
4. DeepSeek JSON/empty-response smoke；
5. source pool freeze；
6. P0 parity；
7. one H checkpoint fork；
8. one C checkpoint fork；
9. P0/H/C 各一小任务；
10. telemetry completeness；
11. crash/resume smoke；
12. secret leakage scan。

### 14.4 Phase 1：component screening

对 SCREEN checkpoints 使用 balanced incomplete block：

- 每个 checkpoint 总有 P0；
- 从同 node 候选中分配 2–3 个 P1 variants；
- variant 在 evidence volume、facet、conflict/table/negative strata 中近似平衡；
- 同 checkpoint 内 execution order 随机；
- AB/BA/period 平衡；
- 每 6–8 小时插入固定 P0 sentinel。

Screening 只用于淘汰与选择，不作 headline。

初始容量目标：

```text
48 H boundary states + 48 C boundary states
每 state: P0 + 5 assigned P1/control variants
25% states: 预指定第二 seed，用于 selector stochasticity
```

若 profiling 显示超预算，减少 state 数或每 state variants 必须由 `block_design.py` 在 outcome 不可见时完成，不能删除表现差的 execution。

`block_design.py` 必须：

1. 枚举满足 constraints 的合法 variants；
2. 对 categorical factors做 effect coding；
3. design matrix 包含主效应和预注册交互：
   - chunker × materialized budget；
   - scope × aggregation；
   - output contract × aggregation；
4. 固定 ID-only、typed-coverage、bridge 三个 anchor；
5. 用 Fedorov coordinate exchange 最大化 `log det(X'X + 1e-8 I)`；
6. 再以 information gain 减 exposure/pair/stratum imbalance penalty 分配到 states。

freeze gate：

- design matrix full rank；
- condition number `< 1e4`；
- variant exposure 差不超过 1；
- factor-level standardized stratum imbalance `< 0.1`；
- pair exposure 最大差不超过 2；
- 每个 state 总含 P0；
- 分配器只读 pre-treatment strata，不可读 outcome。

### 14.5 Screening hard gates

以下任一成立立即淘汰 variant：

- out-of-set/dangling ID；
- wrong source/citation namespace；
- exact reconstruction 失败；
- critical item 系统性遗漏；
- contradiction/gap 无法表达且对应 stratum 有伤害；
- bridge 越界或产生无 evidence binding 的实体/数值；
- P0 fallback 语义错误；
- GPU work 的方向明显为正且无预注册质量收益；
- crash/timeout rate 超过 provisional guard。

Screening 可用 80% interval 做非绑定 futility/harm 判断；holdout 使用冻结的 95% interval。

这里的淘汰只裁决**具体 variant**，不能用 exploratory 80% interval 宣称整个 WEBPAGE/CLOSE 设计族 `KILL`。若一个 node 无 candidate 晋级：

- 全部合法 variants 共享同一个确定性协议/结构错误时，可裁决 `KILL_STRUCTURAL`；
- 否则从预冻结规则选一个 best-case family sentinel，进入 confirmatory no-headroom/harm block；
- 若时间不足以确认 sentinel，则 node verdict 是 `NOT_ESTABLISHED / NO_CANDIDATE_ADVANCED`，不是 KILL。

### 14.6 Variant promotion

Promotion 顺序：

1. 结构正确性；
2. selector absolute quality；
3. end-to-end harm screen；
4. complete work saving；
5. 简单性 tie-break。

每 node 最多晋级两个。若两个 candidate 的 utility 差在预冻结 indifference band 内，选择：

```text
ID < TYPED < BRIDGE
stable_union < coverage_budget < global_rerank
separate < fused-with-fallback
```

最后一项只用于 C_FUSED_EXT 内部比较；fused extension 不与 reducer-only candidate 用同一 tie-break 混选。其余情况选择更简单、自由度更小者。

### 14.7 Phase 2：mini end-to-end ITT

先将候选收敛为一个受限 3×3：

```text
H0 = off
H1 = best low-freedom WEBPAGE policy
H2 = best alternative contract/scope

C0 = off
C1 = best C_VISIBLE low-freedom policy
C2 = best C_VISIBLE alternative contract/close mode
```

在 POWER_PILOT dev tasks 上运行 H×C 九格的 balanced incomplete/完整区组（由实测 GPU-hour 决定），以辨别 interaction，而不是只跑看起来最好的组合。还至少保留：

- P0；
- H champion candidate(s)；
- C ID candidate；
- C TYPED/BRIDGE candidate；
- H+C best provisional composition；
- fused 与 separate 若 screening 无法裁决。

`C_REGISTRY` 不占用 primary 3×3 的 C1/C2 位置；它在独立 extension block 与同 checkpoint 的 `C_VISIBLE` 配对。CPU lexical 与 short-prose control 至少在 component screen 和一小组 mini-E2E task 上运行，回答“为什么赢”。

完整任务允许 trajectory 分叉。相同 task 使用相同 frozen source pool 与 sampling envelope；task×seed 配对，执行顺序 AB/BA 随机。

### 14.8 Automatic policy freeze

`promotion.py` 先生成 outcome-free tracked freeze：

```text
protocol/freezes/p1_holdout_v1.json
protocol/freezes/p1_holdout_v1.sha256
```

内容：

- champion H/C variant；
- chunker/budgets；
- prompt/schema hashes；
- aggregation；
- close mode；
- primary endpoints；
- NI margins；
- minimum meaningful work effect；
- strata；
- sample-size calculation；
- holdout task IDs；
- randomization schedule；
- software/data manifest hashes。

Freeze 后任何字段变化必须新建 protocol version，不得覆盖。该 JSON/sha 不是 generated report，必须：

1. 在 clean protocol worktree 中写入；
2. commit；
3. annotated tag message 同时记录 commit SHA、freeze SHA、ledger state hash；
4. coordinator 验证 checkout HEAD/tag/freeze 三者闭合后才请求 holdout gate release。

champion、holdout manifest hash、randomization、approved margins、analysis hash都必须在 tracked freeze 或其 content-addressed referenced objects 中；只有一个指向代码的 tag 不算 design lock。

本地至少创建两个不可移动的 annotated tags/records：

```text
freeze/p1-screening-v1
freeze/p1-holdout-v1
```

holdout 前同时冻结 analysis CLI 与 report schema；notebook 只能探索，不能作为权威分析入口。若 champion 或分析定义改变，创建新 protocol/tag，旧 freeze 永不改写。

若 promotion 不明确：

- 不打开 holdout；
- 将剩余预留 GPU 用于增加 POWER_PILOT paired tasks；
- 到 wall-clock deadline 仍不明确，则 `NOT_ESTABLISHED`，生成 no-go report。

### 14.9 Phase 3：honest holdout

若 H、C 都存活：

```text
2×2: WEBPAGE on/off × C_VISIBLE on/off
```

若只有一个 champion 正常晋级：

```text
P0 vs surviving champion
+ P0 vs non-advanced node 的 pre-frozen best-case sentinel（预算允许）
```

规则：

- HOLDOUT 只解封一次；
- topic/source cluster 是主要独立单位；
- repeated seeds 为 cluster 内重复；
- engine/cache protocol 相同；
- arms 按 Williams/Latin-square 或等价完整区组顺序交错，不得先跑完一整臂；
- timeout、fallback、empty report、judge unavailable 都保留在 all-offered ledger；
- holdout 不调 variant、threshold、budget、strata。
- 预计新 block 的 p90 完成时间若超过阶段剩余预算，不再 admission；禁止留下系统性不完整 arm。

Primary holdout 在 isolated causal mode 运行；冻结的 operational replication 使用同一 task subset 和预生成 arrival trace。报告不得把两种 mode 合池。最终部署 `KEEP` 要求 operational work/quality guards 同时通过；若只有 isolated mechanism 成立，输出 `MECHANISM_ONLY / PROPOSAL_NO_GO`。只有“isolated 单序列”本身被审批为目标部署 regime 且有明确 registry coverage 时，才可成为 deployment-conditional。

### 14.10 Phase 4：live-validity

只有 holdout 完成且剩余预算足够才运行：

- 小规模、方向性；
- 每 arm 真正调用 live Tavily；
- 记录时间与搜索漂移；
- 不用于重新选择 P1；
- 不与 frozen-corpus primary 混合。

---

## 15. 统计分析

### 15.1 三个 estimand

1. **Direct node effect**
   - 同一 H/C checkpoint fork；
   - 回答 immediate representation/work/quality。
2. **Frozen-source end-to-end ITT**
   - 允许 trajectory 分叉；
   - Week-1 的主要产品效应。
3. **Live-web external validity**
   - 仅描述性。

### 15.2 Primary decision

P1 通过必须同时满足：

1. 结构硬门；
2. selector absolute-quality 门；
3. final report 质量非劣；
4. all-offered complete work 明确下降；
5. failure/tail guard 不破；
6. 若声称 CONDITIONAL，frozen-registry eligibility coverage 达到唯一预注册门。

`configs/decision.yaml` 中将 provisional values 明确标为 v0.1；在 mini-ITT freeze 前锁死。第一版建议：

```yaml
structural:
  invalid_id_max: 0
  lineage_error_max: 0
  critical_item_miss_max: 0
selector:
  weighted_recall_min: 0.90
  contradiction_pair_recall_min: 0.85
  negative_gap_recall_min: 0.85
quality_ni_margin:
  weighted_required_atom_recall_pp: -5
  grounded_claim_precision_pp: -3
  citation_correctness_pp: -3
  citation_association_pp: -3
  required_facet_coverage_pp: -5
  qualified_report_rate_pp: -5
  critical_harm_risk_pp_max: 3
  terminal_failure_risk_pp_max: 2
utility:
  minimum_meaningful_work_reduction: 0.10
  preferred_strong_reduction: 0.20
coverage:
  conditional_task_exposure_coverage_lcb_min: 0.30
```

这些值是本次显式 auto-launch 指令采用的 protocol v0.1，不是 coding agent 可自行调节的默认。`protocol/launch_approval.json` 中的 `decision_thresholds_sha` 必须精确匹配。Automatic policy freeze 只能复制这些 margins，不能在 mini 结果后选择或修改它们。

### 15.3 Inference

- checkpoint screen：paired differences，checkpoint/task cluster bootstrap；
- mini/holdout：task/source-cluster bootstrap 或 mixed model；
- 2×2：估计 H main effect、C main effect、H×C interaction；
- binary endpoint：paired risk difference/GEE；
- continuous endpoint：paired mean/median、BCa CI；
- tail：task-cluster bootstrap 的 p90/p95/CVaR；
- 多次 seed 不增加独立 task n；
- quality 与 utility 为 co-primary guard，不能用一个补偿另一个。

holdout 前冻结 confirmatory contrast family：

```text
Primary:
  one champion policy (H, C_VISIBLE, or H+C_VISIBLE) versus P0

Secondary factorial family:
  H simple effect at C=off       = H - P0
  C simple effect at H=off       = C - P0
  joint effect                   = H+C - P0
  interaction                    = (H+C - H) - (C - P0)
```

Primary champion 只有一个，在任何 holdout outcome 前选定。质量 co-primary guards使用 intersection-union：全部 one-sided NI guards 通过才 PASS，不用平均分补偿，因此该组本身不靠多重检验“挑一个过”。四个 secondary contrasts/节点级 benefit claim 构成一个 family，用 Holm-FWER 0.05（或等价 simultaneous bootstrap CI）校正；未经校正的 secondary 只能写 descriptive。若 node 未进入相应 confirmatory contrast，它只能是 `NOT_ESTABLISHED` 或确定性的 `KILL_STRUCTURAL`。

Primary paired work estimand：

```text
R_work = exp(mean_task(log(W_P1 / W_P0)))
saving = 1 - R_work
```

其中 isolated `W` 是 gateway 强制单 upstream in-flight 后的 summed upstream dispatch-to-response service intervals；operational `W` 是固定 arrival trace 的 block makespan/throughput，不把并发 GPU 时间伪分到 request。E2E wall、GPU joules 和 token work 是共同 guards。还必须报告 median、p90、任务级节省 ≥10/25/33% 的比例、quality-qualified Pareto-win rate，以及“同时变慢且伤害质量”的比例。

最终机器判决细分为：

```text
KEEP:
  structural PASS
  AND isolated all-quality NI one-sided 95% guards PASS
  AND isolated W-saving LCB95 >= 10%
  AND operational block work-saving LCB95 >= 10%
  AND operational E2E wall and critical-quality guards PASS

THESIS_GRADE:
  KEEP AND E2E speedup LCB95 >= 1.5x

CONDITIONAL:
  overall KEEP fails
  BUT pre-frozen eligibility rule passes KEEP on holdout
  AND frozen-registry task-exposure coverage LCB95 >= 30%

MECHANISM_ONLY:
  isolated guards PASS
  BUT operational guards fail or are underpowered
  => deployment/proposal NO-GO

KILL_HARM:
  deterministic safety violation or quality harm crosses margin

KILL_NO_HEADROOM:
  quality feasible BUT work-saving UCB95 < 10%

NOT_ESTABLISHED:
  neither benefit nor no-headroom established
```

`THESIS_GRADE` 是 stretch label，不得倒逼修改实验。`NOT_ESTABLISHED` 对 proposal 是 NO-GO，但科学文字不能写成“已证明零效果”。

### 15.4 “什么时候有用”

eligibility 是**node invocation policy**，不是事后给整 task 贴标签。只允许该 boundary 在 selector 调用前可见的 features：

```text
WEBPAGE:
  candidate_evidence_tokens, source_count, span_count
  span_length_quantiles, table_list_fraction, redundancy
  visible_conflict_cue, raw_content_available_fraction

C_VISIBLE:
  visible_message_tokens, visible_tool_output_tokens
  tool_call_count, source_count, query_attempt_count
  table_list_fraction, redundancy, close_reason
```

在 SCREEN/POWER_PILOT 分别学习 H/C 的低自由度 rule；end-to-end policy 对每个 eligible invocation 执行 P1，其余 fallback P0。训练 label 只用同 checkpoint 可观测的 component outcome：

```text
local_quality_pass
  = structural PASS
    AND H raw-evidence recall / C VisibleTruthProjection recall passes

local_saving_pass
  = isolated paired upstream-service saving >=10%

training_success = local_quality_pass AND local_saving_pass
```

operational saving 不归因到 invocation，也不进入 CART label；冻结后的整套 node policy 在 independent TraceBlocks 上验证。

Week-1 没有概率抽样框，因此不得声称覆盖任意外部“目标人口”。唯一可识别的 coverage estimand 是：

```text
invocation_coverage(node)
= eligible node opportunities / all node opportunities

task_exposure_coverage(node)
= tasks with >=1 eligible invocation / all tasks offering that node
```

每个 topic/source cluster 等权，cluster 内 tasks 平分该权重；invocations 在 task 内再平分 task 权重。95% CI 用 task/source-cluster bootstrap。CONDITIONAL 的 30% 门针对 `task_exposure_coverage` LCB；同时强制报告 invocation coverage、每 task eligible count 和 all-offered policy effect。报告必须写成“本冻结 registry 的覆盖率”，外推到论文 workload 需要未来代表性 benchmark。

eligibility learner 冻结为：

- depth ≤2 CART；
- 每 leaf 至少来自 8 个 independent tasks，而非 8 个 correlated boundaries；
- 只用上列 features；
- design split 目标为上述 `training_success`；
- bootstrap rule-selection stability ≥70%；
- 选择目标是 `coverage × saving_LCB`；
- mini 后保存可读 rule、feature thresholds 和 hash；
- stability 不足时禁止 CONDITIONAL claim。

禁止：

- treatment 后 fallback/retry/reuse 作为 eligibility feature；
- 在 holdout 重新切点；
- 高维 subgroup fishing；
- 只报 eligible-only、不报 all-offered；
- 用 P0 实际输出长度作为部署时不可见的 oracle feature。

---

## 16. SQLite ledger、idempotency 与恢复

### 16.1 主键

```text
protocol_sha
split
phase_id
task_id
arm_id
variant_id
replicate_id/seed
checkpoint_hash
stage_version
```

以上字段组成 logical work key；`attempt_ordinal` 只区分重试。同一逻辑 run 只能有一个 terminal accepted attempt。重试产生新 attempt，不覆盖旧记录。

SQLite 至少有：

```text
runs, stages, config_versions
tasks, splits, variants
jobs, leases, attempts, transitions, incidents
external_calls, budget_reservations
artifacts, boundary_snapshots, forks
search_requests, search_snapshots, sources, occurrences
spans, selections, aggregations, preflights, publishes, fallbacks
node_invocations, llm_requests, tool_calls, metric_samples
evaluations, judgments
```

大 blob 不进 SQLite，进入 content-addressed object store；SQLite 只保存 identity/hash/size/location/关系。

### 16.2 Coordinator

```python
while budget.remaining():
    state = ledger.current_state()
    work = planner.next_idempotent_work(state)
    lease = ledger.claim(work, worker_id, ttl)
    try:
        result = execute(work)
        validate(result)
        ledger.commit_terminal(work, result)
    except RetryableError:
        ledger.record_failure(...)
        planner.retry_if_budgeted(...)
    except FatalProtocolError:
        ledger.block_protocol(...)
        break
```

### 16.3 Crash resume

- `PRAGMA journal_mode=WAL`、`synchronous=FULL`、`foreign_keys=ON`，单 writer queue；
- 每个 output 先写 temp，再 fsync/atomic rename；
- terminal record 最后提交；
- stale lease 可回收；
- engine/API/task retries 有独立上限；
- 重启后从 ledger 恢复；
- 不依赖 shell 当前目录；
- 每分钟 heartbeat；
- 每 10 分钟写 `reports/STATUS.json`；
- 不删除失败 attempt。

work item 状态至少为：

```text
PENDING -> CLAIMED -> MATERIALIZED -> VALIDATED -> COMMITTED
                    -> FAILED_RETRYABLE
                    -> FAILED_FINAL
                    -> FAILED_UNKNOWN
                    -> BLOCKED_BUDGET
```

resume 只能跳过“DB 为 COMMITTED 且 object hash 重验通过”的 item。`.tmp` 进入 quarantine；hash mismatch 是 corruption hard-stop，不能自动覆盖。SQLite 备份使用 backup API，不能裸复制 WAL 数据库。

实验恢复的最小单位是预注册 experimental pair/balanced block：

- 一个 arm 中途崩溃，整个 pair/block 标记 system-invalid；
- confirmatory 默认整 pair/block 在同一 engine/provider epoch 下重跑；
- 不允许把崩溃前 P0 与崩溃后 P1 拼成 paired observation；
- 补跑规则在 protocol freeze，不能看结果后决定；
- 原失败 attempt、成本和 artifacts 全部保留。

---

## 17. 自动启动、服务与 watchdog

### 17.1 `bootstrap_and_run.sh`

顺序必须是：

```bash
set -euo pipefail
resolve absolute repo path
verify no existing active coordinator
verify rotated credentials exist in approved backend without printing them
verify protocol budgets are explicit and approved
acquire singleton lock and one GPU UUID lease
verify campaign quota and free-space hard floor
git submodule update --init --recursive
verify vendor commit
materialize pinned submodule into .build/open_deep_research-patched
apply patch and verify patched-tree hash
uv sync --frozen
assert imported open_deep_research.__file__ is under the patched materialization
pytest -q
uv run shapeflow-p1 doctor --config configs/week1.yaml
uv run shapeflow-p1 prepare --config configs/week1.yaml
uv run shapeflow-p1 smoke --config configs/week1.yaml
uv run shapeflow-p1 preflight --approved-protocol-sha ...
uv run shapeflow-p1 run-week1 --config configs/week1.yaml --resume
```

任一步失败就写 BLOCKED report 并退出非零；不得 `|| true`。

`pyproject.toml`/`uv.lock` 使用 `.build/open_deep_research-patched` 的 explicit path source；因此必须先 materialize 再 `uv sync --frozen`。首次生成 lock 时也从同一 submodule commit + patch 构建。不得意外 import server 上全局安装或未 patch 的 vendor；import-origin assertion 和 patched-tree hash 是 launch hard gate。

### 17.2 后台运行

正式 unattended live run 使用 systemd units：

```text
shapeflow-api-provider.service
shapeflow-vllm.service
shapeflow-p1-week1.service
shapeflow-holdout-gate.service
```

有管理员权限时使用 system-level 独立 UID；否则使用 user-systemd process-only isolation并在报告中降级标注。若完全无可用 systemd，写 `BLOCKED_NO_UNATTENDED_SUPERVISOR`；tmux 只允许交互 debug/smoke，不得作为 168h fallback。不得使用裸 `nohup`。其他 supervisor 只有通过“杀 coordinator 后自动 `--resume`、三次 crash 后停止、ledger 不重复提交”的同等 fault test才可自动替代。

Service 配置：

- `Restart=on-failure`；
- `RestartSec=30`，30 分钟最多三次；超过后 `BLOCKED_REPEATED_CRASH`；
- working directory 绝对路径；
- provider unit 使用 `LoadCredentialEncrypted=` 或独立 UID 可读的 root-owned credential；
- `UMask=0077`、`NoNewPrivileges=true`、`PrivateTmp=true`；
- singleton `flock`、精确 `ReadWritePaths=`；
- stdout/stderr 经 redaction 后写日志；
- `ExecStop` 调用 `stop_safely.sh`；
- 不允许多 coordinator。

启动成功的判据不是“拿到 PID”，而是：

1. vLLM readiness 通过；
2. coordinator heartbeat 正常；
3. SQLite 中第一条真实 work item 已落库；
4. `LAUNCH_GATE_PASSED.json` 记录每个 gate 的证据与 protocol SHA。

### 17.3 GPU sentinel

每 15–30 秒记录 GPU/host heartbeat；每 6–8 小时运行固定 P0 sentinel。记录：

- latency；
- tokens/s；
- GPU clock/temp/power；
- model/engine process identity；
- APC state。

GPU 监控还记录 UUID、compute PID、utilization、memory、temperature、power、clocks/throttle、host RAM/swap、disk/inodes。低 GPU utilization 只有在声明为 GPU-active 的 stage 才可判 hang；等待 API 时属于正常。

若相对冻结 baseline 漂移超过 config guard：

- 暂停新 work；
- 重启 engine；
- 重跑受影响的整对/block；
- 不只补表现较差的 arm；
- 记录 `PERIOD_INVALIDATED`。

### 17.4 自动停止

以下情况立即停止：

- GPU ECC/Xid/driver error；
- OOM 连续超过阈值；
- engine/model revision 漂移；
- secret leakage detector 命中；
- snapshot/freeze hash 改变；
- P0 parity 失败；
- all P1 已达到预注册 kill-evidence sufficiency，且 `CHARACTERIZE_NO_GO` 队列完成或不存在合法 best-case；
- wall-clock/GPU/API budget 用尽；
- SQLite integrity failure；
- holdout 被提前读取。

SIGTERM 时依次停止 admission、标记当前 attempt、flush ledger/artifact、终止本 run 的 cgroup/process group、在 timeout 内退出。不得影响共享服务器上的其他进程。

---

## 18. 测试与 acceptance gates

### 18.1 Unit tests

- canonical JSON/hash；
- secret redaction；
- snapshot/occurrence identity；
- chunk offsets/reconstruction；
- table/list chunking；
- selector schema；
- invalid ID/preflight；
- deterministic aggregation/order；
- duplicate bytes + distinct citations；
- gap/query attempt；
- bridge token/binding；
- randomization determinism；
- budget accounting；
- freeze hash；
- ledger idempotency。

### 18.2 Property tests

至少：

- 任意选中 span 可精确重建；
- selector 输出永远是 candidate subset；
- aggregate 后 citation occurrence 不丢；
- 相同输入/config 产生相同 IDs/order；
- P1-ID 不含自由文本；
- P1-BRIDGE 每句有 evidence binding 且不过 cap；
- crash 任意发生点恢复后不重复接受 attempt；
- fake secret 不进入任何 artifact。

### 18.3 Integration tests

- fake Tavily acquisition；
- frozen search 无网络；
- fake OpenAI-compatible model；
- P0 parity fixture；
- H checkpoint fork；
- C checkpoint fork；
- all three close reasons；
- P0/H/C/H+C tiny task；
- DeepSeek empty JSON/retry；
- vLLM kill/restart；
- SQLite crash resume；
- API budget exhaustion；
- holdout access guard。

### 18.4 Fault-injection gates

正式长跑前必须自动注入并断言：

- Tavily 429 + `Retry-After`、500、malformed JSON、timeout-after-send；
- DeepSeek 401/402/422 fail-fast、429/500/503 backoff、empty JSON；
- 第 N 个 dispatch 前预算耗尽；
- artifact 写一半 SIGKILL；
- response 已落盘但 DB 未 commit 时 SIGKILL；
- DB commit 后、report 前 SIGKILL；
- paired block 的一个 arm 中途崩溃；
- corrupt cached response / replay miss；
- disk full/free-floor crossing；
- vLLM OOM、foreign GPU process、repeated service crash；
- fake secret 进入 header、body、exception、child environment。

每项必须证明：不越预算、不泄密、不覆盖旧 attempt、不重复接受已提交 work、不把 incomplete pair 当有效样本、恢复后 hash 可复验。

### 18.5 Real GPU smoke

在大 run 前：

- 2 tasks；
- P0/H/C；
- 1 normal page、1 no-raw-content page；
- 1 natural close、1 forced close；
- APC cached tokens 可见；
- telemetry completeness 100%；
- raw output/metrics/report hashes闭合；
- 无 secret。

---

## 19. CLI 合同

必须提供：

```bash
uv run shapeflow-p1 doctor --config configs/week1.yaml
uv run shapeflow-p1 acquire --config configs/week1.yaml
uv run shapeflow-p1 freeze-snapshots --config configs/week1.yaml
uv run shapeflow-p1 build-truth --config configs/week1.yaml
uv run shapeflow-p1 test-p0-parity --config configs/week1.yaml
uv run shapeflow-p1 smoke --config configs/week1.yaml
uv run shapeflow-p1 run-screen --resume
uv run shapeflow-p1 analyze-screen
uv run shapeflow-p1 run-mini-itt --resume
uv run shapeflow-p1 freeze-policy
uv run shapeflow-p1 run-holdout --resume
uv run shapeflow-p1 run-live-validity --resume
uv run shapeflow-p1 report
uv run shapeflow-p1 status
uv run shapeflow-p1 verify-artifacts
```

`acquire/build-truth/freeze-snapshots` 由 steward UID 执行；`run-*` 由 runner UID；quality evaluation/report 的 truth-reading部分由 evaluator UID。CLI 必须检查 effective UID/peer role，不能只靠文档约定。

pre-launch diagnostic CLI 可支持：

```text
--dry-run
--budget-hours
--max-tasks
--phase
```

这些 override 只能生成 profile/smoke，artifact 强制标 `DIAGNOSTIC_NON_PROTOCOL`。一旦 `LAUNCH_GATE_PASSED.json` 存在：

- mutation CLI 只接受 `--resume --protocol-sha <exact>`；
- `--budget-hours`、`--max-tasks`、`--phase` 一律拒绝；
- 紧急停止用单独 `stop-safely`，不修改 protocol；
- 任一预算、样本、phase/order 变更都生成新 protocol SHA，使旧 approval 失效并重新审批；
- 即便只收紧预算也保留旧 run，另建新 run/estimand，不在原 SHA 下静默改变 planned work。

`bootstrap_and_run.sh` 的正式路径不使用 `--dry-run`，并把 exact approved protocol SHA 传给每个 mutation command。

---

## 20. 自动报告与图表

### 20.1 必交报告

```text
reports/ENVIRONMENT_MANIFEST.json
reports/ACQUISITION_REPORT.md
reports/P0_PARITY_REPORT.md
reports/GPU_SMOKE_REPORT.md
reports/VARIANT_SCREEN_REPORT.md
reports/MINI_ITT_AND_POWER_REPORT.md
reports/P1_POLICY_FREEZE.json
reports/HOLDOUT_REPORT.md
reports/LIVE_VALIDITY_REPORT.md               # 若运行
reports/HUMAN_AUDIT_QUEUE.csv
reports/WEEK1_P1_DECISION.md
reports/WEEK1_P1_DECISION.json
reports/OPERATIONS.md
reports/COST_LEDGER.csv
reports/COST_RECONCILIATION.md
reports/FAILURE_AUDIT.md
reports/DATA_INTEGRITY.md
reports/REPRODUCIBILITY.md
reports/SECURITY_AUDIT.md
reports/ARTIFACT_MANIFEST.json
reports/MANIFEST.sha256
reports/STATUS.json
exports/task_level_results.parquet
exports/request_level_accounting.parquet
```

### 20.2 最终图

- P1 design-space Pareto frontier；
- H/C/H+C end-to-end effect；
- work waterfall；
- selector recall vs selected-token budget；
- final quality NI panel；
- citation correctness/association；
- trajectory shifts；
- harm/fallback/failure；
- eligibility coverage；
- benefit surface；
- GPU sentinel/time drift。

### 20.3 `WEEK1_P1_DECISION.md` 固定结构

Markdown 与 `WEEK1_P1_DECISION.json` 必须从同一个 typed decision object 生成。JSON 至少包含：

```text
WEBPAGE_P1: KEEP | CONDITIONAL | MECHANISM_ONLY | KILL_STRUCTURAL | KILL_HARM | KILL_NO_HEADROOM | NOT_ESTABLISHED
C_VISIBLE: ...
C_REGISTRY: ...
H_PLUS_C_VISIBLE: ...
verdict_status: PROVISIONAL_MACHINE | HUMAN_AUDIT_COMPLETE
champion_variant: exact full specification
eligible_coverage + CI
quality_effects + CI
engine_work/wall/energy effects + CI
critical_harm_rate
where_it_works
where_it_fails
confirmatory_power_shortfall
human_audit_status
```

1. Executive verdict；
2. Scope and frozen stack；
3. What was actually run；
4. P1 design variants；
5. WEBPAGE-P1 verdict；
6. C_VISIBLE verdict；
7. C_REGISTRY extension verdict；
8. H×C interaction；
9. Average/median/tail effect；
10. Quality and critical harms；
11. Work balance；
12. Eligibility envelope and coverage；
13. Failure/fallback accounting；
14. Sensitivity and live validity；
15. Human-audit status；
16. Decision for ShapeFlow proposal；
17. Exact limitations；
18. Reproduction commands；
19. Artifact hashes。

不得只写“P1 平均更快”。必须给：

```text
是否有用
多有用
何时有用
覆盖多少任务
哪个变体有用
质量差多少
最坏情况下发生什么
全成本是否仍然节省
结论是否已有人类审核
```

---

## 21. Git 交付节奏

建议本地 commits：

1. `scaffold standalone p1 study repository`
2. `pin odr and add strategy hooks`
3. `add tavily snapshot acquisition and frozen search`
4. `add evidence ir chunkers and lineage`
5. `add p1 selectors contracts aggregators`
6. `add checkpoint fork and odr arms`
7. `add telemetry and work ledger`
8. `add truth packets judges and quality metrics`
9. `add experiment coordinator promotion and freeze`
10. `add unattended runner watchdog and reports`
11. `freeze week1 protocol and start run`

每个 commit 前运行相应 tests。不得提交：

- API keys；
- `.env`；
- model weights；
- raw Tavily pages；
- run DB/logs；
- judge raw private credentials；
- unreviewed generated reports。

不自动 push。

---

## 22. Definition of done

Coding 阶段完成必须同时满足：

- 独立 repo 已初始化；
- vendor commit 正确；
- tests 全通过；
- secret scan 通过；
- P0 parity 通过；
- Tavily snapshots 已冻结；
- TruthPacket pipeline 可运行；
- H/C checkpoints 可复放；
- P0/H/C/H+C tiny E2E 可运行；
- telemetry 完整；
- coordinator 可 crash-resume；
- freeze/holdout access guard 有测试；
- bootstrap 自动启动 Week-1；
- status/report 实时生成；
- Git worktree 仅含预期 code/config/schema 变更。

实验阶段完成必须满足：

- 每个逻辑 run 在 ledger 中有 terminal 状态；
- 所有失败/all-offered 都入账；
- screen 与 holdout 数据未泄漏；
- policy freeze hash 存在；
- final report 与 JSON decision 一致；
- artifact verifier 通过；
- 结论明确到 H、C、H+C 和 champion；
- limitations 明确；
- 未将 machine-only judge 写成人类真值。
- 所有 planned work item 恰好处于一个 terminal state；
- remote/GPU/wall/disk budgets 均未越界；
- secret scan 为零；
- 报告可在断网状态下由冻结 artifacts 一条命令重建。

---

## 23. Coding agent 开始执行时的第一批动作

1. 在新目录创建 repo，确认不在 drbat/drbo 内；
2. 写 `.gitignore`、`.env.example`、`AGENTS.md`；
3. pin ODR submodule；
4. 写 schemas 与 canonical hashing；
5. 先实现 secret redaction 和 ledger，再接外部 API；
6. 实现 Tavily acquisition/freeze；
7. 实现 P0 hook parity；
8. 实现 H/C checkpoint；
9. 实现最小 P1-ID + stable union；
10. 打通 P0/H/C tiny E2E；
11. 再扩 TYPED/BRIDGE/hierarchical/fused；
12. 最后实现 coordinator、promotion、holdout guard 和 auto report；
13. 全部 acceptance gates 通过后执行：

```bash
./scripts/bootstrap_and_run.sh
```

Coding agent 必须从父任务提供的 credentials 通过 secure runtime injection 自动配置 provider；不再等待用户醒来确认。只有运行环境确实不提供任何不落盘/不入命令行的注入通道时，才停在 `BLOCKED_NO_SECURE_SECRET_INJECTION`；仍不得把明文写进仓库或日志。
