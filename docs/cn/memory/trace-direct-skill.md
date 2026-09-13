# Trace → Skill 直连规格

新建一条 **RawTurn → Skill** 链路，不经过 L2 诱导，也不用 L1 summary 当生成正文。

**旧链原样保留：** L1 → L2 → `skill_crystallization` 的触发、门槛、证据打包一律不改。新链只在 reward 之后**并行加** assign / evolve，不替换、不短路 L2 毕业 Skill。

两条链可能各写一份 Skill（`source` 不同：`worker.skill_crystallization.v7` vs `worker.skill_batch_evolve.v1`）。第一期不去重、不合并。新链只认自己簇上的 `skill_memory_id`，不把 L2 结晶出来的 Skill 当成「簇上已有」。

阈值沿用现有配置，不另起一套：

| 含义 | 配置 | 默认 |
|---|---|---|
| 成功 episode | `skill.outcomeRTaskSuccessThreshold` | `rTask >= 0.5` |
| 失败 episode | `skill.outcomeRTaskFailureThreshold` | `rTask <= -0.15` |
| 中间 / 未知 | 上两者之间，或没有 `rTask` | 不进 steps，也不当失败锚 |

`rTask` 以 episode 关闭后 **reward 写回值为准**。未打分的 episode 不进本链。

---

## 1. 对象

| 对象 | 粒度 | 本链角色 |
|---|---|---|
| RawTurn | 一轮 QA（用户 + 本轮全部工具 + 回复） | **唯一生成证据** |
| Episode | 一次任务，含多轮 RawTurn | 聚类与成败单位 |
| L1 | 默认同一轮 QA 一条 | 只做索引 / embedding 辅特征，不进抽取正文 |
| L2 Policy | 单步经验 | 旧链照常写、照常触发结晶；**新链不读、不写** |
| SkillCluster | 同工作流的 episode 集合 | 一簇一份 Skill + 一份 meta-skill |
| Skill | 现有 `procedure_json` + `invocation_guide` | 本链唯一产物 |

簇上另存（不进检索、不注入执行 agent）：

- `skill_memory_id`：该簇当前那一份 Skill（可空）
- `meta_skill_md`：给下一轮抽取器的指导
- 不做 `rejected_buffer`（第一期；拒因写进 meta-skill）

### 「簇上已有 Skill」怎么判断

**看簇记录，不看 top6 episode 有没有挂 skill。**

```text
已有 Skill  ⟺  cluster.skill_memory_id 非空
              且该行仍是 layer=Skill、未 archived
```

- **有** → `skill_batch_evolve` 走 rebuild，更新这一份。
- **无**（含：从未结晶、只有失败被跳过、Skill 已归档）→ 走 crystallize；没有成功锚则仍不建。

`episode.skillMemoryIds` 只是写回后的反链，用来在面板/检索上从任务点到 Skill。**不能当判定**：

| 若按 top6 的 episode 反链判断 | 会错成 |
|---|---|
| 新 episode 刚入簇，自己还没挂过 Skill | 误判「无 Skill」，再结晶出第二份 |
| 历史集挂过旧 L2 结晶的 Skill | 误判「有」，rebuild 错对象 |
| 只有失败、从未 upsert | 反链为空（这点碰巧对），但判定源仍应是簇字段 |

top6 只决定 **这轮用哪些 RawTurn 当证据**，不决定簇上有没有 Skill。

第一次结晶成功后：`cluster.skill_memory_id = skillId`，并给本批 episode 写 `skillMemoryIds`。之后无论新来的集有没有反链，都算已有 Skill。

---

## 2. 总流程

```text
turn.complete
  → RawTurn 落库、L1 照常捕获（可摘要 / embedding）
  → Episode 关闭
  → reflection
  → reward（写入 episode.rTask）
       ├─ 旧链（不动）：l2_association / l2_induction / skill_crystallization / negative_experience
       └─ 新链（并行，每个 episode 只入队一次）：
            skill_cluster_assign
              → skill_batch_evolve（簇变脏时）
```

`finalizeClosedEpisode` 不改。新 job 挂在 **reward 成功写完 `rTask` 之后**，与 L2 并行，不替代 reflection / reward。

去重：`skill_cluster_assign` 按 `episodeId`；`skill_batch_evolve` 按 `clusterId`（cooldown 内合并）。

```text
                    ┌─ success ─────────────────────────────────────────┐
                    │                                                   ▼
reward(rTask) ─► assign ─► cluster ─► 选 top6 RawTurn batch ─► evolve
                    │                                                   ▲
                    └─ failure ─────────────────────────────────────────┘
                                      （同一簇，分析时再按 outcome 切开）
```

成败 **不拆成两份 Skill、也不拆成两个簇**。同一簇、同一份 `procedure_json`，写入字段不同。

---

## 3. `skill_cluster_assign`

输入：刚打分的 episode `E`（`rawTurnIds`、`rTask`、`userId`、可选 `projectId`）。

### 3.1 特征（只从 RawTurn 算）

```text
EpisodeSkillFeatures
  tools         本集全部 tool name
  artifacts     路径 / 交付物扩展名（xlsx, pptx, pdf, csv…）
  query_vec     第一条用户任务句的 embedding（不用 L1 summary 向量当主特征）
  tool_bigrams  相邻工具对
  scope         userId + 可选 projectId
  outcome       success | failure | unknown
```

L2 的 signature / `l2PolicyIds` 最多当召回先验，**不当分桶键**。

### 3.2 指派（两段：粗簇 + 细簇）

在同 scope 已有簇上逐个比，**不把工具 Jaccard 和向量加成一个分数**：

1. **粗簇（规则硬门）**：`Jaccard(tools)`、artifact 重叠过地板。过不了的族直接排除（避免「都会 bash」并成一坨，也避免 ppt 进 xlsx 族）。
2. **细簇（向量硬门）**：只在过了粗门的候选里比 `cosine(query_vec, centroid)`。
   - 有向量：`cosine ≥ τ_fine`（默认 `clusterJoinThreshold = 0.5`）才加入该细簇；否则在该族下新开细簇。
   - 没有向量：停在粗簇，一族一份（加入粗分最高的已有簇）。
3. 两边都空：没有粗门，只比向量，`τ_empty` 更严（默认 `0.7`）。
4. 否则新开簇（单集允许）。

Skill 挂在**细簇**上，一细簇一份 SOP。成败不参与分簇。

**工具 / 产物为空时（规则，不调 LLM）：**

| 本集 | 已有簇 | 怎么比 |
|---|---|---|
| 有工具，无产物 | 有工具 | 只比工具 Jaccard + `query_vec`；产物门跳过 |
| 无工具，有产物 | 有产物 | 只比产物重叠 + `query_vec`；工具门跳过 |
| 两边都空 | 两边都空 | **没有硬门**，只比 `query_vec`，`τ_join` 更严 |
| 本集无工具 | 簇里有工具 | **不加入**（「帮我看看表」不能进 excel 簇） |
| 本集有工具 | 簇是纯对话 | **不加入** |

两边都空的成功集可以自己成簇、也可以结晶（`tools=[]` 合法）。纯闲聊、没打分或 unknown，仍然不 evolve。

评测可跳过自动指派，改用冻结簇（如 KW v2.1 `transfer_anchors`）。Test episode **不入簇、不 evolve**。

指派完成后：`Episode.meta.skill_cluster_id = cluster.id`，入队 `skill_batch_evolve`。

---

## 4. `skill_batch_evolve`：组 batch

```text
members = cluster 内 closed 且已有 rTask 的 episode
success = rTask >= 0.5
failure = rTask <= -0.15
unknown = 其余（可进检索旁注，不进 steps / 不当失败锚）
```

超过 6 个时取 **top 6**：

1. 最新成功最多 3
2. 最新失败最多 3
3. 不足则用另一侧补齐，再按时间补

证据：`listRawTurnsByEpisode`，按时间排序。

```text
{
  episode_id, r_task, outcome,
  turns: [{
    user, assistant, reasoning,
    tools: [{ name, input, output, success }]
  }]
}
```

- 不把 L1 `summary`、L2 `procedure` 当正文。
- 单条 tool output 可按条 clip（建议 2–4k），**先不摘要再抽**。
- `EVIDENCE_TOOLS` = 本批 RawTurn 里出现过的工具名。

---

## 5. 成功 / 失败写进 Skill 的哪一块

同一份现有结晶合同：

```json
{
  "name", "retrieval_blurb", "trigger_context", "summary",
  "parameters", "preconditions", "steps",
  "examples", "tools",
  "decision_guidance": { "preference", "anti_pattern" },
  "tags"
}
```

| 字段 | 成功 episode | 失败 episode |
|---|---|---|
| `steps` / `tools` / `parameters` / `examples` | 可写（必须能指回 RawTurn） | **禁止**（失败过程不当 SOP） |
| `decision_guidance.preference` | 可写「做对时坚持的做法」 | 不写 |
| `preconditions` / `decision_guidance.anti_pattern` | 一般不写 | **只准写这里** |
| `retrieval_blurb` / `trigger_context` | 用成功侧的真实用户说法 | 可补「何种请求会踩坑」，不代替成功 trigger |

合并规则（SkillOpt / Trace2Skill 的降配）：**失败修正优先**。成功 steps 与失败约束冲突时，留约束，改步骤或加 precondition，禁止删掉 anti_pattern 去迁就一次成功。

---

## 6. 完整分支

先按簇上有没有 Skill，再按本批 outcome。

### 6.1 簇上还没有 Skill → crystallize

System = 现有 `SKILL_CRYSTALLIZE_PROMPT` 的 JSON 合同 + **prepend 该簇 meta-skill**（没有则用全局种子）。
`POLICY` 为空。证据为 RawTurn batch，每条带 `outcome` / `r_task`。

| 本批 | 动作 | 写出 | 硬门不过 |
|---|---|---|---|
| **至少 1 个成功**（可另有失败） | 一次结晶 | 成功 → steps/tools/parameters/examples；失败 → preconditions + anti_pattern | 不建 Skill；拒因写入 meta-skill |
| **只有失败** | **不结晶 SOP** | 不建 Skill。簇保留成员与失败证据，只更新 meta-skill（「这类失败缺什么步骤」） | — |
| **只有 unknown** | 跳过 | 不建 Skill | — |

「只有失败不建 Skill」对齐现有 `hasPolicySuccessAnchor`：没有成功锚点不当可调用 SOP。失败仍留在簇里，等后续成功到来再结晶。

有成功又有失败、且 batch ≥ 4 时，允许两次分析再合并（失败一份约束、成功一份步骤，失败优先合成一份 JSON）。1–2 条则一次 LLM，靠字段规则切开即可。

### 6.2 簇上已有 Skill → rebuild（不看 L2 hash）

改写幅度 **只看本批新 episode 的成败**（新 = 尚未进入该 Skill 的 `evidence_anchor_ids` / 簇已处理集合）。

| 新证据 | `rebuild_scope` | 允许改的字段 |
|---|---|---|
| 无实质新信息 | `retrieval` | `retrieval_blurb`, `summary` |
| **主要是新失败** | `constraints` | `preconditions`, `decision_guidance.anti_pattern`；steps 实质不动 |
| **有新成功** | `workflow` | **改 `steps`**；必要时同步 `tools` / `parameters`。已有 anti_pattern 不得无故删。工具集合没变也要改步骤——同一簇里新题可以是另一种 SOP |

对应到旧 rebuild 名字（仅便于对照，实现请用 `rebuild_scope`）：

- `retrieval` ≈ 旧 L0
- `constraints` ≈ 旧 L1（外科手术，但按失败而不是「新 L1 条数」）
- `workflow` ≈ 旧 L2（允许改步骤；触发条件是本批有新成功，不是 policy hash，也不是工具 Jaccard）

更新方式：第一期仍吐整份 crystallize JSON + `changed_sections`，按 `rebuild_scope` 做字段合并。约束列表有硬顶（`preconditions` ≤ 12，`anti_pattern` ≤ 5），失败优先：

- `constraints`：本批 draft 非空则 **整表替换** 旧 `preconditions` / `anti_pattern`，不做并集。draft 为空则保留旧表再截断。
- `workflow`：`unique(draft + 旧表)`（新失败在前）再截断；仍不得无故丢掉仍有效的 anti_pattern。
- 前置条件只写簇级纪律。实例 id、路径、单元格、commit、工单号进 `parameters` / `examples`。抽取器 prompt 不写某个 bench 的评分细则；`data_only` / patch 测试等以 EVIDENCE 里的 verifier 为准。

传给 LLM 的是 **旧 Skill 快照 + 本批新 episode 的 RawTurn**（成功写步骤，失败只当约束），不是整簇重放、也不是 L2。`name` 默认锁死。

---

## 7. 逐条链路（成功 / 失败）

### 7.1 成功 episode（`rTask >= 0.5`）

```text
E_success 关闭并打分
  → assign：按 RawTurn 特征加入或新开簇 C
  → evolve：
      打包 C 的 top6 RawTurn（成功优先占位，失败仍带上）
      若 C 无 Skill 且本批有成功
        → crystallize（meta-skill prepend）
        → 硬门
        → upsert Skill，`evidence_anchor_ids` **只挂成功 episode**
        → E.skillMemoryIds += skillId
        → 更新 C.meta_skill（这次哪些具体约束写进了 steps）
      若 C 无 Skill 且本批没有成功
        → 不会走到这里（本集是成功）
      若 C 已有 Skill
        → 有新成功 → rebuild_scope=workflow
        → 只有已处理过的证据 → retrieval
        → 硬门 → 过则覆盖 Skill（保留 name）
        → 更新 meta-skill
```

成功 episode 的 RawTurn 是 **steps 的唯一合法来源**。

### 7.2 失败 episode（`rTask <= -0.15`）

```text
E_fail 关闭并打分
  → assign：与成功同一套特征，进同一工作流簇（不成「失败专用簇」）
  → evolve：
      打包 top6（失败占位，成功若有也带上）
      若 C 无 Skill 且簇内仍无任何成功
        → 不 upsert Skill
        → 只更新 meta-skill：失败机制、缺的前置、下次成功结晶时必须写进 anti_pattern 的点
      若 C 无 Skill 但簇内已有历史成功（本批 top6 能带到）
        → 走 6.1「至少 1 个成功」：成功写 steps，本集失败写约束
      若 C 已有 Skill
        → rebuild_scope=constraints
        → 用本批约束表替换 preconditions / anti_pattern（最多 12 / 5）
        → 硬门（新 steps 不得抄失败过程）
        → 过则更新 Skill；不过则保留旧 Skill，拒因写入 meta-skill
```

失败 **可以入簇、可以更新已有 Skill 的边界，不能单独把失败过程结晶成 SOP**。

### 7.3 同簇一成一败（最常见的「该分源」情况）

```text
C = { E_ok, E_fail }
  → top6 两集都进
  → 分析：成功 → 可执行步骤 / 工具参数 / 验收
          失败 → 约束 / anti_pattern（失败优先）
  → 一份 procedure_json
  → 硬门：steps 能指回 E_ok 的 RawTurn；anti_pattern 能指回 E_fail；
          E_fail 的命令序列不得出现在 steps
  → upsert 同一份 Skill；`evidence_anchor_ids` / 反链只挂成功 episode。失败只进分析，不挂 evidence
```

### 7.4 中间分（`-0.15 < rTask < 0.5`）

入簇（特征仍有用），evolve 时标 `unknown`：不当成功锚、不当失败锚。已有 Skill 时最多触 `retrieval`。

---

## 8. Meta-skill

概念借自 **SkillOpt 的 optimizer-side `m_meta`**：只写给未来抽取器，不随 Skill 发给执行 agent。  
不借：等 epoch≥2 才更新、用 \(D_{sel}\) 上「同一批题新旧 skill 对照」、rejected-edit 列表。我们按 **每次 `skill_batch_evolve` 结束** 更新，对照的是本簇这一轮前后的 `procedure_json` / 硬门结果。

每簇一份 `cluster.meta_skill_md`。下一轮 crystallize / rebuild /（可选）analyst 的 system **前面 prepend 它**。不进 Skill 检索。

### 8.1 种子（簇刚创建、还没有 evolve 过）

不是 SkillOpt 仓库里的某一份运行时 meta。是我们按它的职责写的 **抽取纪律**，钉在 memmy 的 `procedure_json` 上：

- 有用的 skill：执行 agent 按 steps 能复现成功轨迹里的关键动作和验收，而不是复述题面。
- `steps.body` 必须落到本批成功 RawTurn 的命令、列/表、验收、失败恢复。
- 禁止套话：「仔细检查 / 验证结果 / 按需处理」。这类不算有用 skill。
- 一次性文件名进 `parameters`，不写进 `name`。
- 失败只进 `anti_pattern` / `preconditions`，不进 steps。
- 抽象停在可再执行的程序，不要停在「用表格工具处理数据」。

第一次结晶就用这份种子。只有失败、没建 Skill 时，evolve 结束仍跑一次 meta 更新（记下失败机制），供以后有成功锚时用。

### 8.2 什么时候更新

`skill_batch_evolve` **每次跑完都更新**（收下或拒绝都更新）。  
不更新：assign 之后判定簇未变脏、cooldown 跳过 evolve。

### 8.3 怎么更新

一次小 LLM 调用，合同对齐 SkillOpt `meta_skill.md`：对 **future optimizer** 说话，禁止输出给执行 agent 的任务 SOP。

输入：

| 字段 | 内容 |
|---|---|
| `previous_meta` | 更新前的 `meta_skill_md`（第一轮是种子） |
| `skill_before` | 本轮 evolve 前的 `procedure_json`（无 Skill 则为空） |
| `skill_after` | 过硬门后的新 JSON；被拒则为候选草稿或空 |
| `accepted` | 是否 upsert |
| `reject_reason` | 硬门原因（若有） |
| `batch_outcomes` | 本批各 episode 的 `outcome` / `r_task`、用了哪些工具 |
| `rebuild_scope` | `retrieval` / `constraints` / `workflow` / `crystallize` / `skip_no_success_anchor` |

输出（只收这个 JSON）：

```json
{
  "reasoning": "本轮哪种写法有用或有害",
  "meta_skill_content": "给下一轮抽取器的短原则，覆盖或删掉过时条目"
}
```

`meta_skill_content` **整份覆盖** `cluster.meta_skill_md`（SkillOpt 也是重写 optimizer memory，不是往 Skill 里 append）。要求短、可执行；上一轮里被这轮证伪的原则要删掉。

下一轮 evolve 只读新的 `meta_skill_md`，不再读旧版。这替代 rejected_buffer：被拒的套话写进 meta，下次结晶/rebuild 直接看见。

---

## 9. 硬门（规则，不重跑任务）

过了才 upsert。不过：**不覆盖已有 Skill**，拒因写入 meta-skill。

1. JSON 合法；`name` 为 snake_case。
2. `tools ⊆` 本批 RawTurn 工具名（覆盖率 1.0，不再用 0.5）。
3. 若本批有成功：`steps` 非空，且能指回成功 RawTurn 的具体片段（命令 / 参数 / 验收）。
4. 套话 steps（仅「检查 / 验证 / 处理」）→ 拒。
5. 失败 episode 的过程不得进入 `steps`。
6. 只有失败、没有成功锚 → 根本不应走到 upsert。

`verifySkillDraft` 的 L1 token resonance **不再当主门**。第一期 **不做** hold-out 重跑（1 集簇没有 \(D_{sel}\)；test 不当门）。簇内 ≥ 2 成功时，可选留 1 条不进 evolve 做弱选门，后加。

---

## 10. 写回

复用现有落库，不改 Skill schema：

- `coerceSkillProcedureJson` → `procedure_json`
- `renderSkillInvocationGuide` → `memoryValue` / `invocation_guide`
- `layer: Skill`，`source: worker.skill_batch_evolve.v1`
- `source_policy_ids` 为空
- `evidence_anchor_ids` = **本批成功 episodeId**（失败可进 LLM 当约束，不进 evidence）
- `appendEpisodeDerivedMemory(episodeId, "Skill", skillId)`
- 需要检索时再 enqueue skill embedding

不关闭、不改 L2 的 `enqueue skill_crystallization`。

---

## 11. 和旧链的关系

```text
RawTurn     完整观测     Skill 的唯一生成证据
L1          检索锚点     聚类可用 embedding 作辅，不作文
L2          单步经验     面板 / Repair / 旧结晶；新链不把它当输入
Skill       簇级 SOP     只由 skill_batch_evolve 写
meta-skill  抽取器记忆   不注入
```

---

## 12. 实现切面（不含本规格的实现）

1. 表 `skill_clusters` / `skill_cluster_members`。
2. Job `skill_cluster_assign`、`skill_batch_evolve`。
3. Reward 末尾每个 episode 入队一次 assign。
4. Evolve 复用结晶 JSON 的 coerce / render / upsert；新建 RawTurn 打包与按 outcome 的字段路由。
5. 不改 `l2_induction` / `associateL2` / `crystallizeSkill`，也不加切断旧链的开关。

第一期不做：全量重聚类、slow-update 写入 `invocation_guide`、rejected_buffer、40×4 rollout、用 test 当门。
