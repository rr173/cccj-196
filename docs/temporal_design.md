# 双时间物流宽表（temporal-enrich）设计与产物格式说明

在保留 `inspect` / `scan` / `cache-clear` / `diff` / `diff-explain` /
`quality-audit` / `quality-explain` 全部既有行为的前提下，新增**双时间物流宽表**：
给定一份事实分片清单和一份维表修订日志分片清单（各自钉死版本），系统按
"当时知道到哪一刻"还原每个业务键的窗口状态，把事实分片和维表修订物化成一份
可直接用 PyArrow 读取的宽表。消费者不需要拼两份快照、解释变更记录或跑二次
连接。

## 1. 输入：作业清单（JSON）

```json
{
  "spec_version": 1,
  "fact_shards": [
    {"shard_id": "f0", "url": "http://h/f0.parquet",
     "etag": "\"ab…\"", "length": 12345}
  ],
  "revision_shards": [
    {"shard_id": "r0", "url": "http://h/r0.parquet",
     "etag": "\"cd…\"", "length": 23456}
  ],
  "fact_columns": ["event_id", "payload"],
  "dim_columns": ["service_area", "owner"],
  "output_path": "/out/wide.parquet",
  "memory_limit": 8388608,
  "spill_dir": "/out/.pqr-temporal-tmp",
  "batch_rows": 4096
}
```

- 两个数组都是**有序**的：事实按清单分片顺序、分片内物理行顺序输出。
- 每个分片条目有作业内唯一的 `shard_id`、规范 URL、强 ETag 和对象长度，
  与 diff 的快照清单同构；清单或必要 schema 错误在读取行数据前被拒绝。
- `fact_columns` 是要保留的事实列（可为空），`dim_columns` 是至少一个待附
  维表列；各自不重复，且不得占用输出保留列名（`_match_status`、`_dim`、
  `_top_window_count`、`_fact_source`、`_dim_sources`）。
- `memory_limit` 是工作内存预算（字节）；`spill_dir` 是临时目录（缺省为
  输出目录下的 `.pqr-temporal-tmp`）；`batch_rows` 是解码/校验微批大小。

## 2. 固定列与数据契约

两个角色的 `entity_key` 与修订表的 `window_id` 均为非空 UTF-8 字符串。
事实表 `event_at` 是事件时间，`known_at` 是回填允许看到的登记时间上界。
修订表 `recorded_at` 是登记时间，另有字符串列 `op`、窗口起止 `valid_from`/
`valid_to`、优先级 `priority` 及待附列。时间和优先级均为有符号 64 位整数
（UTC 微秒）；`event_at`、`known_at`、`recorded_at`、`op` 不得为 null。

- `upsert` 必须带 `valid_from` 及 `priority`；`valid_to=null` 表示无结束
  时间，有结束时间时必须严格晚于 `valid_from`。
- `withdraw` 撤销整个窗口，其区间、优先级和待附列必须为 null。
- 所选业务列至少支持顶层 UTF-8 字符串、整数、布尔、二进制、decimal128 和
  Arrow timestamp（date/time/float 作为便利超集）；同一角色各分片的必要列
  必须精确同型，输出保留逻辑类型和值。

## 3. 双时间匹配语义

对每条事实 `(entity_key, event_at, known_at)`：

1. 只看相同 `entity_key` 的修订。
2. 对每个 `window_id`，取 `recorded_at <= known_at` 的最后一条修订；找不到
   或取到 `withdraw` 时该窗口不存在。取到 `upsert` 才判断
   `valid_from <= event_at < valid_to`（无结束时间不设上界）；最新修订移动
   窗口后，不能因新范围不含事件就回退到旧范围。
3. 留下的窗口只按 `priority` 取最大值：一个最高优先窗口时附上其业务列；
   多个同优先窗口时标为 `ambiguous`（不任选赢家，即使业务列相同）；完全
   没有时标为 `unmatched`。
4. 修订中 `(entity_key, window_id, recorded_at)` 重复是作业错误（跨分片、
   内容完全相同、或尚未被任何事实看见，都一样），诊断须含来源位置。

## 4. 输出：单个完整 Parquet 宽表

每条事实恰好一行，按事实清单顺序、分片内物理行顺序排列，重复事实也保留。
输出列：

- 所选事实列（保留逻辑类型和值）；
- `_match_status`：`matched` / `ambiguous` / `unmatched`；
- `_dim`：按选择列组织的可空 struct（`ambiguous`/`unmatched` 时为 null，
  不能与匹配成功但某业务列为 null 混淆）；
- `_top_window_count`：最高优先级窗口数（`unmatched` 为 0）；
- `_fact_source`：事实来源坐标（分片 ID、行组编号、分片内从 0 开始的全局
  行号）；
- `_dim_sources`：维表来源列表。`matched` 记录赢家；`ambiguous` 按
  `window_id` 的 UTF-8 字节序给出最小两个并列窗口的胜出修订来源；
  `unmatched` 为 null。

文件元数据（Parquet footer 的 key-value metadata）保存本次分片 ID 与精确
对象身份（URL、强 ETag、长度、version_id）的映射，使这些坐标脱离运行日志
也可追溯。

## 5. 双时间解析与匹配的实现（无外部 SQL/连接引擎）

管道（无整包下载、无全表驻留、无二次连接）：

1. **身份钉死与 schema 合约**：对全部事实/修订分片做条件身份核对（条件
   HEAD + 一字节 Range 探测，之后每个数据 GET 都带同一 `If-Match`）；随后
   只读 footer，做精确类型合约检查（必要列存在、同角色各分片同型、业务列
   类型受支持），在读取行数据前拒绝。
2. **抽取与外部排序**：只读实际使用的列，逐行校验（null/空键、op 合法、
   upsert/withdraw 规则），把记录编码成可排序字节键（`entity_key` 长度分帧
   + `recorded_at`/`known_at` 符号位翻转大端 + `window_id`/`seq`），按
   （`entity_key`、`recorded_at`、`window_id`）和（`entity_key`、`known_at`、
   `seq`）分别外部排序，落盘为带格式版本、schema、运行 ID 和 SHA-256 的
   校验和段。
3. **按键合并 + 知识时间扫描**：对排序后的修订和事实做 merge-join。每个
   业务键：先把修订流和事实流落盘（热键不驻留内存），第一遍收集窗口边界
   坐标；第二遍按 `recorded_at` 重放修订，用压缩线段树维护"当前可见窗口"
   的 stabbing-max 结构，按 `known_at` 顺序回答每条事实。每条事实的查询
   是 `O(log² n)`，绝不是 `O(该键全部修订)`，也绝不物化候选笛卡尔积。
4. **结果重排 + 原子发布**：结果按 `seq` 重排回事实清单顺序，以同目录
   暂存后替换的方式交付单个完整 Parquet；读取失败、坏输入、取消、写入
   失败或替换前进程中止都不能覆盖已有目标。

## 6. 预算的计量边界

`memory_limit` 计量：外部排序/重排缓冲（溢出即落盘）、单键缓冲（热键
溢出即落盘）、单键匹配工作集（坐标索引、线段树节点成员、活动窗口表，
按条目数计量并设上限，超限报 `TemporalResourceError`）、解码微批
（`batch_rows`）。不计量：HTTP 块缓存（独立的 `--mem-budget`）、
Python/Arrow 基础开销、PyArrow 解码内部缓冲。运行统计报告实际网络字节、
临时空间峰值、峰值 RSS、主要匹配工作量（stabbing 查询数、episode 数、
窗口状态峰值）。

## 7. 运行统计与可复核性

每次运行返回/打印一份 JSON 报告：分片身份映射、事实/修订计数、三种状态
计数、网络/缓存统计、峰值资源、落盘字节与段数、键/窗口/查询计数、
temp 目录清理恢复信息。CLI 以退出码区分失败类：3=清单/请求错误，
4=schema 错误，5=发布失败，6=数据错误（坏行/重复修订），7=取消，
8=资源超限，2=其他。

## 8. 测试与验证

- 小数据对拍：独立本地 Parquet 解码 + 逐条直算（不调用产品的窗口选择
  函数），覆盖时序场景（A 窗口移动/撤销/重开、低优先窗口始终覆盖、
  高优先窗口只在其中一个时刻与 A 重叠）、并列计数/位置、左闭右开边界。
- 规模夹具：同一键至少十万条窗口修订和十万条事实，窗有重叠、登记时间
  乱序；在低于处理中间数据大小的预算下完成，展示落盘工作量与内存证据；
  预计结果由生成规律或小样本独立核对。
- 真实本地 Range 服务观察条件请求（每个 GET 带 `If-Match`）和列投影
  （未用列不取，线上字节远小于整包）。
- 跨分片重复、对象换版、写出中断、取消、临时物清理，都核对退出结果及
  旧目标不变。
