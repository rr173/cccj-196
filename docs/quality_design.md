# 单快照质量审计（quality-audit）设计与合约格式说明

在保留 `inspect` / `scan` / `diff` / `diff-explain` / `cache-clear` 全部既有行为的
前提下，新增**单快照质量审计**：数据团队把数据约束写成一份 JSON 合约，系统对
一个版本锁定的远端 Parquet 对象逐行执行合约规则，产出可直接定位问题记录的
违规明细（JSONL）与完整审计报告（JSON），两者作为一个可恢复发布单元原子发布。

## 1. 输入：审计请求

```json
{
  "spec_version": 1,
  "object":  {"url": "http://h/f.parquet", "etag": "\"ab…\"", "length": 17746572},
  "contract": { ...inline contract... },
  "contract_path": "quality-contract.json",
  "violations_path": "/out/violations.jsonl",
  "report_path":     "/out/report.json",
  "batch_rows": 4096
}
```

- `object` 是钉死的输入身份：规范 URL + 强 ETag + 实体长度（与 diff 的快照清单
  完全同构）。`contract`（内联）与 `contract_path`（文件）必须且只能给一个。
- `batch_rows` 是解码/求值微批大小，约束常驻内存；CLI 侧另有 `--mem-budget`
  等既有缓存预算参数。
- 请求文件与合约文件的 JSON 解析都**拒绝重复键**（防止静默覆盖）。

## 2. 合约格式

```json
{
  "spec_version": 1,
  "fields": {
    "id": "int64", "price": "float64", "qty": "uint8",
    "name": "string", "created": "timestamp[us, tz=UTC]",
    "updated": "timestamp[us, tz=UTC]", "day": "date32"
  },
  "rules": [
    {"id": "id-non-null", "kind": "not_null", "field": "id"},
    {"id": "price-range", "kind": "range", "field": "price",
     "min": 0, "max": 1000000, "min_inclusive": true, "max_inclusive": false,
     "null_is_violation": false, "nan_is_violation": true},
    {"id": "ccy-enum", "kind": "enum", "field": "ccy",
     "values": ["USD", "EUR", "JPY"]},
    {"id": "name-pattern", "kind": "pattern", "field": "name",
     "pattern": "^[A-Z][a-z0-9_]{2,15}$"},
    {"id": "window-order", "kind": "order",
     "left": "created", "right": "updated", "allow_equal": false}
  ]
}
```

### 2.1 逻辑类型

`fields` 声明**必需字段**及其逻辑类型，支持：`bool`；`int8/16/32/64`；
`uint8/16/32/64`；`float32/64`；`string` / `large_string`；`binary` /
`large_binary`；`date32` / `date64`；`timestamp[s|ms|us|ns]` 与
`timestamp[<unit>, tz=<name>]`。未知类型字符串在装载即报错。

### 2.2 规则

每条规则有稳定字符串 `id`（合约内唯一）和 `kind`：

| kind | 参数 | 语义 |
| --- | --- | --- |
| `not_null` | `field` | 报告该字段的 null（NaN 是值，不算 null） |
| `range` | `field`, `min`?, `max`?, `min_inclusive`(默认 true), `max_inclusive`(默认 true) | 数值/时间范围，端点开闭分别指定；`min`/`max` 至少一个 |
| `enum` | `field`, `values`（非空且不重复） | 值必须属于枚举集合 |
| `pattern` | `field`, `pattern` | 字符串必须匹配模式（search 语义，见 §2.4） |
| `order` | `left`, `right`, `allow_equal`(默认 false) | 两列有序关系：`left < right`（或允许相等） |

`range`/`enum`/`pattern`/`order` 还可以带 `null_is_violation`、
`nan_is_violation`（均默认 false）。

### 2.3 装载期校验（读取任何行数据之前）

以下问题全部在**读取行数据前**以带合约位置（如 `rules[2].min`、
`fields.price`）的 `ContractError` 抛出，绝不静默跳过规则、绝不把坏合约
当作全部通过、绝不自动转换字段类型：

- 未知顶层键/规则键、重复 JSON 键、`spec_version` 不支持；
- 未知规则 kind、规则 `id` 重复（报出首次使用位置）；
- 规则引用未声明字段；
- 参数类型与声明的 Arrow 字段类型不相容：整数字段的边界/枚举成员必须是
  JSON 整数且在类型宽度内（如 `uint8` 的 256、负值立即报错）；浮点边界必须
  是 JSON 数（拒绝 NaN）；字符串字段上的 range、非字符串字段上的 pattern、
  二进制字段上的 enum 等；
- 时间戳边界必须是 ISO-8601 字符串，且**时区感知性与字段一致**（带 tz 的
  字段必须配带偏移的边界，naive 字段必须配 naive 边界）；
- `order` 要求两个**不同**字段且类型完全相同（不做有符号/无符号混合比较）；
- `nan_is_violation` 只在涉及浮点字段时合法；
- 字符串模式在装载时编译（见 §2.4），不支持的结构立即报错。

### 2.4 字符串模式：装载期验证 + 有界执行

模式**不**交给 Python 回溯式 `re` 执行，而是编译为 Thompson NFA，用
Pike-VM 模拟匹配，每个值的代价严格为 `O(值长 × 状态数)`——不存在回溯，
恶意模式（如 `(a+)+$`）在毫秒内完成，不可能拖垮审计。

- 支持：字面量与转义、`.`、字符类 `[a-z]`/`[^…]`、ASCII 预定义类
  `\d \D \w \W \s \S`、锚点 `^`/`$`（`$` 为严格串尾，等同 `\Z`）、
  量词 `* + ? {m} {m,} {m,n}`（上限 1000）、分组 `(...)`/`(?:...)`、
  交替 `|`、`\n \t \r \f \v \a` 与 `\xHH`。
- 装载时拒绝：反向引用、前后环视、命名组、内联 flag、原子组、占有/懒惰
  量词、`\b \B \A \Z \z \G`、八进制/Unicode 转义、重复量词（`a**`）等。
- 硬上限：模式长度 ≤ 2048 字符、展开后 NFA 状态 ≤ 1024、被测值长度 ≤
  65536 字符（超长值以独立原因码 `pattern_input_too_long` 记为违规，
  绝不静默截断）。
- 匹配语义为 **search**（同 SQL `REGEXP_LIKE`）：要全值匹配请自己写
  `^...$`。`\d \w \s` 均为 ASCII 语义。

### 2.5 null / NaN / 时区 / 无符号语义

- `not_null` 只报告 null；NaN 是合法值。
- 其余规则默认把 null（以及浮点字段的 NaN）视为**不适用**：跳过、计数
  （报告中的 `null_skipped` / `nan_skipped`）、不计违规；只有规则显式写
  `null_is_violation: true` / `nan_is_violation: true` 时才记为违规
  （原因码 `null` / `nan`）。
- 比较不依赖 Python 真值、不做不安全的有符号转换：时间戳/日期先转成其
  存储整数（瞬时值，时区在装载期已校验）再比较；`uint64` 以 Python 任意
  精度整数比较；浮点按 IEEE 语义比较（NaN 已先行分流）。
- 边界与枚举成员在装载期转换到与值完全相同的比较域（时间字段 → 存储
  整数），转换经过 PyArrow 精确完成。

## 3. 身份固定与稀疏读取

- 启动即对 `object` 做**条件**身份核对（`If-Match` 条件 HEAD + 1 字节条件
  Range 探测）；之后每个数据 GET 都带同一个 `If-Match`。ETag 不符、区间
  不符、实体长度不符、收到字节数不符、弱/缺 ETag、200 整包回退、412、
  短响应、有限重试耗尽，一律使审计失败。
- 只读取**规则引用的列**（按顶层字段投影，嵌套但未引用的列绝不取）和
  **实际需要的行组**（空行组跳过）；绝不整包下载。按 `batch_rows` 微批
  解码求值，违规记录流式写出——不会随规则数/批次增长而保留与全表成比例
  的行对象或全部违规对象。

## 4. 产物

### 4.1 违规明细（UTF-8 JSONL，每行一条）

字段顺序固定，记录按 **(行组, 文件内全局行号, 合约中的规则顺序)** 严格
递增排列；同一行违反多条规则时逐条记录，不合并、不丢失规则身份：

```json
{"identity":{"url":"…","etag":"\"…\"","length":…,"version_id":"…"},
 "rule_id":"price-range","rule_kind":"range","fields":["price"],
 "row_group":2,"row":12345,
 "observed":{"price":{"type":"double","value":-3.5}},
 "reason":"below_min"}
```

- `reason` 是稳定的机器可读原因码：`null`、`nan`、`below_min`、
  `above_max`、`not_in_enum`、`pattern_mismatch`、`pattern_input_too_long`、
  `order_violated`。
- `observed` 是**类型化**观测值：`{"type": <Arrow 类型>, "value": …}`；
  null 为 `{"type": t, "value": null}`；NaN/±Inf 为标记字符串；时间戳渲染
  为 UTC 瞬时的 ISO-8601（保留字段单位精度，字段自身时区见 `type`）；
  日期为 ISO 日历日；二进制为 hex。`order` 规则的 `fields`/`observed`
  同时包含左右两列。

### 4.2 审计报告（单个 JSON 文档）

含：报告格式/版本/完成状态、起止时间、**输入身份**（url/etag/length/
version_id）、**合约 SHA-256**（规范序列化）、匹配的 schema 摘要（匹配
字段 + 文件字段数/行数）、**规则顺序**、扫描行数/行组数、**每条规则的
违规数**（及 null/NaN 跳过数）、**总违规数**、真实网络/缓存统计（物理
请求数、线上字节、缓存命中等）、峰值内存（max RSS 与批大小）、启动恢复
摘要，以及明细文件的**路径/记录数/字节数/SHA-256**。

### 4.3 原子发布与崩溃恢复

- 明细与报告先写同目录暂存文件（`.quality-violations-*.tmp` /
  `.quality-report-*.tmp`），flush+fsync 后从磁盘重读明细并复核
  SHA-256/记录数/每规则计数，全部一致才进入发布；
- 两段 `os.replace` 带 `.old` 备份与完整回滚：任何失败都不覆盖既有成功
  结果，也不留下看似成功的新结果；
- 每次审计启动时 `recover_publish_unit` 检查发布单元：清理暂存残片、把
  中断的交换**恢复**到上一份有效审计（旧的 `.old` 备份在新 pair 未完整
  提交前绝不删除）、把不一致或孤立的产物**隔离**到 `quarantine/`（绝不
  静默删除数据，也绝不让半个新结果被解释为成功）。

### 4.4 离线分页读取

`read_violations(report_path, rule_id=None, offset=0, limit=100)` 与
`quality-explain` 命令**只做本地 I/O**：重新计算明细 SHA-256、记录数、
每规则计数与 (行组, 行号, 规则序) 严格递增顺序，全部与报告核对一致后，
按规则分页返回违规样例；任何篡改都会以 `AuditPublishError` 失败。

## 5. 命令与 Python 接口

```bash
# 请求文件（或全量 flag：--object-url/--object-etag/--object-length
#   --contract-file --violations-out --report-out [--audit-batch-rows N]）
python3 bin/pqr.py quality-audit audit-request.json
# 离线复核 + 按规则分页（不触网）
python3 bin/pqr.py quality-explain out/report.json --rule price-range \
    --offset 0 --limit 20
```

```python
from parquet_range import (
    RemoteParquetClient, QualityAuditEngine, QualityAuditRequest,
    QualityContract, read_violations, explain_quality,
)

request = QualityAuditRequest.from_json_file("audit-request.json")
with RemoteParquetClient(disk_dir="./cache") as client:
    report = QualityAuditEngine(client, request).run()
page = read_violations(request.report_path, rule_id="price-range",
                       offset=0, limit=20)   # 纯本地复核
```

退出状态（`quality-audit`）：0 成功；3 请求/合约错误
（`AuditRequestError`/`ContractError`）；4 合约与文件 schema 不符
（`AuditSchemaError`）；5 发布失败（`AuditPublishError`）；7 取消；
2 协议/版本/短包/重试耗尽等其余错误。任何失败都不产生正式结果，既有
成功结果保持字节不变。

## 6. 模块边界

```
quality_regex.py     有界 NFA 正则引擎（装载期验证 + O(n·m) 执行）
quality_contract.py  合约解析/校验/schema 精确匹配（全部在取数前完成）
quality_rules.py     五种规则的逐行求值器（显式 null/NaN 语义）
quality_values.py    观测值的类型化 JSON 渲染
quality_request.py   审计请求（身份 + 合约 + 路径 + 批大小）
quality_publish.py   流式明细写出、原子发布、崩溃恢复、离线分页读取
quality_engine.py    编排：身份核对 -> 投影扫描 -> 规则求值 -> 发布
parquet_reader.py    新增 scan_batches（按行组预取 + 微批解码）
bin/generate_quality_parquet.py  运行时大资料 + 合约 + 请求模板生成器
tests/quality_oracle.py          独立参考实现（不调用产品规则执行器）
```
