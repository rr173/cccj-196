# 版本差异（diff）功能设计与产物格式说明

在保留单对象稀疏读取（`inspect` / `scan` / `cache-clear`）全部行为的前提下，
新增**两份已锁定快照之间的版本审计**能力。本文档描述身份固定、键规范、
有界外部排序、确定性差异事实与原子发布的完整契约。

## 1. 输入：两份快照清单

每次 diff 由两份 *snapshot manifest* 驱动，每份清单把一个具体对象版本钉死为
三元组 `(canonical_url, strong_etag, length)`，请求还声明唯一键、比较字段、
结果路径与内存上限：

```json
{
  "spec_version": 1,
  "left":  {"url": "http://h/base.parquet",   "etag": "\"ab…\"", "length": 12345},
  "right": {"url": "http://h/target.parquet", "etag": "\"cd…\"", "length": 12901},
  "key_column": "kid",
  "compare_columns": ["code", "tags", "meta", "cat", "blob"],
  "diff_path":   "/out/facts.jsonl",
  "report_path": "/out/report.json",
  "memory_limit": 8388608
}
```

### 1.1 启动即核对身份（`transport.validate_pinned`）

- 条件 **HEAD** 带 `If-Match: <清单强 ETag>`；412 / 弱或缺失 ETag /
  实体长度与清单不符，一律在取数前失败（`VersionMismatchError` 等）。
- 随后一次**一字节**条件 Range 探测（同样带 `If-Match`）证明服务器真支持
  Range；200 整包回退、坏 `Content-Range`、短包均失败。
- 此后**每个数据 GET** 都带 `Range` 与同一个 `If-Match`。206 的 ETag、
  `Content-Range` 区间、实体长度、实收字节数任一不符即失败；412 不重试；
  瞬时 5xx/网络错误做有界重试且版本不漂移，耗尽抛 `RetryExhaustedError`。
- 同一 URL 的两份清单是**两个明确身份**：`RemoteParquetClient.open_identity`
  不覆盖客户端的“已知版本”表，条件请求各自钉死自己的 ETag，无法用连续探测
  把不同时刻的内容冒充为左右输入。

## 2. 业务键：可排序、无碰撞的规范编码

键列必须是**非空的顶层 Arrow 标量**（字符串、各宽度有/无符号整数、布尔、
date/time/timestamp、二进制、decimal 等）；浮点、null 类型、list/struct/map
不允许作为键。空键抛 `NullKeyError`，每份资料内键必须唯一（见 §4）。

编码 `parquet_range/diff_key.py` 是**自描述字节串**，不是字符串拼接、也不是
无证明的裸哈希：

1. 首字节为固定**类型标签**——不同类型永不碰撞，并有确定的全局顺序；
2. 变长载荷用 **8 字节大端长度 + 载荷**做长度分帧，边界无歧义；
3. 有符号整数做符号位翻转的大端 `ordered` 变换，字节序即数值序；
4. timestamp 把单位、是否带时区、时区名（长度分帧）与瞬时值一并编码；
5. decimal 编码 precision/scale 与保序的无符号缩放整数（保留全部小数位）；
6. 单独的 null 标签仅用于精确报错，合法键不会出现。

外部排序与 JSONL 的稳定顺序都直接基于这些字节的字典序。

## 3. Schema 契约与字段级 drift 诊断（`diff_drift.py`）

取数前对两侧 schema 与声明字段做契约检查，**只有合约兼容才允许发布**：

- 声明字段任一侧缺失 → 记录 `missing on left/right` 诊断并拒绝；
- 键列类型两侧不一致、或不是允许的标量类型 → 拒绝；
- 比较字段两侧 Arrow 逻辑类型不一致（含 dictionary 的 index/value 类型、
  dictionary 与普通类型之差）→ 记录 `type mismatch` 诊断并拒绝；
- **按字段名而非位置匹配**：字段排列不同不算 drift，也不会误判相等；
- 绝不静默转换、不补空字段。聚合错误以 `SchemaDriftError.diagnostics`
  （含字段名、角色、两侧类型、问题描述）抛出，且不产出任何正式结果。

## 4. 有界外部排序与归并（`diff_segment.py` / `diff_engine.py`）

不假设两个文件排序相同，也不把全量表装入内存：

1. 按行组稀疏扫描**仅声明的**键 + 比较列（复用现有版本锁定/缓存/取消组件）；
2. 微批校验并编码键，dictionary 先解码；空键在此失败；
3. `RunBuilder` 仅在内存上限内累积记录，超限即把当前运行排序后落一个
   **临时段**；大于内存上限的输入必然产生受控 spill；远端对象从不整体落盘；
4. 每侧的多个有序段用**有界 k 路堆归并**输出全局有序记录（每路只驻留队首）。

### 4.1 临时段格式与自检

```
PQRSEG01\n
<u64 BE header 长度><header JSON>
<Arrow IPC stream：schema + 有序 record batches>
<trailer JSON><u64 BE trailer 长度>
PQRSEG01\n
```

段头记录**格式版本、Arrow schema 摘要、左右两个身份、run_id**；段尾记录
**记录数、批次数、IPC 体的 SHA-256**。读取时重新校验魔数/分帧/schema
digest/双侧身份/计数/校验和，任何不符抛 `SegmentIntegrityError` 并隔离，绝不
合并不可信段。记录列固定为 `_pkey`（规范键字节）、`_kmeta`（键 JSON）、
`_rg`（行组）、`_row`（资料内全局行号）+ 声明的比较列。

### 4.2 唯一性与重复键定位

归并时对同键记录分组：同一份资料内一键出现多次即抛 `DuplicateKeyError`，
`occurrences` 给出每条的**来源身份（url/etag/length/version_id）、行组、行号、
键 hex**，便于离线定位，且不产出正式结果。

### 4.3 取消、异常与进程终止

- 取消（复用 `QueryToken`）在抽取与归并各检查点生效，抛 `DiffCancelled`，
  `finally` 中删除本 run 的整个临时目录；
- 任何异常同样清理本 run 临时物；`quarantine/` 保留损坏段供取证；
- 启动时 `cleanup_stale_runs` 扫描临时根：属于其它 run 的完好段删除、
   framing 损坏的段隔离、`.tmp` 残片清除——不可信临时段永不参与合并。

## 5. 左右对齐与 Arrow 语义比较（`diff_engine.py`）

归并后的左右有序流按键对齐，比较值始终以 Arrow 标量保留到最后一刻，用
`Scalar.equals` 判定，明确区分：null、空 list/struct、dictionary **解码后**
的值、二进制（逐字节）。键只在左 → `delete`，只在右 → `insert`，两侧皆有且
相等 → `no-change`，否则 `update`。

## 6. 产物：JSONL 差异事实 + JSON 审计报告

### 6.1 差异文件（UTF-8 JSONL，每行一条）

字段顺序由声明固定，结果按规范键稳定排序：

```json
{"kind":"update","key":{"type":"int64","value":2,"hex":"13…"},"before":{…},"after":{…}}
```

- `kind` ∈ `insert` / `delete` / `update` / `no-change`，四类事实都会写入
  JSONL（报告同时给出四类计数）；
- `key` 含类型、人类可读值与规范编码 hex；
- `insert` 仅含 `after`，`delete` 仅含 `before`，`update` 两者皆有；
- 值渲染（`diff_values.py`）：null 为 JSON `null`；空容器为
  `{"type": <arrow 类型>, "value": []/{}}`，与 null 可区分；二进制为
  `{"type":"binary","value":"<hex>"}`；日期/时间为带类型标记的 ISO 字符串。

### 6.2 审计报告（单个 JSON 文档）

含：报告格式/版本/完成状态、起止时间、键与比较字段、**左右身份与 schema
摘要**、两侧输入行数、唯一键数、**四类计数**、比较值摘要（每字段类型、
dictionary 标记、两侧 null/空容器计数）、`resources`（真实网络/缓存统计、
spill 段数与字节、驻留峰值、进程 RSS、内存上限）、以及差异文件的路径/记录数/
字节数/SHA-256。

### 6.3 原子发布与离线复核（`diff_publish.py`）

- 先写同目录临时文件，JSONL 完整 flush+fsync 后从磁盘重读，逐行解析并复核
  SHA-256、记录数与报告计数一致，才进入发布；
- 两段 `os.replace`：先事实后报告；报告替换失败会把事实与报告都回滚到上一
  成功版本，任何失败都不覆盖既有成功结果、也不留看似完整的目标文件；
- `diff-explain` / `explain_diff()` **只读**已生成的报告与差异文件，重新
  校验校验和/计数/顺序，不访问服务。

## 7. 失败模式与退出状态

| 场景 | 异常 / CLI 退出码 |
| --- | --- |
| 清单畸形、弱/缺 ETag、坏 URL | `ManifestError` / 3 |
| schema/type drift、不支持的键型、空键、重复键 | `SchemaDriftError`/`UnsupportedKeyTypeError`/`NullKeyError`/`DuplicateKeyError` / 4 |
| 原子发布失败 | `DiffPublishError` / 5 |
| 临时段校验失败 | `SegmentIntegrityError` / 6 |
| 取消 | `DiffCancelled` / 7 |
| 协议/版本/短包/重试耗尽等 | 既有 `ParquetRangeError` 子类 / 2 |

任何失败都不产生正式结果；既有成功结果保持字节不变。

## 8. 模块边界

```
diff_manifest.py   快照清单与请求的解析/规范化（无 I/O）
diff_key.py        规范键编码（类型标签 + 长度分帧 + 保序变换）
diff_drift.py      schema 契约与字段级 drift 诊断
diff_segment.py    有界 RunBuilder、自检临时段、k 路归并、崩溃恢复
diff_values.py     Arrow -> JSON 值渲染（null/空容器/dict/二进制可区分）
diff_publish.py    原子发布、校验和复核、离线 explain
diff_engine.py     编排：身份核对 -> 抽取 -> 外部排序 -> 对齐比较 -> 发布
transport.py       新增 validate_pinned（条件 HEAD + 条件探测）
client.py          新增 open_identity（同 URL 两个明确身份）
```

## 9. 命令与 Python 接口

```bash
# 请求文件
python3 bin/pqr.py diff request.json
# 或全量 flag
python3 bin/pqr.py diff \
  --left-url  … --left-etag '"…"' --left-length 123 \
  --right-url … --right-etag '"…"' --right-length 456 \
  --key-column kid --compare-columns code,tags,meta \
  --diff-out out/f.jsonl --report-out out/report.json \
  --diff-memory-limit 8388608
# 离线复核（不触网）
python3 bin/pqr.py diff-explain out/report.json
```

```python
from parquet_range import RemoteParquetClient, DiffEngine, DiffRequest, explain_diff

with RemoteParquetClient(disk_dir="./cache") as client:
    report = DiffEngine(client, DiffRequest.from_json_file("request.json")).run()
explanation = explain_diff("out/report.json")   # 纯本地
```
