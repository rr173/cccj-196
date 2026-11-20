# parquet-range

基于 HTTP Range 的稀疏远端 Parquet 读取器。分析任务只取需要的几个行组和少量列，
并发查询共享已取字节，并严格固定对象版本。**不会**为图方便整包下载，也**不会**
把两个不同版本的 footer 与列块拼在一起。

- 远端 I/O：HTTPX 自写的可 seek/read 稀疏对象 + 请求调度（无 fsspec / S3
  filesystem / 远端 dataset scanner）。
- PyArrow：仅负责 Parquet 元数据与列块解码。
- 版本身份：规范 URL + 强 ETag + 长度，缓存键含版本，旧版数据按版本隔离。
- 支持嵌套 list/struct 的 leaf 映射、dictionary 页、null、空列/空行组。

## 目录

```
parquet_range/   生产包（transport/cache/disk_cache/sparse_file/parquet_reader/...）
                 版本审计：diff_manifest/diff_key/diff_drift/diff_segment/
                 diff_values/diff_publish/diff_engine
                 质量审计：quality_regex/quality_contract/quality_rules/
                 quality_values/quality_request/quality_publish/quality_engine
                 双时间宽表：temporal_request/temporal_schema/temporal_tree/
                 temporal_segment/temporal_publish/temporal_engine
bin/pqr.py                可运行 CLI（inspect / scan / cache-clear / diff /
                          diff-explain / quality-audit / quality-explain /
                          temporal-enrich / serve / generate / stats）
bin/generate_large_parquet.py  大档案生成脚本
bin/generate_diff_parquet.py   大资料左右快照 + diff 请求模板生成脚本
bin/generate_quality_parquet.py 大资料 + 质量合约 + 审计请求模板生成脚本
bin/generate_temporal_parquet.py 偏斜双时间数据集 + 请求模板生成脚本
tests/           真实本地 HTTP 服务上的 pytest 套件（含屏障与故障注入，非 mock）
docs/design.md   协议与架构设计说明
docs/diff_design.md  版本审计（diff）设计与产物格式说明
docs/quality_design.md 单快照质量审计设计与合约格式说明
docs/temporal_design.md 双时间物流宽表（temporal-enrich）设计与产物格式说明
```

## 快速开始

```bash
# 1. 生成一个含嵌套 list/struct、dictionary、null、多行组的大档案
python3 bin/generate_large_parquet.py data/big.parquet \
    --rows-per-group 50000 --num-groups 8

# 2. 起真实 Range HTTP 服务
python3 bin/pqr.py serve data --port 8765

# 3. 另开终端：inspect（首次接触校验长度/强 ETag/Range 能力）
python3 bin/pqr.py inspect http://127.0.0.1:8765/big.parquet

# 4. 稀疏 scan：明确行组编号 + 顶层列，资源统计打印到 stderr
python3 bin/pqr.py --block-size 65536 scan \
    http://127.0.0.1:8765/big.parquet \
    --columns id,category,meta --row-groups 2 --format arrow

# 5. 清缓存（只删未被读者 pin 的覆盖）
python3 bin/pqr.py cache-clear http://127.0.0.1:8765/big.parquet
```

## 版本审计：两份锁定快照的 diff

给定基线与目标两份快照清单（规范 URL + 强 ETag + 长度），系统做键对齐的
版本审计，原子产出 UTF-8 JSONL 差异事实与 JSON 审计报告。完整契约见
[`docs/diff_design.md`](docs/diff_design.md)。

```bash
# 1. 生成运行时大资料左右快照（多行组、重排、insert/delete/update/no-change、
#    嵌套/dictionary/null/空容器/二进制），并得到带 ETag/长度的请求模板
python3 bin/generate_diff_parquet.py data/diff-pairs \
    --rows-per-group 20000 --num-groups 8

# 2. 起真实 Range 服务后，按模板里的 URL 运行（小内存会真实产生多段 spill）
python3 bin/pqr.py --block-size 65536 diff data/diff-pairs/diff-request.json

# 3. 离线复核已发布结果（不重新访问服务，重新校验 checksum/计数/顺序）
python3 bin/pqr.py diff-explain data/diff-pairs/results/report.json
```

差异文件每行一条，字段顺序固定、按规范键稳定排序：

```json
{"kind":"update","key":{"type":"int64","value":2,"hex":"…"},"before":{…},"after":{…}}
```

关键保证：

- 启动即对两侧做**条件**身份核对，之后每次取数都带对应精确 `If-Match`；
  同一 URL 也按两个明确身份处理，412/弱 ETag/整包回退/短包/重试耗尽都判失败。
- 键为非空顶层 Arrow 标量，编码携带**类型、长度与空值边界**，可排序且无碰撞；
  重复键报错带来源身份、行组、行号。
- 不假设两侧同序、不整表入内存：按批抽取 + 有界外部排序/归并，临时段带格式
  版本、schema、双侧身份、记录数与 SHA-256；取消/异常/进程终止都会清理或隔离。
- schema/字段类型 drift 给字段级诊断并拒绝发布，不静默转换、不补空字段、
  不因字段排列不同而误判相等。
- 仅当 JSONL 完整 flush 且报告计数/校验一致才原子替换目标路径，失败不覆盖
  既有成功结果。

## 单快照质量审计：合约驱动的逐行检查

给定一份钉死的输入身份（规范 URL + 强 ETag + 长度）和一份 JSON 质量合约
（必需字段及其逻辑类型 + 带稳定 ID 的行规则），系统只读取合约需要的列，
逐行执行规则，原子产出违规明细 JSONL 与审计报告。完整契约见
[`docs/quality_design.md`](docs/quality_design.md)。

```bash
# 1. 生成运行时大资料（多行组、有/无符号整数、NaN、时区时间、嵌套/宽列），
#    并得到带真实 ETag/长度的合约与审计请求模板
python3 bin/generate_quality_parquet.py data/quality-audit \
    --rows-per-group 20000 --num-groups 8

# 2. 起真实 Range 服务后运行审计（小内存/小批均可，违规流式落盘）
python3 bin/pqr.py serve data/quality-audit --port 8765 &
python3 bin/pqr.py --block-size 65536 quality-audit \
    data/quality-audit/quality-audit-request.json

# 3. 离线复核并按规则分页读取违规样例（不重新请求远端对象）
python3 bin/pqr.py quality-explain \
    data/quality-audit/results/report.json --rule score-range --limit 5
```

违规文件每行一条，字段顺序固定，按（行组、全局行号、合约规则序）稳定排列，
同一行违反多条规则时逐条记录：

```json
{"identity":{"url":"…","etag":"\"…\"","length":…,"version_id":"…"},
 "rule_id":"price-range","rule_kind":"range","fields":["price"],
 "row_group":2,"row":12345,
 "observed":{"price":{"type":"double","value":-3.5}},
 "reason":"below_min"}
```

关键保证：

- 合约在**读取行数据前**全量校验：未知规则、重复规则 ID、引用不存在字段、
  参数类型与 Arrow 字段不相容、坏模式，全部带合约位置报错；不静默跳过、
  不把坏合约当全部通过、不自动转换类型。合约类型须与 Parquet 顶层字段
  **精确**匹配，字段排列不影响匹配。
- 规则覆盖：非空、数值/时间范围（端点开闭分别指定）、枚举、字符串模式、
  两列有序关系。null/NaN/时区/无符号语义明确一致：非空规则报告 null；
  其余规则默认把 null/NaN 视为不适用，显式要求才计违规。
- 字符串模式装载期编译为 NFA，每个值 `O(值长×状态数)` 有界执行，恶意
  模式拖不垮审计。
- 审计基于启动时验证的精确对象身份；每个 Range 请求带同一条件身份，
  换版/412/短包/重试耗尽即失败，既有结果字节不变。
- 明细与报告作为可恢复发布单元原子替换；崩溃后重启能识别、清理或隔离
  未完成发布物，绝不误删上一份有效审计。

### 质量审计的 Python 接口

```python
from parquet_range import (
    RemoteParquetClient, QualityAuditEngine, QualityAuditRequest,
    QualityContract, read_violations,
)

request = QualityAuditRequest.from_json_file("quality-audit-request.json")
with RemoteParquetClient(disk_dir="./.cache") as client:
    report = QualityAuditEngine(client, request).run()   # 原子发布明细 + 报告
page = read_violations(request.report_path, rule_id="score-range",
                       offset=0, limit=20)                # 纯本地复核，不触网
```

## 双时间物流宽表：按"当时知道到哪一刻"还原维表归属

给定一份事实分片清单和一份维表修订日志分片清单（各自钉死版本），系统按
"当时知道到哪一刻"还原每个业务键的窗口状态，把事实分片和维表修订物化成
一份可直接用 PyArrow 读取的宽表。完整契约见
[`docs/temporal_design.md`](docs/temporal_design.md)。

```bash
# 1. 生成偏斜数据集（热键 10 万修订 + 10 万事实，窗重叠、登记时间乱序）
#    并得到带真实 ETag/长度的请求模板
python3 bin/generate_temporal_parquet.py data/temporal \
    --hot-facts 100000 --hot-windows 1000 --hot-revisions-per-window 100

# 2. 起真实 Range 服务后运行（小内存会真实产生多段 spill）
python3 bin/pqr.py serve data/temporal --port 8765 &
python3 bin/pqr.py --block-size 65536 temporal-enrich \
    data/temporal/temporal-request.json
```

关键保证：

- 同一业务键、同一事件时间，只改"当时知道到哪一刻"，就可能得到不同的
  服务区；最新修订移动窗口后，不能因新范围不含事件就回退到旧范围。
- 留下的窗口只按 `priority` 取最大值：一个最高优先窗口时附上其业务列；
  多个同优先窗口时标为 `ambiguous`（不任选赢家，即使业务列相同）；
  完全没有时标为 `unmatched`。
- 修订中 `(entity_key, window_id, recorded_at)` 重复是作业错误（跨分片、
  内容完全相同、或尚未被任何事实看见，都一样），诊断须含来源位置。
- 每条事实恰好一行，按事实清单顺序、分片内物理行顺序排列，重复事实也
  保留；文件元数据保存分片 ID 与精确对象身份的映射。
- 以同目录暂存后替换的方式交付单个完整 Parquet；读取失败、坏输入、
  取消、写入失败或替换前进程中止都不能覆盖已有目标。

### 双时间宽表的 Python 接口

```python
from parquet_range import (
    RemoteParquetClient, TemporalEnrichEngine, TemporalEnrichRequest,
)

request = TemporalEnrichRequest.from_json_file("temporal-request.json")
with RemoteParquetClient(disk_dir="./.cache") as client:
    report = TemporalEnrichEngine(client, request).run()  # 原子发布宽表
# 结果：单个完整 Parquet，直接用 PyArrow 读取
import pyarrow.parquet as pq
table = pq.read_table(request.output_path)
```

## 作为库使用

```python
from parquet_range import RemoteParquetClient, ClientConfig

with RemoteParquetClient(
    disk_dir="./.cache",
    config=ClientConfig(block_size=64 * 1024),
) as client:
    obj = client.open("http://127.0.0.1:8765/big.parquet")  # 固定 (url, etag, length)
    info = obj.inspect()
    table = obj.scan(columns=["id", "events", "meta"], row_groups=[0, 3])  # Arrow 批次

    # 可取消的查询：取消只解除它自己的需要
    token = client.new_token()
    # ... 在另一线程调用 token.cancel()
```

### 版本审计的 Python 接口

```python
from parquet_range import RemoteParquetClient, DiffEngine, DiffRequest, explain_diff

request = DiffRequest.from_json_file("diff-request.json")
with RemoteParquetClient(disk_dir="./.cache") as client:
    report = DiffEngine(client, request).run()      # 原子发布 facts + report
explanation = explain_diff(request.report_path)      # 纯本地复核，不触网
```

同 URL 换版后，旧 `obj` 的条件 GET 会收到 412/ETag 不符而失败；用
`client.open_fresh(url)` 接受新版，旧版缓存仍按版本隔离、可读。

## 运行测试

```bash
python3 -m pytest tests/ -q
```

测试覆盖（全部跑在真实 `http.server.ThreadingHTTPServer` 上，记录每个 Range/条件头、
返回字节与版本，并用可控屏障做并发验收）：

- 与本地 PyArrow 全量参考解码逐值/schema/行数一致（嵌套、dictionary、null、多行组）
- Q1/Q2 读重叠块时取消 Q1，Q2 完整成功且只发生必要物理取数
- 两次 Range 之间换 ETag，旧查询失败而非交付混合表；新版另开查询、旧版隔离
- 尾部短于最小 footer、无 Range、错误 Content-Range、短包、请求中途取消、重试耗尽
- 缓存命中后退出重开（磁盘证据核验）、断电残片拒绝、回收时仍有读者
- 选少量列/行组时线上传输显著小于整包（实测约整包的 2.7%）

版本审计（diff）测试以**独立本地 PyArrow 实现为 oracle**（绝不调用产品 diff）：

- 多行组、重排记录、insert/delete/update/no-change 的逐行事实与独立期望一致，
  重复运行字节级稳定；嵌套 list/struct、dictionary、null、空 list/struct、二进制、
  多种键类型（string/binary/有/无符号整数/bool）全覆盖
- 重复键（含身份/行组/行号）、空键、schema/type drift、坏身份、412、短包、
  重试耗尽都有明确退出状态且不产出正式结果
- 执行到一半切换 ETag 不能把前后版本合成一份差异；同 URL 两个 ETag 作为两个
  明确身份处理；同版本可命中隔离缓存、跨版本绝不串用
- 小内存下确实多段落盘、k 路归并后仍与期望一致并记录驻留峰值/RSS；取消、写出
  失败、进程终止后临时物被清理或隔离；段校验和/身份篡改被拒绝
- 服务日志证明只取声明字段/行组，并保留条件头、区间、字节数与 206 退出状态

质量审计（quality-audit）测试同样以**独立本地 PyArrow 实现为 oracle**（绝不
调用产品规则执行器）：

- 多列多规则同一行命中逐条记录、不同 Arrow 数值类型、无符号边界、时区时间、
  null/NaN 显式语义、嵌套但未被引用的列、缺列/类型不符、坏模式、错参数，
  全部与独立期望一致；结果按（行组、行号、规则序）确定排列，重复运行字节稳定
- 审计中途切换版本、注入错误区间/短包/412/重试耗尽，审计失败且原有结果逐字节
  不变；合约/schema 错误在读取行数据前失败
- 小预算运行大于预算的数据：峰值 RSS 有界、只投影合约列、传输少于整对象；
  取消、磁盘写失败、报告与明细发布之间强制退出后重开，恢复状态正确且旧结果
  可读；离线分页不触网；空表/零违规报告正确；每条命令退出状态与服务日志一致

双时间宽表（temporal-enrich）测试以**独立本地 PyArrow 直算为 oracle**（绝不
调用产品的窗口选择函数）：

- 时序场景：A 窗口移动/撤销/重开、低优先窗口始终覆盖、高优先窗口只在其中
  一个时刻与 A 重叠；并列计数/位置、左闭右开边界；重复事实保留、空表、
  修订-only 键、matched-null 列与 unmatched 区分
- 规模夹具：同一键至少十万条窗口修订和十万条事实，窗有重叠、登记时间乱序；
  在低于处理中间数据大小的预算下完成，展示落盘工作量与内存证据；预计结果
  由生成规律或小样本独立核对
- 跨分片重复、对象换版、写出中断、取消、临时物清理，都核对退出结果及旧
  目标不变；真实本地 Range 服务观察条件请求和列投影

## 资源统计

每次 `scan` 后 CLI 在 stderr 打印物理请求数、线上字节、内存/磁盘命中、合并命中、
驱逐与预算等待等计数；库中通过 `client.stats.render()` / `client.stats.snapshot()`
获取。
