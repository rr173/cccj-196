# 协议与架构设计说明

稀疏 HTTP Range Parquet 读取器。所有远端 I/O 由本包基于 HTTPX 自写的层完成；
PyArrow 只负责 Parquet 元数据解析与列块解码，**不接触任何网络、文件系统或
fsspec/S3 filesystem/dataset scanner**。

## 1. 核心不变量

1. **版本身份固定**：一个查询对象 `VersionedObject` 的身份在其整个生命周期内
   固定为三元组 `(canonical_url, strong_etag, length)`。
   - URL 经规范化（scheme/host 小写、默认端口去除、点段归一化、百分号编码统一）。
   - ETag 必须是**强验证器**。`W/"..."` 弱 ETag 或缺失 ETag 在首次接触即失败，
     因为它们无法证明字节级身份，版本固定将失去意义。
2. **版本隔离**：每个缓存键都嵌入 `sha256(canonical_url + etag + length)` 的版本
   哈希。同 URL 的两个版本落到不同键空间，永远不可能把旧 footer 和新列块拼接。
3. **只有校验成功的完整字节范围才能成为可读覆盖**。在途/部分数据放在独立的
   in-flight 表中，绝不进入可读 overlay；晚到的包在等待者已离开时被丢弃。
4. **禁止整包下载**：任何 200 回退（即使带 Range 头）都明确报错；不存在「为了
   方便先下完整文件」的代码路径。

## 2. 首次接触校验（`transport.validate_object`）

1. `HEAD` 取 `Content-Length` 与强 `ETag`，要求 200。
2. 发一次**一字节**条件探测 `Range: bytes=0-0` + `If-Match: <etag>`，证明服务器
   真的支持 Range（有些服务器 HEAD 宣称 `Accept-Ranges` 但 GET 回 200）。
   - 探测要求 206，校验 `Content-Range: bytes 0-0/<length>`、ETag 一致、body 恰为
     1 字节。探测本身的 5xx 走有界重试。
3. 由此构造 `ObjectIdentity`。同 URL 再次 `open()` 若 ETag 已变，抛
   `VersionMismatchError` 并提示用 `open_fresh()` 接受新版；旧版本缓存保持隔离。

## 3. 数据 GET 协议（`transport.get_range`）

每个数据请求都带：

```
Range: bytes=<start>-<end>
If-Match: <pinned strong etag>
```

成功路径必须**同时**满足，否则失败（绝不静默拼接旧缓存）：

| 检查项 | 失败行为 |
| --- | --- |
| HTTP 206 | 200 → `RangeProtocolError`（禁止整包回退）；416 → 协议错误 |
| `Content-Range: bytes start-end/total` 精确匹配 | `RangeProtocolError` |
| `Content-Length` 与区间大小一致 | `RangeProtocolError` |
| ETag 存在且为强、且等于固定 ETag | 弱/缺 → `WeakOrMissingETagError`；不一致 → `VersionMismatchError` |
| 实际收到字节数恰为 `end-start+1` | 短包 → `TruncatedResponseError`；超长 → `RangeProtocolError` |
| HTTP 412 | `VersionMismatchError`，**不重试** |

**有界重试**：仅网络错误（`httpx.HTTPError`/`OSError`）与可重试状态码
（408/425/429/500/502/503/504）重试，最多 `max_retries` 次，指数退避且封顶 2s。
重试始终携带**同一个** `If-Match`，不漂移版本；耗尽抛 `RetryExhaustedError`。
协议违规（200/坏 Content-Range/短包/ETag 不符）是确定性错误，不重试、不无限循环。

## 4. 块覆盖缓存与调度（`cache.py`）

### 地址空间与覆盖
- 对象按固定 `block_size`（默认 1 MiB，CLI/测试可调小）切分为块。
- 三级读取：内存 LRU overlay → 磁盘证据缓存 → （合并后的）条件 GET。
- 一个块只有在物理 GET 完整返回且全部协议校验通过后，才经 `_publish` 成为可读
  覆盖。`_publish` 在唤醒任何等待者**之前**完成，因此读侧不会看到半成品覆盖。

### 在途合并（coalescing）
- 每个块在 `_inflight` 表登记一个 `_Inflight`（含等待者计数、event、cancel event）。
- 并发查询命中同一块时，第二个查询只增加 `waiters` 并等待同一 event，物理 GET
  只有一个（`coalesced_hits` 计数）。
- **有界相邻合并**：发起者可把相邻的缺失块并入一次 GET，但跨越的空洞块数不超过
  `max_merge_gap_blocks`，因此不会为了消除重叠而把巨大空洞整体读回。

### 取消语义（`QueryToken`）
- 每个查询持一个 token，登记它正在等的 `_Inflight`。
- `cancel()` 只对该 token 等待的每个请求做**一次** waiter 递减（`_drop_once`
  防重复扣减）。物理请求只要还有别的查询等待就继续，不被杀。
- 当且仅当最后一个等待者离开，才 set 该请求的 cancel event 中止传输；此时即便
  数据随后到达，`_run_physical` 也因 `waiters<=0/cancel` 丢弃结果、不发布覆盖。
- `scheduler.read` 入口与逐块循环都检查 token，已取消即抛 `QueryCancelled`。
- 预算信号量（在途字节/请求数）在 `finally` 中释放，取消不泄漏预算。

### 预算与 LRU
- 内存总字节额度、磁盘总字节额度、在途字节额度（按 1 MiB 粒度获取信号量）、
  在途请求数额度。
- 内存 `OrderedDict` LRU：超预算时淘汰最旧且**未 pin** 的块；被读者 pin 的块跳过
  （宁可暂时超预算也不破坏进行中的读）。
- 磁盘超预算时按证据文件 mtime 做 LRU 回收，pinned 块不回收。

## 5. 磁盘缓存与重启证据（`disk_cache.py`）

每个块 B 覆盖版本 V 的 `[start,end)`，落两个文件：

```
<root>/<version_hash>.b<index>.blob       原始字节
<root>/<version_hash>.b<index>.evidence   JSON 元数据 + blob 的 sha256
```

- **两阶段原子写**：blob 先写 `*.partial` 并 `fsync`，再 `os.replace`；evidence
  同样 partial→replace。崩溃只会留下 `.partial` 或缺 evidence 的孤儿 blob。
- **重开核验**：新进程扫描目录，`.partial`、无 evidence 的 blob、evidence 解析失败、
  sha256 不符、长度不符——一律隔离删除并计入 `disk_fragments_rejected`，绝不服务。
- 仅当 evidence 与 blob 全部核验通过，块才重新进入可读集。

## 6. Parquet 集成（`parquet_reader.py`，PyArrow 仅解码）

1. **自定位 footer**：读尾部 8 字节 → 校验尾部 `PAR1` magic 与 footer 长度 → 读
   Thrift footer 字节 → 读首部 4 字节校验 leading magic → 重建
   `footer + len + PAR1` 尾部 blob，用 `pq.read_metadata` 解析。短于最小 footer
   信封、magic 错误、footer 长度越界都抛 `FooterError`。
2. **构造零 I/O**：`pq.ParquetFile(sparse_file, metadata=metadata)` 传入预解析元数据，
   PyArrow 构造阶段不发任何读取。
3. **嵌套 leaf 映射**：用每个列块的 `path_in_schema`（如 `events.list.element.ts`、
   `meta.flags.list.element`）把顶层列名映射到全部 leaf 列块索引。
4. **精确列块区间**：对所选行组 × leaf，起点取 `min(data_page_offset,
   dictionary_page_offset)`（含字典页），终点 `start + total_compressed_size`；
   不依赖跨 writer 语义不一致的 `file_offset`。零值空块跳过预取。
5. 先经稀疏层预取全部规划区间（集中享受合并/预算/取消），再让 PyArrow 逐行组
   `read_row_group(..., use_threads=False)` 解码。
6. **空语义明确**：`columns=[]` 返回带正确行数的零列表；零行行组返回空 batch；
   零行组文件 `inspect` 可见但选行组时报 `BadSelectionError`。

## 7. 失败模式对照表

| 场景 | 结果 |
| --- | --- |
| 弱/缺 ETag | `WeakOrMissingETagError`（首次接触失败） |
| 不支持 Range / 探测回 200 | `RangeProtocolError` |
| 两次 Range 间换 ETag | 旧查询 `VersionMismatchError`；可另开新查询接受新版，旧数据版本隔离 |
| HTTP 412 | `VersionMismatchError`，不重试 |
| 200 整包回退 | `RangeProtocolError` |
| 坏 Content-Range / 实体长度不符 | `RangeProtocolError` |
| 短包 / 超长包 | `TruncatedResponseError` / `RangeProtocolError` |
| 请求中途取消（最后等待者离开） | `QueryCancelled`，结果丢弃不发布，预算释放 |
| 取消但仍有其他等待者 | 仅取消方退出，物理请求继续，另一方完整成功 |
| 瞬时 5xx / 网络错误 | 有界重试，固定原版 |
| 重试耗尽 | `RetryExhaustedError` |
| 断电残片 / evidence 篡改 | 重启核验拒绝并隔离，重新走网络 |
| 回收时块仍被读者 pin | 不回收该块 |
| footer 过短 / magic 错 / 长度越界 | `FooterError` |

## 8. 模块边界

```
transport.py    唯一的远端 I/O：HTTPX 条件 Range GET、严格 206 校验、有界重试
cache.py        块 overlay、LRU、在途合并、预算、取消调度
disk_cache.py   两阶段原子磁盘缓存 + 重启证据核验
sparse_file.py  可 seek/read 的 io.RawIOBase，PyArrow 看到的唯一文件对象
parquet_reader.py footer 定位、leaf 映射、列块区间规划、scan/inspect
identity.py     规范化 URL、强 ETag、版本身份与版本化缓存键
client.py       对外 open/open_fresh/scan/inspect/cache_clear 生命周期
test_server.py  真实本地 Range HTTP 服务 + 屏障 + 故障注入（非 mock）
```
