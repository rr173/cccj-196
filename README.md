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
bin/pqr.py                可运行 CLI（inspect / scan / cache-clear / serve / generate / stats）
bin/generate_large_parquet.py  大档案生成脚本
tests/           真实本地 HTTP 服务上的 pytest 套件（含屏障与故障注入，非 mock）
docs/design.md   协议与架构设计说明
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

## 资源统计

每次 `scan` 后 CLI 在 stderr 打印物理请求数、线上字节、内存/磁盘命中、合并命中、
驱逐与预算等待等计数；库中通过 `client.stats.render()` / `client.stats.snapshot()`
获取。
