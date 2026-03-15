---
title: 第 9 章：Scheduler 架构
---

# 第 9 章：Scheduler 架构

> "调度器是推理引擎的大脑——它决定在每个时刻，哪些请求应当获得宝贵的 GPU 计算资源。"

## 调度器的核心角色

在 SGLang 的架构中，Scheduler 是连接前端请求处理与后端模型执行的关键组件。它负责管理请求的生命周期，决定哪些请求进入计算流水线，协调 prefill 和 decode 阶段的资源分配。调度器的核心代码位于 `python/sglang/srt/managers/scheduler.py`。

## 请求的生命周期

### Req 类

每个到达系统的推理请求被封装为一个 `Req` 对象，定义在 `python/sglang/srt/managers/schedule_batch.py` 中：

```python
# python/sglang/srt/managers/schedule_batch.py
class Req:
    def __init__(self, rid, origin_input_text, origin_input_ids, ...):
        self.rid = rid                        # 请求唯一标识
        self.origin_input_ids = origin_input_ids  # 原始输入 token
        self.output_ids = []                   # 已生成的 token
        self.sampling_params = ...             # 采样参数
        self.prefix_indices = None             # 前缀匹配的 KV Cache 索引
        self.extend_input_len = 0              # 需要新计算的 token 数
        self.finished = False                  # 是否完成生成
```

`Req` 对象贯穿请求的整个生命周期，从进入等待队列到完成生成。其中 `prefix_indices` 和 `extend_input_len` 反映了 RadixCache 前缀匹配的结果——前者是已匹配的 KV Cache 索引，后者是仍需计算的 token 数量。

### Batch 类

调度器将多个请求组装为一个 `ScheduleBatch`，作为一次 GPU 计算的基本单位：

```python
# python/sglang/srt/managers/schedule_batch.py
class ScheduleBatch:
    def __init__(self, reqs, ...):
        self.reqs = reqs                  # 批中的请求列表
        self.batch_size = len(reqs)       # 批大小
        self.input_ids = ...              # 合并后的输入 token
        self.seq_lens = ...               # 每个请求的序列长度
        self.prefix_lens = ...            # 每个请求的前缀匹配长度
```

`ScheduleBatch` 将多个请求的数据组织为模型执行所需的张量格式，使 GPU 能够并行处理批中的所有请求。

## 调度器的双队列结构

SGLang 调度器维护两个核心队列：

### 等待队列（Waiting Queue）

新到达的请求首先进入等待队列。等待队列中的请求尚未开始 prefill 计算，等待调度器分配资源。

```python
# python/sglang/srt/managers/scheduler.py
class Scheduler:
    def __init__(self, ...):
        self.waiting_queue = []     # 等待 prefill 的请求
        self.running_batch = None   # 当前正在运行的批
        self.cur_batch = None
```

### 运行批（Running Batch）

通过调度决策后的请求进入运行批。运行批中的请求处于 decode 阶段，每次迭代生成一个新 token。

## 主调度循环

调度器的核心逻辑在主循环中执行，每次迭代做出一系列关键决策：

```python
# python/sglang/srt/managers/scheduler.py
def event_loop_normal(self):
    while True:
        recv_reqs = self.recv_requests()  # 接收新请求
        self.process_input_requests(recv_reqs)

        batch = self.get_next_batch_to_run()

        if batch:
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)
```

每次循环迭代包含四个阶段：

1. **接收请求**：从前端接收新的推理请求，加入等待队列。
2. **构建批次**：根据当前资源状况决定下一个要执行的批次。
3. **执行计算**：将批次交给模型运行器（Model Runner）执行前向计算。
4. **处理结果**：收集生成结果，判断请求是否完成，更新状态。

## 调度决策：get_next_batch_to_run

这是调度器最核心的决策函数。它需要在有限的 GPU 资源下做出最优的调度选择：

```python
# python/sglang/srt/managers/scheduler.py
def get_next_batch_to_run(self):
    # 1. 检查是否有正在运行的 decode 批
    if self.running_batch is not None:
        # 尝试从等待队列中选择新请求加入
        can_run_list = self.get_new_prefill_batch()
        if can_run_list:
            # 合并 prefill 和 decode（混合批处理）
            return self.build_mixed_batch(can_run_list)
        else:
            return self.running_batch  # 继续 decode

    # 2. 没有运行批时，从等待队列选择请求启动 prefill
    if self.waiting_queue:
        can_run_list = self.get_new_prefill_batch()
        if can_run_list:
            return self.build_prefill_batch(can_run_list)

    return None
```

### 资源约束判断

调度器在选择请求时需要考虑多重资源约束：

- **显存容量**：新请求的 KV Cache 不能超出可用显存。通过 `TokenToKVPool` 的剩余容量判断。
- **最大批大小**：受限于 GPU 计算能力和显存带宽。
- **序列长度限制**：单个请求的序列长度不能超过模型的最大上下文窗口。

```python
# python/sglang/srt/managers/scheduler.py
def get_new_prefill_batch(self):
    running_tokens = self._get_running_tokens()
    available_tokens = self.token_to_kv_pool.available_size()
    can_run_list = []

    for req in self.waiting_queue:
        # 执行前缀匹配
        prefix_indices, last_node, matched_len = \
            self.tree_cache.match_prefix(req.origin_input_ids)
        req.prefix_indices = prefix_indices
        req.extend_input_len = len(req.origin_input_ids) - matched_len

        # 检查资源是否充足
        needed = req.extend_input_len + req.max_new_tokens
        if running_tokens + needed > available_tokens:
            break

        can_run_list.append(req)
        running_tokens += needed

    return can_run_list
```

注意这里的关键步骤：在决定是否调度一个请求时，调度器首先通过 RadixCache 进行前缀匹配。匹配结果直接影响资源评估——如果前缀命中率高，`extend_input_len` 就小，需要的新显存就少，从而允许更多请求被同时调度。

## 与模型运行器的交互

调度器将构建好的 `ScheduleBatch` 交给模型运行器执行：

```python
# python/sglang/srt/managers/scheduler.py
def run_batch(self, batch):
    if batch.forward_mode.is_extend():
        # Prefill 阶段：处理输入 token
        logits, next_token_ids = self.tp_worker.forward_batch_extend(batch)
    else:
        # Decode 阶段：生成下一个 token
        logits, next_token_ids = self.tp_worker.forward_batch_decode(batch)
    return logits, next_token_ids
```

模型运行器（`TpModelWorker`）封装了实际的模型前向计算，处理张量并行等底层细节。调度器只关心批次的组织和结果的处理，不涉及具体的模型计算逻辑。

## 请求完成与清理

当一个请求满足终止条件（生成了 EOS token、达到最大长度等），调度器将其从运行批中移除并进行清理：

```python
# 处理完成的请求
def process_batch_result(self, batch, result):
    for req in batch.reqs:
        if req.finished:
            # 释放请求资源
            self.tree_cache.cache_finished_req(req)
            # 发送结果给前端
            self.send_result(req)
            batch.reqs.remove(req)
```

`cache_finished_req` 将完成请求的 KV Cache 保留在 RadixCache 中（降低引用计数但不立即释放），使其可被后续请求复用。

## 本章小结

本章分析了 SGLang Scheduler 的架构设计。调度器通过等待队列和运行批的双队列结构管理请求生命周期；`Req` 和 `ScheduleBatch` 类封装了请求和批次的核心数据；主调度循环通过接收请求、构建批次、执行计算和处理结果四个阶段驱动系统运转；调度决策结合了 RadixCache 前缀匹配结果与资源约束判断，实现了缓存感知的智能调度。下一章我们将深入探讨 Continuous Batching 机制。
