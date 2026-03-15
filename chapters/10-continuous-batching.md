---
title: 第 10 章：Continuous Batching
---

# 第 10 章：Continuous Batching

> "静态批处理像公交车——到站才能上下客；Continuous Batching 像出租车——随时接送，永不空跑。"

## 从静态批处理说起

### 静态批处理的局限

在传统的推理服务中，批处理（Batching）是最基本的吞吐量优化手段。静态批处理（Static Batching）将多个请求组成一个固定的批次，统一进行前向计算。然而，它有一个根本性的缺陷：整个批次必须等待最慢的请求完成后才能释放资源。

考虑一个包含 4 个请求的批次：

```
请求 A: 生成 10 个 token   → 第 10 步完成
请求 B: 生成 50 个 token   → 第 50 步完成
请求 C: 生成 30 个 token   → 第 30 步完成
请求 D: 生成 20 个 token   → 第 20 步完成
```

在静态批处理中，所有请求必须等到第 50 步才能统一结束。请求 A 在第 10 步就已完成生成，但其占用的 GPU 资源和显存在接下来的 40 步中完全浪费。更严重的是，等待队列中的新请求无法加入正在运行的批次，导致延迟和吞吐量都受到影响。

## Continuous Batching 的核心思想

Continuous Batching（也称为 Iteration-Level Scheduling）将调度粒度从"批次级"细化到"迭代级"。在每次 decode 迭代后，调度器都可以做出新的调度决策：

- **移除已完成的请求**：一旦某个请求完成生成，立即将其从批次中移除并释放资源。
- **加入新的请求**：利用释放的资源，将等待队列中的新请求加入当前批次。

```
迭代 1-10:  [A, B, C, D] → A 完成，移除
迭代 11:    [B, C, D, E] → E 新加入
迭代 11-20: [B, C, D, E] → D 完成，移除
迭代 21:    [B, C, E, F] → F 新加入
...
```

GPU 始终保持满负载运行，资源利用率大幅提升。

## SGLang 中的 Continuous Batching 实现

### 迭代级调度循环

SGLang 的调度器天然支持 Continuous Batching。其主循环在每次迭代后都会重新评估调度决策，这在 `python/sglang/srt/managers/scheduler.py` 中体现：

```python
# python/sglang/srt/managers/scheduler.py
def event_loop_normal(self):
    while True:
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)

        batch = self.get_next_batch_to_run()
        if batch:
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)
        else:
            # 无可运行的批次，等待新请求
            self.wait_for_requests()
```

每次循环都是一次完整的调度决策周期。`process_batch_result` 会识别并移除已完成的请求，`get_next_batch_to_run` 会尝试将等待队列中的新请求加入运行批次。

### 动态序列管理

Continuous Batching 的关键挑战之一是处理批次中序列长度的动态变化。随着 decode 的进行，每个请求的序列长度都在增长，新加入的请求可能处于 prefill 阶段而非 decode 阶段。

SGLang 通过 `ScheduleBatch` 类管理这种动态性：

```python
# python/sglang/srt/managers/schedule_batch.py
class ScheduleBatch:
    def prepare_for_extend(self, ...):
        # 准备 prefill 阶段的输入数据
        ...

    def prepare_for_decode(self, ...):
        # 准备 decode 阶段的输入数据
        ...

    def filter_batch(self, finished_indices):
        # 从批次中移除已完成的请求
        ...

    def merge_batch(self, other_batch):
        # 将新的请求批次合并到运行批次中
        ...
```

`filter_batch` 和 `merge_batch` 是实现 Continuous Batching 的两个关键方法。前者负责"下客"——移除已完成的请求并释放其资源；后者负责"上客"——将新调度的请求合并到运行中的批次。

### 混合批处理：Prefill 与 Decode 共存

在 Continuous Batching 中，一个批次可能同时包含处于 prefill 阶段的新请求和处于 decode 阶段的旧请求。SGLang 支持这种混合批处理模式：

```python
# python/sglang/srt/managers/scheduler.py
def get_next_batch_to_run(self):
    if self.running_batch and self.waiting_queue:
        # 尝试构建混合批次
        new_reqs = self.get_new_prefill_batch()
        if new_reqs:
            prefill_batch = self.build_prefill_batch(new_reqs)
            # 混合批次：新请求做 prefill，旧请求做 decode
            return self.build_mixed_batch(prefill_batch)
    ...
```

混合批处理面临的技术挑战是 prefill 和 decode 的计算模式不同——prefill 处理多个输入 token，而 decode 每次只处理一个 token。SGLang 通过 FlashInfer 等高效 attention 内核来处理这种不对称性，使两种模式能在同一次 GPU 调用中高效执行。

## 性能分析：Continuous vs Static

### 吞吐量提升

Continuous Batching 的吞吐量提升来自两个方面：

1. **消除气泡（Bubble）**：静态批处理中，短请求完成后的空闲计算槽位（气泡）被消除。
2. **持续满载**：GPU 始终处理接近最大容量的请求数，计算资源利用率最大化。

在生成长度差异较大的工作负载中，Continuous Batching 的吞吐量提升可达 2-3 倍。即使在生成长度较为均匀的场景中，由于能持续接纳新请求，延迟分布也更加均匀。

### 延迟优化

对于单个请求而言，Continuous Batching 带来的延迟改善同样显著：

- **排队时间缩短**：新请求无需等待整个批次完成，只要有空闲槽位就可以加入。
- **TTFT 降低**：请求的等待时间从"下一个完整批次开始时"缩短为"下一次迭代中有空闲资源时"。

### 与 RadixCache 的协同效应

Continuous Batching 与 RadixCache 之间存在良性的协同效应。当一个请求完成并离开批次时，其 KV Cache 不会被立即释放，而是保留在 RadixCache 中。后续加入的新请求如果与已完成请求共享前缀，可以直接复用这些缓存，进一步减少 prefill 计算量。

```
时间线:
t=0:  请求 A (系统提示 + 用户1) 开始 prefill
t=5:  请求 A 完成 prefill，开始 decode
t=10: 请求 A 完成生成，KV Cache 保留在 RadixCache
t=11: 请求 B (系统提示 + 用户2) 加入
      → 系统提示部分命中 RadixCache，只需 prefill 用户2 的输入
```

## 实现细节：请求的进出管理

### 请求加入

当新请求被调度加入运行批次时，需要完成以下步骤：

1. 通过 RadixCache 进行前缀匹配，确定可复用的 KV Cache。
2. 从 `TokenToKVPool` 分配新的 KV Cache 空间（仅为未匹配部分）。
3. 执行 prefill 计算，填充新分配的 KV Cache。
4. 将请求添加到运行批次的序列列表中。

### 请求退出

当请求完成生成时：

1. 将请求的 KV Cache 信息写入 RadixCache（`cache_finished_req`）。
2. 降低 RadixCache 中相关节点的引用计数。
3. 从运行批次的序列列表中移除。
4. 将生成结果返回给前端。

## 本章小结

本章深入分析了 Continuous Batching 的原理和 SGLang 中的实现。与静态批处理相比，Continuous Batching 通过迭代级调度消除了短请求完成后的资源空闲，通过动态的请求进出管理保持 GPU 持续满载运行。SGLang 的 `ScheduleBatch` 类通过 `filter_batch` 和 `merge_batch` 方法支持请求的动态增减，并通过混合批处理模式实现了 prefill 与 decode 阶段的共存执行。结合 RadixCache 的前缀复用能力，Continuous Batching 在吞吐量和延迟两个维度都带来了显著的性能提升。
