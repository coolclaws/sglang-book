---
title: 第 11 章：Chunked Prefill 与 PD 分离
---

# 第 11 章：Chunked Prefill 与 PD 分离

> "长序列推理的艺术，在于如何将一个庞然大物拆解为可管理的碎片，同时不丢失全局的连贯性。"

## 长序列 Prefill 的挑战

随着大语言模型上下文窗口的不断扩展——从 4K 到 32K、128K 乃至更长——prefill 阶段的计算压力急剧增大。一个 128K token 的长文档输入需要一次性计算所有 token 的 attention，这带来了多重问题：

1. **显存峰值过高**：attention 计算的中间结果占用大量显存，可能导致 OOM（Out of Memory）。
2. **阻塞 decode 请求**：长 prefill 独占 GPU 计算资源，正在 decode 的请求被迫等待，导致所有用户的生成延迟剧增。
3. **计算不均衡**：一个长 prefill 的计算量可能是几十次 decode 迭代的总和，破坏了 Continuous Batching 的均衡性。

Chunked Prefill 正是为了解决这些问题而提出的优化策略。

## Chunked Prefill 的核心概念

### 分块策略

Chunked Prefill 将一个长序列的 prefill 计算拆分为多个较小的块（Chunk），每个块包含固定数量的 token（例如 2048 或 4096 个），分多次迭代完成。

```
原始序列 (10000 tokens):
[t_0 ..................... t_9999]

分块后 (chunk_size = 2048):
迭代 1: [t_0    ... t_2047]   → 计算前 2048 token 的 KV Cache
迭代 2: [t_2048 ... t_4095]   → 计算接下来 2048 token
迭代 3: [t_4096 ... t_6143]   → ...
迭代 4: [t_6144 ... t_8191]   → ...
迭代 5: [t_8192 ... t_9999]   → 完成 prefill
```

### 与 Continuous Batching 的结合

Chunked Prefill 的真正威力在于它与 Continuous Batching 的结合。在每次迭代中，一个 chunk 的 prefill 计算可以与其他请求的 decode 计算共享 GPU 资源：

```
迭代 N:
  - 请求 A: prefill chunk_3 (2048 tokens)
  - 请求 B: decode 1 token
  - 请求 C: decode 1 token
  - 请求 D: decode 1 token
```

这样，长序列的 prefill 不再阻塞其他请求的 decode，用户体验得到了显著改善。

## SGLang 中的 Chunked Prefill 实现

### 配置与参数

SGLang 通过配置参数控制 chunked prefill 的行为，相关逻辑在 `python/sglang/srt/managers/scheduler.py` 中实现：

```python
# python/sglang/srt/server_args.py
@dataclass
class ServerArgs:
    chunked_prefill_size: int = -1  # -1 表示禁用，正数为 chunk 大小
    ...
```

当 `chunked_prefill_size` 设置为正整数时，调度器会自动将超过该长度的 prefill 请求拆分为多个 chunk。

### 调度器中的分块逻辑

调度器在构建 prefill 批次时检查是否需要分块：

```python
# python/sglang/srt/managers/scheduler.py
def get_new_prefill_batch(self):
    for req in self.waiting_queue:
        prefix_indices, _, matched_len = \
            self.tree_cache.match_prefix(req.origin_input_ids)

        extend_len = len(req.origin_input_ids) - matched_len

        if self.chunked_prefill_size > 0 and \
           extend_len > self.chunked_prefill_size:
            # 只处理前 chunk_size 个 token
            extend_len = self.chunked_prefill_size
            req.is_chunked = True

        req.extend_input_len = extend_len
        ...
```

分块后的请求在完成一个 chunk 的计算后，不会被标记为完成，而是重新回到等待队列（或以特殊状态保留），等待下一次迭代继续处理剩余部分。已计算的 KV Cache 被保留，后续 chunk 从断点处继续。

### Chunked Prefill 与 RadixCache 的交互

Chunked Prefill 与 RadixCache 的配合需要特殊处理。每个 chunk 计算完成后，对应的 KV Cache 会被插入 RadixCache：

```python
# 每个 chunk 完成后将 KV Cache 写入缓存
# 后续 chunk 可以复用前面 chunk 的缓存
# 如果请求中途被抢占，已计算的部分不会丢失
```

这意味着即使一个长序列的 prefill 被中断（例如因为优先级更高的请求到达），已完成 chunk 的 KV Cache 仍然安全地存储在 RadixCache 中，恢复时无需重新计算。

## PD 分离：Prefill-Decode 分离架构

### 为什么要分离 Prefill 和 Decode

Chunked Prefill 缓解了长 prefill 的阻塞问题，但在极高吞吐的场景中，更彻底的解决方案是将 Prefill 和 Decode 阶段部署在不同的 GPU 集群上。这就是 PD 分离（Prefill-Decode Disaggregation）架构。

Prefill 和 Decode 两个阶段有着截然不同的计算特征：

| 特征 | Prefill | Decode |
|------|---------|--------|
| 计算密度 | 高（计算密集型） | 低（显存带宽密集型） |
| token 处理数 | 多（整个输入序列） | 少（每次 1 个 token） |
| GPU 利用模式 | 高算力利用 | 高带宽利用 |
| 延迟敏感度 | 中等（TTFT） | 高（TPOT） |

将两者混合在同一 GPU 上运行时，decode 的低计算密度拖累了整体的 GPU 利用率，而 prefill 的高计算量又干扰了 decode 的延迟稳定性。

### SGLang 的 PD 分离实现

SGLang 支持 PD 分离部署模式，通过将 prefill 和 decode 部署为独立的服务实例来实现。相关的架构组件定义在 `python/sglang/srt/managers/` 目录下：

```python
# PD 分离的基本架构
# Prefill 节点: 专门处理 prefill 计算
# Decode 节点:  专门处理 decode 生成
# 控制层:       协调请求在两者之间的流转
```

PD 分离的工作流程如下：

1. **请求到达 Prefill 节点**：执行前缀匹配和 prefill 计算，生成完整的 KV Cache。
2. **KV Cache 传输**：将计算好的 KV Cache 通过高速互联（如 NVLink 或 RDMA）传输到 Decode 节点。
3. **Decode 节点接管**：在 Decode 节点上进行自回归生成，每步生成一个 token。
4. **结果返回**：生成完成后将结果返回给客户端。

### KV Cache 传输的挑战

PD 分离架构的核心技术挑战是 KV Cache 的跨节点传输。对于一个 Llama-70B 模型，单个序列 1024 token 的 KV Cache 大小约为数百 MB，传输延迟不可忽视。SGLang 通过以下策略优化传输效率：

- **流水线传输**：prefill 计算与 KV Cache 传输并行，layer 级别的流水线减少等待时间。
- **压缩传输**：对 KV Cache 进行量化压缩，减少传输数据量。
- **选择性传输**：结合 RadixCache，只传输 Decode 节点上不存在的 KV Cache 部分。

### 部署配置

SGLang 中 PD 分离的部署通过启动参数配置：

```python
# 启动 Prefill 节点
# python -m sglang.launch_server --dp-size 2 --disaggregation-mode prefill

# 启动 Decode 节点
# python -m sglang.launch_server --dp-size 4 --disaggregation-mode decode
```

Prefill 节点通常配置较少的 GPU 但使用高算力型号，Decode 节点则配置较多的 GPU 以支撑大量并发的 decode 请求。

## Chunked Prefill 与 PD 分离的适用场景

### 何时使用 Chunked Prefill

- 单机或少量 GPU 的部署场景
- 输入长度差异较大的混合工作负载
- 需要平衡 TTFT 和 TPOT 的交互式应用

### 何时使用 PD 分离

- 大规模分布式部署
- 超高吞吐需求的生产环境
- 输入序列普遍较长（如 RAG、文档处理）
- 有充足的硬件资源和高速网络互联

### 两者结合

在 PD 分离架构中，Prefill 节点内部仍然可以使用 Chunked Prefill。这样即使在专用的 Prefill 节点上，超长序列也不会完全阻塞其他 prefill 请求的处理。

## 本章小结

本章讨论了两种应对长序列 prefill 挑战的技术方案。Chunked Prefill 通过将长序列拆分为固定大小的块，使 prefill 计算可以与 decode 交替执行，解决了长 prefill 阻塞 decode 的问题。PD 分离架构则更进一步，将 prefill 和 decode 部署在不同的硬件上，使两个阶段各自按最优方式运行。SGLang 同时支持这两种策略，用户可以根据硬件条件和工作负载特征灵活选择。两者与 RadixCache 的协同配合，确保了 KV Cache 的高效管理和跨请求复用不受分块或分离部署的影响。
