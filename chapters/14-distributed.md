---
title: 第 14 章：分布式执行
---

# 第 14 章：分布式执行

> "单卡的算力终有上限，分布式才是通往更大模型的必经之路。SGLang 的分布式设计，让多 GPU 协作如同单卡般流畅。"

随着大语言模型规模不断增长，单个 GPU 的内存和算力已无法满足需求。SGLang 实现了多种分布式执行策略，包括 Tensor Parallelism（TP）、Pipeline Parallelism（PP）和 Expert Parallelism（EP），以支持多 GPU 甚至多节点部署。本章将详细剖析这些并行策略的实现细节。

## 14.1 分布式架构概览

SGLang 的分布式执行架构由以下核心组件构成：

- **TpModelWorker**：每个 GPU 上运行的工作进程，负责执行模型的一个分片
- **TpModelWorkerClient**：Scheduler 端的代理，负责向 Worker 发送请求
- **NCCL 通信**：GPU 之间的高速数据传输通道

整体架构定义在 `python/sglang/srt/model_executor/` 目录下：

```python
# python/sglang/srt/model_executor/tp_model_worker.py
class TpModelWorker:
    def __init__(self, server_args, tp_rank, tp_size, ...):
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.model_runner = ModelRunner(
            model_config=model_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            ...
        )
```

## 14.2 Tensor Parallelism（TP）

Tensor Parallelism 是 SGLang 最核心的并行策略，它将模型的每一层在多个 GPU 之间进行切分，每个 GPU 只持有权重的一部分。

### 14.2.1 切分策略

TP 的切分发生在模型的线性层。对于 Transformer 中的 Attention 和 FFN 层，SGLang 采用 Megatron-LM 风格的切分方式：

```python
# python/sglang/srt/layers/linear.py
class ColumnParallelLinear(nn.Module):
    """按列切分的线性层，每个 GPU 持有输出维度的 1/tp_size"""
    def __init__(self, input_size, output_size, ...):
        self.output_size_per_partition = output_size // tp_size
        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, input_size)
        )

class RowParallelLinear(nn.Module):
    """按行切分的线性层，每个 GPU 持有输入维度的 1/tp_size"""
    def __init__(self, input_size, output_size, ...):
        self.input_size_per_partition = input_size // tp_size
        self.weight = nn.Parameter(
            torch.empty(output_size, self.input_size_per_partition)
        )
```

### 14.2.2 AllReduce 通信

TP 中最关键的通信操作是 AllReduce。在每个 Transformer 层中，Attention 和 FFN 各需要一次 AllReduce 来聚合各 GPU 的部分结果：

```python
# python/sglang/srt/layers/linear.py
class RowParallelLinear(nn.Module):
    def forward(self, x):
        output = F.linear(x, self.weight)
        # AllReduce：将所有 GPU 上的部分和累加
        if self.tp_size > 1:
            torch.distributed.all_reduce(output)
        return output
```

SGLang 使用 NCCL（NVIDIA Collective Communications Library）执行 AllReduce，NCCL 利用 NVLink 或 PCIe 实现 GPU 间的高带宽通信。

## 14.3 Pipeline Parallelism（PP）

Pipeline Parallelism 将模型按层划分为多个阶段（stage），每个阶段分配到不同的 GPU 上。与 TP 相比，PP 的通信量更小（只需传递激活值），适合跨节点部署。

```python
# PP 的层分配逻辑
# 假设模型有 32 层，pp_size = 4
# GPU 0: layers 0-7
# GPU 1: layers 8-15
# GPU 2: layers 16-23
# GPU 3: layers 24-31
```

SGLang 在 `python/sglang/srt/model_executor/` 中实现了 PP 支持，通过 `pp_rank` 和 `pp_size` 参数控制层的分配。PP 模式下，各 stage 之间通过点对点通信（P2P Send/Recv）传递中间激活值。

### 14.3.1 PP 与 TP 的组合

在实际部署中，PP 和 TP 通常组合使用。例如，对于一个 8 GPU 的部署，可以设置 `tp_size=4, pp_size=2`，意味着每 4 个 GPU 组成一个 TP 组，共 2 个 PP stage：

```
Stage 0: GPU 0,1,2,3 (TP group)
Stage 1: GPU 4,5,6,7 (TP group)
```

## 14.4 Expert Parallelism（EP）

Expert Parallelism 专门为 Mixture of Experts（MoE）模型设计。MoE 模型中每个 Transformer 层包含多个"专家"网络，EP 将不同的专家分布到不同 GPU 上。

```python
# python/sglang/srt/layers/moe/
# EP 模式下的 token 路由
class EPMoE(nn.Module):
    def forward(self, x, router_logits):
        # 1. 计算路由权重，决定每个 token 发送给哪些专家
        routing_weights = self.gate(router_logits)
        # 2. All-to-All 通信：将 token 发送到对应专家所在的 GPU
        dispatched = all_to_all(x, routing_weights, ...)
        # 3. 各 GPU 上的本地专家执行计算
        expert_output = self.experts(dispatched)
        # 4. All-to-All 通信：将结果返回原始 GPU
        output = all_to_all(expert_output, ...)
        return output
```

EP 的核心挑战在于 All-to-All 通信的效率。SGLang 通过以下优化减轻通信开销：

- **Token 选择优化**：减少需要跨 GPU 传输的 token 数量
- **通信计算重叠**：在等待 All-to-All 完成的同时执行其他计算
- **专家负载均衡**：通过辅助损失和容量因子确保各 GPU 工作量均匀

## 14.5 进程启动与通信初始化

SGLang 使用 PyTorch 的 `distributed` 模块管理分布式进程组。在服务启动时，会根据 `tp_size` 和 `pp_size` 参数创建相应数量的工作进程：

```python
# python/sglang/srt/utils.py
def init_distributed_environment(tp_rank, tp_size, ...):
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=tp_size,
        rank=tp_rank,
    )
    torch.cuda.set_device(tp_rank)
```

每个 `TpModelWorker` 运行在独立的进程中，拥有自己的 CUDA context。Scheduler 通过 `TpModelWorkerClient` 向所有 Worker 广播指令，Worker 执行完成后将结果（仅 rank 0 的结果）返回给 Scheduler。

## 本章小结

本章全面分析了 SGLang 的分布式执行机制。Tensor Parallelism 通过切分模型权重实现单层内的并行，是多 GPU serving 的基础方案；Pipeline Parallelism 通过层间划分降低跨节点通信量；Expert Parallelism 则专门针对 MoE 模型的特殊结构进行优化。在实际部署中，这些策略可以灵活组合，以适应不同的硬件拓扑和模型规模。理解这些并行策略的原理和实现，对于大模型的高效部署至关重要。
