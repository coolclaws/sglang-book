---
title: 第 12 章：Model Runner
---

# 第 12 章：Model Runner

> "模型推理的最后一公里，是将高层抽象的请求翻译成 GPU 可以执行的 forward call。Model Runner 正是这座桥梁。"

Model Runner 是 SGLang 执行后端的核心组件，负责将 Scheduler 组织好的批次请求转化为实际的模型 forward 调用。本章将深入分析 `ModelRunner` 的设计与实现，涵盖 forward 调用流程、CUDA Graph 优化、输入元数据准备以及权重管理。

## 12.1 ModelRunner 的定位与职责

在 SGLang 的架构中，Scheduler 负责决定"做什么"，而 `ModelRunner` 负责"怎么做"。它位于 `python/sglang/srt/model_executor/model_runner.py`，承担以下核心职责：

- 加载模型权重并初始化模型结构
- 准备每次 forward 调用所需的输入元数据
- 管理 CUDA Graph 的捕获与重放
- 协调 Attention Backend 的选择与调用

```python
# python/sglang/srt/model_executor/model_runner.py
class ModelRunner:
    def __init__(self, model_config, mem_fraction, tp_rank, ...):
        self.model = None
        self.device = torch.device(f"cuda:{tp_rank}")
        # 初始化模型、分词器、内存池等
```

`ModelRunner` 在初始化阶段完成模型加载、内存分配和 CUDA Graph 预捕获等准备工作，为后续高效推理奠定基础。

## 12.2 Forward 调用流程

每一次推理请求最终都会进入 `ModelRunner` 的 `forward` 方法。这个方法是整个推理管线的关键路径，其调用流程如下：

1. **输入准备**：将 batch 中的 token ids、position ids 等信息组装为张量
2. **元数据构建**：创建 `InputMetadata` 对象，包含 attention mask、序列长度等信息
3. **模型执行**：调用底层模型的 `forward` 方法
4. **采样输出**：对 logits 进行采样，获得下一个 token

```python
# forward 调用的核心逻辑
def forward(self, batch: ScheduleBatch):
    input_metadata = self.prepare_input_metadata(batch)
    logits = self.model.forward(
        input_ids=batch.input_ids,
        positions=batch.positions,
        input_metadata=input_metadata,
    )
    return logits
```

### 12.2.1 InputMetadata 的构建

`InputMetadata`（也称为 `ForwardBatch`）是连接 Scheduler 和模型执行的数据桥梁，定义在 `python/sglang/srt/model_executor/forward_batch_info.py` 中。它封装了一次 forward 所需的全部上下文信息：

```python
# python/sglang/srt/model_executor/forward_batch_info.py
class ForwardBatch:
    batch_size: int
    input_ids: torch.Tensor
    seq_lens: torch.Tensor
    req_pool_indices: torch.Tensor
    out_cache_loc: torch.Tensor
    forward_mode: ForwardMode  # PREFILL 或 DECODE
```

`forward_mode` 字段区分了 prefill 和 decode 两种模式，这对 Attention Backend 的选择至关重要。Prefill 模式处理完整的输入序列，而 decode 模式每次只处理一个新 token。

## 12.3 CUDA Graph 捕获与重放

CUDA Graph 是 SGLang 实现高性能 decode 的关键优化手段。其核心思想是：将一系列 GPU 操作"录制"下来，后续可以直接"重放"，避免了每次调用时的 CPU 端 kernel launch 开销。

### 12.3.1 捕获阶段

在服务启动时，`ModelRunner` 会针对不同的 batch size 预先捕获 CUDA Graph：

```python
# python/sglang/srt/model_executor/cuda_graph_runner.py
class CudaGraphRunner:
    def __init__(self, model_runner, max_batch_size):
        self.graphs = {}
        self.capture_batch_sizes = [1, 2, 4, 8, 16, 32, ...]

    def capture(self, batch_size):
        # 使用虚拟输入执行一次 forward，录制为 CUDA Graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.model_runner.forward_decode(dummy_batch)
        self.graphs[batch_size] = graph
```

### 12.3.2 重放阶段

在实际推理时，根据当前 batch size 找到最接近的已捕获 graph，将真实输入拷贝到 graph 的输入缓冲区，然后直接重放：

```python
def replay(self, batch):
    # 将实际输入拷贝到 graph 的输入缓冲区
    self.input_buffers["input_ids"][:batch_size].copy_(batch.input_ids)
    # 重放 CUDA Graph
    self.graphs[padded_batch_size].replay()
    return self.output_buffers["logits"][:batch_size]
```

CUDA Graph 仅用于 decode 阶段，因为 decode 阶段的计算模式相对固定（每个序列只处理一个 token），非常适合图捕获。Prefill 阶段由于输入长度变化较大，通常不使用 CUDA Graph。

## 12.4 权重加载与内存管理

模型权重的加载是 `ModelRunner` 初始化过程中最重要的步骤之一。SGLang 支持从 HuggingFace Hub 加载模型，并对权重进行分片、转换等处理。

### 12.4.1 权重加载流程

```python
# 权重加载的简化流程
def load_model(self):
    # 1. 根据 model_config 确定模型架构
    model_class = get_model_cls(self.model_config.hf_config)
    # 2. 在 meta device 上初始化模型结构（不分配实际内存）
    with torch.device("meta"):
        model = model_class(self.model_config)
    # 3. 加载权重文件并填充参数
    load_model_weights(model, self.model_config.path)
    return model
```

### 12.4.2 内存管理

`ModelRunner` 需要精心管理 GPU 内存，主要包含以下几个部分：

- **模型权重**：占用固定内存，在初始化时分配
- **KV Cache**：由 `TokenToKVPool` 管理，占用大部分剩余内存
- **CUDA Graph 缓冲区**：每个捕获的 graph 需要额外的输入输出缓冲区
- **临时激活值**：forward 过程中的中间结果

SGLang 通过 `mem_fraction_static` 参数控制模型权重与 KV Cache 之间的内存分配比例，确保在有限的 GPU 内存中实现最优的吞吐量。

## 本章小结

本章详细分析了 `ModelRunner` 的核心设计与实现。`ModelRunner` 作为 SGLang 执行后端的中枢，通过 `ForwardBatch` 组织输入元数据，利用 CUDA Graph 优化 decode 性能，并通过精细的内存管理确保 GPU 资源的高效利用。理解 `ModelRunner` 的工作原理，是深入掌握 SGLang 推理引擎的关键一步。下一章我们将进入 Attention Backend，探索不同注意力计算后端的实现细节。
