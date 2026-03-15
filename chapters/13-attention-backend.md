---
title: 第 13 章：Attention Backend
---

# 第 13 章：Attention Backend

> "注意力机制是 Transformer 的灵魂，而 Attention Backend 决定了这个灵魂运转的效率。选择合适的后端，就是选择性能的上限。"

注意力计算是大语言模型推理中最关键的性能瓶颈之一。SGLang 通过抽象的 Attention Backend 层，支持多种高性能注意力实现，并能根据硬件环境和模型特性灵活切换。本章将深入剖析 FlashInfer、Triton 和 FlashAttention 三大后端的实现与选择策略。

## 13.1 Attention Backend 抽象层

SGLang 在 `python/sglang/srt/layers/attention/` 目录下定义了统一的 Attention Backend 抽象。这套设计允许上层代码无需关心具体的注意力实现细节，只需通过统一接口调用即可。

```python
# python/sglang/srt/layers/attention/base_attention_backend.py
class AttentionBackend:
    def init_forward_metadata(self, forward_batch):
        """准备 forward 所需的注意力元数据"""
        raise NotImplementedError

    def forward(self, q, k, v, forward_batch, ...):
        """执行注意力计算"""
        raise NotImplementedError
```

这种抽象设计的好处是显而易见的：当出现更快的注意力实现时，只需新增一个 Backend 类，而无需修改模型代码。同时，不同的硬件平台（如 NVIDIA A100 与 H100）可以选择各自最优的后端。

### 13.1.1 Backend 的注册与选择

Backend 的选择发生在 `ModelRunner` 初始化阶段，通过配置参数或自动检测来决定：

```python
# python/sglang/srt/layers/attention/__init__.py
def get_attention_backend(server_args):
    if server_args.attention_backend == "flashinfer":
        return FlashInferAttnBackend
    elif server_args.attention_backend == "triton":
        return TritonAttnBackend
    elif server_args.attention_backend == "flash_attn":
        return FlashAttentionBackend
```

用户可以通过 `--attention-backend` 参数显式指定后端，也可以让系统根据已安装的库自动选择。

## 13.2 FlashInfer Backend

FlashInfer 是 SGLang 默认推荐的 Attention Backend，也是性能最优的选择之一。它由 SGLang 团队深度参与开发，专为 LLM serving 场景优化。

```python
# python/sglang/srt/layers/attention/flashinfer_backend.py
class FlashInferAttnBackend(AttentionBackend):
    def init_forward_metadata(self, forward_batch):
        # 构建 ragged tensor 索引，用于描述 batch 中各序列的长度
        self.prefill_wrapper.plan(
            qo_indptr=forward_batch.qo_indptr,
            kv_indptr=forward_batch.kv_indptr,
            kv_indices=forward_batch.kv_indices,
            ...
        )
```

FlashInfer 的核心优势在于：

- **Ragged Tensor 支持**：原生处理不等长序列的 batch，无需 padding
- **Paged KV Cache**：与 SGLang 的 token 级别 KV Cache 管理完美配合
- **融合算子**：将 RoPE、softmax 等操作与注意力计算融合，减少显存访问
- **Decode 优化**：针对 decode 阶段（每个序列只有一个 query token）做了专门的 kernel 优化

### 13.2.1 Prefill 与 Decode 的分离处理

FlashInfer 对 prefill 和 decode 使用不同的 wrapper，因为两者的计算模式截然不同：

```python
# prefill：多个 query token，使用 ragged attention
self.prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(...)

# decode：每个序列一个 query token，使用 paged KV cache
self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(...)
```

## 13.3 Triton Backend

Triton Backend 基于 OpenAI 的 Triton 编译器实现，是一个纯 Python 编写的注意力后端。它的主要价值在于跨平台兼容性和可调试性。

```python
# python/sglang/srt/layers/attention/triton_backend.py
class TritonAttnBackend(AttentionBackend):
    def forward(self, q, k, v, forward_batch, ...):
        if forward_batch.forward_mode.is_decode():
            return triton_decode_attention(q, k, v, ...)
        else:
            return triton_prefill_attention(q, k, v, ...)
```

Triton Backend 的实现位于 `python/sglang/srt/layers/attention/triton_ops/` 目录下，包含手写的 Triton kernel。虽然性能不如 FlashInfer 的手工 CUDA kernel，但它具备以下优势：

- **无需额外安装**：只要有 Triton（PyTorch 自带），就可以使用
- **易于修改**：Triton kernel 用 Python 编写，便于研究者实验新的注意力变体
- **跨硬件支持**：理论上可以运行在任何 Triton 支持的 GPU 上

## 13.4 FlashAttention Backend

FlashAttention Backend 封装了 Dao-AILab 的 FlashAttention 库，这是业界最知名的高效注意力实现之一。

```python
# python/sglang/srt/layers/attention/flashattn_backend.py
class FlashAttentionBackend(AttentionBackend):
    def forward(self, q, k, v, forward_batch, ...):
        # 调用 flash_attn 库的接口
        output = flash_attn_with_kvcache(
            q, k_cache, v_cache,
            cache_seqlens=forward_batch.seq_lens,
            softmax_scale=self.scale,
            causal=True,
        )
        return output
```

FlashAttention 的核心算法通过分块计算（tiling）和重计算（recomputation）来减少 HBM 访问次数，实现了 IO-aware 的注意力计算。

## 13.5 性能特征对比

三种后端各有特色，适用于不同场景：

| 特性 | FlashInfer | Triton | FlashAttention |
|------|-----------|--------|----------------|
| Decode 性能 | 最优 | 中等 | 良好 |
| Prefill 性能 | 优秀 | 中等 | 优秀 |
| 安装依赖 | 需安装 flashinfer | 无额外依赖 | 需安装 flash-attn |
| Paged KV Cache | 原生支持 | 自定义实现 | 部分支持 |
| 自定义扩展 | 困难 | 容易 | 困难 |

在生产环境中，推荐使用 FlashInfer 以获得最佳性能。在开发和调试阶段，Triton Backend 则是更灵活的选择。

## 本章小结

本章分析了 SGLang 的 Attention Backend 抽象层及三种主要实现。通过统一的 `AttentionBackend` 接口，SGLang 实现了注意力计算的可插拔设计。FlashInfer 以其原生的 paged KV cache 支持和优化的 decode kernel 成为默认首选；Triton Backend 提供了良好的可移植性和可扩展性；FlashAttention 则作为成熟的第三方方案提供了稳定可靠的性能。理解这三种后端的特性与适用场景，有助于在实际部署中做出最优选择。
