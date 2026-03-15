---
title: 第 8 章：与 PagedAttention 的对比
---

# 第 8 章：与 PagedAttention 的对比

> "理解一项技术的最佳方式，是将它与同领域的其他方案放在一起比较。"

## 引言

在 LLM 推理优化领域，vLLM 提出的 PagedAttention 和 SGLang 提出的 RadixAttention 是两个具有代表性的 KV Cache 管理方案。两者都致力于解决 KV Cache 的内存效率问题，但出发点和设计哲学截然不同。本章将从多个维度进行深入对比。

## PagedAttention 回顾

### 核心思想

PagedAttention 借鉴了操作系统中虚拟内存的分页机制。它将 KV Cache 分割为固定大小的"页"（Block），每页存储固定数量 token 的 KV 数据（通常为 16 个 token）。每个请求维护一个页表（Block Table），记录其 KV Cache 分布在哪些物理页中。

```
逻辑视图:  [token_0 ... token_15] [token_16 ... token_31] ...
             |                      |
             v                      v
物理页:     Block #7              Block #23              ...
```

### 解决的核心问题

传统 KV Cache 管理要求为每个序列预分配连续的显存空间，按最大可能长度分配。这导致了严重的内存碎片和浪费——一个可能生成 2048 个 token 的请求在实际只生成 100 个 token 时，浪费了 95% 的预分配空间。

PagedAttention 通过按需分配页来消除这种浪费，每次只分配当前需要的页数，随着生成的进行逐步追加新页。

## 对比维度一：内存利用率

### PagedAttention 的优势

PagedAttention 将内存浪费控制在最后一页的内部碎片（不足一页的部分），内存利用率接近最优。vLLM 的实验数据表明，相比传统方案，PagedAttention 可将内存浪费从 60-80% 降低至 4% 以下。

### RadixAttention 的优势

RadixAttention 同样基于 token 级别的内存池管理（`TokenToKVPool`），避免了连续分配的浪费。但它的额外优势在于——共享前缀只存储一份。当 100 个请求共享 500 token 的系统提示词时：

| 方案 | 该前缀占用的 KV Cache 空间 |
|------|--------------------------|
| PagedAttention（无共享） | 500 × 100 = 50,000 token |
| RadixAttention | 500 × 1 = 500 token |

这一差异在前缀共享率高的场景中意味着数量级的显存节省。

## 对比维度二：前缀共享能力

这是两者最本质的区别。

### PagedAttention 的前缀共享

vLLM 在后续版本中也引入了前缀共享支持（Automatic Prefix Caching），但其实现基于页级别的哈希匹配。每个物理页计算其包含 token 的哈希值，新请求通过哈希匹配来发现可共享的页。

这种方式存在局限：

1. **页对齐约束**：前缀必须恰好对齐到页边界才能被匹配，否则需要额外处理。
2. **哈希冲突**：理论上存在哈希冲突的可能性（尽管概率极低）。
3. **非增量式匹配**：无法高效处理部分匹配的情况。

### RadixAttention 的前缀共享

RadixAttention 天然支持任意粒度的前缀共享：

```python
# python/sglang/srt/mem_cache/radix_cache.py
# match_prefix 返回精确匹配的 token 数量
value, node, matched_len = cache.match_prefix(token_ids)
# matched_len 可以是任意值，不受页大小限制
```

Radix Tree 的结构保证了前缀匹配的精确性和高效性，时间复杂度与序列长度成线性关系，且不存在对齐约束。

## 对比维度三：架构设计哲学

### PagedAttention：通用优化

PagedAttention 的设计目标是通用的内存效率优化。它不假设请求之间存在任何关联，每个请求被独立对待。前缀共享是后续添加的功能增强，而非核心设计驱动力。

这种设计适合以下场景：
- 请求之间前缀重叠较少
- 单轮独立对话为主
- 请求长度差异大，碎片化严重

### RadixAttention：前缀复用驱动

RadixAttention 的设计从一开始就以前缀复用为核心目标。整个系统架构——从 Radix Tree 数据结构到调度策略——都围绕最大化前缀命中率来设计。

SGLang 的调度器在选择下一批处理的请求时，会优先选择与已有缓存前缀重叠度高的请求，这种"缓存感知调度"（Cache-Aware Scheduling）在 `python/sglang/srt/managers/scheduler.py` 中实现：

```python
# python/sglang/srt/managers/scheduler.py
# 调度器在选择请求时考虑前缀匹配长度
# 优先调度能最大化利用已有 KV Cache 的请求
```

## 对比维度四：性能基准分析

根据 SGLang 论文和社区基准测试，两种方案在不同工作负载下的表现差异明显：

### 前缀共享密集场景

在共享系统提示词的多用户并发场景中，SGLang 的 RadixAttention 相比 vLLM 可实现：
- TTFT（首 token 延迟）降低 3-5 倍
- 吞吐量提升 2-4 倍

关键原因是大量重复的 prefill 计算被完全避免。

### 独立请求场景

在请求之间无前缀重叠的场景中，两者的性能差异较小。RadixAttention 仍然通过 token 级内存管理保持良好的显存利用率，但无法发挥前缀共享的优势。此时 RadixCache 的树管理开销成为轻微的额外成本。

## 两者的融合趋势

值得注意的是，两种方案并非完全互斥。SGLang 的底层内存管理同样采用了类似 PagedAttention 的分块管理思想，通过 `TokenToKVPool` 和 `ReqToTokenPool` 实现高效的显存分配：

```python
# python/sglang/srt/mem_cache/memory_pool.py
class ReqToTokenPool:
    """管理请求到 token 槽位的映射"""
    ...

class TokenToKVPool:
    """管理 token 槽位到实际 KV Cache 显存的映射"""
    ...
```

可以说，SGLang 在 PagedAttention 的内存管理基础上，叠加了 Radix Tree 的前缀索引层，实现了两种优化的融合。

## 如何选择

| 考虑因素 | 推荐方案 |
|---------|---------|
| 大量共享前缀（聊天、RAG） | RadixAttention（SGLang） |
| 无前缀共享的独立请求 | 两者差异不大 |
| 多轮对话场景 | RadixAttention（SGLang） |
| 生态和工具链集成 | 根据具体需求评估 |
| Few-shot 批量推理 | RadixAttention（SGLang） |

## 本章小结

本章从内存利用率、前缀共享能力、架构设计哲学和性能基准四个维度对比了 PagedAttention 与 RadixAttention。PagedAttention 通过分页机制解决了连续分配的内存浪费问题，是一项通用性强的优化。RadixAttention 则以前缀复用为核心驱动力，通过 Radix Tree 实现了高效的跨请求 KV Cache 共享。在前缀共享普遍存在的实际生产场景中，RadixAttention 展现出显著的性能优势。两种技术并非互斥，SGLang 的实现实际上融合了两者的优点。
