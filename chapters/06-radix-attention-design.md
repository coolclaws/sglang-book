---
title: 第 6 章：RadixAttention 设计原理
---

# 第 6 章：RadixAttention 设计原理

> "在 LLM 推理中，最昂贵的计算并非生成本身，而是反复计算那些本可复用的前缀。"

## 引言：为什么需要前缀共享

在大语言模型的实际服务场景中，大量请求共享相同的前缀。典型的例子包括：

- **共享系统提示词（System Prompt）**：同一个应用的所有用户请求都携带相同的系统指令，这些指令可能长达数千 token。
- **多轮对话（Multi-turn Conversation）**：用户与模型的多轮交互中，历史对话内容在每次请求时都需要重新处理。
- **少样本提示（Few-shot Prompting）**：多个请求使用相同的示例集合作为前缀。

传统的 LLM 推理框架对每个请求独立计算 KV Cache，即使两个请求的前 500 个 token 完全相同，也需要各自进行一次完整的 prefill 计算。这意味着大量的 GPU 算力和显存被浪费在重复计算上。

SGLang 提出的 RadixAttention 正是为了解决这一核心瓶颈。

## Radix Tree 的基本概念

Radix Tree（基数树），也称为压缩前缀树（Compressed Trie），是一种空间优化的前缀树数据结构。与普通的 Trie 相比，Radix Tree 将只有单个子节点的节点进行合并压缩，从而大幅减少树的深度和节点数量。

在 SGLang 中，Radix Tree 被用来管理 KV Cache 的存储与复用。每个节点代表一段连续的 token 序列及其对应的 KV Cache 数据。树的路径从根节点到叶节点代表一个完整（或部分）的 token 序列。

```
          Root
         /    \
    [System Prompt]    [另一个前缀]
      /       \
 [用户A对话]  [用户B对话]
```

在上述结构中，System Prompt 的 KV Cache 只需计算和存储一次，所有共享该前缀的请求都可以直接复用，无需重复计算。

## RadixAttention 的核心设计

### 与传统 KV Cache 的本质区别

传统的 KV Cache 管理方式是"每请求独占"——每个请求拥有自己独立的 KV Cache 空间，请求结束后立即释放。这种设计简单直接，但完全无法利用请求之间的前缀共享特性。

RadixAttention 将 KV Cache 的生命周期从"跟随请求"转变为"跟随内容"。相同内容的 KV Cache 只存储一份，多个请求通过树节点的引用来共享访问。其核心设计原则包括：

1. **内容寻址**：以 token 序列的内容作为索引键，而非请求 ID。
2. **自动前缀匹配**：新请求到达时，自动沿 Radix Tree 匹配已有的最长前缀。
3. **增量计算**：只需为未匹配到的后缀部分计算新的 KV Cache。

### 树结构与节点管理

RadixAttention 的树结构定义在 `python/sglang/srt/mem_cache/radix_cache.py` 中。每个 `TreeNode` 包含以下关键信息：

```python
# python/sglang/srt/mem_cache/radix_cache.py
class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.key = None           # token 序列片段
        self.value = None         # 对应的 KV Cache 索引
        self.lock_ref = 0         # 引用计数
        self.last_access_time = 0 # LRU 驱逐用
```

节点通过 `lock_ref` 引用计数来追踪有多少活跃请求正在使用该节点的 KV Cache。当引用计数降为零时，该节点变为可驱逐状态，但不会立即释放——它仍然保留在树中，以便后续请求可能的复用。

### 前缀匹配的工作流程

当一个新请求到达时，RadixAttention 执行以下步骤：

1. **前缀匹配**：从根节点开始，逐步匹配请求的 token 序列，找到最长的已有前缀。
2. **节点分裂**：如果匹配在某个节点的中间位置停止（即只匹配了节点 token 序列的一部分），则需要将该节点分裂为两个节点。
3. **插入新节点**：为未匹配的后缀创建新节点，并分配对应的 KV Cache 空间。
4. **增加引用**：对匹配路径上的所有节点增加引用计数。

这一流程确保了 KV Cache 的最大化复用，同时保持了树结构的一致性。

## 实际场景中的收益分析

### 多轮对话场景

假设系统提示词为 200 个 token，用户已有 3 轮对话共计 800 个 token。在传统方案中，第 4 轮对话需要对全部 1000 个 token 重新计算 KV Cache。而在 RadixAttention 下，这 1000 个 token 的 KV Cache 已经存在于 Radix Tree 中，新请求只需匹配前缀即可直接复用，仅需为新增的用户输入计算 KV Cache。

### 批量请求场景

当多个用户同时使用相同的 few-shot 模板时，模板部分的 KV Cache 在第一个请求计算后即被缓存。后续请求的 prefill 时间可减少 50% 以上，TTFT（Time To First Token）显著降低。

## RadixAttention 的设计权衡

RadixAttention 并非没有代价。引入 Radix Tree 带来了额外的管理开销：

- **内存碎片化**：树节点管理的 KV Cache 块可能分散在显存中，需要与内存池配合管理。
- **并发控制**：多个请求同时访问和修改树结构时需要正确的同步机制。
- **驱逐策略复杂度**：需要在缓存命中率和显存利用率之间取得平衡。

尽管如此，在前缀共享普遍存在的实际工作负载中，RadixAttention 带来的性能收益远大于其管理开销。

## 本章小结

本章介绍了 SGLang 的核心创新之一——RadixAttention 的设计原理。通过将 Radix Tree 数据结构应用于 KV Cache 管理，SGLang 实现了跨请求的自动前缀共享，从根本上消除了重复的 prefill 计算。RadixAttention 以"内容寻址"取代"请求独占"的 KV Cache 管理模式，通过前缀匹配、节点分裂和引用计数等机制，在保持系统一致性的同时最大化了缓存复用率。下一章我们将深入 `radix_cache.py` 的具体实现，分析 RadixCache 类的核心算法细节。
