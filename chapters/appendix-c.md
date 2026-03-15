---
title: 附录 C：名词解释
---

# 附录 C：名词解释

本附录收录了本书中频繁出现的专业术语，按字母顺序排列，供读者随时查阅。

### Chunked Prefill（分块预填充）

将长 prompt 的 prefill 计算分割为多个较小的 chunk 逐步执行的技术。这样做可以避免长 prompt 独占 GPU 计算资源，使 prefill 和 decode 阶段能够交替执行，从而降低 decode 请求的排队延迟。

### Continuous Batching（连续批处理）

一种动态批处理策略，与传统的静态批处理不同。当批次中的某个请求完成生成后，新的请求可以立即加入当前批次，无需等待整个批次完成。这大幅提升了 GPU 利用率和系统吞吐量。

### CUDA Graph

NVIDIA 提供的一种优化技术，将一系列 GPU kernel 调用预先录制为一个图（Graph），后续执行时可以一次性提交整个图，避免逐个 kernel 的 CPU 端启动开销。SGLang 在 decode 阶段使用 CUDA Graph 来减少 kernel launch 延迟。

### Expert Parallelism（专家并行）

针对 Mixture of Experts（MoE）模型的并行策略。将不同的 expert 子网络分布到不同的 GPU 上，每个 GPU 只负责处理被路由到其上的 expert 计算，通过 All-to-All 通信交换 token 数据。

### FlashInfer

一个高性能的注意力计算库，为 LLM 推理场景优化。SGLang 默认使用 FlashInfer 作为注意力计算后端，它支持 Paged KV Cache、Ragged Tensor 等多种内存布局，提供了优于 FlashAttention 的推理性能。

### FSM（Finite State Machine，有限状态机）

在结构化输出中用于约束 token 生成的控制机制。将正则表达式或 JSON Schema 编译为有限状态机，在每一步解码时根据当前状态过滤合法的 token，确保输出严格符合指定格式。

### Jump Forward Decoding（跳跃前进解码）

SGLang 原创的结构化输出加速技术。当 FSM 进入确定性状态（即只有唯一合法的后续 token 序列）时，直接跳过逐 token 解码过程，一次性填充确定性内容。这在生成包含固定格式元素（如 JSON 的花括号、引号等）时能显著提升速度。

### KV Cache

Key-Value Cache 的缩写。在 Transformer 模型的自回归生成过程中，缓存已计算的 Key 和 Value 矩阵，避免对历史 token 的重复计算。KV Cache 的管理效率直接影响推理引擎的性能和内存利用率。

### PagedAttention（分页注意力）

由 vLLM 提出的 KV Cache 管理技术，借鉴了操作系统的虚拟内存分页机制。将 KV Cache 分割为固定大小的 block（页），通过 block table 维护逻辑到物理的映射关系，消除了内存碎片问题。

### Pipeline Parallelism（流水线并行）

将模型的不同层分布到不同的 GPU 上，数据在 GPU 之间按层序流动的并行方式。适用于单卡无法容纳完整模型但不需要 Tensor Parallelism 的场景。可以与 Tensor Parallelism 结合使用。

### RadixAttention

SGLang 原创的 KV Cache 管理与复用技术。使用 Radix Tree（基数树）数据结构存储和索引 KV Cache，使得具有共同前缀的请求能够自动共享缓存。这在系统提示复用、多轮对话、Few-shot 学习等场景中可以大幅减少重复计算。

### Tensor Parallelism（张量并行）

将模型单层内部的权重矩阵沿特定维度切分到多个 GPU 上并行计算的策略。每个 GPU 持有权重的一部分，通过 All-Reduce 通信同步中间结果。适用于单卡显存不足以容纳模型权重的场景。

### TTFT（Time To First Token）

首 Token 延迟，指从收到请求到返回第一个生成 token 的时间。这一指标主要受 prefill 阶段的计算时间影响，在交互式应用中是衡量用户体验的关键指标。

### TPOT（Time Per Output Token）

每个输出 Token 的平均生成时间，衡量 decode 阶段的速度。TPOT 越低，用户感知到的生成速度越快。
