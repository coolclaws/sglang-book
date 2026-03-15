---
title: 第 1 章：项目概览与设计哲学
---

# 第 1 章：项目概览与设计哲学

> "The best way to predict the future is to invent it." —— Alan Kay。SGLang 正是以这种精神，重新定义了大语言模型推理服务的边界。

## 为什么需要 SGLang

随着 GPT、LLaMA、DeepSeek 等大语言模型（LLM）的快速发展，推理服务（inference serving）成为了工业界和学术界共同面对的核心挑战。传统的推理框架往往在吞吐量和延迟之间难以兼顾，尤其在面对复杂的多轮对话、结构化输出、以及多模态任务时，性能瓶颈尤为突出。

在生产环境中，一个典型的 LLM 应用可能涉及以下复杂场景：用户发起多轮对话，每一轮都需要在前一轮的上下文基础上继续生成；系统需要从多个候选回复中选择最佳答案；生成的内容必须严格符合 JSON Schema 或特定的正则表达式格式。这些场景中，如果每次交互都独立发起推理请求，不仅浪费了大量已计算的中间结果，还会导致端到端延迟的显著增加。

SGLang（全称 Structured Generation Language）由 LMSYS 团队发起，定位为一个高性能的大语言模型推理服务框架。它支持从单 GPU 到分布式集群的多种部署方式，目前已在全球超过 40 万张 GPU 上运行，被 xAI、AMD、NVIDIA 等知名组织广泛采用。SGLang 项目的核心代码仓库托管在 GitHub 上的 `sgl-project/sglang`，使用 Python 作为主要开发语言，底层算子则使用 CUDA 和 Triton 实现高性能计算。

## SGLang 与 vLLM 的对比

vLLM 是当前最流行的 LLM 推理框架之一，以 PagedAttention 作为核心创新，通过类似操作系统虚拟内存的方式管理 KV Cache。SGLang 与 vLLM 的主要差异体现在以下几个方面：

### 核心缓存策略的差异

vLLM 使用 PagedAttention，将 KV Cache 按固定大小的 page 进行分配和回收，解决了内存碎片化问题，但本质上仍然是以单个请求为粒度管理缓存。SGLang 则提出了 **RadixAttention**，采用 radix tree（基数树）数据结构来管理跨请求的 KV Cache 前缀复用。这意味着当多个请求共享相同的 system prompt 或对话前缀时，SGLang 能够自动识别并复用已计算的 KV Cache，显著减少重复计算。

在实际的多轮对话场景中，RadixAttention 的优势尤为明显。假设有 100 个用户同时使用同一个应用，且这些请求共享相同的系统提示词（system prompt），RadixAttention 可以在基数树中缓存这段共享前缀的 KV Cache，所有后续请求直接从缓存中读取，而无需重复计算注意力。这在高并发场景下可以带来数倍的吞吐量提升。

### 编程模型与运行时协同设计

vLLM 主要提供 API 级别的推理服务，不涉及编程语言抽象。SGLang 的独特之处在于它提供了一套 **前端 DSL**（Domain-Specific Language），通过 `gen`、`select`、`fork`、`join` 等原语，让用户可以像编写普通 Python 程序一样描述复杂的 LLM 交互逻辑。前端 DSL 与后端运行时的协同设计（co-design）是 SGLang 最核心的设计哲学。这种设计使得运行时能够"看到"用户程序的完整结构，从而进行全局优化——例如提前识别可以复用的前缀、并行化独立的生成分支、以及批量调度相关联的请求。

### 调度策略的创新

SGLang 的 Scheduler 实现了 overlap 模式，能够将 CPU 调度与 GPU 计算交叠执行，减少空闲等待时间。在传统的推理框架中，CPU 端的请求调度和 GPU 端的模型计算是串行进行的，导致 GPU 在等待调度决策时处于空闲状态。SGLang 通过将调度逻辑放入独立的执行流（stream），使得当前批次的结果处理与下一批次的调度准备可以并行进行。同时，SGLang 还支持 prefill-decode 分离（disaggregation）、推测解码（speculative decoding）、连续批处理（continuous batching）等多种高级优化策略。

## 整体架构：前端与后端

SGLang 的架构清晰地分为三个层次：

```
┌─────────────────────────────────────────┐
│              Frontend (DSL)             │
│   python/sglang/lang/                   │
│   gen / select / fork / join            │
├─────────────────────────────────────────┤
│           Runtime Frontend              │
│   python/sglang/srt/entrypoints/        │
│   Engine / Runtime / HTTP Server        │
├─────────────────────────────────────────┤
│              Backend (SRT)              │
│   python/sglang/srt/managers/           │
│   Scheduler / ModelRunner / Cache       │
└─────────────────────────────────────────┘
```

- **前端层**（`python/sglang/lang/`）：提供 DSL 原语和 `SglFunction` 装饰器，负责将用户编写的结构化生成程序转换为可被运行时优化的中间表示（IR）。这一层是 SGLang 区别于其他框架的标志性特性。
- **Runtime 前端**（`python/sglang/srt/entrypoints/`）：包含 `Engine` 类和 HTTP 服务入口，负责接收来自用户或前端 DSL 的请求，进行分词处理，并将请求分发到后端调度器。它同时支持 HTTP 服务模式和嵌入式（in-process）模式两种部署方式。
- **后端层**（`python/sglang/srt/managers/`）：核心调度器 `Scheduler`、模型执行器、KV Cache 管理（包括 RadixAttention 的实现）、采样引擎、结构化输出约束等模块，负责实际的推理计算和资源管理。

## 设计哲学

SGLang 的设计哲学可以总结为以下几点核心原则：

1. **编程模型与运行时的协同设计**：前端 DSL 不只是语法糖，而是为运行时优化提供了关键的结构信息，例如前缀共享、分支合并、依赖关系等。运行时可以利用这些信息进行全局调度优化，这是单纯的 API 级框架无法实现的。
2. **RadixAttention 驱动的智能缓存**：通过基数树数据结构自动管理前缀复用，无需用户手动干预。缓存的粒度从"单个请求"提升到"共享前缀"，在多租户、多轮对话等场景下带来显著的性能提升。
3. **零开销调度**：通过 overlap 模式和高效的批处理策略，最大化 GPU 利用率。CPU 调度与 GPU 计算的流水线化消除了传统串行调度带来的性能损耗。
4. **广泛的硬件兼容性**：支持 NVIDIA、AMD、Intel、Google TPU 和华为 Ascend 等多种硬件平台。底层算子通过 `sgl-kernel/` 中的抽象层适配不同硬件。
5. **模型生态的全面支持**：兼容 LLaMA、Qwen、DeepSeek、Gemma 等主流大语言模型，同时支持多模态模型、嵌入模型和扩散模型。

## 性能优势总结

在标准的推理基准测试中，SGLang 相比其他框架展现出以下关键优势：在多轮对话场景中，得益于 RadixAttention 的前缀缓存，SGLang 可以避免重复计算共享前缀，带来显著的延迟降低和吞吐提升；在结构化输出场景中，SGLang 的约束解码引擎与采样流程深度集成，避免了额外的后处理开销；在高并发服务场景中，overlap 调度模式和连续批处理策略确保了 GPU 的高效利用。

## 本章小结

本章介绍了 SGLang 项目的背景、定位和核心设计哲学。SGLang 通过前端 DSL 与后端运行时的协同设计，以 RadixAttention 为核心创新，在复杂 LLM 应用场景中实现了显著的性能优势。与 vLLM 相比，SGLang 在缓存策略、编程抽象和调度优化三个维度上都有差异化的技术路线。在下一章中，我们将深入探索 SGLang 代码仓库的目录结构与模块依赖关系，为后续的源码分析打下基础。
