---
layout: home

hero:
  name: SGLang 源码解析
  text: RadixAttention 与结构化生成推理运行时深度剖析
  tagline: 从前端 DSL 到 RadixAttention，从 Scheduler 到 Attention Backend，全面拆解 SGLang 推理引擎
  image:
    src: /logo.png
    alt: SGLang
  actions:
    - theme: brand
      text: 开始阅读 →
      link: /chapters/01-project-overview
    - theme: alt
      text: GitHub 源码
      link: https://github.com/sgl-project/sglang

features:
  - icon:
      src: /icons/radix-tree.svg
    title: RadixAttention 核心
    details: 深入前缀共享树设计，解析 RadixCache 实现、Token 复用策略与 PagedAttention 对比，理解 SGLang 高效缓存的核心机制。
    link: /chapters/06-radix-attention-design
  - icon:
      src: /icons/batch.svg
    title: 调度与批处理引擎
    details: 剖析 Scheduler 三队列架构、Continuous Batching 动态调度、Chunked Prefill 与 PD 分离推理，掌握高吞吐批处理的实现细节。
    link: /chapters/09-scheduler
  - icon:
      src: /icons/structured.svg
    title: 结构化生成
    details: 解读 Outlines 集成、Jump Forward Decoding 加速、Regex/JSON Schema 约束解码，理解结构化输出的完整实现路径。
    link: /chapters/16-outlines-integration
  - icon:
      src: /icons/runtime.svg
    title: 推理运行时全景
    details: 覆盖前端 DSL、Model Runner、Attention Backend、分布式执行与量化支持，从源码理解 SGLang 的完整推理运行时架构。
    link: /chapters/12-model-runner
---
