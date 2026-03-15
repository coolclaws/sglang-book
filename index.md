---
layout: home

hero:
  name: SGLang 源码解析
  text: RadixAttention 与结构化生成推理运行时深度剖析
  tagline: 从前端 DSL 到 RadixAttention，从 Scheduler 到 Attention Backend，全面拆解 SGLang 推理引擎
  image:
    src: /icons/logo.svg
    alt: SGLang
  actions:
    - theme: brand
      text: 开始阅读 →
      link: /chapters/01-project-overview
    - theme: alt
      text: GitHub 源码
      link: https://github.com/sgl-project/sglang

features:
  - icon: 🌲
    title: 第一部分：宏观认知
    details: 项目概览与设计哲学、Repo 结构与模块依赖，建立全局视角
    link: /chapters/01-project-overview
  - icon: 💬
    title: 第二部分：前端语言
    details: SGLang DSL 原语设计、Runtime 接口、Sampling 参数与结构化输出
    link: /chapters/03-frontend-dsl
  - icon: 🌳
    title: 第三部分：RadixAttention
    details: 前缀共享树、RadixCache 实现、与 PagedAttention 对比
    link: /chapters/06-radix-attention-design
  - icon: ⚡
    title: 第四部分：调度与批处理
    details: Scheduler 架构、Continuous Batching、Chunked Prefill 与 PD 分离
    link: /chapters/09-scheduler
  - icon: 🔧
    title: 第五部分：执行后端
    details: Model Runner、Attention Backend、分布式执行、量化与模型加载
    link: /chapters/12-model-runner
  - icon: 📐
    title: 第六部分：结构化生成
    details: Outlines 集成与 Jump Forward Decoding 加速
    link: /chapters/16-outlines-integration
  - icon: 🌐
    title: 第七部分：API 与部署
    details: OpenAI 兼容 API、多模态支持
    link: /chapters/18-openai-api
  - icon: 📊
    title: 第八部分：生态
    details: 与 vLLM、TRT-LLM 的对比与选型指南
    link: /chapters/20-comparison-guide
---
