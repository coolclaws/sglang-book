---
title: 第 2 章：Repo 结构与模块依赖
---

# 第 2 章：Repo 结构与模块依赖

> "好的架构是项目长期成功的基石。" —— 在阅读源码之前，理解项目的目录结构和模块边界，是最高效的切入方式。

## 顶层目录结构

SGLang 仓库的顶层目录组织清晰地反映了项目的多层次架构。我们首先从全局视角审视整个代码库的组成：

```
sglang/
├── python/           # Python 核心实现（本书分析的主体）
├── sgl-kernel/       # CUDA/Triton 高性能算子库
├── sgl-model-gateway/ # 模型网关组件
├── benchmark/        # 性能基准测试套件
├── docker/           # Docker 部署配置文件
├── docs/             # 项目文档
├── examples/         # 使用示例代码
├── test/             # 端到端测试套件
└── scripts/          # 开发和部署工具脚本
```

其中 `python/sglang/` 是我们源码分析的核心目录，包含了从前端 DSL 到后端运行时的全部 Python 实现。`sgl-kernel/` 则包含了用 CUDA 和 Triton 实现的底层高性能算子，例如自定义的 attention kernel、采样 kernel 等，这些算子是 SGLang 实现高性能的关键基础设施。`sgl-model-gateway/` 提供了模型网关功能，用于统一管理和路由多个模型实例。

## python/sglang 核心目录详解

`python/sglang/` 是整个项目的 Python 主体，其内部结构体现了清晰的分层设计理念：

```
python/sglang/
├── lang/              # 前端 DSL 层（用户编程接口）
├── srt/               # SGLang Runtime（后端核心引擎）
├── cli/               # 命令行接口工具
├── eval/              # 模型评估框架
├── jit_kernel/        # JIT 编译的自定义 kernel
├── multimodal_gen/    # 多模态生成加速模块
├── benchmark/         # 内置基准测试工具
├── test/              # 测试工具集
├── api.py             # 公共 API 入口（gen、select 等函数）
├── launch_server.py   # HTTP 服务启动入口
├── global_config.py   # 全局配置常量定义
├── utils.py           # 通用工具函数
├── check_env.py       # 环境检查工具
└── version.py         # 版本信息管理
```

### 前端层：python/sglang/lang/

`lang/` 目录实现了 SGLang 的前端 DSL（领域特定语言），这是用户编写结构化生成程序的入口。核心文件包括：

- **`ir.py`**：定义了中间表示（Intermediate Representation），包含 `SglFunction`、`SglGen`、`SglSelect`、`SglFork` 等核心 IR 节点类。每一种 DSL 原语在这里都有对应的 IR 类定义，它们共同构成了 SGLang 程序的抽象语法树。
- **`interpreter.py`**：DSL 的解释执行器，负责遍历 IR 节点并将其转换为实际的运行时调用。解释器维护了执行状态（包括对话历史、变量绑定等），逐步执行用户程序中的每一个操作。
- **`backend/`**：后端适配器目录，定义了与不同推理引擎对接的抽象接口。通过这一层抽象，同一个 SGLang 程序可以在不同的后端上运行，无需修改用户代码。

### 后端核心：python/sglang/srt/

`srt/`（SGLang Runtime）是整个系统最庞大也最核心的模块，包含超过 30 个子目录，涵盖了推理服务的各个方面：

```
python/sglang/srt/
├── entrypoints/       # 入口点：Engine 类、HTTP Server、OpenAI 兼容 API
├── managers/          # 核心管理器：Scheduler 调度器
├── model_executor/    # 模型前向计算执行器
├── model_loader/      # 模型权重加载器（支持多种格式）
├── models/            # 各类模型的定义（LLaMA、Qwen、DeepSeek 等）
├── layers/            # 神经网络层实现（Attention、MLP 等）
├── sampling/          # 采样逻辑（temperature、top_p、top_k 等）
├── constrained/       # 结构化输出约束（JSON Schema、正则表达式、EBNF）
├── mem_cache/         # 内存与 KV Cache 管理（RadixAttention 的核心实现）
├── distributed/       # 分布式通信（tensor parallel、pipeline parallel）
├── disaggregation/    # Prefill-Decode 分离部署
├── speculative/       # 推测解码（speculative decoding）
├── lora/              # LoRA 适配器热加载
├── tokenizer/         # 分词器管理与封装
├── compilation/       # 模型编译优化（torch.compile 集成）
├── batch_overlap/     # CPU-GPU 重叠调度实现
├── elastic_ep/        # 弹性专家并行
├── function_call/     # 函数调用解析
├── observability/     # 可观测性（metrics、tracing）
├── server_args.py     # 服务启动参数定义（ServerArgs 数据类）
└── constants.py       # 运行时常量定义
```

这些子目录的组织遵循了"关注点分离"的原则：每个目录负责一个独立的功能领域，通过 `Scheduler` 进行协调和集成。

## 关键文件导航

对于初次阅读 SGLang 源码的读者，以下四个文件构成了理解系统运作的核心路径，建议按照这个顺序进行阅读：

### 1. server_args.py —— 配置中枢

```
python/sglang/srt/server_args.py
```

`ServerArgs` 数据类定义了服务启动的全部配置参数。这个文件是理解系统所有可调参数的最佳入口，包括模型路径（`model_path`）、tensor 并行度（`tp_size`）、pipeline 并行度（`pp_size`）、调度策略（`schedule_policy`）、量化方式（`quantization`）、KV Cache 数据类型（`kv_cache_dtype`）、最大运行请求数（`max_running_requests`）等。阅读这个文件可以快速了解 SGLang 提供了哪些系统级别的配置能力。

### 2. Engine 入口 —— 系统枢纽

```
python/sglang/srt/entrypoints/engine.py
```

`Engine` 类是推理引擎的主入口，充当整个系统的枢纽角色。它在初始化时启动三大核心组件：`TokenizerManager` 负责请求的分词处理、`Scheduler` 子进程负责批调度和模型前向计算、`DetokenizerManager` 负责输出的反分词。这三个组件通过 ZMQ（ZeroMQ）消息队列进行高效的异步跨进程通信，避免了 Python GIL 带来的性能瓶颈。

### 3. Scheduler 调度器 —— 系统大脑

```
python/sglang/srt/managers/scheduler.py
```

`Scheduler` 是系统的"大脑"，也是最复杂的单个文件之一。它继承了多个 Mixin 类来组织不同功能：`SchedulerOutputProcessorMixin` 处理输出结果、`SchedulerUpdateWeightsMixin` 管理权重更新、`SchedulerProfilerMixin` 提供性能分析、`SchedulerMetricsMixin` 收集指标数据。调度器维护了 `waiting_queue`（等待队列）和 `running_batch`（运行批次）两个核心数据结构，通过 `get_next_batch_to_run()` 方法实现 prefill 调度、chunked prefill、decode 批处理等核心逻辑。

### 4. 采样参数 —— 生成控制

```
python/sglang/srt/sampling/sampling_params.py
```

`SamplingParams` 类定义了生成时的所有采样参数，是控制模型输出行为的关键接口。每个推理请求都携带一个 `SamplingParams` 实例，其中包含 `temperature`、`top_p`、`top_k`、`json_schema`、`regex` 等参数，决定了模型如何从概率分布中选择 token 以及如何约束输出格式。

## 模块依赖关系

SGLang 的模块依赖遵循严格的分层原则，上层模块依赖下层，同层模块尽量保持独立。整体的依赖关系可以用以下依赖图来概括：

```
api.py / launch_server.py
         │
         ▼
   entrypoints/engine.py
         │
    ┌────┼────────────┐
    ▼    ▼            ▼
Tokenizer  Scheduler  Detokenizer
Manager    │          Manager
           │
     ┌─────┼──────┐
     ▼     ▼      ▼
  Model  KV Cache  Sampling
  Runner Manager   Engine
     │
     ▼
  models/ + layers/
```

这个依赖图体现了以下关键的设计原则：

- **上层模块依赖下层**：`Engine` 依赖 `Scheduler`，`Scheduler` 依赖 `ModelRunner` 和 `MemCache`，`ModelRunner` 依赖 `models/` 和 `layers/` 中的具体模型实现。依赖方向始终是从上到下，不存在循环依赖。
- **前端不依赖后端实现细节**：`lang/` 通过抽象的 backend 接口与 `srt/` 解耦。这意味着前端 DSL 代码不直接引用 `Scheduler` 或 `ModelRunner` 的具体类，而是通过 backend 适配器间接调用。这种解耦设计使得前端可以灵活对接不同的推理后端。
- **横向模块相互独立**：`sampling/`、`constrained/`、`lora/`、`speculative/` 等模块各自封装独立的功能逻辑，通过 `Scheduler` 进行协调和集成。这使得每个模块可以独立开发、测试和演进，不会相互影响。
- **硬件抽象层隔离**：`sgl-kernel/` 中的高性能算子通过统一接口暴露给上层，`distributed/` 模块封装了多种并行策略的通信逻辑，使得上层代码无需关心具体的硬件差异。

## 构建与安装

SGLang 的 Python 包通过标准的 `pip install` 安装，同时 `sgl-kernel` 作为独立的包需要单独安装以获得最佳性能。项目使用 `pyproject.toml` 管理依赖，支持多种安装配置来适配不同的硬件平台和使用场景。

## 本章小结

本章系统梳理了 SGLang 仓库的目录结构和关键文件位置。项目采用清晰的前端（`lang/`）与后端（`srt/`）分层架构，`srt/` 内部又进一步划分为入口点、调度器、模型执行、缓存管理、采样、约束输出等十余个子模块。四个关键文件——`server_args.py`、`engine.py`、`scheduler.py`、`sampling_params.py`——构成了理解系统的核心阅读路径。理解这些模块的边界和依赖关系，是深入源码分析的前提。下一章我们将聚焦前端 DSL 的设计，详解 `gen`、`select`、`fork`、`join` 等核心原语的实现。
