---
title: 第 2 章：Repo 结构与模块依赖
---

# 第 2 章：Repo 结构与模块依赖

> "好的架构是项目长期成功的基石。" —— 在阅读源码之前，理解项目的目录结构和模块边界，是最高效的切入方式。

## 顶层目录结构

SGLang 仓库的顶层目录组织如下：

```
sglang/
├── python/           # Python 核心实现
├── sgl-kernel/       # CUDA/Triton 高性能算子
├── sgl-model-gateway/ # 模型网关组件
├── benchmark/        # 性能基准测试
├── docker/           # Docker 部署配置
├── docs/             # 文档
├── examples/         # 示例代码
├── test/             # 测试套件
└── scripts/          # 工具脚本
```

其中 `python/sglang/` 是我们源码分析的核心目录，`sgl-kernel/` 则包含了用 CUDA 和 Triton 实现的底层高性能算子。

## python/sglang 核心目录

`python/sglang/` 是整个项目的 Python 主体，其内部结构体现了清晰的分层设计：

```
python/sglang/
├── lang/              # 前端 DSL 层
├── srt/               # SGLang Runtime（后端核心）
├── cli/               # 命令行接口
├── eval/              # 评估框架
├── jit_kernel/        # JIT 编译的 kernel
├── multimodal_gen/    # 多模态生成加速
├── benchmark/         # 内置基准测试工具
├── test/              # 测试工具
├── api.py             # 公共 API 入口
├── launch_server.py   # 服务启动入口
├── global_config.py   # 全局配置常量
├── utils.py           # 通用工具函数
└── version.py         # 版本信息
```

### 前端层：python/sglang/lang/

`lang/` 目录实现了 SGLang 的前端 DSL，核心文件包括：

- **`ir.py`**：定义了中间表示（IR），包含 `SglFunction`、`SglGen`、`SglSelect`、`SglFork` 等核心类。
- **`interpreter.py`**：DSL 的解释执行器，负责将 IR 节点转换为实际的运行时调用。
- **`backend/`**：后端适配器，支持对接不同的推理引擎。

### 后端核心：python/sglang/srt/

`srt/`（SGLang Runtime）是整个系统最庞大的模块，包含 32 个以上的子目录：

```
python/sglang/srt/
├── entrypoints/       # 入口点（Engine、HTTP Server）
├── managers/          # 核心管理器（Scheduler）
├── model_executor/    # 模型执行器
├── model_loader/      # 模型加载器
├── models/            # 模型定义
├── layers/            # 网络层实现
├── sampling/          # 采样逻辑
├── constrained/       # 结构化输出约束
├── mem_cache/         # 内存与 KV Cache 管理
├── distributed/       # 分布式通信
├── disaggregation/    # Prefill-Decode 分离
├── speculative/       # 推测解码
├── lora/              # LoRA 适配器
├── tokenizer/         # 分词器管理
├── server_args.py     # 服务启动参数定义
└── constants.py       # 运行时常量
```

## 关键文件导航

对于初次阅读 SGLang 源码的读者，以下四个文件构成了理解系统运作的核心路径：

### 1. server_args.py —— 配置中枢

```
python/sglang/srt/server_args.py
```

`ServerArgs` 数据类定义了服务启动的全部配置参数，包括模型路径、tensor 并行度、调度策略、量化方式等。这是理解系统可调参数的最佳入口。

### 2. Engine 入口

```
python/sglang/srt/entrypoints/engine.py
```

`Engine` 类是推理引擎的主入口，负责启动 `TokenizerManager`、`Scheduler` 子进程和 `DetokenizerManager`，并通过 ZMQ 套接字协调它们之间的通信。

### 3. Scheduler 调度器

```
python/sglang/srt/managers/scheduler.py
```

`Scheduler` 是系统的"大脑"，管理请求队列（`waiting_queue`、`running_batch`），实现 prefill 调度、chunked prefill、decode 批处理等核心逻辑。

### 4. 采样参数

```
python/sglang/srt/sampling/sampling_params.py
```

`SamplingParams` 类定义了生成时的采样参数，如 `temperature`、`top_p`、`top_k`、`json_schema` 等，是理解生成控制的关键。

## 模块依赖关系

SGLang 的模块依赖遵循严格的分层原则，可以用以下依赖图来概括：

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
```

关键依赖方向：

- **上层模块依赖下层**：`Engine` 依赖 `Scheduler`，`Scheduler` 依赖 `ModelRunner` 和 `MemCache`。
- **前端不依赖后端实现细节**：`lang/` 通过抽象的 backend 接口与 `srt/` 解耦。
- **横向模块相互独立**：`sampling/`、`constrained/`、`lora/` 等模块各自封装，通过 `Scheduler` 协调。

## 本章小结

本章梳理了 SGLang 仓库的目录结构和关键文件位置。项目采用清晰的前端（`lang/`）与后端（`srt/`）分层架构，`srt/` 内部又进一步划分为入口点、调度器、模型执行、缓存管理等子模块。理解这些模块的边界和依赖关系，是深入源码分析的前提。下一章我们将聚焦前端 DSL 的设计，详解 `gen`、`select`、`fork`、`join` 等核心原语的实现。
