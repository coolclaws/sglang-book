---
title: 第 4 章：Runtime 前端接口
---

# 第 4 章：Runtime 前端接口

> "好的接口设计应当让简单的事情保持简单，复杂的事情变得可能。" —— SGLang 的 Runtime 前端正是这一原则的实践者。

## 架构概览

SGLang 的 Runtime 前端是连接用户请求与后端调度系统的桥梁。它提供了两种使用模式：**HTTP 服务模式** 和 **嵌入式模式**（embedded mode），分别适用于不同的部署场景。无论哪种模式，核心的推理逻辑都由同一个 `Engine` 类驱动。

```
               HTTP 模式                    嵌入式模式
                  │                            │
          launch_server()                  Engine(**kwargs)
                  │                            │
           FastAPI Server                      │
                  │                            │
                  └──────────┬─────────────────┘
                             │
                         Engine 类
                    ┌────────┼────────┐
                    ▼        ▼        ▼
              Tokenizer  Scheduler  Detokenizer
              Manager    (子进程)    Manager
```

## Engine 类：推理引擎的核心

`Engine` 类定义在 `python/sglang/srt/entrypoints/engine.py` 中，是整个推理系统的主入口点。官方文档将其描述为 "The entry point to the inference engine"。

### 初始化流程

`Engine` 的构造函数接受与 `ServerArgs` 相同的参数：

```python
# python/sglang/srt/entrypoints/engine.py
class Engine:
    def __init__(self, **kwargs):
        # 1. 解析 ServerArgs 配置
        # 2. 注册 atexit 子进程清理
        # 3. 启动 Scheduler 子进程
        # 4. 初始化 ZMQ 通信套接字
        # 5. 启动事件循环
```

初始化过程中，`Engine` 会通过 `_launch_scheduler_processes()` 方法创建一个或多个 `Scheduler` 子进程。每个子进程绑定到特定的 GPU，并通过 multiprocessing pipe 和 ZMQ 套接字与主进程通信。

### 三大核心组件

`Engine` 协调三个核心组件的工作：

1. **TokenizerManager**：负责将用户输入的文本进行分词（tokenize），将文本请求转换为 token ID 序列，然后发送给 Scheduler。
2. **Scheduler**（子进程）：负责批调度和模型前向计算，是系统的核心计算单元，定义在 `python/sglang/srt/managers/scheduler.py`。
3. **DetokenizerManager**：负责将模型输出的 token ID 序列转换回文本，并将结果返回给用户。

这三个组件通过 ZMQ 消息队列进行高效的异步通信，避免了 Python GIL 对性能的影响。

### generate 调用链

`Engine` 提供了同步和异步两种生成接口：

```python
# 同步接口
def generate(self, prompt, sampling_params, input_ids,
             image_data, audio_data, video_data, ...) -> Union[Dict, Iterator[Dict]]:
    ...

# 异步接口
async def async_generate(self, ...) -> Union[Dict, AsyncIterator[Dict]]:
    ...
```

一个典型的 `generate` 调用会经历以下链路：

```
Engine.generate()
    │
    ▼
TokenizerManager.tokenize(prompt)
    │
    ▼
发送 tokenized request 到 Scheduler（via ZMQ）
    │
    ▼
Scheduler.recv_requests() 接收请求
    │
    ▼
Scheduler.get_next_batch_to_run() 调度
    │
    ▼
ModelRunner 执行前向计算
    │
    ▼
DetokenizerManager 解码输出
    │
    ▼
返回结果给 Engine.generate()
```

### collective_rpc 分布式调用

`Engine` 还提供了 `collective_rpc()` 方法，用于在所有 Scheduler 子进程上执行远程过程调用：

```python
def collective_rpc(self, method, **kwargs):
    """在所有 Scheduler 进程上执行指定方法"""
    ...
```

这在分布式场景下尤为重要，例如保存模型权重、动态更新 LoRA 适配器等操作，需要所有 GPU Worker 协同执行。

## launch_server：HTTP 服务模式

`python/sglang/launch_server.py` 是 HTTP 服务模式的入口。它基于 FastAPI 框架，启动一个兼容 OpenAI API 的 HTTP 服务：

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-3-8B-Instruct
```

HTTP 服务模式下，请求通过 OpenAI 兼容的 REST API 进入系统。请求和响应的协议定义在 `python/sglang/srt/entrypoints/openai/` 目录下，支持 `/v1/completions`、`/v1/chat/completions` 等标准端点。

## 嵌入式模式

嵌入式模式允许在 Python 进程内直接创建推理引擎，无需启动 HTTP 服务：

```python
import sglang as sgl

engine = sgl.Engine(model_path="meta-llama/Llama-3-8B-Instruct", tp_size=2)

result = engine.generate(
    prompt="请解释量子计算的基本原理。",
    sampling_params={"temperature": 0.7, "max_new_tokens": 256}
)
print(result["text"])

engine.shutdown()
```

嵌入式模式的优势在于：

- **零网络开销**：请求不经过 HTTP 协议栈，直接通过进程内通信传递。
- **更灵活的控制**：可以直接访问 `Engine` 的方法，进行模型权重更新、动态配置等操作。
- **更适合集成**：方便嵌入到已有的 Python 应用或研究 pipeline 中。

## 请求流转全景

综合来看，一个用户请求从进入系统到返回结果，经历了以下完整路径：

```
用户请求（HTTP / Python API）
    │
    ▼
Engine.generate() / async_generate()
    │
    ▼
TokenizerManager：文本 → token IDs
    │
    ▼
ZMQ 消息传递
    │
    ▼
Scheduler：请求入队 → 批调度 → 前向计算
    │
    ▼
ZMQ 消息传递
    │
    ▼
DetokenizerManager：token IDs → 文本
    │
    ▼
返回响应
```

这条链路中，ZMQ 消息队列是跨进程通信的核心基础设施，它使得 Tokenizer、Scheduler、Detokenizer 三个组件可以在不同的进程（甚至不同的机器）上独立运行，实现了计算与 IO 的解耦。

## 本章小结

本章详细分析了 SGLang Runtime 前端的架构设计。`Engine` 类作为推理引擎的统一入口，协调 TokenizerManager、Scheduler 和 DetokenizerManager 三大组件。系统支持 HTTP 服务和嵌入式两种部署模式，通过 ZMQ 实现高效的跨进程异步通信。下一章我们将深入 `SamplingParams` 类，探索采样参数和结构化输出的实现机制。
