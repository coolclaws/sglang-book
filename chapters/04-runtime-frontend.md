---
title: 第 4 章：Runtime 前端接口
---

# 第 4 章：Runtime 前端接口

> "好的接口设计应当让简单的事情保持简单，复杂的事情变得可能。" —— SGLang 的 Runtime 前端正是这一原则的实践者。

## 架构概览

SGLang 的 Runtime 前端是连接用户请求与后端调度系统的桥梁。它提供了两种使用模式：**HTTP 服务模式** 和 **嵌入式模式**（embedded mode），分别适用于不同的部署场景。无论哪种模式，核心的推理逻辑都由同一个 `Engine` 类驱动，确保了行为的一致性。

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

这种双模式设计使得 SGLang 既可以作为独立的推理服务对外提供 API，也可以作为库嵌入到用户的 Python 应用中，满足从生产部署到研究实验的多种需求。

## Engine 类：推理引擎的核心

`Engine` 类定义在 `python/sglang/srt/entrypoints/engine.py` 中，是整个推理系统的主入口点。官方文档将其描述为 "The entry point to the inference engine"。无论是 HTTP 服务还是嵌入式调用，最终都汇聚到 `Engine` 的方法中。

### 初始化流程

`Engine` 的构造函数接受与 `ServerArgs`（定义在 `python/sglang/srt/server_args.py`）相同的关键字参数：

```python
# python/sglang/srt/entrypoints/engine.py
class Engine:
    def __init__(self, **kwargs):
        # 1. 解析 ServerArgs 配置参数
        # 2. 注册 atexit 钩子，确保子进程在退出时被正确清理
        # 3. 调用 _launch_scheduler_processes() 启动 Scheduler 子进程
        # 4. 初始化 ZMQ 通信套接字
        # 5. 启动 TokenizerManager 和 DetokenizerManager
        # 6. 创建事件循环，启动后台异步任务
```

初始化过程中最关键的步骤是 `_launch_scheduler_processes()` 方法。这个方法会根据配置的 tensor 并行度（`tp_size`）和 pipeline 并行度（`pp_size`）创建一个或多个 `Scheduler` 子进程。每个子进程被绑定到特定的 GPU 设备，拥有独立的 CUDA 上下文，并通过 multiprocessing pipe 和 ZMQ 套接字与主进程建立双向通信通道。子进程启动后，会通过 pipe 发送就绪信号和配置信息（如最大请求输入长度等参数），主进程收到确认后才继续后续的初始化步骤。

### 三大核心组件

`Engine` 协调三个核心组件的协同工作，它们运行在不同的进程中，通过消息传递进行通信：

1. **TokenizerManager**：运行在主进程中，负责将用户输入的文本进行分词（tokenize），将文本形式的 prompt 转换为 token ID 序列。它还负责处理多模态输入（图像、音频、视频），将不同模态的数据编码为模型可以理解的格式。处理完成后，TokenizerManager 将结构化的请求通过 ZMQ 发送给 Scheduler。
2. **Scheduler**（独立子进程）：负责批调度（batching）和模型前向计算，是系统的核心计算单元。它从 TokenizerManager 接收请求，维护等待队列和运行批次，决定哪些请求进入当前批次，然后驱动模型执行前向计算。调度器的详细实现在 `python/sglang/srt/managers/scheduler.py` 中，其设计复杂度远超其他组件。
3. **DetokenizerManager**：同样运行在独立进程中，负责将模型输出的 token ID 序列转换回人类可读的文本。它从 Scheduler 接收生成的 token ID，执行反分词操作，并将最终的文本结果返回给用户。

这三个组件通过 ZMQ（ZeroMQ）消息队列进行高效的异步通信。ZMQ 的无锁设计和零拷贝传输确保了跨进程通信的极低开销，同时避免了 Python GIL 对多线程并发的限制。

### generate 调用链

`Engine` 提供了同步和异步两种生成接口，支持文本和多模态输入：

```python
# 同步接口
def generate(self, prompt, sampling_params, input_ids,
             image_data, audio_data, video_data,
             return_logprob, logprob_start_len,
             stream, ...) -> Union[Dict, Iterator[Dict]]:
    ...

# 异步接口
async def async_generate(self, ...) -> Union[Dict, AsyncIterator[Dict]]:
    ...
```

一个典型的 `generate` 调用会经历以下完整的处理链路：

```
Engine.generate() 接收用户请求
    │
    ▼
TokenizerManager.tokenize(prompt)
    │  文本 → token IDs
    │  处理多模态输入（如有）
    ▼
发送 tokenized request 到 Scheduler（via ZMQ）
    │
    ▼
Scheduler.recv_requests() 接收请求
    │  请求入队 waiting_queue
    ▼
Scheduler.get_next_batch_to_run() 批调度
    │  从 waiting_queue 选取请求组成 batch
    ▼
ModelRunner 执行模型前向计算
    │  在 GPU 上执行 attention + MLP
    ▼
采样引擎根据 SamplingParams 选择 token
    │
    ▼
结果通过 ZMQ 发送到 DetokenizerManager
    │
    ▼
DetokenizerManager 执行反分词
    │  token IDs → 文本
    ▼
返回结果给 Engine.generate() 调用方
```

当开启流式输出（`stream=True`）时，这条链路会持续运行，每生成一个或多个 token 就立即将增量结果推送给用户，无需等待完整生成结束。

### collective_rpc 分布式调用

`Engine` 还提供了 `collective_rpc()` 方法，用于在所有 Scheduler 子进程上同步执行远程过程调用：

```python
def collective_rpc(self, method, **kwargs):
    """在所有 Scheduler 进程上执行指定方法"""
    ...
```

这在分布式场景下尤为重要。例如，当需要保存模型权重时，所有持有权重分片的 GPU Worker 必须协同执行保存操作；当需要动态加载或卸载 LoRA 适配器时，也需要所有 Worker 同步更新。`collective_rpc` 提供了一个简洁的接口来实现这种全局协调操作。

## launch_server：HTTP 服务模式

`python/sglang/launch_server.py` 是 HTTP 服务模式的启动入口。它基于 FastAPI 框架构建，启动一个兼容 OpenAI API 的 HTTP 推理服务：

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-3-8B-Instruct \
    --tp 4 --port 30000
```

HTTP 服务模式下，请求通过 OpenAI 兼容的 REST API 进入系统。协议定义在 `python/sglang/srt/entrypoints/openai/` 目录下，支持以下标准端点：

- `/v1/completions`：文本补全接口
- `/v1/chat/completions`：对话补全接口
- `/v1/embeddings`：文本嵌入接口
- `/v1/models`：模型信息查询接口

这种兼容性设计使得现有基于 OpenAI API 构建的应用可以无缝迁移到 SGLang，只需修改 API 端点地址即可。此外，SGLang 还支持 gRPC 模式（通过 `--grpc-mode` 参数启用）和 SSL/TLS 加密通信，满足生产环境的安全需求。

## 嵌入式模式

嵌入式模式允许在 Python 进程内直接创建推理引擎，无需启动独立的 HTTP 服务：

```python
import sglang as sgl

# 创建引擎实例
engine = sgl.Engine(
    model_path="meta-llama/Llama-3-8B-Instruct",
    tp_size=2,
    dtype="float16"
)

# 直接调用 generate
result = engine.generate(
    prompt="请解释量子计算的基本原理。",
    sampling_params={"temperature": 0.7, "max_new_tokens": 256}
)
print(result["text"])

# 使用完毕后关闭引擎
engine.shutdown()
```

嵌入式模式的优势体现在多个方面。首先是 **零网络开销**：请求不经过 HTTP 协议栈和网络传输，直接通过进程内的 ZMQ 通道传递，消除了序列化和网络延迟。其次是 **更灵活的控制能力**：用户可以直接访问 `Engine` 的所有方法，包括 `collective_rpc` 进行权重更新、动态调整配置参数等高级操作，这些在 HTTP 模式下通常不暴露或受到限制。第三是 **更适合集成场景**：嵌入式模式方便将 SGLang 集成到已有的训练流水线、评估框架或研究实验代码中，无需管理额外的服务进程。

## 请求流转全景

综合来看，一个用户请求从进入系统到返回结果，在两种模式下的流转路径如下：

```
用户请求
  │
  ├── HTTP 模式：FastAPI 路由 → 请求解析 → 参数校验
  │
  └── 嵌入式模式：直接调用 Engine.generate()
         │
         ▼
   Engine 统一入口
         │
         ▼
   TokenizerManager：文本 → token IDs → 多模态编码
         │
         ▼
   ZMQ 消息传递（跨进程）
         │
         ▼
   Scheduler：请求入队 → 批调度 → 前向计算 → 采样
         │
         ▼
   ZMQ 消息传递（跨进程）
         │
         ▼
   DetokenizerManager：token IDs → 文本 → 停止检测
         │
         ▼
   返回响应给用户
```

在这条链路中，ZMQ 消息队列是跨进程通信的核心基础设施。它使得 TokenizerManager、Scheduler、DetokenizerManager 三个组件可以在不同的进程中独立运行和扩展，实现了计算密集型操作（GPU 推理）与 IO 密集型操作（分词、反分词、网络通信）的彻底解耦。这种进程级别的隔离还带来了额外的稳定性优势：即使 Scheduler 因为 GPU OOM 崩溃，主进程仍然可以检测到异常并尝试恢复，而不会导致整个服务不可用。

## 本章小结

本章详细分析了 SGLang Runtime 前端的架构设计。`Engine` 类作为推理引擎的统一入口，协调 TokenizerManager、Scheduler 和 DetokenizerManager 三大组件，通过 ZMQ 实现高效的跨进程异步通信。系统支持 HTTP 服务和嵌入式两种部署模式，前者通过 OpenAI 兼容的 REST API 提供标准化接口，后者则提供零网络开销的进程内调用。完整的请求处理链路从文本分词开始，经过批调度和 GPU 计算，最终通过反分词输出结果。下一章我们将深入 `SamplingParams` 类，探索采样参数和结构化输出的实现机制。
