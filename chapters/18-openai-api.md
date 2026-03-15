---
title: 第 18 章：OpenAI 兼容 API
---

# 第 18 章：OpenAI 兼容 API

> "最好的 API 设计是让用户感觉不到它的存在——兼容即是最大的便利。"

在大语言模型的生态系统中，OpenAI 的 API 格式已经成为事实上的行业标准。SGLang 通过提供完整的 OpenAI 兼容 API 层，使得用户可以在几乎不修改客户端代码的情况下，将请求从 OpenAI 服务无缝迁移到 SGLang 推理引擎。本章将深入分析这一兼容层的实现原理。

## FastAPI 服务器架构

SGLang 的 API 服务基于 FastAPI 框架构建，入口位于 `python/sglang/srt/entrypoints/openai/api_server.py`。服务器启动时，会注册一系列 OpenAI 兼容的路由端点：

```python
# python/sglang/srt/entrypoints/openai/api_server.py
app = FastAPI()

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    ...

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    ...

@app.get("/v1/models")
async def list_models():
    ...
```

FastAPI 本身运行在 uvicorn 之上，提供异步 HTTP 服务能力。每个请求进入后，会经过参数校验、格式转换、引擎调度、结果封装等一系列流程。

## 核心端点实现

### /v1/chat/completions

Chat Completions 端点是最常用的接口。它接收符合 OpenAI 格式的聊天消息列表，内部需要将其转换为模型可处理的 prompt 格式。转换过程主要依赖 tokenizer 的 chat template：

```python
# 请求格式转换的核心逻辑
# python/sglang/srt/entrypoints/openai/protocol.py
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    ...
```

消息列表通过 `tokenizer.apply_chat_template()` 方法转换为单一的文本 prompt，随后传入推理引擎处理。这一步骤确保了不同模型的 chat template 差异对用户透明。

### /v1/completions

Completions 端点接收原始的 prompt 文本，处理逻辑相对简单。它直接将 prompt 传递给引擎，无需经过 chat template 转换。

## 采样参数映射

OpenAI API 的采样参数（如 `temperature`、`top_p`、`max_tokens`、`frequency_penalty` 等）需要映射到 SGLang 内部的 `SamplingParams` 对象：

```python
# python/sglang/srt/sampling/sampling_params.py
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_new_tokens: int = 128
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    ...
```

值得注意的是，SGLang 使用 `max_new_tokens` 而非 OpenAI 的 `max_tokens`，API 层会自动完成这一命名转换。此外，SGLang 还支持 OpenAI 尚未提供的额外参数，如 `top_k` 和 `regex` 约束等。

## 流式响应（SSE）实现

流式响应是 LLM API 中至关重要的功能，SGLang 通过 Server-Sent Events（SSE）协议实现。当请求中 `stream=True` 时，服务器不再等待完整生成结果，而是逐 token 返回：

```python
# 流式响应的核心实现
async def generate_stream_response(request, generator):
    async for chunk in generator:
        data = create_stream_chunk(chunk)
        yield f"data: {json.dumps(data)}\n\n"
    yield "data: [DONE]\n\n"
```

每个 SSE chunk 遵循 OpenAI 的 delta 格式，包含增量生成的 token 内容。客户端可以使用标准的 SSE 客户端库或 `aiohttp` 逐块接收数据。

## API 层与引擎的连接

API 层与推理引擎之间通过 `TokenizerManager` 组件进行桥接。`TokenizerManager` 位于 `python/sglang/srt/managers/tokenizer_manager.py`，它承担了请求预处理和响应后处理的职责：

```
客户端请求 → FastAPI 路由 → TokenizerManager（tokenize）
    → Scheduler（调度与推理） → TokenizerManager（detokenize）
        → FastAPI 响应
```

这种分层设计使得 API 层保持轻量，专注于协议转换和格式兼容，而将计算密集型的工作完全交给后端引擎。`TokenizerManager` 通过 ZMQ 消息队列与 Scheduler 进程通信，实现了前后端的解耦。

## 错误处理与兼容性

SGLang 的 API 层还实现了与 OpenAI 一致的错误响应格式。当请求参数不合法、模型不存在或推理过程出错时，会返回标准的 JSON 错误对象：

```json
{
  "error": {
    "message": "Invalid request: temperature must be >= 0",
    "type": "invalid_request_error",
    "code": 400
  }
}
```

此外，`/v1/models` 端点返回当前加载的模型列表，使得 OpenAI 官方 SDK 可以无缝对接。

## 本章小结

本章详细分析了 SGLang 的 OpenAI 兼容 API 层实现。基于 FastAPI 的异步架构提供了高性能的 HTTP 服务；标准化的端点设计（`/v1/chat/completions`、`/v1/completions`）确保了与 OpenAI SDK 的即插即用兼容性；SSE 流式响应实现了低延迟的逐 token 输出；而 `TokenizerManager` 作为 API 层和推理引擎之间的桥梁，实现了关注点分离。这种设计让 SGLang 在保持内部架构灵活性的同时，为用户提供了熟悉且稳定的接口体验。
