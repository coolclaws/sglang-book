---
title: 附录 B：核心参数速查
---

# 附录 B：核心参数速查

本附录汇总了 SGLang 中最常用的配置参数，方便读者在开发和部署过程中快速查阅。参数定义主要位于 `python/sglang/srt/server_args.py` 和 `python/sglang/srt/sampling/sampling_params.py`。

## ServerArgs 服务器参数

以下参数在启动服务时通过命令行指定：

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--model-path` | str | 必填 | 模型路径或 HuggingFace 模型名 |
| `--tp-size` | int | 1 | Tensor Parallelism 并行度 |
| `--dp-size` | int | 1 | Data Parallelism 并行度 |
| `--mem-fraction-static` | float | 0.88 | GPU 显存中用于 KV Cache 的静态分配比例 |
| `--max-running-requests` | int | auto | 最大并发运行请求数 |
| `--max-total-tokens` | int | auto | KV Cache 池的最大 token 总数 |
| `--chunked-prefill-size` | int | 8192 | Chunked Prefill 的分块大小 |
| `--host` | str | "0.0.0.0" | 服务器监听地址 |
| `--port` | int | 30000 | 服务器监听端口 |
| `--tokenizer-path` | str | None | 自定义 tokenizer 路径（默认与模型相同） |
| `--context-length` | int | auto | 模型最大上下文长度 |
| `--schedule-policy` | str | "lpm" | 调度策略：lpm / random / fcfs / dfs-weight |
| `--disable-radix-cache` | bool | False | 是否禁用 RadixAttention 前缀缓存 |
| `--enable-torch-compile` | bool | False | 是否启用 torch.compile 优化 |
| `--dtype` | str | "auto" | 模型数据类型：auto / float16 / bfloat16 |
| `--quantization` | str | None | 量化方式：gptq / awq / fp8 等 |
| `--enable-dp-attention` | bool | False | 启用 Data Parallel Attention |
| `--lora-paths` | list | None | LoRA 适配器路径列表 |

## SamplingParams 采样参数

以下参数在每次请求中指定，控制文本生成行为：

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `temperature` | float | 1.0 | 采样温度，0 表示贪心解码 |
| `top_p` | float | 1.0 | Top-p (nucleus) 采样阈值 |
| `top_k` | int | -1 | Top-k 采样，-1 表示不限制 |
| `max_new_tokens` | int | 128 | 最大生成 token 数 |
| `min_new_tokens` | int | 0 | 最小生成 token 数 |
| `frequency_penalty` | float | 0.0 | 频率惩罚，抑制重复 token |
| `presence_penalty` | float | 0.0 | 存在惩罚，鼓励新 token |
| `repetition_penalty` | float | 1.0 | 重复惩罚系数 |
| `stop` | list | None | 停止词列表 |
| `regex` | str | None | 正则表达式约束 |
| `json_schema` | str | None | JSON Schema 约束 |
| `n` | int | 1 | 每个请求生成的候选数 |
| `logprobs` | int | None | 返回 top logprobs 的数量 |

## 常用环境变量

| 环境变量 | 说明 |
|----------|------|
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN` | 允许覆盖模型默认上下文长度 |
| `CUDA_VISIBLE_DEVICES` | 指定可用的 GPU 设备编号 |
| `NCCL_P2P_DISABLE` | 禁用 NCCL P2P 通信（某些网络环境需要） |
| `SGLANG_BLOCK_NONFINITE_TOKENS` | 阻止生成非有限值（NaN/Inf）的 token |

## 启动示例

```bash
# 基础启动
python -m sglang.launch_server --model-path meta-llama/Llama-3-8B-Instruct

# 双卡 Tensor Parallelism
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-70B-Instruct \
    --tp-size 2 \
    --mem-fraction-static 0.9

# 启用量化和 LoRA
python -m sglang.launch_server \
    --model-path model_path \
    --quantization gptq \
    --lora-paths adapter1=path/to/lora
```
