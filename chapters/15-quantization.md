---
title: 第 15 章：量化与模型加载
---

# 第 15 章：量化与模型加载

> "量化是在精度与效率之间寻找最佳平衡点的艺术。用更少的比特表达同样的智能，这是工程师的浪漫。"

模型量化通过降低权重和激活值的数值精度来减少内存占用和计算开销，是大模型部署的重要优化手段。SGLang 支持多种量化方案，包括 AWQ、FP8、INT4 等，并提供了灵活的模型加载流程。本章将详细分析量化方案的实现与模型加载机制。

## 15.1 量化概述与支持方案

SGLang 的量化支持建立在 `python/sglang/srt/layers/quantization/` 目录下的模块化设计之上。每种量化方法都实现为独立的类，通过统一的注册机制接入系统。

```python
# python/sglang/srt/layers/quantization/__init__.py
QUANTIZATION_METHODS = {
    "awq": AWQConfig,
    "fp8": Fp8Config,
    "gptq": GPTQConfig,
    "marlin": MarlinConfig,
    "squeezellm": SqueezeLLMConfig,
    ...
}

def get_quantization_config(quantization: str):
    return QUANTIZATION_METHODS[quantization]
```

### 15.1.1 量化的基本原理

量化的核心思想是将 FP16/BF16 的浮点权重映射到低比特整数或低精度浮点数。以 INT4 量化为例：

```
原始权重 (FP16): [-0.5, 0.3, -0.1, 0.8, ...]  (每个值 16 bit)
量化权重 (INT4):  [-4, 2, -1, 6, ...]           (每个值 4 bit)
缩放因子 (FP16): 0.125                           (per-group)
反量化: INT4_value × scale ≈ 原始值
```

这种表示将模型大小压缩为原来的 1/4，同时通过分组量化（group quantization）保持合理的精度。

## 15.2 AWQ 量化

AWQ（Activation-aware Weight Quantization）是一种先进的 INT4 量化方法，通过分析激活值的分布来保护重要的权重通道。SGLang 通过集成 AutoAWQ 库来支持 AWQ 格式。

```python
# python/sglang/srt/layers/quantization/awq.py
class AWQLinearMethod:
    def apply(self, layer, x):
        # AWQ 量化的线性层计算
        out = awq_dequantize_and_gemm(
            x,
            layer.qweight,    # INT4 量化权重
            layer.qzeros,     # 零点
            layer.scales,     # 缩放因子
            layer.group_size, # 量化分组大小
        )
        return out
```

AWQ 的优势在于它不需要校准数据的反向传播，只通过前向推理统计激活值分布，从而决定哪些权重通道更重要，给予这些通道更高的量化精度。

### 15.2.1 AutoAWQ 集成

SGLang 直接支持加载 AutoAWQ 格式的预量化模型。用户只需指定量化后的模型路径，系统会自动检测量化配置：

```python
# 模型配置中的量化检测
# config.json 中包含:
# "quantization_config": {
#     "quant_method": "awq",
#     "bits": 4,
#     "group_size": 128
# }
```

## 15.3 FP8 量化

FP8（8-bit Floating Point）量化是 NVIDIA Hopper 架构（H100）引入的原生支持格式。与整数量化不同，FP8 保留了浮点数的表示结构，在简单性和精度之间取得了良好平衡。

```python
# python/sglang/srt/layers/quantization/fp8.py
class Fp8LinearMethod:
    def apply(self, layer, x):
        # 将输入动态量化为 FP8
        x_fp8, x_scale = per_token_quant(x, torch.float8_e4m3fn)
        # 使用 FP8 GEMM
        output = torch._scaled_mm(
            x_fp8, layer.weight_fp8,
            scale_a=x_scale,
            scale_b=layer.weight_scale,
            out_dtype=torch.bfloat16,
        )
        return output
```

FP8 量化有两种主要格式：

- **E4M3**（4 位指数 + 3 位尾数）：适用于权重和前向激活
- **E5M2**（5 位指数 + 2 位尾数）：动态范围更大，适用于梯度

在 SGLang 的推理场景中，主要使用 E4M3 格式。FP8 的量化误差通常小于 INT8，同时在 H100 上可以利用 FP8 Tensor Core 获得接近 2 倍的计算加速。

## 15.4 INT4 与 GPTQ/Marlin

除了 AWQ 之外，SGLang 还支持 GPTQ 格式的 INT4 量化。更重要的是，SGLang 支持 Marlin 格式——一种专门为 GPU 推理优化的 INT4 权重布局：

```python
# python/sglang/srt/layers/quantization/marlin.py
class MarlinLinearMethod:
    def apply(self, layer, x):
        # Marlin 使用优化的 CUDA kernel
        output = marlin_gemm(
            x, layer.B_packed,  # 特殊布局的 INT4 权重
            layer.s,            # 缩放因子
            layer.workspace,
        )
        return output
```

Marlin kernel 通过精心设计的内存访问模式和寄存器分配，实现了接近理论带宽上限的 INT4 矩阵乘法性能。

## 15.5 模型加载流程

SGLang 的模型加载流程高度灵活，支持从 HuggingFace Hub、本地路径或 S3 等多种来源加载模型。

### 15.5.1 加载流程详解

```python
# python/sglang/srt/model_executor/model_runner.py
# 简化的模型加载流程
def load_model(self):
    # 1. 读取模型配置
    hf_config = AutoConfig.from_pretrained(model_path)

    # 2. 检测量化配置
    quant_config = hf_config.quantization_config
    quant_method = get_quantization_config(quant_config["quant_method"])

    # 3. 初始化模型架构（使用量化后的线性层）
    model = build_model(hf_config, quant_method)

    # 4. 加载权重文件
    for name, weight in load_weights_iterator(model_path):
        # 对权重进行必要的转换（分片、转置等）
        param = model.get_parameter(name)
        loaded_weight = process_weight(weight, quant_method)
        param.data.copy_(loaded_weight)
```

### 15.5.2 权重格式处理

SGLang 需要处理多种权重存储格式：

- **safetensors**：HuggingFace 推荐的安全格式，支持内存映射和零拷贝加载
- **PyTorch bin**：传统的 pickle 序列化格式
- **GGUF**：来自 llama.cpp 的量化格式

```python
# 根据文件后缀选择加载方式
if weight_path.endswith(".safetensors"):
    weights = safetensors.torch.load_file(weight_path)
elif weight_path.endswith(".bin"):
    weights = torch.load(weight_path, map_location="cpu")
```

在 TP 模式下，权重加载还需要进行分片处理——每个 GPU 只加载属于自己的那部分权重，这通过 `weight_loader` 回调函数实现，确保每个 TP rank 获得正确的权重切片。

## 本章小结

本章详细分析了 SGLang 的量化方案与模型加载机制。AWQ 通过激活感知的策略实现了高质量的 INT4 量化；FP8 在 Hopper 架构上提供了原生的硬件加速；Marlin 则通过极致的 kernel 优化最大化 INT4 推理性能。模型加载流程通过统一的抽象支持多种权重格式和量化方案，结合 TP 分片机制实现了高效的分布式权重加载。选择合适的量化方案，需要综合考虑模型精度要求、硬件支持和部署预算。
