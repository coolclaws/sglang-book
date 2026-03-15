---
title: 第 19 章：多模态支持
---

# 第 19 章：多模态支持

> "语言模型的边界不应止于文本——当视觉与语言融合，智能才真正完整。"

随着 LLaVA、Qwen-VL、InternVL 等视觉语言模型（VLM）的兴起，推理引擎对多模态输入的支持变得越来越重要。SGLang 从架构层面提供了对多模态模型的原生支持，本章将深入解析其图像处理管线和多模态推理的核心实现。

## 多模态架构概览

SGLang 的多模态支持涉及多个模块的协作。从一张图像进入系统到最终参与注意力计算，需要经历以下阶段：

```
图像输入 → 图像预处理 → Vision Encoder 编码
    → Image Embedding 注入 → 与文本 Token 合并 → 模型推理
```

这一流程的实现分布在模型定义、请求处理和调度器等多个组件中。

## 图像预处理管线

当用户通过 API 发送包含图像的请求时（如 OpenAI 的 vision 格式），图像首先在 `TokenizerManager` 中被提取和预处理。SGLang 支持多种图像输入方式：Base64 编码、URL 链接和本地文件路径。

```python
# python/sglang/srt/managers/image_processor.py
class ImageProcessor:
    def __init__(self, hf_config, server_args, tokenizer):
        self.processor = get_processor(
            hf_config, server_args, tokenizer
        )

    async def process_images_async(self, images, input_text):
        # 使用模型对应的 processor 进行预处理
        pixel_values = self.processor(
            images=images, return_tensors="pt"
        ).pixel_values
        return pixel_values
```

不同的多模态模型使用不同的图像处理器。例如 LLaVA 系列使用 CLIP 的图像处理器，将图像缩放到固定分辨率并归一化；而 Qwen-VL 支持动态分辨率，会将图像分割为多个 patch。SGLang 通过 HuggingFace 的 `AutoProcessor` 机制自动适配不同模型的处理需求。

## Vision Token 的处理

多模态模型中，图像信息通过特殊的 vision token 占位符嵌入到文本序列中。以 LLaVA 为例，输入文本中的 `<image>` 占位符会被替换为一系列 image token：

```python
# python/sglang/srt/models/llava.py
class LlavaForCausalLM(nn.Module):
    def __init__(self, config, ...):
        self.vision_tower = CLIPVisionModel(config.vision_config)
        self.mm_projector = MultiModalProjector(config)
        self.language_model = LlamaForCausalLM(config)

    def encode_images(self, pixel_values):
        # 通过 Vision Encoder 提取图像特征
        image_features = self.vision_tower(pixel_values)
        # 通过投影层将视觉特征映射到语言模型的维度
        projected = self.mm_projector(image_features)
        return projected
```

关键步骤在于投影层（`mm_projector`）的设计——它将 Vision Encoder 输出的特征向量从视觉空间映射到语言模型的嵌入空间。常见的投影方式包括线性投影和 MLP 投影。

## Image Embedding 的注入机制

在 prefill 阶段，SGLang 需要将图像 embedding 注入到输入序列的正确位置。这一过程在模型的 `forward` 方法中完成：

```python
# 简化的 embedding 注入逻辑
def forward(self, input_ids, pixel_values=None, image_offsets=None, ...):
    # 获取文本 embedding
    text_embeds = self.embed_tokens(input_ids)

    if pixel_values is not None:
        # 编码图像
        image_embeds = self.encode_images(pixel_values)
        # 在 image_offsets 指定的位置替换文本 embedding
        for i, offset in enumerate(image_offsets):
            num_image_tokens = image_embeds[i].shape[0]
            text_embeds[offset:offset + num_image_tokens] = image_embeds[i]

    return self.language_model(inputs_embeds=text_embeds)
```

`image_offsets` 记录了每张图像在 token 序列中的起始位置，确保图像特征被插入到正确的位置。这些位置信息在 tokenize 阶段就已经计算好，并作为请求元数据传递给模型。

## 多模态模型的 KV Cache 管理

多模态请求在 KV Cache 管理方面有其特殊性。图像 token 通常会占据大量的序列位置（例如 LLaVA-1.5 中一张图像对应 576 个 token），这意味着：

1. **更大的 prefill 开销**：图像 token 的 prefill 计算量显著增加
2. **更多的 KV Cache 占用**：每张图像需要额外的 KV Cache 存储空间
3. **RadixAttention 的适配**：图像 token 的 KV Cache 也可以被复用

SGLang 的 RadixAttention 对多模态场景提供了天然的优势——当多个请求共享相同的系统提示和图像时，图像 token 的 KV Cache 可以在 Radix Tree 中被高效复用，避免重复计算。

## 支持的多模态模型

SGLang 通过模型注册机制支持多种多模态架构，定义在 `python/sglang/srt/models/` 目录下：

| 模型系列 | 对应文件 | 特点 |
|---------|---------|------|
| LLaVA | `llava.py` | CLIP + LLaMA，经典 VLM 架构 |
| LLaVA-NeXT | `llava.py` | 支持动态分辨率 |
| Qwen-VL | `qwen2_vl.py` | 动态 patch，位置感知 |
| InternVL | `internvl2.py` | 高分辨率，多 patch 支持 |

每个模型文件实现了标准的 `forward` 接口，通过统一的注册机制（`ModelRegistry`）被调度器识别和调用。

## 注意力计算中的图像 Token

在注意力计算层面，图像 token 与文本 token 的处理方式完全一致——它们共享相同的注意力矩阵和 KV Cache 机制。这是因为图像特征在注入时已经被投影到与文本 embedding 相同的空间。FlashInfer 或 Triton 后端在计算注意力时，并不区分 token 的来源类型，这保证了多模态推理的计算效率。

## 本章小结

本章系统分析了 SGLang 的多模态支持实现。从图像预处理管线到 Vision Encoder 编码，从投影层的维度映射到 embedding 注入机制，SGLang 通过模块化设计在不改变核心推理流程的前提下，优雅地集成了多模态能力。RadixAttention 对图像 token KV Cache 的复用进一步提升了多模态场景的效率。随着视觉语言模型的快速发展，SGLang 的多模态架构展现了良好的可扩展性。
