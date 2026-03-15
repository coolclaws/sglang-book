---
title: 第 5 章：Sampling 参数与结构化输出
---

# 第 5 章：Sampling 参数与结构化输出

> "控制生成的艺术，在于精确地告诉模型什么该说、什么不该说。" —— 采样参数和结构化输出约束是实现这一目标的两大武器。

## SamplingParams 概览

`SamplingParams` 类定义在 `python/sglang/srt/sampling/sampling_params.py` 中，是 SGLang 控制文本生成行为的核心数据结构。每一个推理请求都会携带一个 `SamplingParams` 实例，决定模型如何从概率分布中选择下一个 token。

```python
# python/sglang/srt/sampling/sampling_params.py
class SamplingParams:
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    min_new_tokens: int = 0
    n: int = 1
    ...
```

## 核心采样参数详解

### temperature：温度

`temperature` 控制输出分布的随机性。设为 0 时等价于贪心解码（greedy decoding），始终选择概率最高的 token；设为较高值（如 1.5）时，分布更加平坦，输出更具多样性。

```
logits = logits / temperature
probs = softmax(logits)
```

在实际应用中，需要确定性输出的场景（如 JSON 生成、代码补全）通常设 `temperature=0`，而创意写作类场景则适合设为 0.7-1.0。

### top_p：核采样

`top_p`（nucleus sampling）按概率从高到低累加，仅保留累积概率达到 `top_p` 阈值的 token 集合。默认值 1.0 表示不进行截断。

例如，设 `top_p=0.9` 时，模型只从覆盖 90% 累积概率的 token 子集中采样，过滤掉尾部低概率的噪声 token。

### top_k：Top-K 采样

`top_k` 直接限制候选 token 的数量。设 `top_k=50` 时，只保留概率最高的 50 个 token。默认值 -1 表示不进行 Top-K 截断。

### min_p：最小概率过滤

`min_p` 是一种相对过滤策略：任何概率低于 `最高概率 × min_p` 的 token 将被排除。这比固定的 `top_k` 更加自适应，因为它根据当前分布的"确定性"动态调整候选集大小。

### 惩罚参数

SGLang 提供三种重复控制机制：

- **`frequency_penalty`**（默认 0.0）：按 token 出现次数进行惩罚，出现越多惩罚越大。
- **`presence_penalty`**（默认 0.0）：只要 token 出现过就施加固定惩罚，不计次数。
- **`repetition_penalty`**（默认 1.0）：对已出现 token 的 logit 除以惩罚系数，值大于 1.0 时降低重复概率。

这三个参数可以组合使用，在不同场景下精细控制生成的多样性与连贯性。

## 生成控制参数

### 停止条件

`SamplingParams` 提供了多层次的停止控制：

```python
stop: Optional[Union[str, List[str]]] = None        # 文本级停止词
stop_token_ids: Optional[List[int]] = None           # Token ID 级停止
stop_regex: Optional[Union[str, List[str]]] = None   # 正则表达式停止
ignore_eos: bool = False                             # 是否忽略 EOS token
no_stop_trim: bool = False                           # 是否保留停止词
```

`stop` 支持字符串或字符串列表，当生成文本包含任一停止词时立即终止。`stop_regex` 则支持正则表达式匹配，提供更灵活的停止条件。

### 长度控制

- **`max_new_tokens`**（默认 128）：最大生成 token 数。
- **`min_new_tokens`**（默认 0）：最小生成 token 数，在此之前不会触发停止条件。

### 多候选生成

`n` 参数（默认 1）指定为单个 prompt 生成多少个独立的候选结果。当 `n > 1` 时，系统会为同一个输入创建多个采样路径，返回多个完成结果。

## 结构化输出

结构化输出是 SGLang 的重要特性之一，允许用户通过约束条件确保模型输出符合特定格式。相关的约束实现位于 `python/sglang/srt/constrained/` 目录。

### JSON Schema 约束

通过 `json_schema` 参数，用户可以指定输出必须符合的 JSON Schema：

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "hobbies": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["name", "age"]
}

result = engine.generate(
    prompt="请生成一个用户信息的 JSON。",
    sampling_params={"json_schema": json.dumps(schema), "temperature": 0}
)
```

系统会在每一步生成时，根据 JSON Schema 构建有限状态自动机（FSM），只允许符合语法规则的 token 被采样。这由 `python/sglang/srt/constrained/` 下的 grammar backend 实现。

### 正则表达式约束

`regex` 参数允许通过正则表达式约束输出格式：

```python
result = engine.generate(
    prompt="请生成一个电话号码。",
    sampling_params={"regex": r"1[3-9]\d{9}", "temperature": 0}
)
```

正则表达式同样被编译为 FSM，在解码过程中逐 token 进行合法性检查。

### EBNF 语法约束

`ebnf` 参数支持更复杂的上下文无关文法（CFG）约束，适用于编程语言语法、DSL 等场景。

### Structural Tag

`structural_tag` 参数提供了一种更高级的结构化输出方式，允许在生成过程中插入结构化标记。

## 参数在系统中的流转

`SamplingParams` 从用户请求到实际生效的完整路径如下：

```
用户指定 sampling_params（Dict 或 SamplingParams）
    │
    ▼
Engine.generate() 接收参数
    │
    ▼
SamplingParams.verify()  —— 参数合法性校验
SamplingParams.normalize() —— 参数标准化
    │
    ▼
Scheduler 将参数附加到 Request 对象
    │
    ▼
Sampling Engine（python/sglang/srt/sampling/）
    │  根据 temperature/top_p/top_k 等执行采样
    │
    ├── Constrained Backend（python/sglang/srt/constrained/）
    │   根据 json_schema/regex/ebnf 过滤非法 token
    │
    ▼
输出合法 token
```

`verify()` 方法会检查参数的合法性，例如 `temperature` 不能为负数、`top_p` 必须在 0 到 1 之间等。`normalize()` 方法则进行标准化处理，例如当 `temperature` 为 0 时自动切换为贪心解码模式。

## 本章小结

本章系统性地介绍了 SGLang 的采样参数体系和结构化输出机制。`SamplingParams` 类提供了 `temperature`、`top_p`、`top_k`、`min_p` 等丰富的采样控制参数，以及 `json_schema`、`regex`、`ebnf` 等结构化输出约束。这些参数从用户请求出发，经过校验和标准化，最终在采样引擎和约束后端中生效，确保生成结果既满足质量要求又符合格式规范。
