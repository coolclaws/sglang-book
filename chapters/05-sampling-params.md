---
title: 第 5 章：Sampling 参数与结构化输出
---

# 第 5 章：Sampling 参数与结构化输出

> "控制生成的艺术，在于精确地告诉模型什么该说、什么不该说。" —— 采样参数和结构化输出约束是实现这一目标的两大武器。

## SamplingParams 概览

`SamplingParams` 类定义在 `python/sglang/srt/sampling/sampling_params.py` 中，是 SGLang 控制文本生成行为的核心数据结构。每一个推理请求都会携带一个 `SamplingParams` 实例，它决定了模型如何从概率分布中选择下一个 token，以及何时停止生成。

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
    json_schema: Optional[str] = None
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    structural_tag: Optional[str] = None
    ...
```

这个类不仅定义了参数字段，还包含 `verify()` 方法用于参数合法性校验和 `normalize()` 方法用于参数标准化处理。这两个方法确保了无论用户传入什么参数组合，系统都能正确且一致地处理。

## 核心采样参数详解

### temperature：温度控制

`temperature` 是最常用的采样参数，控制输出概率分布的"尖锐程度"。数学上，它作用于 softmax 之前的 logits 值：

```
logits = logits / temperature
probs = softmax(logits)
```

当 `temperature` 设为 0 时，等价于贪心解码（greedy decoding），模型始终选择概率最高的 token，输出完全确定性。当 `temperature` 设为较高值（如 1.5 或 2.0）时，概率分布变得更加平坦，低概率的 token 也有较大机会被选中，输出更具随机性和多样性。默认值 1.0 表示使用模型原始的概率分布，不做任何缩放。

在实际应用中，参数选择取决于具体场景：需要确定性输出的任务（如 JSON 生成、代码补全、信息抽取）通常设 `temperature=0`；创意写作、头脑风暴等场景则适合设为 0.7 到 1.0；而探索性的实验场景可以使用更高的温度值。

### top_p：核采样

`top_p`（也称 nucleus sampling，核采样）是另一种控制输出多样性的策略。它按概率从高到低对所有 token 进行排序并累加，仅保留累积概率刚好达到 `top_p` 阈值的最小 token 集合。默认值 1.0 表示保留所有 token，不进行截断。

例如，设 `top_p=0.9` 时，系统会找到概率最高的那些 token，使得它们的概率之和恰好达到 0.9，然后只从这个子集中采样。这样就过滤掉了尾部的低概率噪声 token，同时保留了足够的多样性。与固定的 `top_k` 相比，`top_p` 的优势在于它能够自适应地调整候选集大小：当模型非常确定时（一个 token 的概率很高），候选集会很小；当模型不确定时，候选集会自动扩大。

### top_k：Top-K 采样

`top_k` 直接限制候选 token 的数量。设 `top_k=50` 时，只保留概率最高的 50 个 token，其余全部被屏蔽。默认值 -1 表示不进行 Top-K 截断。这是最简单直接的多样性控制方式，适合对采样行为有明确预期的场景。需要注意的是，`top_k` 和 `top_p` 可以同时使用，系统会依次应用两种过滤策略。

### min_p：最小概率过滤

`min_p` 是一种更加自适应的过滤策略，默认值为 0.0（不启用）。它的工作方式是：首先找到当前分布中概率最高的 token，然后将其概率乘以 `min_p` 得到一个阈值，所有概率低于这个阈值的 token 将被排除。

这种相对过滤比 `top_k` 的绝对过滤更加灵活。当模型非常确定（最高概率接近 1.0）时，阈值较高，大量低概率 token 被过滤；当模型不确定（概率分布较平坦）时，阈值较低，更多 token 被保留。

### 重复惩罚参数

SGLang 提供三种互补的重复控制机制，可以单独或组合使用：

- **`frequency_penalty`**（默认 0.0）：按 token 出现次数施加线性惩罚，出现越多惩罚越大。适合抑制高频重复词汇。
- **`presence_penalty`**（默认 0.0）：只要 token 曾经出现过就施加固定惩罚，与出现次数无关。适合鼓励模型探索新话题和新词汇。
- **`repetition_penalty`**（默认 1.0）：对已出现 token 的 logit 值除以惩罚系数。值大于 1.0 时降低重复概率，值小于 1.0 则增加重复倾向。这种方式直接操作 logits，与 temperature 的作用机制不同。

## 生成停止控制

`SamplingParams` 提供了多层次的停止条件控制，满足不同粒度的需求：

```python
stop: Optional[Union[str, List[str]]] = None        # 文本级停止词
stop_token_ids: Optional[List[int]] = None           # Token ID 级停止
stop_regex: Optional[Union[str, List[str]]] = None   # 正则表达式匹配停止
ignore_eos: bool = False                             # 是否忽略 EOS token
no_stop_trim: bool = False                           # 是否在输出中保留停止词
max_new_tokens: int = 128                            # 最大生成长度限制
min_new_tokens: int = 0                              # 最小生成长度保障
```

`stop` 支持单个字符串或字符串列表，当生成的文本中出现任意一个停止词时立即终止生成。`stop_token_ids` 在 token 级别进行检查，效率更高。`stop_regex` 支持正则表达式匹配，适合更复杂的停止条件。`min_new_tokens` 确保生成至少产出指定数量的 token，在此之前不会触发任何停止条件，这对于避免过短回复非常有用。

### 多候选生成

`n` 参数（默认 1）指定为单个 prompt 生成多少个独立的候选结果。当 `n > 1` 时，系统会为同一输入创建多条独立的采样路径，每条路径使用相同的采样参数但不同的随机种子，返回多个完成结果。这在实现 best-of-n 采样、self-consistency 投票等策略时非常有用。配合 `sampling_seed` 参数，可以精确控制每次生成的随机性，确保实验的可复现性。

## 结构化输出

结构化输出是 SGLang 最重要的差异化特性之一，允许用户通过声明式的约束条件确保模型输出严格符合预定义的格式。相关的约束实现位于 `python/sglang/srt/constrained/` 目录，通过 `ServerArgs` 中的 `grammar_backend` 参数选择具体的实现引擎。

### JSON Schema 约束

通过 `json_schema` 参数，用户可以指定输出必须符合的 JSON 结构：

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

系统在收到包含 `json_schema` 约束的请求后，会将 JSON Schema 编译为有限状态自动机（Finite State Automaton, FSA）。在每一步解码时，FSA 根据当前状态计算出合法的 token 集合，将所有非法 token 的概率设为负无穷，确保采样只能选择符合语法规则的 token。这种约束方式在 token 级别生效，保证输出结果一定是合法的 JSON 且符合给定的 Schema 定义。

### 正则表达式约束

`regex` 参数允许通过正则表达式精确约束输出的文本格式：

```python
result = engine.generate(
    prompt="请生成一个中国大陆手机号码。",
    sampling_params={"regex": r"1[3-9]\d{9}", "temperature": 0}
)
```

正则表达式同样被编译为有限状态自动机。在每一步解码过程中，系统检查当前 FSA 的状态，确定哪些 token 能够使自动机转移到下一个合法状态，只允许这些 token 参与采样。正则约束适用于电话号码、邮箱地址、日期格式、枚举值等固定格式的输出场景。

### EBNF 语法约束

`ebnf` 参数支持扩展巴科斯-诺尔范式（Extended Backus-Naur Form），能够表达更复杂的上下文无关文法。这适用于需要生成符合编程语言语法、自定义 DSL 或复杂数据格式的场景。EBNF 的表达能力远超正则表达式，可以描述嵌套结构、递归模式等复杂语法规则。

### Structural Tag

`structural_tag` 参数提供了一种基于标签的结构化输出方式，允许在生成过程中定义结构化的输出区域。这种方式在需要混合自由文本和结构化数据的场景中特别有用。

## 参数在系统中的完整流转

`SamplingParams` 从用户指定到实际影响生成结果，经历了一条清晰的流转路径：

```
用户指定参数（Dict 字典或 SamplingParams 对象）
    │
    ▼
Engine.generate() 接收并封装参数
    │
    ▼
SamplingParams.verify()  —— 参数合法性校验
    │  检查 temperature >= 0, 0 <= top_p <= 1 等约束
    ▼
SamplingParams.normalize() —— 参数标准化
    │  例如 temperature=0 时切换为贪心模式
    ▼
Scheduler 将 SamplingParams 附加到 Request 对象
    │
    ▼
采样引擎（python/sglang/srt/sampling/）
    │  根据 temperature/top_p/top_k/min_p 执行采样
    │
    ├── 约束后端（python/sglang/srt/constrained/）
    │   根据 json_schema/regex/ebnf 构建 FSA
    │   过滤不合法的 token，修改 logits 掩码
    │
    ▼
输出符合所有约束条件的合法 token
```

在这条流转路径中，`verify()` 和 `normalize()` 是两个关键的预处理步骤。`verify()` 会检查所有参数的合法性，例如 `temperature` 不能为负数、`top_p` 必须在 0 到 1 之间、`n` 必须为正整数等，对不合法的参数抛出明确的错误提示。`normalize()` 则进行语义层面的标准化处理，例如当 `temperature` 设为 0 时自动启用贪心解码逻辑，避免除以零的数值问题。

采样引擎和约束后端在 Scheduler 的每一步解码中协同工作：采样引擎首先根据温度缩放 logits，然后依次应用 top_k、top_p、min_p 过滤；如果存在结构化输出约束，约束后端会进一步修改 logits 掩码，将所有违反语法规则的 token 的分数设为负无穷；最终从过滤后的分布中采样得到下一个 token。

## 本章小结

本章系统性地介绍了 SGLang 的采样参数体系和结构化输出机制。`SamplingParams` 类提供了 `temperature`、`top_p`、`top_k`、`min_p` 等丰富的采样控制参数，以及 `frequency_penalty`、`presence_penalty`、`repetition_penalty` 三种重复惩罚机制。在结构化输出方面，`json_schema`、`regex`、`ebnf`、`structural_tag` 四种约束方式覆盖了从简单格式到复杂语法的全部需求。这些参数从用户请求出发，经过校验和标准化，最终在采样引擎和约束后端中协同生效，确保生成结果既满足质量要求又严格符合格式规范。
