---
title: 第 16 章：Outlines 集成
---

# 第 16 章：Outlines 集成

> "让语言模型按照规则说话，不是限制它的自由，而是赋予它在受限空间中精确表达的能力。"

结构化生成是 LLM 应用中的重要需求——我们经常需要模型输出符合特定格式的文本，如 JSON、SQL 或满足某种正则表达式的字符串。SGLang 通过集成 Outlines 库，利用有限状态机（FSM）实现了高效的受约束解码。本章将剖析这一集成的架构与实现细节。

## 16.1 结构化生成的挑战

大语言模型的自回归生成过程是逐 token 进行的。在每一步，模型输出一个概率分布，然后从中采样得到下一个 token。如果我们希望最终输出符合某种语法约束（如合法的 JSON），需要在每一步采样时"屏蔽"掉不合法的 token。

这个问题的核心挑战在于：如何高效地在每一步计算出哪些 token 是合法的？

```python
# 结构化生成的基本流程
for step in generation:
    logits = model.forward(input_ids)
    # 关键步骤：根据语法约束计算合法 token 集合
    allowed_tokens = grammar.get_allowed_tokens(current_state)
    # 将非法 token 的 logits 设为 -inf
    logits[~allowed_tokens] = float("-inf")
    next_token = sample(logits)
```

## 16.2 GrammarBackend 抽象

SGLang 在 `python/sglang/srt/constrained/` 目录下定义了统一的 Grammar Backend 抽象，支持多种结构化生成引擎：

```python
# python/sglang/srt/constrained/base_grammar_backend.py
class BaseGrammarBackend:
    def accept_token(self, token: int):
        """接受一个 token，更新内部状态"""
        raise NotImplementedError

    def try_jump_forward(self, tokenizer):
        """尝试跳过确定性 token（Jump Forward）"""
        raise NotImplementedError

    def fill_vocab_mask(self, vocab_mask: torch.Tensor):
        """填充 token 掩码，标记合法 token"""
        raise NotImplementedError
```

这个抽象层使得 SGLang 可以支持 Outlines、XGrammar 等多种结构化生成后端，而上层解码逻辑无需修改。

## 16.3 Outlines 的 FSM 机制

Outlines 的核心是将语法约束（如正则表达式或 JSON Schema）编译为有限状态机（Finite State Machine，FSM）。FSM 的每个状态对应生成过程中的一个"位置"，状态之间的转移由 token 触发。

### 16.3.1 从正则表达式到 FSM

```python
# Outlines 的 FSM 构建流程
# 1. 将正则表达式编译为 FSM
regex = r'"name":\s*"[a-zA-Z]+"'
fsm = RegexFSM(regex, tokenizer)

# 2. 预计算每个状态的合法 token 集合
# state_to_tokens: Dict[int, Set[int]]
# 状态 0 -> {token_id('"')}
# 状态 1 -> {token_id('n'), token_id('na'), token_id('name'), ...}
# ...
```

### 16.3.2 FSM 索引的预构建

Outlines 最重要的优化是**预构建 FSM 索引**。在开始生成之前，Outlines 会遍历 tokenizer 的整个词表，计算每个状态下每个 token 能触发的转移：

```python
# python/sglang/srt/constrained/outlines_backend.py
class OutlinesGrammarBackend(BaseGrammarBackend):
    def __init__(self, regex_string, tokenizer):
        # 预构建 FSM 索引 —— 这一步可能较慢
        self.fsm = RegexGuide(regex_string, tokenizer)
        self.state = 0  # 初始状态

    def fill_vocab_mask(self, vocab_mask):
        # 查表获取当前状态的合法 token
        allowed = self.fsm.get_next_instruction(self.state)
        vocab_mask.fill_(0)
        for token_id in allowed.tokens:
            vocab_mask[token_id] = 1
```

这种预计算策略将运行时的 token 过滤从"遍历整个词表"简化为"查表"操作，极大提升了解码效率。

## 16.4 Token 掩码机制

Token 掩码是结构化生成的核心执行机制。在每一步解码时，Grammar Backend 生成一个与词表大小相同的布尔掩码，标记哪些 token 可以被选择。

```python
# 在 SGLang 的采样流程中应用掩码
# python/sglang/srt/sampling/sampling_batch_info.py
def apply_grammar_mask(self, logits, grammar_backends):
    for i, backend in enumerate(grammar_backends):
        if backend is not None:
            vocab_mask = torch.zeros(vocab_size, dtype=torch.bool)
            backend.fill_vocab_mask(vocab_mask)
            # 将非法 token 的 logits 设为负无穷
            logits[i][~vocab_mask] = float("-inf")
    return logits
```

掩码操作在 softmax 之前进行，确保非法 token 的采样概率为零。这种方法的优雅之处在于它不改变合法 token 之间的相对概率分布。

## 16.5 JSON Schema 约束

在实际应用中，JSON Schema 是最常用的结构化输出约束。SGLang 支持通过 JSON Schema 自动生成对应的正则表达式，再编译为 FSM：

```python
# 从 JSON Schema 到 FSM 的转换链
json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}

# JSON Schema → 正则表达式 → FSM
regex = json_schema_to_regex(json_schema)
fsm = RegexFSM(regex, tokenizer)
```

这种转换使得用户只需提供声明式的 Schema 定义，系统就能自动保证模型输出符合要求。

## 16.6 与解码循环的集成

结构化生成约束与 SGLang 的解码循环紧密集成。在 `Scheduler` 的每一步中：

1. 模型 forward 得到 logits
2. Grammar Backend 生成 token 掩码
3. 掩码应用到 logits 上
4. 采样得到下一个 token
5. Grammar Backend 接受该 token，更新 FSM 状态

```python
# 解码循环中的集成（简化）
next_token = sample(masked_logits)
for req in batch.requests:
    if req.grammar_backend:
        req.grammar_backend.accept_token(next_token)
```

## 本章小结

本章详细分析了 SGLang 与 Outlines 的集成机制。通过 `BaseGrammarBackend` 抽象层，SGLang 实现了结构化生成后端的可插拔设计。Outlines 的 FSM 机制将语法约束预编译为状态机索引，使得运行时的 token 过滤高效快速。Token 掩码机制优雅地与自回归解码流程结合，在不改变模型本身的前提下实现了输出格式的精确控制。下一章我们将探讨 Jump Forward Decoding——一种利用约束结构进一步加速生成的技术。
