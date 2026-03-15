---
title: 第 17 章：Jump Forward Decoding
---

# 第 17 章：Jump Forward Decoding

> "当未来已经注定，何必逐步前行？Jump Forward Decoding 让模型学会跳过已知的答案，直达真正需要思考的地方。"

在上一章中，我们了解了 SGLang 如何通过 FSM 实现结构化生成。然而，标准的受约束解码有一个效率问题：即使下一个 token 是确定的（只有一个合法选择），模型仍然需要执行完整的 forward pass。Jump Forward Decoding 正是为了解决这个问题而设计的——它跳过所有确定性 token，直接推进到下一个需要采样的位置。

## 17.1 问题的提出

考虑一个 JSON 生成的场景。假设模型已经输出了 `{"name": "Ali`，根据 JSON 语法约束，在字符串值结束之前，模型仍有多种合法选择（任意字母）。但当模型输出 `"` 关闭字符串后，接下来必须输出 `, "age": ` 这样的确定性序列（假设 Schema 要求）。

在传统的受约束解码中：

```
Step 1: 输出 ","   → 需要完整 forward（但结果确定）
Step 2: 输出 " "   → 需要完整 forward（但结果确定）
Step 3: 输出 """   → 需要完整 forward（但结果确定）
Step 4: 输出 "age" → 需要完整 forward（但结果确定）
Step 5: 输出 """   → 需要完整 forward（但结果确定）
Step 6: 输出 ":"   → 需要完整 forward（但结果确定）
Step 7: 输出 " "   → 需要完整 forward（但结果确定）
Step 8: 输出数字    → 需要完整 forward（这次结果不确定！）
```

前 7 步的 forward 计算完全是浪费的，因为结果已经由语法约束决定了。

## 17.2 Jump Forward 的核心思想

Jump Forward Decoding 的核心思想非常直观：**当 FSM 在某个状态只有唯一的转移路径时，直接沿着这条路径"跳"到下一个有多个选择的状态**。

```python
# Jump Forward 的基本逻辑
def try_jump_forward(self, tokenizer):
    """尝试跳过确定性 token 序列"""
    jump_tokens = []
    state = self.current_state

    while True:
        next_states = self.fsm.get_transitions(state)
        if len(next_states) != 1:
            # 有多个可能的转移，停止跳跃
            break
        # 唯一的转移，直接跳过
        token, next_state = next_states[0]
        jump_tokens.append(token)
        state = next_state

    return jump_tokens  # 返回可以跳过的 token 序列
```

## 17.3 在 SGLang 中的实现

SGLang 在 `python/sglang/srt/constrained/` 目录下实现了 Jump Forward 机制。这个实现与 Grammar Backend 和 Scheduler 紧密协作。

### 17.3.1 Grammar Backend 中的跳跃检测

```python
# python/sglang/srt/constrained/outlines_backend.py
class OutlinesGrammarBackend(BaseGrammarBackend):
    def try_jump_forward(self, tokenizer):
        # 检查当前 FSM 状态是否存在确定性路径
        jump_forward_str = self.fsm.find_jump_forward_string()
        if jump_forward_str:
            # 将确定性字符串编码为 token 序列
            jump_tokens = tokenizer.encode(jump_forward_str)
            # 更新 FSM 状态
            for token in jump_tokens:
                self.accept_token(token)
            return jump_forward_str
        return None
```

### 17.3.2 与解码循环的集成

Jump Forward 的触发发生在每次采样之后。当一个 token 被采样并接受后，系统会检查是否可以继续跳跃：

```python
# 在 Scheduler 的解码循环中（简化）
def process_batch_result(self, batch, result):
    for req in batch.requests:
        # 1. 接受采样得到的 token
        req.grammar_backend.accept_token(result.next_token)

        # 2. 尝试 Jump Forward
        jump_str = req.grammar_backend.try_jump_forward(self.tokenizer)
        if jump_str:
            # 3. 将跳跃的 token 直接追加到序列中
            jump_token_ids = self.tokenizer.encode(jump_str)
            req.extend_input_ids(jump_token_ids)
            # 4. 这些 token 将在下一次 prefill 中被处理
            req.need_chunked_prefill = True
```

关键点在于：跳跃得到的 token 不需要经过模型的逐个 decode，而是作为已知输入在下一次 forward 中以 prefill 的方式一次性处理。这本质上将多次 decode forward 合并为一次 prefill forward。

## 17.4 性能收益分析

Jump Forward Decoding 的性能收益取决于约束中确定性 token 的比例。以几个典型场景为例：

### 17.4.1 JSON 输出

JSON 格式中包含大量确定性结构：花括号、引号、冒号、逗号、键名等。对于一个有多个字段的 JSON Schema，确定性 token 可能占总输出的 40%-60%。

```
不使用 Jump Forward:  每个 token 一次 forward → N 次 forward
使用 Jump Forward:    跳过确定性 token      → ~0.5N 次 forward
```

### 17.4.2 正则表达式约束

正则表达式中的固定字符串部分（如前缀、分隔符）都可以被跳过：

```python
# 正则表达式: r"Answer: (yes|no)\. Confidence: [0-9]+%"
# 确定性部分: "Answer: ", ". Confidence: ", "%"
# 非确定性部分: "(yes|no)", "[0-9]+"
```

### 17.4.3 性能提升的量化

根据 SGLang 论文中的实验数据，Jump Forward Decoding 在典型的结构化生成任务上可以带来显著的端到端加速：

- **JSON 生成**：约 1.5x - 3x 加速
- **严格格式约束**：约 2x - 5x 加速
- **松散约束**（大部分 token 非确定性）：加速较小

## 17.5 实现细节与边界情况

### 17.5.1 Token 边界对齐

Jump Forward 面临的一个技术挑战是 token 边界对齐问题。确定性字符串在 tokenize 时可能产生与逐个 decode 不同的 token 序列：

```python
# 逐个 decode: ["}", ",", " ", "\"", "age", "\"", ":"]
# 整体 tokenize: ["}, \"", "age", "\":"]
# 两者的 token 序列可能不同！
```

SGLang 通过特殊处理确保跳跃后的 KV Cache 与逐步 decode 的结果保持一致。具体来说，跳跃的 token 序列会作为 prefill 输入重新通过模型，确保 KV Cache 被正确填充。

### 17.5.2 与 RadixAttention 的协作

Jump Forward 与 RadixAttention 的前缀缓存可以产生协同效应。当多个请求使用相同的 JSON Schema 时，确定性前缀部分的 KV Cache 可以被共享：

```
请求 A: {"name": "Alice", "age": 30}
请求 B: {"name": "Bob", "age": 25}
共享前缀: {"name": "  → KV Cache 可复用
```

## 17.6 与 XGrammar 的结合

除了 Outlines 之外，SGLang 也支持使用 XGrammar 作为结构化生成后端。XGrammar 同样支持 Jump Forward 机制，并且在某些场景下提供了更高效的 FSM 实现：

```python
# python/sglang/srt/constrained/xgrammar_backend.py
class XGrammarGrammarBackend(BaseGrammarBackend):
    def try_jump_forward(self, tokenizer):
        # XGrammar 的跳跃实现
        result = self.matcher.find_jump_forward_string()
        return result
```

## 本章小结

Jump Forward Decoding 是 SGLang 在结构化生成领域的标志性优化。通过利用 FSM 中的确定性转移路径，它将不必要的 decode forward 调用转化为高效的 prefill 操作，在 JSON 生成等典型场景下带来了显著的性能提升。这一优化与 RadixAttention 的前缀缓存机制形成协同，进一步放大了 SGLang 在结构化生成任务中的性能优势。Jump Forward Decoding 的设计体现了 SGLang 团队"从约束中寻找加速机会"的工程哲学——有时候，限制反而是优化的起点。
