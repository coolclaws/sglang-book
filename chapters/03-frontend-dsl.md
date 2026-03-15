---
title: 第 3 章：SGLang 前端设计
---

# 第 3 章：SGLang 前端设计

> "程序即请求。" —— SGLang 前端 DSL 的本质，是将复杂的 LLM 交互逻辑压缩为一个可被运行时深度优化的结构化程序。

## 从 API 调用到编程模型

传统的 LLM 推理框架将每次交互视为一个独立的 API 调用：发送 prompt，接收 completion。但在实际应用中，LLM 的使用模式远比这复杂——多轮对话、分支选择、并行生成、结构化输出等场景层出不穷。每个场景如果都用独立的 API 调用来实现，不仅代码冗长，更错失了大量的优化机会。

SGLang 提出了 **"program as request"**（程序即请求）的范式：用户不再逐个发送 API 请求，而是编写一个完整的结构化生成程序，由运行时统一调度和优化。这个程序使用 SGLang 提供的 DSL 原语来描述，定义在 `python/sglang/lang/ir.py` 中。

## 核心原语

### gen —— 文本生成

`SglGen` 是最基础的原语，表示一次模型生成操作：

```python
# python/sglang/lang/ir.py
class SglGen:
    def __init__(self, name, max_new_tokens, min_new_tokens,
                 temperature, top_p, top_k, stop, regex,
                 json_schema, ...):
        ...
```

在用户代码中通过 `gen()` 函数调用：

```python
s += sgl.gen("answer", max_new_tokens=256, temperature=0.7)
```

`gen` 原语将生成请求与采样参数打包为一个 IR 节点，交由运行时执行。通过 `name` 参数，用户可以在后续代码中引用生成的结果。

### select —— 约束选择

`SglSelect` 原语让模型从预定义的选项列表中选择：

```python
# python/sglang/lang/ir.py
class SglSelect:
    def __init__(self, name, choices, temperature, choices_method):
        ...
```

使用示例：

```python
s += sgl.select("sentiment", choices=["积极", "消极", "中性"])
```

与直接让模型自由生成不同，`select` 利用运行时对每个选项计算 log-probability，选择概率最高的选项。这比正则约束更高效，因为不需要逐 token 约束生成。

### fork —— 分支执行

`SglFork` 原语创建多个并行的执行分支：

```python
# python/sglang/lang/ir.py
class SglFork:
    def __init__(self, number, position_ids_offset):
        ...
```

使用场景示例：

```python
forks = s.fork(3)
for f in forks:
    f += sgl.gen("response", temperature=1.0)
s += sgl.join(forks)
```

`fork` 操作会复制当前的对话状态（包括 KV Cache 前缀），创建多个独立的执行路径。由于 RadixAttention 的前缀共享机制，这些分支可以复用相同的 KV Cache 前缀，避免重复计算。

### join —— 分支合并

`join` 操作将 `fork` 创建的多个分支合并回主执行流。在合并时，运行时可以收集所有分支的生成结果，供后续逻辑使用。

## SglFunction 装饰器

`SglFunction` 是 SGLang 程序的核心封装，定义在 `python/sglang/lang/ir.py` 中：

```python
class SglFunction:
    def bind(self, **kwargs):
        """创建预绑定参数的新函数"""
        ...

    def run(self, **kwargs):
        """同步执行函数"""
        ...

    def run_batch(self, batch_kwargs, **kwargs):
        """批量执行，共享采样参数"""
        ...
```

用户通过装饰器或显式包装来定义 SGLang 程序：

```python
@sgl.function
def multi_turn_qa(s, question1, question2):
    s += sgl.system("你是一个有用的助手。")
    s += sgl.user(question1)
    s += sgl.gen("answer1", max_new_tokens=256)
    s += sgl.user(question2)
    s += sgl.gen("answer2", max_new_tokens=256)
```

`SglFunction` 不仅是一个装饰器，更是一个完整的执行单元。它通过 `run()` 方法驱动解释器（`python/sglang/lang/interpreter.py`）逐步执行 IR 节点，将每个 `gen`、`select` 操作翻译为对后端引擎的实际调用。

## DSL 到运行时的映射

整个前端的执行流程可以概括为：

```
用户代码 (@sgl.function)
    │
    ▼
SglFunction.run()
    │
    ▼
Interpreter 逐步执行 IR 节点
    │
    ├── SglGen   → 调用 backend.generate()
    ├── SglSelect → 调用 backend.select()
    └── SglFork  → 创建并行 session
```

解释器通过后端适配器（`python/sglang/lang/backend/`）与具体的推理引擎解耦。这意味着同一个 SGLang 程序既可以在本地 SGLang Runtime 上执行，也可以对接其他兼容的推理服务。

## 多轮程序示例

以下是一个综合运用多个原语的 SGLang 程序：

```python
@sgl.function
def essay_judge(s, topic):
    # 第一轮：生成文章
    s += sgl.user(f"请以'{topic}'为题写一篇短文。")
    s += sgl.gen("essay", max_new_tokens=512)

    # 第二轮：自我评估
    s += sgl.user("请对上面的文章进行评分。")
    s += sgl.select("rating", choices=["优秀", "良好", "一般", "较差"])

    # 第三轮：并行生成改进建议
    forks = s.fork(2)
    for i, f in enumerate(forks):
        f += sgl.user(f"请从第{i+1}个角度提出改进建议。")
        f += sgl.gen("suggestion", max_new_tokens=256)
```

在这个例子中，运行时可以自动识别：第二轮的 prompt 与第一轮共享前缀，`fork` 产生的两个分支也共享前缀，从而通过 RadixAttention 实现大幅度的 KV Cache 复用。这就是 **编程模型与运行时协同设计** 带来的性能红利。

## 本章小结

本章详细介绍了 SGLang 前端 DSL 的设计理念和核心原语。`gen`、`select`、`fork`、`join` 四个原语构成了结构化生成的完整表达能力，`SglFunction` 则提供了程序级别的封装与执行能力。前端 DSL 与 RadixAttention 的协同是 SGLang 区别于其他推理框架的关键特性。下一章我们将深入 Runtime 前端接口，探索 `Engine`、`Runtime` 等类如何将前端请求引入调度系统。
