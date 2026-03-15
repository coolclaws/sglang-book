---
title: 第 3 章：SGLang 前端设计
---

# 第 3 章：SGLang 前端设计

> "程序即请求。" —— SGLang 前端 DSL 的本质，是将复杂的 LLM 交互逻辑压缩为一个可被运行时深度优化的结构化程序。

## 从 API 调用到编程模型

传统的 LLM 推理框架将每次交互视为一个独立的 API 调用：发送 prompt，接收 completion。但在实际应用中，LLM 的使用模式远比这复杂——多轮对话、分支选择、并行生成、结构化输出等场景层出不穷。每个场景如果都用独立的 API 调用来实现，不仅代码冗长，更重要的是错失了大量的运行时优化机会。

举一个具体的例子：假设我们需要实现一个"生成文章 → 评判质量 → 给出改进建议"的三步流程。使用传统框架，开发者需要手动发起三次独立的 API 调用，并在应用层管理上下文传递。每次调用都会重新处理完整的对话历史，造成大量的计算浪费。更重要的是，运行时无法提前知道后续还有哪些操作，因此无法进行跨请求的全局优化。

SGLang 提出了 **"program as request"**（程序即请求）的范式：用户不再逐个发送 API 请求，而是编写一个完整的结构化生成程序，由运行时统一调度和优化。这个程序使用 SGLang 提供的 DSL 原语来描述，核心的 IR（中间表示）节点定义在 `python/sglang/lang/ir.py` 中。运行时可以分析程序的完整结构，识别可复用的前缀、可并行的分支，从而进行全局最优的调度决策。

## 核心原语

SGLang 的前端 DSL 由四个核心原语构成，每一个原语在 `python/sglang/lang/ir.py` 中都有对应的 IR 类定义。

### gen —— 文本生成

`SglGen` 是最基础也是最常用的原语，表示一次模型文本生成操作：

```python
# python/sglang/lang/ir.py
class SglGen:
    def __init__(self, name, max_new_tokens, min_new_tokens,
                 temperature, top_p, top_k, stop, regex,
                 json_schema, ...):
        ...
```

在用户代码中，通过 `gen()` 函数来创建生成节点：

```python
s += sgl.gen("answer", max_new_tokens=256, temperature=0.7)
```

`gen` 原语将生成请求与采样参数打包为一个 IR 节点，交由运行时执行。其中 `name` 参数为生成结果指定一个名称，用户可以在后续代码中通过这个名称引用生成的文本内容。`gen` 支持丰富的参数配置，包括最大生成长度、采样温度、停止条件、正则约束、JSON Schema 约束等，这些参数会被传递到后端的 `SamplingParams` 中。

### select —— 约束选择

`SglSelect` 原语让模型从预定义的选项列表中进行选择，而不是自由生成文本：

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

与直接让模型自由生成再进行后处理不同，`select` 利用运行时的底层机制对每个候选选项分别计算 log-probability（对数概率），选择概率最高的选项作为结果。这种方式比正则约束更加高效，因为它不需要逐 token 进行约束生成，而是通过一次前向计算就能得到所有候选的概率分布。`choices_method` 参数允许用户指定不同的选择策略，例如基于首 token 概率还是完整序列概率来做决策。

### fork —— 分支执行

`SglFork` 原语创建多个并行的执行分支，每个分支可以独立进行不同的生成操作：

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

`fork` 操作会复制当前的对话状态（包括 KV Cache 前缀），创建指定数量的独立执行路径。这个操作与 RadixAttention 的前缀共享机制天然契合：所有分支共享 fork 之前的完整 KV Cache 前缀，避免了重复计算。在基数树中，这些分支从同一个前缀节点分叉出去，各自延伸生成独立的内容。`position_ids_offset` 参数用于在特定场景下调整分支的位置编码偏移量，确保注意力计算的正确性。

### join —— 分支合并

`join` 操作将 `fork` 创建的多个分支合并回主执行流。在合并时，运行时会收集所有分支的生成结果，供后续逻辑使用。合并后，用户可以访问每个分支各自生成的内容，进行比较、选择或进一步处理。这形成了一个完整的"分散-收集"（scatter-gather）模式，是实现 tree-of-thought、self-consistency 等高级推理策略的基础。

## SglFunction 装饰器

`SglFunction` 是 SGLang 程序的核心封装单元，定义在 `python/sglang/lang/ir.py` 中。它将一个普通的 Python 函数包装为可被 SGLang 运行时识别和执行的结构化生成程序：

```python
class SglFunction:
    def bind(self, **kwargs):
        """创建预绑定参数的新函数实例"""
        ...

    def run(self, **kwargs):
        """同步执行函数，返回完整结果"""
        ...

    def run_batch(self, batch_kwargs, **kwargs):
        """批量执行，多组输入共享采样参数"""
        ...

    def trace(self):
        """追踪程序执行路径，用于分析和调试"""
        ...
```

用户通过 `@sgl.function` 装饰器来定义 SGLang 程序：

```python
@sgl.function
def multi_turn_qa(s, question1, question2):
    s += sgl.system("你是一个有用的助手。")
    s += sgl.user(question1)
    s += sgl.gen("answer1", max_new_tokens=256)
    s += sgl.user(question2)
    s += sgl.gen("answer2", max_new_tokens=256)
```

`SglFunction` 不仅是一个装饰器，更是一个完整的执行单元。它通过 `run()` 方法驱动解释器（`python/sglang/lang/interpreter.py`）逐步执行 IR 节点，将每个 `gen`、`select` 操作翻译为对后端引擎的实际调用。`run_batch()` 方法则支持批量执行，将多组不同的输入参数打包成一个批次，共享相同的采样配置，充分利用 GPU 的并行计算能力。`bind()` 方法支持函数式编程中的偏函数（partial application）模式，预先绑定部分参数创建新的函数实例。

## DSL 到运行时的映射

整个前端的执行流程形成了一条清晰的从用户代码到底层推理引擎的调用链：

```
用户代码 (@sgl.function 装饰的函数)
    │
    ▼
SglFunction.run() 启动执行
    │
    ▼
Interpreter 逐步遍历 IR 节点
    │
    ├── SglGen   → 调用 backend.generate()  发起文本生成
    ├── SglSelect → 调用 backend.select()   执行约束选择
    ├── SglFork  → 创建多个并行 session      分支执行
    └── SglJoin  → 收集分支结果              合并分支
```

解释器通过后端适配器（`python/sglang/lang/backend/`）与具体的推理引擎解耦。后端适配器定义了统一的接口规范，包括 `generate()`、`select()` 等方法。这意味着同一个 SGLang 程序既可以在本地的 SGLang Runtime 上执行，也可以对接 OpenAI API 或其他兼容的推理服务，只需切换不同的 backend 实现即可。

## 多轮程序综合示例

以下是一个综合运用多个原语的完整 SGLang 程序，展示了 DSL 在实际场景中的表达能力：

```python
@sgl.function
def essay_judge(s, topic):
    # 第一轮：生成文章
    s += sgl.user(f"请以'{topic}'为题写一篇短文。")
    s += sgl.gen("essay", max_new_tokens=512, temperature=0.8)

    # 第二轮：自我评估（使用 select 进行约束选择）
    s += sgl.user("请对上面的文章进行评分。")
    s += sgl.select("rating", choices=["优秀", "良好", "一般", "较差"])

    # 第三轮：并行生成多个角度的改进建议
    forks = s.fork(3)
    for i, f in enumerate(forks):
        f += sgl.user(f"请从第{i+1}个角度提出改进建议。")
        f += sgl.gen("suggestion", max_new_tokens=256)
    s += sgl.join(forks)
```

在这个例子中，运行时可以自动识别并利用以下优化机会：第二轮的 prompt 与第一轮共享完整前缀，无需重新计算第一轮的 KV Cache；`fork` 产生的三个分支共享前两轮的全部 KV Cache 前缀，只需各自计算分支后新增的部分；`select` 操作通过概率比较而非自由生成来完成评分选择，更加高效和可靠。这就是 **编程模型与运行时协同设计** 带来的性能红利——用户只需自然地表达意图，运行时自动发现和利用优化空间。

## 本章小结

本章详细介绍了 SGLang 前端 DSL 的设计理念和核心原语。`gen`、`select`、`fork`、`join` 四个原语构成了结构化生成的完整表达能力，分别对应文本生成、约束选择、分支执行和分支合并四种基本操作。`SglFunction` 装饰器提供了程序级别的封装与执行能力，通过解释器和后端适配器将 DSL 程序映射为底层推理调用。前端 DSL 与 RadixAttention 的协同设计是 SGLang 区别于其他推理框架的关键特性，它使得运行时能够"理解"用户程序的结构并进行全局优化。下一章我们将深入 Runtime 前端接口，探索 `Engine`、`Runtime` 等类如何将前端请求引入调度系统。
