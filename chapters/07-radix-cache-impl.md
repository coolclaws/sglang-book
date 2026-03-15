---
title: 第 7 章：RadixCache 实现
---

# 第 7 章：RadixCache 实现

> "优雅的数据结构设计，需要同样精巧的工程实现来承载。"

## 从设计到实现

上一章我们讨论了 RadixAttention 的设计思想，本章将深入 SGLang 的源码，分析 `RadixCache` 的具体实现。核心代码位于 `python/sglang/srt/mem_cache/radix_cache.py`，这个文件包含了 Radix Tree 的完整生命周期管理。

## TreeNode 类详解

`TreeNode` 是 Radix Tree 的基本构建单元。在 SGLang 的实现中，每个节点存储一段 token 序列及其对应的 KV Cache 在内存池中的索引。

```python
# python/sglang/srt/mem_cache/radix_cache.py
class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.key = None
        self.value = None
        self.lock_ref = 0
        self.last_access_time = 0
```

关键字段解读：

- **`key`**：该节点代表的 token 序列片段，以 tuple 形式存储。
- **`value`**：对应的 KV Cache 在 `TokenToKVPool` 中的索引数组，用于定位实际的显存位置。
- **`children`**：子节点字典，以下一个 token 的值作为键。
- **`lock_ref`**：引用计数器，正在被活跃请求使用的节点不可驱逐。
- **`last_access_time`**：最后访问时间戳，用于 LRU 驱逐策略。

## 核心操作：match_prefix

`match_prefix` 是 RadixCache 最关键的操作之一，负责在树中查找与给定 token 序列匹配的最长前缀。

```python
# python/sglang/srt/mem_cache/radix_cache.py
def match_prefix(self, key, **kwargs):
    if len(key) == 0:
        return [], self.root_node, 0

    node = self.root_node
    matched_len = 0
    value = []

    while matched_len < len(key):
        first_token = key[matched_len]
        if first_token not in node.children:
            break
        child = node.children[first_token]
        child_key = child.key

        prefix_len = _common_prefix_len(child_key, key[matched_len:])

        if prefix_len < len(child_key):
            # 部分匹配，需要分裂节点
            new_node = self._split_node(child, prefix_len)
            value.extend(new_node.value)
            matched_len += prefix_len
            node = new_node
            break

        value.extend(child.value)
        matched_len += len(child_key)
        node = child

    return value, node, matched_len
```

这段代码揭示了几个重要的实现细节：

1. **逐层遍历**：从根节点出发，通过 `first_token` 索引子节点，逐层向下匹配。
2. **部分匹配处理**：当一个节点的 key 只有部分被匹配时，调用 `_split_node` 将节点分裂。
3. **返回值**：返回匹配到的 KV Cache 索引列表、最后匹配的节点以及匹配的 token 数量。

### 节点分裂机制

节点分裂是 Radix Tree 维护的核心操作。当新的 token 序列与某个节点的 key 部分匹配时，需要将该节点一分为二：

```python
def _split_node(self, node, split_pos):
    # 创建新的中间节点
    new_node = TreeNode()
    new_node.key = node.key[:split_pos]
    new_node.value = node.value[:split_pos]
    new_node.parent = node.parent

    # 更新原节点
    node.key = node.key[split_pos:]
    node.value = node.value[split_pos:]
    node.parent = new_node

    # 重新连接树结构
    new_node.children[node.key[0]] = node
    return new_node
```

分裂后，原节点变为新中间节点的子节点，树的逻辑结构保持不变，但增加了一个新的分支点。

## 插入操作：insert

当 prefill 计算完成后，新的 KV Cache 需要被插入到 Radix Tree 中：

```python
# python/sglang/srt/mem_cache/radix_cache.py
def insert(self, key, value=None):
    if len(key) == 0:
        return 0

    matched_value, node, matched_len = self.match_prefix(key)

    if matched_len == len(key):
        return matched_len  # 完全匹配，无需插入

    # 创建新节点存储剩余部分
    new_key = key[matched_len:]
    new_node = TreeNode()
    new_node.key = new_key
    new_node.value = value[matched_len:] if value else None
    new_node.parent = node
    node.children[new_key[0]] = new_node

    return matched_len
```

插入操作首先尝试匹配前缀，然后只为未匹配的部分创建新节点。这确保了树中不会出现重复的前缀存储。

## 驱逐策略：LRU on Leaf Nodes

显存是有限资源，当可用空间不足时，RadixCache 需要驱逐一些不再活跃的节点来释放 KV Cache 空间。SGLang 采用了基于叶节点的 LRU（Least Recently Used）驱逐策略。

```python
# python/sglang/srt/mem_cache/radix_cache.py
def evict(self, num_tokens):
    leaves = self._collect_evictable_leaves()
    # 按最后访问时间排序
    leaves.sort(key=lambda x: x.last_access_time)

    evicted = 0
    for leaf in leaves:
        if evicted >= num_tokens:
            break
        if leaf.lock_ref > 0:
            continue  # 跳过被锁定的节点
        tokens_freed = len(leaf.key)
        self._delete_leaf(leaf)
        self.token_to_kv_pool.free(leaf.value)
        evicted += tokens_freed
    return evicted
```

驱逐策略的核心要点：

- **仅驱逐叶节点**：内部节点可能被多个分支共享，不能随意删除。
- **LRU 排序**：优先驱逐最久未使用的叶节点。
- **锁定保护**：`lock_ref > 0` 的节点正在被活跃请求使用，不可驱逐。
- **级联清理**：删除叶节点后，如果其父节点也变成了无子节点的叶节点且未被锁定，可以继续驱逐。

## 内存池集成

RadixCache 与 `TokenToKVPool`（定义在 `python/sglang/srt/mem_cache/memory_pool.py`）紧密配合。内存池负责实际的 GPU 显存分配和管理，而 RadixCache 负责逻辑层面的 KV Cache 索引管理。

```python
# python/sglang/srt/mem_cache/memory_pool.py
class TokenToKVPool:
    def alloc(self, num_tokens):
        # 从预分配的显存块中分配 token 槽位
        ...
    def free(self, indices):
        # 释放 token 槽位
        ...
```

RadixCache 中的 `value` 字段存储的是内存池中的索引。当 RadixCache 驱逐节点时，它调用内存池的 `free` 方法归还显存空间；当插入新节点时，从内存池的 `alloc` 方法获取空间。

## Token 序列到树路径的映射

一个 token 序列 `[t1, t2, t3, ..., tn]` 在 Radix Tree 中的存储方式取决于与已有节点的前缀重叠情况。假设树中已有路径 `[t1, t2, t3, t4]`，新序列 `[t1, t2, t5, t6]` 插入后，树的结构变为：

```
Root -> [t1, t2] -> [t3, t4]  (原有路径)
                  -> [t5, t6]  (新路径)
```

原来的单一节点 `[t1, t2, t3, t4]` 被分裂为 `[t1, t2]` 和 `[t3, t4]` 两个节点，新路径 `[t5, t6]` 作为 `[t1, t2]` 的另一个子节点插入。这就是 Radix Tree 的压缩与分裂机制在实际运行中的体现。

## 本章小结

本章深入分析了 `RadixCache` 的实现细节。`TreeNode` 作为基本构建单元承载 token 片段及 KV Cache 索引；`match_prefix` 通过逐层遍历实现最长前缀匹配，必要时触发节点分裂；`insert` 操作在匹配基础上只为增量部分创建新节点；驱逐策略采用叶节点 LRU 方式，通过引用计数保护活跃数据。这些组件共同构成了一个高效的 KV Cache 复用系统。下一章我们将把 RadixCache 与 vLLM 的 PagedAttention 进行对比分析。
