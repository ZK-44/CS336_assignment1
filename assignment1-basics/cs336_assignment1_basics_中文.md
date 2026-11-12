# CS336 作业 1（基础）：构建 Transformer 语言模型

**版本 26.0.3**
**CS336 课程组**
**2026 年春季学期**

> 译者注：本文为课程作业说明文档 `cs336_assignment1_basics.pdf`（Version 26.0.3）的中文翻译，仅供学习参考；如有歧义，请以英文原版为准。请注意原文件第 1 节中关于 AI 工具使用的课程政策。

---

## 1 作业概述

在本作业中，你将从零构建训练一个标准 Transformer 语言模型（LM）所需的全部组件，并训练一些模型。

**你将实现的内容**

1. 字节对编码（BPE）分词器（第 2 节）
2. Transformer 语言模型（LM）（第 3 节）
3. 交叉熵损失函数与 AdamW 优化器（第 4 节）
4. 训练循环，支持序列化与加载模型及优化器状态（第 5 节）

**你将运行的内容**

1. 在 TinyStories 数据集上训练 BPE 分词器。
2. 使用训练好的分词器对数据集进行编码，将其转换为整数 ID 序列。
3. 在 TinyStories 数据集上训练 Transformer LM。
4. 使用训练好的 Transformer LM 进行采样生成，并评估困惑度（perplexity）。
5. 在 OpenWebText 上训练模型，并将你达到的困惑度提交到排行榜。

**你可以使用的工具**

我们要求你从零构建每个组件。特别地，除了以下内容之外，你**不得**使用 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的任何定义：

- `torch.nn.Parameter`
- `torch.nn` 中的容器类（例如 `Module`、`ModuleList`、`Sequential` 等）<sup>1</sup>
- `torch.optim.Optimizer` 基类

你可以使用 PyTorch 的任何其他定义。如果你想使用某个函数或类，但不确定是否被允许，欢迎在 Slack 上提问。如有疑问，请考虑使用它是否会损害本作业"从零实现"的精神。

<sup>1</sup> 完整列表见 pytorch.org/docs/stable/nn.html#containers。

**关于 AI 工具的声明**

AI 可以完全自主地解决作业中的许多部分。这使得人们更难深入接触课程材料并从中学习。

允许使用 AI 工具来回答高层的概念性问题，或提供低层级的编程文档（如函数签名和库 API）。但是，**不允许**使用 AI 工具来实现作业的任何部分。这包括编程智能体（如 Cursor Agents、Codex、Claude Code）和 AI 自动补全（如 Cursor Tab、GitHub Copilot）。在使用 AI 智能体时，请确保它使用了所提供的 AGENTS.md 文件；使用聊天机器人时，也应附上相应的提示词。

我们强烈建议你在完成作业时在 IDE 中禁用 AI 自动补全（不过非 AI 的自动补全——例如自动补全函数名——完全没问题）。以往的学生反馈说，禁用 AI 自动补全让他们更容易深入地学习材料。

完整的 AI 政策请参阅此文档。

**代码结构**

作业代码以及本文档均可在 GitHub 上获取：
github.com/stanford-cs336/assignment1-basics

请 git clone 该仓库。如果有任何更新，我们会通知你，你可以通过 git pull 获取最新版本。

1. `cs336_basics/*`：这是你编写代码的地方。注意这里没有任何代码——你可以完全从零开始，做任何你想做的事！
2. `adapters.py`：这里定义了一组你的代码必须实现的功能。对于每一项功能（例如 scaled dot product attention），通过简单地调用你自己的代码来填充相应的实现（例如 `run_scaled_dot_product_attention`）。注意：你对 `adapters.py` 的修改不应包含任何实质性逻辑；这只是胶水代码。
3. `test_*.py`：这里包含了你必须通过的所有测试（例如 `test_scaled_dot_product_attention`），这些测试会调用 `adapters.py` 中定义的钩子。不要编辑测试文件。

**如何提交**

如需提交，请运行 `make_submission.sh` 来构建提交用的 zip 文件。如果你有不想包含在提交 zip 中的大型数据文件或检查点（checkpoint），请务必将它们添加到脚本的排除列表中。

你将向 Gradescope 提交以下文件：

- `writeup.pdf`：回答所有书面问题。请对你的回答进行排版（typeset）。
- `code.zip`：包含你编写的所有代码。

如需提交到排行榜，请向以下仓库提交 PR：
github.com/stanford-cs336/assignment1-basics-leaderboard

详细的提交说明请参阅排行榜仓库中的 README.md。

**在哪里获取数据集**

本作业将使用两个预处理过的数据集：TinyStories [R. Eldan et al., 2023] 和 OpenWebText [A. Gokaslan et al., 2019]。两个数据集都是单个大型纯文本文件。

如果你随课程一起完成作业，可以在计算指南（compute guide）中找到下载数据集的说明。

如果你在家自学，可以使用 README.md 中的命令下载这些文件。

> **低资源提示**
>
> 在本课程的作业文档中，我们将针对如何在更少或没有 GPU 资源的情况下完成作业的各部分提供建议。例如，我们有时会建议缩小数据集或模型规模，或者解释如何在 Mac 集成 GPU 或 CPU 上运行训练代码。你会在蓝色方框（就像这个）中找到这些"低资源提示"。即使你是可以使用课程机器的斯坦福在读学生，这些提示也可能帮助你更快地迭代、节省时间，因此我们建议阅读它们！

> **低资源提示：在 Apple Silicon 或 CPU 上完成作业 1**
>
> 使用助教的参考实现代码，我们可以在 Apple M4 Max 芯片（36 GB 内存）上训练出一个能生成相当流畅文本的 LM：在 Metal GPU（MPS）上不到 5 分钟，使用 CPU 约 30 分钟。如果你不明白这些词是什么意思，不用担心！只要知道：如果你有一台配置较新的笔记本电脑，且你的实现正确且高效，你就能训练出一个能以不错的流畅度生成简单儿童故事的小型 LM。
>
> 在作业的后半部分，我们将解释如果你使用 CPU 或 MPS 需要做哪些改动。

---

## 2 字节对编码（BPE）分词器

在作业的第一部分，我们将训练并实现一个基于字节的字节对编码（BPE）分词器 [R. Sennrich et al., 2016; C. Wang et al., 2019]。特别地，我们将把任意（Unicode）字符串表示为字节序列，并在这个字节序列上训练我们的 BPE 分词器。之后，我们将使用这个分词器把文本（字符串）编码为词元（token，即整数序列），用于语言建模。

### 2.1 Unicode 标准

Unicode 是一种文本编码标准，它将字符映射为整数码点（code point）。截至 Unicode 17.0（2025 年 9 月发布），该标准定义了 172 种文字（scripts）共 159,801 个字符。例如，字符"s"的码点是 115（通常记作 U+0073，其中 U+ 是约定俗成的前缀，0073 是 115 的十六进制表示），字符"牛"的码点是 29275。在 Python 中，你可以使用 `ord()` 函数将单个 Unicode 字符转换为其整数表示。`chr()` 函数则将一个整数 Unicode 码点转换为包含相应字符的字符串。

```python
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

> **问题（unicode1）：理解 Unicode（1 分）**
>
> (a) `chr(0)` 返回什么 Unicode 字符？
> **提交要求：** 一句话回答。
>
> (b) 这个字符的字符串表示（`__repr__()`）与其打印（print）出来的表示有何不同？
> **提交要求：** 一句话回答。
>
> (c) 当这个字符出现在文本中时会发生什么？在你的 Python 解释器中摆弄下面的代码、看看结果是否符合你的预期，可能会有所帮助：
>
> ```python
> >>> chr(0)
> >>> print(chr(0))
> >>> "this is a test" + chr(0) + "string"
> >>> print("this is a test" + chr(0) + "string")
> ```
>
> **提交要求：** 一句话回答。

### 2.2 Unicode 编码

虽然 Unicode 标准定义了从字符到码点（整数）的映射，但直接在 Unicode 码点上训练分词器并不实际，因为词表会大得离谱（约 15 万项）且稀疏（许多字符非常罕见）。相反，我们将使用 Unicode 编码，它把 Unicode 字符转换为字节序列。Unicode 标准本身定义了三种编码：UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网上的主导编码（超过 98% 的网页使用它）。

要将 Unicode 字符串编码为 UTF-8，我们可以使用 Python 中的 `encode()` 函数。要访问 Python `bytes` 对象底层的字节值，可以对它进行迭代（例如调用 `list()`）。最后，我们可以使用 `decode()` 函数将 UTF-8 字节串解码为 Unicode 字符串。

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # 获取编码后字符串的字节值（0 到 255 的整数）。
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129,
161, 227, 129, 175, 33]
>>> # 一个字节不一定对应一个 Unicode 字符！
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

通过将 Unicode 码点转换为字节序列（例如通过 UTF-8 编码），我们本质上是把一个码点序列（21 位整数，共有 159,801 个有效值）变换成了一个字节值序列（0 到 255 范围内的整数）。长度为 256 的字节词表要容易处理得多。使用字节级分词时，我们不必担心词表外（out-of-vocabulary）词元，因为我们知道任何输入文本都可以表示为 0 到 255 的整数序列。

> **问题（unicode2）：Unicode 编码（3 分）**
>
> (a) 相比 UTF-16 或 UTF-32，在 UTF-8 编码的字节上训练分词器有哪些优势？比较这些编码对各种输入字符串的输出可能会有所帮助。
> **提交要求：** 一到两句话的回答。
>
> (b) 考虑下面这个（错误的）函数，它本意是将 UTF-8 字节串解码为 Unicode 字符串。为什么这个函数是错误的？请给出一个会产生错误结果的字节串输入示例。
>
> ```python
> def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
>     return "".join([bytes([b]).decode("utf-8") for b in bytestring])
> >>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
> 'hello'
> ```
>
> **提交要求：** 一个使 `decode_utf8_bytes_to_str_wrong` 产生错误输出的字节串输入示例，并用一句话解释该函数为什么错误。
>
> (c) 给出一个无法解码为任何 Unicode 字符的两字节序列。
> **提交要求：** 一个示例，并附一句话解释。

### 2.3 子词（Subword）分词

虽然字节级分词可以缓解词级分词器面临的词表外问题，但将文本切分为字节会产生极长的输入序列。这会拖慢模型训练，因为在词级语言模型中只有 10 个词元的句子，在字符级模型中可能会有 50 个甚至更多词元（取决于词的长度）。处理这些更长的序列需要在模型的每一步进行更多计算。此外，在字节序列上做语言建模也很困难，因为更长的输入序列会在数据中造成长程依赖。

子词分词是词级分词器与字节级分词器之间的折中。注意字节级分词器的词表有 256 个条目（字节值为 0 到 255）。子词分词器以更大的词表规模换取对输入字节序列更好的压缩。例如，如果字节序列 `b'the'` 在我们的原始文本训练数据中频繁出现，把它加入词表就能把这个 3 词元序列缩减为单个词元。

我们如何选出要加入词表的这些子词单元呢？R. Sennrich et al. [3] 提出使用字节对编码（BPE；P. Gage [5]），这是一种压缩算法，它迭代地将出现频率最高的一对字节替换（"合并"）为一个新的未使用的索引。注意，该算法通过向词表中加入子词词元来最大化输入序列的压缩——如果一个词在输入文本中出现得足够频繁，它就会被表示为单个子词单元。

通过 BPE 构建词表的子词分词器通常称为 BPE 分词器。在本作业中，我们将实现一个字节级 BPE 分词器，其词表条目是字节或合并后的字节序列，这让我们在词表外处理和可管理的输入序列长度两方面兼得优势。构建 BPE 分词器词表的过程称为"训练"BPE 分词器。

### 2.4 BPE 分词器训练

BPE 分词器训练过程包含三个主要步骤。

**词表初始化**

分词器词表是字节串词元到整数 ID 的一对一映射。由于我们训练的是字节级 BPE 分词器，初始词表就是所有字节的集合。因为字节值共有 256 种可能，所以初始词表大小为 256。

**预分词（Pre-tokenization）**

有了词表之后，原则上你可以统计文本中每个字节与相邻字节共同出现的频率，并从最频繁的字节对开始合并。然而，这样做的计算成本很高，因为每次合并都需要对整个语料库完整遍历一遍。此外，直接在整个语料库上合并字节可能会产生仅标点符号不同的词元（例如 `dog!` 与 `dog.`）。这些词元会得到完全不同的词元 ID，尽管它们的语义很可能高度相似（因为它们只差一个标点）。

为了避免这种情况，我们对语料库进行预分词。你可以把它看作对语料库进行的粗粒度分词，帮助我们统计字符对出现的频率。例如，单词 'text' 可能是一个出现了 10 次的预分词（pre-token）。这样，当我们统计字符 't' 和 'e' 相邻出现的频率时，就会发现单词 'text' 中 't' 和 'e' 相邻，从而可以把它们的计数加 10，而不必再遍历整个语料库。由于我们训练的是字节级 BPE 模型，每个预分词都表示为一个 UTF-8 字节序列。

R. Sennrich et al. [3] 的原始 BPE 实现仅按空白字符切分进行预分词（即 `s.split(" ")`）。基于 SentencePiece 的分词器（例如 Llama 1 和 2 的分词器）中仍能找到这种方法。

大多数现代分词器使用基于正则表达式的预分词器，这一做法来自 GPT-2（A. Radford et al. [6]）。我们将使用原始正则表达式的一个略更美观的版本，取自 github.com/openai/tiktoken/pull/234/files：

```python
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

用这个预分词器对一些文本进行交互式切分，可能有助于你更好地理解它的行为：

```python
>>> # 需要 `regex` 包
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

然而在代码中使用它时，你应该使用 `re.finditer`，以避免在构建从预分词到其计数的映射时把所有预分词后的词都存储下来。

**计算 BPE 合并**

现在我们已经把输入文本转换成了预分词，并将每个预分词表示为 UTF-8 字节序列，接下来就可以计算 BPE 合并（即训练 BPE 分词器）了。概括地说，BPE 算法迭代地统计每一对字节，找出频率最高的一对（"A"、"B"）。然后把出现频率最高的这对（"A"、"B"）的每一处出现都进行合并，即替换为一个新词元 "AB"。这个新合并出的词元会被加入词表；因此，BPE 训练后的最终词表大小等于初始词表大小（在我们的例子中是 256）加上训练过程中执行的 BPE 合并操作次数。为了在 BPE 训练时保证效率，我们不考虑跨越预分词边界的词对。<sup>2</sup> 在计算合并时，若词对频率出现并列，则按字典序（lexicographically）较大的词对优先的规则确定性地打破平局。例如，如果词对 ("A", "B")、("A", "C")、("B", "ZZ") 和 ("BA", "A") 的频率同为最高，我们应合并 ("BA", "A")：

```python
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```

<sup>2</sup> 注意，原始 BPE 的表述（R. Sennrich et al. [3]）中包含了词尾（end-of-word）词元。我们在训练字节级 BPE 模型时不添加词尾词元，因为所有字节（包括空白和标点）都已包含在模型词表中。由于我们显式地表示了空格和标点，学到的 BPE 合并会自然地反映这些词的边界。

**特殊词元（Special tokens）**

有些字符串（例如 `<|endoftext|>`）常用于编码元数据（例如文档之间的边界）。在对文本进行编码时，通常希望把某些字符串当作"特殊词元"处理，它们永远不应被切分为多个词元（即始终被保留为单个词元）。例如，序列结束字符串 `<|endoftext|>` 应始终被保留为单个词元（即单个整数 ID），这样我们就知道何时应停止从语言模型生成。这些特殊词元必须被加入词表，以便它们拥有对应的固定词元 ID。

R. Sennrich et al. [3] 的算法 1 包含了一个低效的 BPE 分词器训练实现（基本遵循我们上面概述的步骤）。作为第一个练习，实现并测试这个函数来检验你的理解可能会很有用。

> **示例（bpe_example）：BPE 训练示例**
>
> 这是 R. Sennrich et al. [3] 中的一个示例。考虑由以下文本构成的语料库
>
> ```
> low low low low low
> lower lower widest widest widest
> newest newest newest newest newest newest
> ```
>
> 词表中有一个特殊词元 `<|endoftext|>`。
>
> **词表**
>
> 我们用在特殊词元 `<|endoftext|>` 和 256 个字节值上初始化词表。
>
> **预分词**
>
> 为简单起见，并聚焦于合并过程，本例中假设预分词仅仅是按空白切分。预分词并计数后，我们得到频率表
>
> ```
> {low: 5, lower: 2, widest: 3, newest: 6}
> ```
>
> 把它表示为 `dict[tuple[bytes, ...], int]` 会很方便，例如 `{(l,o,w): 5, …}`。注意，在 Python 中即使单个字节也是 `bytes` 对象。Python 中没有表示单个字节的 byte 类型，就像没有表示单个字符的 char 类型一样。
>
> **合并**
>
> 我们首先查看每对相邻字节，并累加它们所在单词的频率 `{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}`。词对 ('e', 's') 和 ('s', 't') 并列，因此我们取字典序较大的词对 ('s', 't')。然后合并预分词，得到 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}`。
>
> 第二轮中，我们看到 (e, st) 是最常见的词对（计数为 9），合并后得到 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}`。如此继续，最终得到的合并序列为 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']`。
>
> 如果取 6 次合并，即 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']`，我们的词表条目将是 `[<|endoftext|>, [...256 个字节字符], st, est, ow, low, west, ne]`。
>
> 使用这个词表和这组合并，单词 newest 将被分词为 `[ne, west]`。

---

### 2.5 BPE 分词器训练实验

让我们在 TinyStories 数据集上训练一个字节级 BPE 分词器。查找/下载数据集的说明见第 1 节。开始之前，我们建议先浏览一下 TinyStories 数据集，感受一下数据的内容。

**并行化预分词**

你会发现一个主要瓶颈是预分词步骤。你可以用内置库 `multiprocessing` 并行化代码来加速预分词。具体而言，我们建议在预分词的并行实现中对语料库进行分块（chunking），并确保分块边界出现在特殊词元的开头。你可以直接原样使用以下链接中的入门代码来获取分块边界，然后用它把任务分配到各个进程：

https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py

这种分块方式总是有效的，因为我们永远不希望跨越文档边界进行合并。就本作业而言，你总是可以这样切分。不必担心语料库非常大且不含 `<|endoftext|>` 的边界情况。

**预分词之前移除特殊词元**

在用正则表达式（使用 `re.finditer`）进行预分词之前，你应该先把语料库（或并行实现下的分块）中所有的特殊词元剥离掉。确保按特殊词元进行切分，这样它们所分隔的文本之间不会发生合并。例如，如果你的语料库（或分块）形如 `[Doc 1]<|endoftext|>[Doc 2]`，你应该在特殊词元 `<|endoftext|>` 处切分，并分别对 `[Doc 1]` 和 `[Doc 2]` 进行预分词，这样就不会发生跨越文档边界的合并。换言之，特殊词元在训练期间定义了硬切分边界，但它们自身不应计入合并统计。这可以使用 `re.split` 并以 `"|".join(special_tokens)` 作为分隔符来实现（注意谨慎使用 `re.escape`，因为特殊词元中可能出现 `|`）。测试 `test_train_bpe_special_tokens` 会检验这一点。

**优化合并步骤**

上面风格化示例中 BPE 训练的朴素实现很慢，因为每做一次合并，都要遍历所有字节对才能找出频率最高的词对。然而，每次合并后只有与合并词对重叠的词对计数会发生变化。因此，可以通过为所有词对的计数建立索引并增量更新这些计数，来加速 BPE 训练，而不必显式地遍历每一对字节来统计词对频率。这种缓存机制可以带来显著加速；不过我们指出，BPE 训练的合并部分在 Python 中是无法并行化的。

> **低资源提示：性能分析（Profiling）**
>
> 你应该使用 cProfile 或 py-spy 等性能分析工具来找出实现中的瓶颈，并专注于优化这些瓶颈。

> **低资源提示："缩小规模"（Downscaling）**
>
> 不要一上来就在完整的 TinyStories 数据集上训练分词器，我们建议先在一小部分数据上训练：一个"调试数据集"。例如，你可以改用 TinyStories 验证集来训练分词器，它只有 2.2 万个文档，而不是 212 万个。这体现了一个通用策略：尽可能缩小规模以加快开发迭代，例如使用更小的数据集、更小的模型等。选择调试数据集或超参数配置的规模需要仔细权衡：你希望调试集足够大，使其具有与完整配置相同的瓶颈（这样你所做的优化才能泛化），但又不能大到运行起来没完没了。

> **问题（train_bpe）：BPE 分词器训练（15 分）**
>
> **提交要求：** 编写一个函数，给定输入文本文件的路径，训练一个（字节级）BPE 分词器。你的 BPE 训练函数至少应处理以下输入参数：
>
> | 输入 | 说明 |
> | --- | --- |
> | `input_path: str` | BPE 分词器训练数据的文本文件路径 |
> | `vocab_size: int` | 正整数，定义最终词表的最大大小（包括初始字节词表、合并产生的词表条目，以及任何特殊词元） |
> | `special_tokens: list[str]` | 要加入词表的字符串列表。训练期间，将它们视为硬边界，阻止跨越其跨度进行合并，但在计算合并统计时不包含它们 |
>
> 你的 BPE 训练函数应返回得到的词表和合并（merges）：
>
> | 输出 | 说明 |
> | --- | --- |
> | `vocab: dict[int, bytes]` | 分词器词表，从 int（词表中的词元 ID）到 bytes（词元字节）的映射 |
> | `merges: list[tuple[bytes, bytes]]` | 训练得到的 BPE 合并列表。每个列表项是一个 bytes 元组 `(<token1>, <token2>)`，表示 `<token1>` 与 `<token2>` 合并。合并应按创建顺序排列。 |
>
> 若要对照我们提供的测试来测试你的 BPE 训练函数，你首先需要实现测试适配器 [adapters.run_train_bpe]。然后运行 `uv run pytest tests/test_train_bpe.py`。你的实现应能通过所有测试。可选地（这可能需要投入大量时间），你可以用某种系统级语言（例如 C++，考虑 cppyy 或 nanobind；或 Rust，使用 PyO3）实现训练方法的关键部分。如果这样做，请注意哪些操作需要拷贝、哪些可以直接从 Python 内存读取，并确保留下构建说明，或确保它仅使用 pyproject.toml 即可构建。还要注意，GPT-2 的正则表达式在大多数正则引擎中都得不到良好支持，在支持它的引擎中也会太慢。我们已验证 Oniguruma 速度相当快且支持否定前瞻（negative lookahead），而 Python 的 regex 包（如果有什么不同的话）甚至更快。

> **问题（train_bpe_tinystories）：在 TinyStories 上训练 BPE（2 分）**
>
> (a) 在 TinyStories 数据集上训练一个字节级 BPE 分词器，最大词表大小为 10,000。确保将 TinyStories 的 `<|endoftext|>` 特殊词元加入词表。将得到的词表和合并序列化到磁盘，以便进一步检查。训练花费了多少时间和内存？词表中最长的词元是什么？它合理吗？
>
> **资源要求：** ≤ 30 分钟（无需 GPU），≤ 30 GB 内存
>
> > **提示：** 通过在预分词时使用多进程，并利用以下两个事实，你应该能把 BPE 训练时间控制在 2 分钟以内：
> > (a) `<|endoftext|>` 词元在数据文件中分隔文档。
> > (b) `<|endoftext|>` 词元在应用 BPE 合并之前就已作为特殊情况处理。
>
> **提交要求：** 一到两句话的回答。
>
> (b) 对你的代码做性能分析。分词器训练过程中哪一部分耗时最多？
> **提交要求：** 一到两句话的回答。

接下来，我们将尝试在 OpenWebText 数据集上训练一个字节级 BPE 分词器。与之前一样，我们建议先浏览一下数据集，以更好地理解其内容。

> **问题（train_bpe_expts_owt）：在 OpenWebText 上训练 BPE（2 分）**
>
> (a) 在 OpenWebText 数据集上训练一个字节级 BPE 分词器，最大词表大小为 32,000。将得到的词表和合并序列化到磁盘，以便进一步检查。词表中最长的词元是什么？它合理吗？
>
> **资源要求：** ≤ 12 小时（无需 GPU），≤ 100 GB 内存
>
> **提交要求：** 一到两句话的回答。
>
> (b) 对比并讨论你在 TinyStories 上训练的分词器与在 OpenWebText 上训练的分词器。
> **提交要求：** 一到两句话的回答。

### 2.6 BPE 分词器：编码与解码

在作业的前一部分，我们实现了一个在输入文本上训练 BPE 分词器的函数，以获得分词器词表和 BPE 合并列表。现在，我们将实现一个 BPE 分词器，它加载给定的词表和合并列表，并使用它们在文本与词元 ID 之间进行编码和解码。

#### 2.6.1 文本编码

用 BPE 编码文本的过程与我们训练 BPE 词表的方式相呼应，主要有几个步骤。

**第 1 步：预分词。** 我们首先对序列进行预分词，并将每个预分词表示为 UTF-8 字节序列，就像 BPE 训练中做的那样。我们将在每个预分词内部把这些字节合并为词表条目，独立地处理每个预分词（不跨预分词边界合并）。

**第 2 步：应用合并。** 然后，我们取 BPE 训练期间生成的词表元素合并序列，并按创建的先后顺序应用到我们的预分词上。

> **示例（bpe_encoding）：BPE 编码示例**
>
> 例如，假设输入字符串为 'the cat ate'，词表为 `{0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}`，学到的合并为 `[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]`。首先，预分词器会将该字符串切分为 `['the', ' cat', ' ate']`。然后，我们逐个查看每个预分词并应用 BPE 合并。
>
> 第一个预分词 'the' 最初表示为 `[b't', b'h', b'e']`。查看合并列表，我们找到第一个适用的合并是 `(b't', b'h')`，用它把预分词变换为 `[b'th', b'e']`。然后回到合并列表，找到下一个适用的合并是 `(b'th', b'e')`，把预分词变换为 `[b'the']`。最后再看合并列表，发现没有更多适用于该字符串的合并了（因为整个预分词已被合并为单个词元），应用 BPE 合并的过程到此结束。对应的整数序列为 `[9]`。
>
> 对其余预分词重复这一过程：预分词 ' cat' 在应用 BPE 合并后表示为 `[b' c', b'a', b't']`，对应整数序列 `[7, 1, 5]`。最后一个预分词 ' ate' 在应用合并后为 `[b' at', b'e']`，对应整数序列 `[10, 3]`。因此，编码输入字符串的最终结果为 `[9, 7, 1, 5, 10, 3]`。

**特殊词元**

你的分词器应能在编码文本时正确处理用户定义的特殊词元（在构造分词器时提供）。

**内存考量**

假设我们想对一个无法全部装入内存的大文本文件进行分词。为了高效地对这样的大文件（或任何其他数据流）进行分词，我们需要把它切分为可管理大小的块并逐块处理，这样内存复杂度就是常数级，而不是随文本大小线性增长。在这样做时，我们必须确保词元不会跨越块边界，否则我们会得到与将整个序列放在内存中分词不同的结果。

#### 2.6.2 文本解码

要将整数词元 ID 序列解码回原始文本，我们只需在词表中查找每个 ID 对应的条目（一个字节序列），把它们拼接起来，然后把字节解码为 Unicode 字符串。注意，输入的 ID 并不保证能映射到合法的 Unicode 字符串（因为用户可以输入任何整数序列）。当输入的词元 ID 无法产生合法的 Unicode 字符串时，你应把格式错误的字节替换为官方的 Unicode 替换字符 U+FFFD。<sup>3</sup> `bytes.decode` 的 `errors` 参数控制 Unicode 解码错误的处理方式，使用 `errors='replace'` 会自动用替换标记替换格式错误的数据。

<sup>3</sup> 关于 Unicode 替换字符的更多信息，参见 en.wikipedia.org/wiki/Specials_(Unicode_block)#Replacement_character。

> **问题（tokenizer）：实现分词器（15 分）**
>
> **提交要求：** 实现一个 `Tokenizer` 类，给定词表和合并列表，将文本编码为整数 ID，并将整数 ID 解码为文本。你的分词器还应支持用户提供的特殊词元（如果它们尚不在词表中，则将其追加到词表）。我们推荐以下接口：
>
> - `def __init__(self, vocab, merges, special_tokens=None)`：根据给定的词表、合并列表和（可选的）特殊词元列表构造分词器。该函数应接受以下参数：
>   - `vocab: dict[int, bytes]`
>   - `merges: list[tuple[bytes, bytes]]`
>   - `special_tokens: list[str] | None = None`
> - `def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)`：类方法，从序列化保存的词表和合并列表（格式与你的 BPE 训练代码输出相同）构造并返回一个 `Tokenizer`，并（可选地）接受一组特殊词元。该方法还应接受以下参数：
>   - `vocab_filepath: str`
>   - `merges_filepath: str`
>   - `special_tokens: list[str] | None = None`
> - `def encode(self, text: str) -> list[int]`：将输入文本编码为词元 ID 序列。
> - `def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]`：给定一个字符串可迭代对象（例如 Python 文件句柄），返回一个惰性产生（lazily yield）词元 ID 的生成器。这是我们无法直接载入内存的大文件进行省内存分词所必需的。
> - `def decode(self, ids: list[int]) -> str`：将词元 ID 序列解码为文本。
>
> 若要对照我们提供的测试来测试你的 `Tokenizer`，你首先需要实现测试适配器 [adapters.get_tokenizer]。然后运行 `uv run pytest tests/test_tokenizer.py`。你的实现应能通过所有测试。

### 2.7 实验

> **问题（tokenizer_experiments）：分词器实验（4 分）**
>
> (a) 从 TinyStories 和 OpenWebText 中各抽取 10 个文档。使用你之前训练好的 TinyStories 和 OpenWebText 分词器（词表大小分别为 10K 和 32K），将这些抽样的文档编码为整数 ID。每个分词器的压缩率（字节/词元）是多少？
> **提交要求：** 一到两句话的回答。
>
> (b) 如果你用 TinyStories 分词器对 OpenWebText 样本进行分词，会发生什么？比较压缩率，和/或定性描述现象。
> **提交要求：** 一到两句话的回答。
>
> (c) 估算你的分词器的吞吐量（例如以字节/秒计）。对 Pile 数据集（825GB 文本）进行分词需要多长时间？
> **提交要求：** 一到两句话的回答。
>
> (d) 使用你的 TinyStories 和 OpenWebText 分词器，将各自的训练集和开发集编码为整数词元 ID 序列。我们稍后将用它来训练语言模型。我们建议把词元 ID 序列化为数据类型为 `uint16` 的 NumPy 数组。为什么 `uint16` 是一个合适的选择？
> **提交要求：** 一到两句话的回答。

---

## 3 Transformer 语言模型架构

语言模型以批量的整数词元 ID 序列（即形状为 `(batch_size, sequence_length)` 的 `torch.Tensor`）作为输入，并返回对词表的（批量）归一化概率分布（即形状为 `(batch_size, sequence_length, vocab_size)` 的 PyTorch Tensor），其中每个输入词元的预测分布都是关于下一个词的。训练语言模型时，我们利用这些下一词预测来计算真实下一个词与预测下一个词之间的交叉熵损失。在推理时从语言模型生成文本，我们取最后一个时间步（即序列中的最后一项）的预测下一词分布来生成序列中的下一个词元（例如取概率最高的词元、从分布中采样等），把生成的词元加入输入序列，然后重复这一过程。

在作业的这一部分，你将从零构建这个 Transformer 语言模型。我们先对模型进行高层次的描述，然后逐步细化各个组件。

### 3.1 Transformer LM

给定一个词元 ID 序列，Transformer 语言模型使用输入嵌入（input embedding）把词元 ID 转换为稠密向量，让嵌入后的词元通过 `num_layers` 个 Transformer 块，然后应用一个学习得到的线性投影（"输出嵌入"或 "LM head"）来产生预测的下一个词元 logits。示意图见图 1。

**图 1：我们的 Transformer 语言模型概览。**（输入：词元嵌入 → 重复 `num_layers` 次的 Transformer 块 → 线性（输出嵌入）→ Softmax → 输出概率；输入/输出张量形状均为 `(batch_size, seq_len, d_model)`）

**图 2：Pre-norm Transformer 块。**（输入张量形状 `(batch_size, seq_len, d_model)` → Norm → 带 RoPE 的因果多头自注意力 → Add → Norm → 逐位置前馈网络 → Add → 输出张量形状 `(batch_size, seq_len, d_model)`）

**词元嵌入（Token Embeddings）**

在第一步中，Transformer 将（批量的）词元 ID 序列嵌入为包含词元身份信息的向量序列（图 1 中的红色块）。

更具体地说，给定一个词元 ID 序列，Transformer 语言模型使用词元嵌入层来产生一个向量序列。每个嵌入层接收一个形状为 `(batch_size, sequence_length)` 的整数张量，产生一个形状为 `(batch_size, sequence_length, d_model)` 的向量序列。

**Pre-norm Transformer 块**

嵌入之后，激活值会经过若干个结构相同的神经网络层处理。一个标准的仅解码器（decoder-only）Transformer 语言模型由 `num_layers` 个完全相同的层组成（通常称为 Transformer"块"）。每个 Transformer 块接收形状为 `(batch_size, sequence_length, d_model)` 的输入，返回同样形状的输出。每个块通过自注意力在序列范围内聚合信息，并通过前馈层对其进行非线性变换。

经过 `num_layers` 个 Transformer 块之后，我们把最终的激活值转换为对词表的一个分布。

我们将实现 "pre-norm" Transformer 块（详见第 3.4 节），它另外要求在最后一个 Transformer 块之后使用层归一化（详见下文），以确保其输出具有合适的尺度。

归一化之后，我们将使用一个标准的学习得到的线性变换，把 Transformer 块的输出转换为预测的下一个词元 logits（参见例如 A. Radford et al. [7] 的公式 2）。

### 3.2 备注：批处理、Einsum 与高效计算

在整个 Transformer 中，我们将对许多"类批（batch-like）"输入执行相同的计算。以下是几个例子：

- 批内的各元素：我们对每个批元素应用相同的 Transformer 前向操作。
- 序列长度：像 RMSNorm 和前馈网络这样的"逐位置（position-wise）"操作，对序列的每个位置都执行相同的操作。
- 注意力头：注意力操作在多个注意力头之间按批处理，构成"多头"注意力操作。

拥有一种符合人体工学（ergonomic）的方式来执行这类操作很有用——既要充分利用 GPU，又要易于阅读和理解。许多 PyTorch 操作可以接受张量开头多余的"类批"维度，并在这些维度上高效地重复/广播该操作。

例如，假设我们要做一个逐位置的批处理操作。有一个形状为 `(batch_size, sequence_length, d_model)` 的"数据张量" $D$，我们想让它与一个形状为 `(d_model, d_model)` 的矩阵 $A$ 做批处理的向量-矩阵乘法。此时 `D @ A` 会执行批处理矩阵乘法——这是 PyTorch 中一个高效的原语，其中 `(batch_size, sequence_length)` 两个维度被当作批处理维度。

因此，假设你的函数可能接收到额外的类批维度，并把这些维度保留在 PyTorch 形状的开头，会很有帮助。为了让张量能以这种方式批处理，可能需要通过很多步 `view`、`reshape` 和 `transpose` 来调整形状。这可能有点痛苦，而且经常会让代码难以阅读、让人难以看出张量的形状。

更符合人体工学的选择是在 `torch.einsum` 中使用 einsum 记法，或者使用 einops 或 einx 这类框架无关的库。两个关键操作是：`einsum`，可以对输入张量的任意维度做张量缩并（tensor contraction）；`rearrange`，可以重排、拼接和拆分任意维度。事实上，机器学习中几乎所有的操作都是"维度变换"和"张量缩并"的某种组合，偶尔（通常是逐点）加上非线性函数。这意味着使用 einsum 记法可以让你的很多代码更易读、更灵活。

我们强烈建议学习并在课程中使用 einsum 记法。之前没接触过 einsum 记法的同学应使用 einops（文档在此），已经熟悉 einops 的同学应学习更通用的 einx（在此）。<sup>4</sup> 这两个包在我们提供的环境中都已经安装好了。

<sup>4</sup> 值得注意的是，einops 得到了广泛支持，而 einx 还没有经过充分的实战检验。如果你发现 einx 有任何局限或 bug，欢迎回退到 einops 加一些更朴素的 PyTorch 写法。

下面给出一些使用 einsum 记法的例子。这些是对 einops 文档的补充，你应该先阅读 einops 文档。

> **示例（einstein_example1）：用 einops.einsum 做批处理矩阵乘法**
>
> ```python
> import torch
> from einops import rearrange, einsum
>
> ## 基本实现
> Y = D @ A.T
> # 很难看出输入/输出形状及其含义。
> # D 和 A 可以是什么形状？其中是否有意想不到的行为？
>
> ## Einsum 是自文档化（self-documenting）且稳健的
> #                          D                A     ->          Y
> Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")
>
> ## 或者批处理版本：D 可以有任意前导维度，但 A 受限
> Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
> ```

> **示例（einstein_example2）：用 einops.rearrange 做广播操作**
>
> 我们有一批图像，想为每张图像基于某个缩放因子生成 10 个变暗版本：
>
> ```python
> images = torch.randn(64, 128, 128, 3)  # (batch, height, width, channel)
> dim_by = torch.linspace(start=0.0, end=1.0, steps=10)
>
> ## Reshape 并相乘
> dim_value = rearrange(dim_by,    "dim_value              -> 1 dim_value 1 1 1")
> images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
> dimmed_images = images_rearr * dim_value
>
> ## 或者一步到位：
> dimmed_images = einsum(
>     images, dim_by,
>     "batch height width channel, dim_value -> batch dim_value height width channel"
> )
> ```

> **示例（einstein_example3）：用 einops.rearrange 做像素混合**
>
> 假设我们有一批图像，表示为形状 `(batch, height, width, channel)` 的张量，我们想对图像的所有像素做线性变换，但该变换应对每个通道独立进行。我们的线性变换表示为一个形状为 `(height * width, height * width)` 的矩阵 $B$。
>
> ```python
> channels_last = torch.randn(64, 32, 32, 3)  # (batch, height, width, channel)
> B = torch.randn(32*32, 32*32)
>
> ## 调整图像张量形状以便跨所有像素混合
> channels_last_flat = channels_last.view(
>     -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
> )
> channels_first_flat = channels_last_flat.transpose(1, 2)
> channels_first_flat_transformed = channels_first_flat @ B.T
> channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
> channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)
> ```
>
> 而使用 einops：
>
> ```python
> height = width = 32
> ## rearrange 替代笨拙的 torch view + transpose
> channels_first = rearrange(
>     channels_last,
>     "batch height width channel -> batch channel (height width)"
> )
> channels_first_transformed = einsum(
>     channels_first, B,
>     "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
> )
> channels_last_transformed = rearrange(
>     channels_first_transformed,
>     "batch channel (height width) -> batch height width channel",
>     height=height, width=width
> )
> ```
>
> 或者，如果你想疯狂一点：用 `einx.dot`（等价于 einops.einsum 的 einx 版本）一步搞定
>
> ```python
> height = width = 32
> channels_last_transformed = einx.dot(
>     "batch row_in col_in channel, (row_out col_out) (row_in col_in)"
>     "-> batch row_out col_out channel",
>     channels_last, B,
>     col_in=width, col_out=width
> )
> ```
>
> 这里的第一种实现可以通过在前后加上注释来说明输入和输出形状，但这很笨拙且容易出 bug。使用 einsum 记法，文档就是实现本身！

Einsum 记法可以处理任意输入批处理维度，还有一个关键优点是自文档化。在使用 einsum 记法的代码中，输入和输出张量的相关形状要清晰得多。对于其余的张量，你可以考虑使用张量类型注解，例如使用 jaxtyping 库（并非 JAX 专用）。

我们将在作业 2 中更多地讨论使用 einsum 记法的性能影响，但现在只需知道：它几乎总是比替代方案更好！

#### 3.2.1 数学记法与内存排布

许多机器学习论文在记法中使用行向量，这使得表示方式与 NumPy 和 PyTorch 默认使用的行主序（row-major）内存排布很好地契合。使用行向量时，线性变换形如

$$y = xW^\top, \quad (1)$$

其中行主序的 $W \in \mathbb{R}^{d_{out} \times d_{in}}$，行向量 $x \in \mathbb{R}^{1 \times d_{in}}$。注意这使我们能够通过增大 $x$ 的最外层维度来对输入进行批处理，也就是说可以把向量输入 $x$ 替换为矩阵输入 $X \in \mathbb{R}^{batch \times d_{in}}$。

在线性代数中，更常见的做法是使用列向量，此时线性变换形如

$$y = Wx, \quad (2)$$

给定行主序的 $W \in \mathbb{R}^{d_{out} \times d_{in}}$ 和列向量 $x \in \mathbb{R}^{d_{in}}$。在这种设定下对输入做批处理时，$x$ 的批维度必须放在最后，所以 $x$ 需要被替换为矩阵 $\tilde{X} \in \mathbb{R}^{d_{in} \times batch}$。

在本作业中，我们将主要使用列向量进行数学记法，因为数学上通常遵循这种记法。请记住，如果你想使用朴素的矩阵乘法记法，就必须像式 (1) 的行向量约定中那样对矩阵应用一次转置，因为 PyTorch 使用行主序内存排布。如果你用 einsum 做线性代数运算，只要正确标记坐标轴，这就不是问题。顺便一提，值得注意的是 Matlab、Julia 和 Fortran 等其他语言/线性代数包都使用列主序内存排布（即批处理维度放在最后），但 Python 及其相关包采用的是 C 语言标准的行主序排布。

### 3.3 基本构建模块：Linear 与 Embedding 模块

#### 3.3.1 参数初始化

有效地训练神经网络通常需要仔细地初始化模型参数——糟糕的初始化可能导致梯度消失或梯度爆炸等不良行为。Pre-norm Transformer 对初始化异常地稳健，但初始化仍然会对训练速度和收敛性产生显著影响。由于本作业已经很长，我们把细节留到作业 3，这里直接给出一些近似初始化，它们在大多数情况下效果良好。暂时请使用：

- Linear 权重：$\mathcal{N}(\mu = 0, \sigma^2 = \frac{2}{d_{in} + d_{out}})$，截断在 $[-3\sigma, 3\sigma]$。
- Embedding：$\mathcal{N}(\mu = 0, \sigma^2 = 1)$，截断在 $[-3, 3]$。
- RMSNorm：全 1。

你应该使用 `torch.nn.init.trunc_normal_` 来初始化截断正态权重。

#### 3.3.2 Linear 模块

线性层是 Transformer 以及一般神经网络的基本构建模块。首先，你将实现自己的 `Linear` 类，它继承自 `torch.nn.Module` 并执行线性变换：

$$y = Wx. \quad (3)$$

注意，按照大多数现代 LLM 的做法，我们不包含偏置（bias）项。

> **问题（linear）：实现 linear 模块（1 分）**
>
> **提交要求：** 实现一个继承自 `torch.nn.Module` 的 `Linear` 类，执行线性变换。你的实现应遵循 PyTorch 内置 `nn.Linear` 模块的接口，但不带 bias 参数。我们推荐以下接口：
>
> - `def __init__(self, in_features, out_features, device=None, dtype=None)`：构造线性变换模块。该函数应接受以下参数：
>   - `in_features: int`：输入的最后一维维度
>   - `out_features: int`：输出的最后一维维度
>   - `device: torch.device | None = None`：存储参数的设备
>   - `dtype: torch.dtype | None = None`：参数的数据类型
> - `def forward(self, x: torch.Tensor) -> torch.Tensor`：对输入应用线性变换。
>
> 请确保：
> - 继承 `nn.Module`
> - 调用父类构造函数
> - 构造并以 𝑊（而不是 𝑊⊤）的形式把你的参数存储在 `nn.Parameter` 中
> - 当然，不要使用 `nn.Linear` 或 `nn.functional.linear`
>
> 初始化时，使用上面的设置，并用 `torch.nn.init.trunc_normal_` 初始化权重。
>
> 若要测试你的 `Linear` 模块，请在 [adapters.run_linear] 处实现测试适配器。适配器应把给定的权重加载到你的 `Linear` 模块中。你可以为此使用 `Module.load_state_dict`。然后运行 `uv run pytest -k test_linear`。

#### 3.3.3 Embedding 模块

如上所述，Transformer 的第一层是嵌入层，它把整数词元 ID 映射到 `d_model` 维的向量空间。我们将实现一个继承自 `torch.nn.Module` 的自定义 `Embedding` 类（因此你不应使用 `nn.Embedding`）。`forward` 方法应通过对一个形状为 `(vocab_size, d_model)` 的嵌入矩阵进行索引，为每个词元 ID 选取对应的嵌入向量；索引用的是一个形状为 `(batch_size, sequence_length)` 的词元 ID `torch.LongTensor`。

> **问题（embedding）：实现 embedding 模块（1 分）**
>
> **提交要求：** 实现继承自 `torch.nn.Module` 的 `Embedding` 类，执行嵌入查找。你的实现应遵循 PyTorch 内置 `nn.Embedding` 模块的接口。我们推荐以下接口：
>
> - `def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)`：构造嵌入模块。该函数应接受以下参数：
>   - `num_embeddings: int`：词表大小
>   - `embedding_dim: int`：嵌入向量的维度，即 $d_{model}$
>   - `device: torch.device | None = None`：存储参数的设备
>   - `dtype: torch.dtype | None = None`：参数的数据类型
> - `def forward(self, token_ids: torch.Tensor) -> torch.Tensor`：查找给定词元 ID 对应的嵌入向量。
>
> 请确保：
> - 继承 `nn.Module`
> - 调用父类构造函数
> - 把你的嵌入矩阵初始化为 `nn.Parameter`
> - 存储嵌入矩阵时 `d_model` 作为最后一维
> - 当然，不要使用 `nn.Embedding` 或 `nn.functional.embedding`
>
> 同样，初始化时使用上面的设置，并用 `torch.nn.init.trunc_normal_` 初始化权重。
>
> 若要测试你的实现，请在 [adapters.run_embedding] 处实现测试适配器。然后运行 `uv run pytest -k test_embedding`。

### 3.4 Pre-norm Transformer 块

每个 Transformer 块有两个子层：一个多头自注意力机制和一个逐位置前馈网络（[A. Vaswani et al., 2017] 第 3.1 节）。

在最初的 Transformer 论文中，模型在两个子层周围各使用一个残差连接，后接层归一化。这种架构通常被称为 "post-norm" Transformer，因为层归一化应用在子层输出上。然而，许多工作发现，把层归一化从每个子层的输出移到每个子层的输入（并在最后一个 Transformer 块之后额外加一个层归一化）可以提高 Transformer 训练的稳定性 [T. Q. Nguyen et al., 2019; R. Xiong et al., 2020]——图 2 直观展示了这种 "pre-norm" Transformer 块。然后，每个 Transformer 块子层的输出通过残差连接加到子层输入上（A. Vaswani et al. [8] 第 5.4 节）。Pre-norm 的一种直觉是：从输入嵌入到 Transformer 最终输出之间存在一条没有任何归一化的干净"残差流（residual stream）"，据称这可以改善梯度流动。这种 pre-norm Transformer 如今是语言模型中使用的标准做法（如 GPT-3、LLaMA、PaLM 等），因此我们将实现这个变体。我们将逐一讲解 pre-norm Transformer 块的每个组件，并按顺序实现它们。

#### 3.4.1 均方根层归一化（RMSNorm）

A. Vaswani et al. [8] 的原始 Transformer 实现使用层归一化 [J. L. Ba et al., 2016] 来归一化激活值。遵循 H. Touvron et al. [12]，我们将使用均方根层归一化（RMSNorm；B. Zhang et al. [13] 公式 4）作为层归一化。给定激活向量 $a \in \mathbb{R}^{d_{model}}$，RMSNorm 将每个激活 $a_i$ 重新缩放如下：

$$\mathrm{RMSNorm}(a_i) = \frac{a_i}{\mathrm{RMS}(a)} g_i, \quad (4)$$

其中 $\mathrm{RMS}(a) = \sqrt{\frac{1}{d_{model}} \sum_{i=1}^{d_{model}} a_i^2 + \varepsilon}$。这里 $g_i$ 是可学习的"增益"参数（总共 $d_{model}$ 个这样的参数），$\varepsilon$ 是一个通常固定为 1e-5 的超参数。

你应把输入上转为 `torch.float32`，以防止对输入平方时发生溢出。总体而言，你的 `forward` 方法应形如：

```python
  in_dtype = x.dtype
  x = x.to(torch.float32)
  # 在这里执行 RMSNorm 的代码
  ...
  result = ...
  # 以原始 dtype 返回结果
  return result.to(in_dtype)
```

> **问题（rmsnorm）：均方根层归一化（1 分）**
>
> **提交要求：** 将 RMSNorm 实现为一个 `torch.nn.Module`。我们推荐以下接口：
>
> - `def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)`：构造 RMSNorm 模块。该函数应接受以下参数：
>   - `d_model: int`：模型隐藏维度
>   - `eps: float = 1e-5`：用于数值稳定性的 epsilon 值
>   - `device: torch.device | None = None`：存储参数的设备
>   - `dtype: torch.dtype | None = None`：参数的数据类型
> - `def forward(self, x: torch.Tensor) -> torch.Tensor`：处理形状为 `(batch_size, sequence_length, d_model)` 的输入张量，返回同样形状的张量。
>
> 注意：如上文所述，记住在执行归一化之前把输入上转为 `torch.float32`（之后再下转回原始 dtype）。
>
> 若要测试你的实现，请在 [adapters.run_rmsnorm] 处实现测试适配器。然后运行 `uv run pytest -k test_rmsnorm`。

#### 3.4.2 逐位置前馈网络

**图 3：SiLU（又称 Swish）与 ReLU 激活函数的比较。**（SiLU: $f(x) = x\,\sigma(x)$；恒等（Identity）: $f(x) = x$；ReLU: $f(x) = \max(0, x)$）

在最初的 Transformer 论文（A. Vaswani et al. [8] 第 3.3 节）中，Transformer 的前馈网络由两个线性变换及其间的 ReLU 激活（$\mathrm{ReLU}(x) = \max(0, x)$）组成。在最初的架构中，前馈内层的维度通常是输入维度的 4 倍。

然而，与现代设计相比，现代语言模型通常有两个主要变化：使用另一种激活函数，并采用门控（gating）机制。具体而言，我们将实现 Llama 3 [A. Grattafiori et al., 2024] 和 Qwen 2.5 [A. Yang et al., 2024] 等 LLM 采用的 "SwiGLU" 激活函数，它把 SiLU（常称 Swish）激活与一种称为门控线性单元（GLU）的门控机制结合起来。我们还将省略线性层中有时使用的偏置项，这是自 PaLM [A. Chowdhery et al., 2022] 和 LLaMA [H. Touvron et al., 2023] 以来大多数现代 LLM 的做法。

SiLU 或 Swish 激活函数 [D. Hendrycks et al., 2016; S. Elfwing et al., 2017] 定义如下：

$$\mathrm{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}. \quad (5)$$

从图 3 可以看出，SiLU 激活函数与 ReLU 激活函数类似，但在零点附近是光滑的。

门控线性单元（GLU）最初由 Y. N. Dauphin et al. [19] 定义为：一个经过 sigmoid 函数的线性变换与另一个线性变换的逐元素乘积：

$$\mathrm{GLU}(x, W_1, W_2) = \sigma(W_1 x) \odot W_2 x, \quad (6)$$

其中 $\odot$ 表示逐元素相乘。门控线性单元被认为可以"在保留非线性能力的同时为梯度提供一条线性通路，从而缓解深层架构的梯度消失问题"。

把 SiLU/Swish 与 GLU 结合起来，我们就得到了 SwiGLU，我们将用它作为前馈网络：

$$\mathrm{FFN}(x) = \mathrm{SwiGLU}(x, W_1, W_2, W_3) = W_2\,(\mathrm{SiLU}(W_1 x) \odot W_3 x), \quad (7)$$

其中 $x \in \mathbb{R}^{d_{model}}$，$W_1, W_3 \in \mathbb{R}^{d_{ff} \times d_{model}}$，$W_2 \in \mathbb{R}^{d_{model} \times d_{ff}}$，并且按惯例 $d_{ff} = \frac{8}{3} d_{model}$。对于具体实现，为了硬件效率把它取整到邻近的 64 的倍数也是可以的。

N. Shazeer [20] 首次提出把 SiLU/Swish 激活与 GLU 结合，其实验表明 SwiGLU 在语言建模任务上优于 ReLU、SiLU（无门控）等基线。在作业稍后部分，你将比较 SwiGLU 和 SiLU。虽然我们提到了这些组件的一些启发式论据（相关论文提供了更多支持证据），但保持实证视角是有益的：Shazeer 论文中一句如今名言的话是

> "对于这些架构为何有效，我们不提供任何解释；我们把它们的成功，一如其他一切，归因于上天的仁慈。"

> **问题（positionwise_feedforward）：实现逐位置前馈网络（2 分）**
>
> **提交要求：** 实现由 SiLU 激活函数和 GLU 组成的 SwiGLU 前馈网络。
>
> 注意：在这个特定情况下，你可以放心地在实现中使用 `torch.sigmoid` 以获得数值稳定性。
>
> 在你的实现中，应将 $d_{ff}$ 设为约 $\frac{8}{3} \times d_{model}$，同时确保前馈内层维度是 64 的倍数，以充分利用你的硬件。若要对照我们提供的测试来测试你的实现，你需要在 [adapters.run_swiglu] 处实现测试适配器。然后运行 `uv run pytest -k test_swiglu` 来测试你的实现。

#### 3.4.3 相对位置嵌入

为了向模型注入位置信息，我们将实现旋转位置嵌入（Rotary Position Embeddings）[J. Su et al., 2021]，通常称为 RoPE。对于位于词元位置 $i$ 的查询词元 $q^{(i)} = W_q x^{(i)} \in \mathbb{R}^d$，我们应用一个成对旋转矩阵 $R_i$，得到 $q'^{(i)} = R_i q^{(i)} = R_i W_q x^{(i)}$。这里 $R_i$ 将嵌入元素对 $q^{(i)}_{2k-1:2k}$ 作为二维向量旋转角度 $\theta_{i,k} = i\,\Theta^{(2k-2)/d}$（其中 $k \in \{1, \dots, d/2\}$，$\Theta$ 为某个常数）。因此，我们可以把 $R_i$ 看作一个 $d \times d$ 的块对角矩阵，其对角块为 $R_i^k$（$k \in \{1, \dots, \frac{d}{2}\}$），其中

$$R_i^k = \begin{pmatrix} \cos(\theta_{i,k}) & \sin(\theta_{i,k}) \\ -\sin(\theta_{i,k}) & \cos(\theta_{i,k}) \end{pmatrix}. \quad (8)$$

于是我们得到完整的旋转矩阵

$$R_i = \begin{pmatrix} R_i^1 & 0 & 0 & \cdots & 0 \\ 0 & R_i^2 & 0 & \cdots & 0 \\ 0 & 0 & R_i^3 & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & R_i^{d/2} \end{pmatrix}, \quad (9)$$

其中 0 表示 $2 \times 2$ 的零矩阵。虽然你可以构造完整的 $d \times d$ 矩阵，但一个好的解答应利用这个矩阵的性质来更高效地实现该变换。由于我们只关心序列内词元之间的相对旋转，我们可以在各层之间、不同批次之间复用为 $\cos(\theta_{i,k})$ 和 $\sin(\theta_{i,k})$ 计算的值。如果你想优化它，可以使用一个被所有层共享的单一 RoPE 模块，它可以在 `init` 时用 `self.register_buffer(persistent=False)` 创建一个预先计算好的二维 sin 和 cos 值缓冲区，而不是用 `nn.Parameter`（因为我们不想学习这些固定的余弦和正弦值）。接着对 $k^{(j)}$ 执行与 $q^{(i)}$ 完全相同的旋转过程，按对应的 $R_j$ 旋转。注意，这一层没有任何可学习参数。

> **问题（rope）：实现 RoPE（2 分）**
>
> **提交要求：** 实现一个 `RotaryPositionalEmbedding` 类，对输入张量应用 RoPE。
>
> 推荐以下接口：
>
> - `def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)`：构造 RoPE 模块，并在需要时创建缓冲区。
>   - `theta: float`：RoPE 的 $\Theta$ 值
>   - `d_k: int`：查询和键向量的维度
>   - `max_seq_len: int`：将输入的最大序列长度
>   - `device: torch.device | None = None`：存储缓冲区的设备
> - `def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor`：处理形状为 `(..., seq_len, d_k)` 的输入张量，返回同样形状的张量。注意你应容忍 $x$ 带有任意数量的批处理维度。你应该假设词元位置是一个形状为 `(..., seq_len)` 的张量，它指定 $x$ 在序列维度上的词元位置。
>
> 你应该用词元位置在序列维度上切片你的（可能预计算好的）cos 和 sin 张量。
>
> 若要测试你的实现，请完成 [adapters.run_rope]，并确保它通过 `uv run pytest -k test_rope`。

#### 3.4.4 缩放点积注意力

现在我们将实现 A. Vaswani et al. [8]（第 3.2.1 节）中描述的缩放点积注意力。作为预备步骤，注意力（Attention）操作的定义会用到 softmax——一个将未归一化的分数向量转换为归一化分布的操作：

$$\mathrm{softmax}(v)_i = \frac{\exp(v_i)}{\sum_{j=1}^{n} \exp(v_j)}. \quad (10)$$

注意，当数值很大时 $\exp(v_i)$ 可能变为 inf（于是 inf/inf = NaN）。我们可以通过注意到 softmax 操作对给所有输入加上任意常数 $c$ 具有不变性来避免这一点。我们可以利用这一性质来保证数值稳定性——通常，我们从 $v$ 的所有元素中减去 $v$ 的最大元素，使新的最大元素为 0。现在你将实现 softmax，并使用这个数值稳定性技巧。

> **问题（softmax）：实现 softmax（1 分）**
>
> **提交要求：** 编写一个对张量应用 softmax 操作的函数。你的函数应接受两个参数：一个张量和一个维度 $i$，并对输入张量的第 $i$ 维应用 softmax。输出张量应与输入张量形状相同，但其第 $i$ 维现在是归一化的概率分布。使用从第 $i$ 维的所有元素中减去该维最大值的技巧，以避免数值稳定性问题。
>
> 若要测试你的实现，请完成 [adapters.run_softmax]，并确保它通过 `uv run pytest -k test_softmax_matches_pytorch`。

现在我们可以从数学上定义注意力（Attention）操作：

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V, \quad (11)$$

其中 $Q \in \mathbb{R}^{n \times d_k}$，$K \in \mathbb{R}^{m \times d_k}$，$V \in \mathbb{R}^{m \times d_v}$。这里 $Q$、$K$ 和 $V$ 都是该操作的输入——注意它们不是可学习参数。

**掩码（Masking）：** 有时对注意力操作的输出进行掩码会很方便。掩码应具有形状 $M \in \{\mathrm{True}, \mathrm{False}\}^{n \times m}$，这个布尔矩阵的每一行 $i$ 指示查询 $i$ 应该关注哪些键。按照惯例（有点令人困惑），位置 $(i, j)$ 上值为 True 表示查询 $i$ 关注键 $j$，值为 False 表示查询不关注该键。换言之，"信息流"发生在值为 True 的 $(i, j)$ 对上。例如，考虑一个元素为 `[[True, True, False]]` 的 $1 \times 3$ 掩码矩阵，这个唯一的查询向量只关注前两个键。

在计算上，使用掩码比 subsequences 上计算注意力要高效得多。具体做法是：取 softmax 之前的值（$QK^\top/\sqrt{d_k}$），并在掩码矩阵中值为 False 的任意条目上加上 $-\infty$。

> **问题（scaled_dot_product_attention）：实现缩放点积注意力（5 分）**
>
> **提交要求：** 实现缩放点积注意力函数。你的实现应处理形状为 `(batch_size, ..., seq_len, d_k)` 的键和查询，以及形状为 `(batch_size, ..., seq_len, d_v)` 的值，其中 `...` 表示任意数量的其他类批处理维度（如果有的话）。实现应返回形状为 `(batch_size, ..., seq_len, d_v)` 的输出。关于类批处理维度的讨论见第 3.2 节。
>
> 你的实现还应支持一个可选的、用户提供的布尔掩码，形状为 `(seq_len, seq_len)`。掩码值为 True 的位置，其注意力概率之和应为 1；掩码值为 False 的位置，其注意力概率应为零。
>
> 若要对照我们提供的测试来测试你的实现，你需要在 [adapters.run_scaled_dot_product_attention] 处实现测试适配器。`uv run pytest -k test_scaled_dot_product_attention` 会在三阶输入张量上测试你的实现，而 `uv run pytest -k test_4d_scaled_dot_product_attention` 会在四阶输入张量上测试你的实现。

#### 3.4.5 因果多头自注意力

我们将实现 A. Vaswani et al. [8] 第 3.2.2 节中描述的多头自注意力。回顾多头注意力的数学定义：

$$\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h), \quad (12)$$

$$\mathrm{head}_i = \mathrm{Attention}(Q_i, K_i, V_i), \quad (13)$$

其中 $Q_i$、$K_i$、$V_i$ 分别是 $Q$、$K$、$V$ 的第 $i \in \{1, \dots, h\}$ 个切片（大小为 $d_k$ 或 $d_v$），而 Attention 是第 3.4.4 节定义的缩放点积注意力操作。由此可以构成多头自注意力操作：

$$\mathrm{MultiHeadSelfAttention}(x) = W^O\,\mathrm{MultiHead}(W^Q x, W^K x, W^V x). \quad (14)$$

这里的可学习参数为 $W^Q \in \mathbb{R}^{h d_k \times d_{model}}$、$W^K \in \mathbb{R}^{h d_k \times d_{model}}$、$W^V \in \mathbb{R}^{h d_v \times d_{model}}$ 和 $W^O \in \mathbb{R}^{d_{model} \times h d_v}$。由于在多头注意力操作中 $Q$、$K$、$V$ 被切分为多个头，我们可以认为 $W^Q$、$W^K$ 和 $W^V$ 在输出维度上是按头分离的。实现正确后，你应该用总共三次矩阵乘法来计算键、值和查询投影。<sup>5</sup>

<sup>5</sup> 作为拓展目标，试着把键、查询和值投影合并到单个权重矩阵中，这样你只需要一次矩阵乘法。

**因果掩码（Causal masking）**

你的实现应阻止模型关注序列中未来的词元。换言之，如果模型拿到词元序列 $t_1, \dots, t_n$，我们想计算前缀 $t_1, \dots, t_i$（其中 $i < n$）的下一词预测，模型不应能访问（关注）位置 $t_{i+1}, \dots, t_n$ 处的词元表示，因为在推理生成文本时它无法访问这些词元（而且这些未来词元会泄露真实下一词身份的信息，使语言建模预训练目标变得平凡）。对于输入词元序列 $t_1, \dots, t_n$，朴素的做法是运行 $n$ 次多头自注意力（对序列的 $n$ 个唯一前缀各跑一次）来阻止对未来词元的访问。相反，我们将使用因果注意力掩码，它允许词元 $i$ 关注序列中所有位置 $j \le i$。你可以使用 `torch.triu` 或广播索引比较来构造这个掩码，并且你应该利用第 3.4.4 节的缩放点积注意力实现已经支持注意力掩码这一事实。

**应用 RoPE**

RoPE 应应用于查询和键向量，但**不**应用于值向量。此外，头的维度应作为批处理维度处理，因为在多头注意力中，注意力在每个头上是独立应用的。这意味着必须对每个头的查询和键向量应用完全相同的 RoPE 旋转。

> **问题（multihead_self_attention）：实现因果多头自注意力（5 分）**
>
> **提交要求：** 将因果多头自注意力实现为一个 `torch.nn.Module`。你的实现应至少接受以下参数：
> - `d_model: int`：Transformer 块输入的维度。
> - `num_heads: int`：多头自注意力中使用的头数。
>
> 遵循 A. Vaswani et al. [8]，令 $d_k = d_v = \frac{d_{model}}{h}$。若要对照我们提供的测试来测试你的实现，请在 [adapters.run_multihead_self_attention] 处实现测试适配器。然后运行 `uv run pytest -k test_multihead_self_attention` 来测试你的实现。

### 3.5 完整的 Transformer LM

我们先从组装 Transformer 块开始（回头参考图 2 会有帮助）。一个 Transformer 块包含两个"子层"，一个用于多头自注意力，另一个用于 SwiGLU 前馈网络。在每个子层中，我们先执行 RMSNorm，然后是主操作（MHA/FF），最后加回残差连接。

具体而言，Transformer 块的第一个半区（第一个"子层"）应对输入 $x$ 实现以下更新以产生输出 $y$：

$$y = x + \mathrm{MultiHeadSelfAttention}(\mathrm{RMSNorm}(x)). \quad (15)$$

> **问题（transformer_block）：实现 Transformer 块（3 分）**
>
> 实现第 3.4 节描述、图 2 所示的 pre-norm Transformer 块。你的 Transformer 块应至少接受以下参数：
> - `d_model: int`：Transformer 块输入的维度。
> - `num_heads: int`：多头自注意力中使用的头数。
> - `d_ff: int`：逐位置前馈内层的维度。
>
> 若要测试你的实现，请实现适配器 [adapters.run_transformer_block]。然后运行 `uv run pytest -k test_transformer_block` 来测试你的实现。
>
> **提交要求：** 通过所提供测试的 Transformer 块代码。

现在我们把各块组装起来，遵循图 1 的高层示意图。按照我们在第 3.1.0.1 节中对嵌入的描述，将其输入 `num_layers` 个 Transformer 块，然后把结果传入最终的层归一化和 LM head，得到对词表的未归一化分布（logits）。

> **问题（transformer_lm）：实现 Transformer LM（3 分）**
>
> 是时候把所有东西拼起来了！实现第 3.1 节描述、图 1 所示的 Transformer 语言模型。你的实现至少应接受 Transformer 块的上述所有构造参数，以及以下额外参数：
> - `vocab_size: int`：词表大小，用于确定词元嵌入矩阵的维度。
> - `context_length: int`：最大上下文长度，用于确定 RoPE sin 和 cos 缓冲区的维度。
> - `num_layers: int`：使用的 Transformer 块数量。
>
> 若要对照我们提供的测试来测试你的实现，你需要首先在 [adapters.run_transformer_lm] 处实现测试适配器。然后运行 `uv run pytest -k test_transformer_lm` 来测试你的实现。
>
> **提交要求：** 通过上述测试的 Transformer LM 模块。

**资源核算（Resource accounting）**

能够理解 Transformer 的各个部分如何消耗计算和内存是很有用的。我们将介绍进行一些基本"FLOPs 核算"的步骤。Transformer 中绝大部分 FLOPS 都是矩阵乘法，因此我们核心方法很简单：

1. 写下一个 Transformer 前向传播中的所有矩阵乘法。
2. 把每个矩阵乘法换算成所需的 FLOPs。

对于第二步，以下事实会很有用：

**规则：** 给定 $A \in \mathbb{R}^{m \times n}$ 和 $B \in \mathbb{R}^{n \times p}$，矩阵-矩阵乘积 $AB$ 需要 $2mnp$ FLOPs。

要理解这一点，注意 $(AB)[i,j] = A[i,:] \cdot B[:,j]$，而这个点积需要 $n$ 次加法和 $n$ 次乘法（共 $2n$ FLOPs）。由于矩阵-矩阵乘积 $AB$ 有 $m \times p$ 个元素，FLOPS 总数为 $(2n)(mp) = 2mnp$。

在做下一题之前，先逐个过一遍你的 Transformer 块和 Transformer LM 的每个组件，列出所有矩阵乘法及其相关的 FLOPs 成本，可能会很有帮助。

> **问题（transformer_accounting）：Transformer LM 资源核算（5 分）**
>
> (a) 考虑一个使用我们作业架构、GPT-2 XL 规模的模型，其配置如下：
> - `vocab_size:` 50,257
> - `context_length:` 1,024
> - `num_layers:` 48
> - `d_model:` 1,600
> - `num_heads:` 25
> - `d_ff:` 4,288（最接近 $\frac{8}{3} \times 1{,}600$ 的 64 的倍数）
>
> 假设我们用这个配置来构建模型。我们的模型会有多少可训练参数？假设每个参数都用单精度浮点数表示，仅加载这个模型需要多少内存？
> **提交要求：** 一到两句话的回答。
>
> (b) 找出一个 GPT-2 XL 规模模型完成一次前向传播所需的矩阵乘法。这些矩阵乘法总共需要多少 FLOPs？假设我们的输入序列有 `context_length` 个词元。
> **提交要求：** 矩阵乘法列表（附说明），以及所需的 FLOPs 总数。
>
> (c) 基于你的上述分析，模型的哪些部分需要的 FLOPs 最多？
> **提交要求：** 一到两句话的回答。
>
> (d) 对 GPT-2 small（12 层，768 d_model，12 头）、GPT-2 medium（24 层，1024 d_model，16 头）和 GPT-2 large（36 层，1280 d_model，20 头）重复你的分析。随着模型规模增大，Transformer LM 的哪些部分在总 FLOPs 中占的比例变大或变小？
> **提交要求：** 对每个模型，给出各模型组件的分解及其相关 FLOPs（占前向传播总 FLOPs 的比例）。此外，用一到两句话描述模型规模的变化如何改变每个组件的 FLOPs 占比。
>
> (e) 取 GPT-2 XL 并把上下文长度增大到 16,384。一次前向传播的总 FLOPs 如何变化？模型各组件的 FLOPs 相对贡献如何变化？
> **提交要求：** 一到两句话的回答。

---

## 4 训练 Transformer LM

现在我们已经有了预处理数据（通过分词器）和模型（Transformer）的步骤。剩下的就是构建支持训练的所有代码，包括：

- **损失（Loss）：** 我们需要定义损失函数（交叉熵）。
- **优化器（Optimizer）：** 我们需要定义最小化该损失的优化器（AdamW）。
- **训练循环：** 我们需要所有支持性基础设施，包括加载数据、保存检查点和管理训练。

### 4.1 交叉熵损失

回顾 Transformer 语言模型为每个长度 $m+1$ 的序列 $x$ 和 $i = 1, \dots, m$ 定义了一个分布 $p_\theta(x_{i+1} \mid x_{1:i})$。给定由长度 $m+1$ 的序列组成的训练集 $D$，我们定义标准的交叉熵（负对数似然）损失函数：

$$\ell(\theta; D) = \frac{1}{|D|} \sum_{x \in D} \sum_{i=1}^{m} -\log p_\theta(x_{i+1} \mid x_{1:i}). \quad (16)$$

（注意，Transformer 的一次前向传播就能得到所有 $i = 1, \dots, m$ 的 $p_\theta(x_{i+1} \mid x_{1:i})$。）

特别地，Transformer 对每个位置 $i$ 计算 logits $o_i \in \mathbb{R}^{\text{vocab\_size}}$，于是：<sup>6</sup>

$$p(x_{i+1} \mid x_{1:i}) = \mathrm{softmax}(o_i)[x_{i+1}] = \frac{\exp(o_i[x_{i+1}])}{\sum_{a=1}^{\text{vocab\_size}} \exp(o_i[a])}. \quad (17)$$

交叉熵损失通常是相对于 logits 向量 $o_i \in \mathbb{R}^{\text{vocab\_size}}$ 和目标 $x_{i+1}$ 来定义的。<sup>7</sup>

<sup>6</sup> 注意 $o_i[k]$ 指的是向量 $o_i$ 中索引 $k$ 处的值。

<sup>7</sup> 这对应于 $x_{i+1}$ 上的 Dirac delta 分布与预测的 $\mathrm{softmax}(o_i)$ 分布之间的交叉熵。

与 softmax 的情况一样，实现交叉熵损失时也需要留意数值问题。

> **问题（cross_entropy）：实现交叉熵（1 分）**
>
> **提交要求：** 编写一个计算交叉熵损失的函数，它接收预测的 logits（$o_i$）和目标（$x_{i+1}$），计算交叉熵 $\ell_i = -\mathrm{logsoftmax}(o_i)[x_{i+1}]$。你的函数应处理以下事项：
> - 减去最大元素以保证数值稳定性。
> - 尽可能约去 log 和 exp。
> - 处理任意额外的批处理维度，并返回批上的平均值。与第 3.2 节一样，我们假设类批处理维度总是排在词表维度之前。
>
> 实现 [adapters.run_cross_entropy]，然后运行 `uv run pytest -k test_cross_entropy` 来测试你的实现。

**困惑度（Perplexity）**

交叉熵对训练来说足够了，但评估模型时，我们还希望报告困惑度。对于长度为 $m$、交叉熵损失为 $\ell_1, \dots, \ell_m$ 的序列：

$$\text{perplexity} = \exp\!\left(\frac{1}{m} \sum_{i=1}^{m} \ell_i\right). \quad (18)$$

### 4.2 SGD 优化器

现在我们有了损失函数，开始探索优化器。最简单的基于梯度的优化器是随机梯度下降（SGD）。我们从随机初始化的参数 $\theta_0$ 开始，然后对每个步数 $t = 0, \dots, T-1$，执行以下更新：

$$\theta_{t+1} \leftarrow \theta_t - \alpha_t \nabla L(\theta_t; B_t), \quad (19)$$

其中 $B_t$ 是从数据集 $D$ 中采样的随机批次，学习率 $\alpha_t$ 和批量大小 $|B_t$| 是超参数。

#### 4.2.1 在 PyTorch 中实现 SGD

为了实现我们的优化器，我们将继承 PyTorch 的 `torch.optim.Optimizer` 类。一个 `Optimizer` 子类必须实现两个方法：

`def __init__(self, params, ...)` 应初始化你的优化器。这里，`params` 是要优化的参数的集合（或者是参数组，以防用户想对模型的不同部分使用不同的超参数，如学习率）。务必把 `params` 传给基类的 `__init__` 方法，基类会存储这些参数供 `step` 使用。你可以根据优化器的需要接受额外的参数（例如学习率就是一个常见参数），并把它们以字典形式传给基类构造函数，字典的键是你为这些参数选择的名称（字符串）。

`def step(self)` 应对参数做一次更新。在训练循环中，它会在反向传播之后被调用，因此你可以访问上一批数据的梯度。该方法应遍历每个参数张量 `p` 并原地修改它们，即根据梯度 `p.grad`（保存损失对该参数的梯度的张量，如果存在的话）设置与该参数相关联的张量 `p.data`。

PyTorch 优化器 API 有一些微妙之处，用一个例子来解释更容易。为了让例子更丰富，我们将实现 SGD 的一个稍加变体的版本：学习率随训练衰减，从初始学习率 $\alpha$ 开始，随时间步逐步减小：

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{t+1}} \nabla L(\theta_t; B_t). \quad (20)$$

看看这个版本的 SGD 如何作为 PyTorch `Optimizer` 实现：

```python
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # 获取学习率。
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # 获取与 p 相关联的状态。
                t = state.get("t", 0)  # 从状态中获取迭代次数，没有则为 0。
                grad = p.grad.data  # 获取损失关于 p 的梯度。
                p.data -= lr / math.sqrt(t + 1) * grad  # 原地更新权重张量。
                state["t"] = t + 1  # 迭代次数加 1。
        return loss
```

在 `__init__` 中，我们把参数和默认超参数传给基类构造函数（参数可能以组的形式传入，每组有不同的超参数）。如果参数只是单个 `torch.nn.Parameter` 对象的集合，基类构造函数会创建一个组并赋予其默认超参数。然后，在 `step` 中，我们遍历每个参数组，再遍历组中的每个参数，并应用式 (20)。这里，我们把迭代次数作为与每个参数相关联的状态来保存：先读取该值，在梯度更新中使用它，然后再更新它。API 规定用户可能会传入一个可调用对象 `closure`，以便在优化器步骤之前重新计算损失。我们要用的优化器不需要这个，但为了符合 API 我们还是加上了。

要看到这个例子的运行效果，可以使用下面这个最小化的训练循环示例：

```python
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)
for t in range(100):
    opt.zero_grad()  # 重置所有可学习参数的梯度。
    loss = (weights**2).mean()  # 计算标量损失值。
    print(loss.cpu().item())
    loss.backward()  # 运行反向传播，计算梯度。
    opt.step()  # 运行优化器步骤。
```

这就是训练循环的典型结构：在每次迭代中，我们计算损失并运行一步优化器。训练语言模型时，我们的可学习参数来自模型（在 PyTorch 中，`m.parameters()` 给出这个集合）。损失将在一个采样批数据上计算，但训练循环的基本结构相同。

> **问题（learning_rate_tuning）：调节学习率（1 分）**
>
> 正如我们将看到的，对训练影响最大的超参数之一就是学习率。让我们在这个玩具示例中实践一下。用另外三个学习率值运行上面的 SGD 示例：1e1、1e2 和 1e3，只训练 10 次迭代。每种学习率下损失会发生什么变化？它衰减得更快、更慢，还是发散（即在训练过程中增大）？
> **提交要求：** 一到两句话的回答，说明你观察到的行为。

### 4.3 AdamW

现代语言模型通常用更复杂的优化器来训练，而不是 SGD。最近使用的大多数优化器都是 Adam 优化器 [D. P. Kingma et al., 2015] 的衍生版本。我们将使用 AdamW [I. Loshchilov et al., 2019]，它在最近的工作中被广泛使用。AdamW 对 Adam 提出了一种改进：通过（在每次迭代时把参数往 0 拉）添加权重衰减（weight decay）来改进正则化，并且以与梯度更新解耦（decoupled）的方式实现。我们将按照 I. Loshchilov et al. [23] 的算法 2 来实现 AdamW。

AdamW 是有状态的：它为每个参数维护一阶矩和二阶矩的滑动估计。因此，AdamW 以额外的内存换取了更好的稳定性和收敛性。除了学习率 $\alpha$ 之外，AdamW 还有一对控制矩估计更新的超参数 $(\beta_1, \beta_2)$，以及权重衰减率 $\lambda$。典型应用将 $(\beta_1, \beta_2)$ 设为 $(0.9, 0.999)$，但 LLaMA [H. Touvron et al., 2023]、GPT-3 [T. B. Brown et al., 2020] 等大型语言模型通常用 $(0.9, 0.95)$ 训练。该算法可写为如下形式，其中 $\varepsilon$ 是一个小值（例如 $10^{-8}$），用于在 $v$ 中出现极小时提高数值稳定性：

**算法 1：AdamW 优化器**

```
 1: init(𝜃)                                        ▷ 初始化可学习参数
 2: m ← 0                                          ▷ 一阶矩向量的初始值；形状与 𝜃 相同
 3: v ← 0                                          ▷ 二阶矩向量的初始值；形状与 𝜃 相同
 4: for t = 1,…,T do
 5:     采样一批数据 B_t
 6:     g ← ∇_𝜃 ℓ(𝜃; B_t)                          ▷ 计算损失的梯度
 7:     α_t ← α · √(1−β_2^t) / (1−β_1^t)           ▷ 计算第 t 次迭代的调整后的 α
 8:     𝜃 ← 𝜃 − α·λ·𝜃                              ▷ 应用权重衰减
 9:     m ← β_1·m + (1−β_1)·g                     ▷ 更新一阶矩估计
10:     v ← β_2·v + (1−β_2)·g²                    ▷ 更新二阶矩估计
11:     𝜃 ← 𝜃 − α_t · m / (√v + ε)               ▷ 应用矩调整后的权重更新
12: end for
```

注意 $t$ 从 1 开始。现在你将实现这个优化器。

> **问题（adamw）：实现 AdamW（2 分）**
>
> **提交要求：** 将 AdamW 优化器实现为 `torch.optim.Optimizer` 的子类。你的类应在 `__init__` 中接收学习率 $\alpha$，以及 $\beta$、$\varepsilon$ 和 $\lambda$ 超参数。为了帮你保存状态，基类 `Optimizer` 提供了一个字典 `self.state`，它把 `nn.Parameter` 对象映射到存储该参数所需信息的字典（对 AdamW 而言就是一阶、二阶矩估计）。实现 [adapters.get_adamw_cls]，并确保它通过 `uv run pytest -k test_adamw`。

> **问题（adamw_accounting）：用 AdamW 训练的资源核算（2 分）**
>
> 我们来计算运行 AdamW 需要多少内存和计算。假设每个张量都使用 float32。
>
> (a) 运行 AdamW 的峰值内存是多少？请按参数、激活值、梯度和优化器状态分别分解你的答案。用 `batch_size` 和模型超参数（`vocab_size`、`context_length`、`num_layers`、`d_model`、`num_heads`）表达你的答案。假设 $d_{ff} = \frac{8}{3} \times d_{model}$。
>
> 为简单起见，计算激活值的内存占用时，只考虑以下组件：
> - Transformer 块
>   - RMSNorm（各次）
>   - 多头自注意力子层：$QKV$ 投影、$QK^\top$ 矩阵乘法、softmax、值的加权和、输出投影。
>   - 逐位置前馈（SwiGLU）：$W_1$、$W_2$、门控分支上的 SiLU、逐元素乘积、$W_3$。
> - 最后的 RMSNorm
> - 输出嵌入
> - logits 上的交叉熵
>
> **提交要求：** 分别给出参数、激活值、梯度和优化器状态的代数表达式，以及总量。
>
> (b) 把你的答案代入一个 GPT-2 XL 规模的模型，得到一个只依赖于 `batch_size` 的表达式。在 80GB 内存内你能使用的最大批量是多少？
> **提交要求：** 形如 $a \cdot \text{batch\_size} + b$（$a$、$b$ 为数值）的表达式，以及最大批量的数值。
>
> (c) 运行一步 AdamW 需要多少 FLOPs？
> **提交要求：** 一个代数表达式，并附简要推导。
>
> (d) 模型 FLOPs 利用率（MFU）定义为观测到的吞吐量（每秒词元数）相对于硬件理论峰值 FLOP 吞吐量的比值 [A. Chowdhery et al., 2022]。NVIDIA H100 GPU 的"float32"（实际上是 TensorFloat-32，现实中是"bfloat19"）运算理论峰值为 495 teraFLOP/s。假设你能达到 50% MFU，在单张 H100 上训练一个 GPT-2 XL、400K 步、批量大小 1024 需要多长时间？遵循 J. Kaplan et al. [25] 和 J. Hoffmann et al. [26]，假设反向传播的 FLOPs 是前向传播的两倍。
> **提交要求：** 训练所需的小时数，并附简要推导。

### 4.4 学习率调度

能让损失下降最快的学习率取值在训练过程中往往会变化。在训练 Transformer 时，通常使用学习率调度（learning rate schedule）：开始时用较大的学习率，以便在初期进行更快的更新，然后随着模型训练逐渐衰减到较小的值。<sup>8</sup> 在本作业中，我们将实现用于训练 LLaMA [H. Touvron et al., 2023] 的余弦退火（cosine annealing）调度。

<sup>8</sup> 有时也会使用学习率回升（重启）以帮助跳出局部最小值的调度，这并不少见。

调度器就是一个函数，它接收当前步数 $t$ 和其他相关参数（如初始和最终学习率），并返回第 $t$ 步梯度更新所用的学习率。最简单的调度是常数函数，它对任何 $t$ 都返回相同的学习率。

余弦退火学习率调度接收：(i) 当前迭代 $t$，(ii) 最大学习率 $\alpha_{max}$，(iii) 最小（最终）学习率 $\alpha_{min}$，(iv) 预热（warm-up）迭代次数 $T_w$，(v) 余弦退火的最终迭代 $T_c$。第 $t$ 次迭代的学习率定义为：

- **（预热）** 若 $t < T_w$，则 $\alpha_t = \frac{t}{T_w} \alpha_{max}$。
- **（余弦退火）** 若 $T_w \le t \le T_c$，则 $\alpha_t = \alpha_{min} + \frac{1}{2}\left(1 + \cos\left(\frac{t - T_w}{T_c - T_w}\pi\right)\right)(\alpha_{max} - \alpha_{min})$。
- **（退火后）** 若 $t > T_c$，则 $\alpha_t = \alpha_{min}$。

> **问题（learning_rate_schedule）：实现带 warm-up 的余弦学习率调度（1 分）**
>
> 编写一个函数，接收 $t$、$\alpha_{max}$、$\alpha_{min}$、$T_w$ 和 $T_c$，并按上述调度器返回学习率 $\alpha_t$。然后实现 [adapters.get_lr_cosine_schedule]，并确保它通过 `uv run pytest -k test_get_lr_cosine_schedule`。

### 4.5 梯度裁剪

在训练中，我们有时会遇到产生很大梯度的训练样本，这可能使训练失稳。为了缓解这一问题，实践中常用的一个技术是梯度裁剪（gradient clipping）。其思想是：在每次反向传播之后、执行优化器步骤之前，对梯度的范数强制施加一个上限。

给定（所有参数的）梯度 $g$，我们计算其 $\ell_2$ 范数 $\|g\|_2$。如果该范数小于最大值 $M$，则 $g$ 保持不变；否则，我们按系数 $\frac{M}{\|g\|_2 + \varepsilon}$ 缩小 $g$（其中加上一个很小的 $\varepsilon$，例如 $10^{-6}$，以保证数值稳定性）。注意，得到的范数将刚好小于 $M$。

> **问题（gradient_clipping）：实现梯度裁剪（1 分）**
>
> 编写一个实现梯度裁剪的函数。你的函数应接收一个参数列表和一个最大 $\ell_2$ 范数。它应原地修改每个参数的梯度。使用 $\varepsilon = 10^{-6}$（PyTorch 默认值）。然后实现适配器 [adapters.run_gradient_clipping]，并确保它通过 `uv run pytest -k test_gradient_clipping`。

---

## 5 训练循环

现在，我们终于要把目前构建的主要组件组合起来：分词后的数据、模型和优化器。

### 5.1 数据加载器

分词后的数据（例如你在 tokenizer_experiments 中准备的）是一个词元序列 $x = (x_1, \dots, x_n)$。即使源数据可能由独立的文档组成（例如不同的网页或源代码文件），一个常见的做法是把它们全部拼接成单个词元序列，并在其间加上分隔符（例如 `<|endoftext|>` 词元）。

数据加载器把它转换为一个批次数据流，每个批次由 $B$ 个长度为 $m$ 的序列组成，并配上对应的同样长度为 $m$ 的下一词元。例如，当 $B=1, m=3$ 时，`([x2, x3, x4], [x3, x4, x5])` 就是一个可能的批次。

以这种方式加载数据简化了训练，原因有几个。首先，任何 $1 \le i \le n-m$ 都给出一个有效的训练序列，因此采样训练序列非常平凡。由于所有训练序列长度相同，无需对输入序列做填充，这提高了硬件利用率（也通过增大批量 $B$）。最后，我们也不需要加载完整数据集就能采样训练数据，因此很容易处理可能放不进内存的大数据集。

> **问题（data_loading）：实现数据加载（2 分）**
>
> **提交要求：** 编写一个函数，接收一个 numpy 数组 $x$（包含词元 ID 的整数数组）、一个 `batch_size`、一个 `context_length` 和一个 PyTorch 设备字符串（例如 `'cpu'` 或 `'cuda:0'`），并返回一对张量：采样得到的输入序列和对应的下一词元目标。两个张量都应具有形状 `(batch_size, context_length)`，包含词元 ID，并且都放置在请求的设备上。若要对照我们提供的测试来测试你的实现，你需要首先在 [adapters.run_get_batch] 处实现测试适配器。然后运行 `uv run pytest -k test_get_batch` 来测试你的实现。

> **低资源提示：在 CPU 或 Apple Silicon 上进行数据加载**
>
> 如果你打算在 CPU 或 Apple Silicon 上训练你的 LM，你需要把数据移动到正确的设备上（同样，之后你的模型也应使用同一个设备）。
>
> 如果你在 CPU 上，可以使用 `'cpu'` 设备字符串；在 Apple Silicon（M* 芯片）上，可以使用 `'mps'` 设备字符串。
>
> 关于 MPS 的更多内容，请参阅以下资源：
> - https://docs.pytorch.org/docs/stable/mps.html
> - https://docs.pytorch.org/docs/stable/notes/mps.html
> - https://developer.apple.com/documentation/metalperformanceshaders

如果数据集太大、无法装入内存怎么办？我们可以使用一个名为 mmap 的 Unix 系统调用，它把磁盘上的文件映射到虚拟内存，并在访问该内存位置时惰性加载文件内容。这样，你就可以"假装"整个数据集都在内存中。NumPy 通过 `np.memmap` 实现这一点（或者如果你最初是用 `np.save` 保存的数组，可以用 `np.load` 的 `mmap_mode='r'` 标志），它会返回一个类 numpy 数组对象，在你访问其中的元素时按需加载。在训练期间从数据集（即 numpy 数组）采样时，请确保以内存映射模式加载数据集（通过 `np.memmap`，或通过 `np.load` 的 `mmap_mode='r'` 标志，取决于你的保存方式）。务必同时指定一个与你所加载数组匹配的 `dtype`。显式验证内存映射数据是否正确（例如，不包含超出预期词表大小的值）可能会有所帮助。

### 5.2 检查点（Checkpointing）

除了加载数据，我们还需要在训练过程中保存模型。运行任务时，我们常常希望能恢复中途停止的训练（例如由于作业超时、机器故障等）。即使一切顺利，我们也可能希望以后能访问中间的模型（例如，事后研究训练动态、从训练不同阶段采样等）。

一个检查点应包含恢复训练所需的全部状态。我们至少希望能恢复模型权重。如果使用有状态的优化器（如 AdamW），我们还需要保存优化器的状态（对 AdamW 而言就是矩估计）。最后，为了恢复学习率调度，我们需要知道停止时的迭代次数。PyTorch 让这一切变得很容易：每个 `nn.Module` 都有 `state_dict()` 方法，返回一个包含所有可学习权重的字典；之后我们可以用它的姊妹方法 `load_state_dict()` 恢复这些权重。`torch.optim.Optimizer` 同理。最后，`torch.save(obj, dest)` 可以把一个对象（例如一个以张量为部分值的字典，也可以是普通的 Python 对象如整数）转储到文件（路径）或类文件对象中，然后可以用 `torch.load(src)` 将其加载回内存。

> **问题（checkpointing）：实现模型检查点（1 分）**
>
> 实现以下两个函数来加载和保存检查点：
>
> `def save_checkpoint(model, optimizer, iteration, out)` 应把来自模型、优化器和迭代次数的所有状态转储到类文件对象 `out` 中。你可以使用模型和优化器的 `state_dict` 方法获取它们的相关状态，并使用 `torch.save(obj, out)` 把 `obj` 转储到 `out` 中（PyTorch 这里既支持路径也支持类文件对象）。一个典型的选择是让 `obj` 为一个字典，但只要之后能加载你的检查点，你可以使用任何格式。
>
> 该函数应接受以下参数：
> - `model: torch.nn.Module`
> - `optimizer: torch.optim.Optimizer`
> - `iteration: int`
> - `out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
>
> `def load_checkpoint(src, model, optimizer)` 应从 `src`（路径或类文件对象）加载检查点，然后从该检查点恢复模型和优化器状态。你的函数应返回到保存到检查点中的迭代次数。你可以使用 `torch.load(src)` 恢复你在 `save_checkpoint` 实现中保存的内容，并使用模型和优化器中的 `load_state_dict` 方法让它们回到之前的状态。
>
> 该函数应接受以下参数：
> - `src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
> - `model: torch.nn.Module`
> - `optimizer: torch.optim.Optimizer`
>
> 实现 [adapters.run_save_checkpoint] 和 [adapters.run_load_checkpoint] 适配器，并确保它们通过 `uv run pytest -k test_checkpointing`。

### 5.3 训练循环

现在，终于到了把你实现的所有组件组装进主训练脚本的时候。值得把"用不同超参数启动训练运行"这件事做得容易些（例如通过命令行参数传入），因为之后你会多次这样做，以研究不同选择对训练的影响。

> **问题（training_together）：组装起来（4 分）**
>
> **提交要求：** 编写一个脚本，在用户提供的输入上运行训练循环来训练你的模型。特别地，我们建议你的训练脚本至少允许以下功能：
> - 能够配置和控制各种模型和优化器超参数。
> - 用 `np.memmap` 省内存地加载大型训练和验证数据集。
> - 把检查点序列化到用户提供的路径。
> - 定期记录训练和验证性能（例如输出到控制台和/或 Weights and Biases 等外部服务）。<sup>9</sup>

<sup>9</sup> wandb.ai

---

## 6 生成文本

现在我们可以训练模型了，最后需要的一项能力是从模型生成文本。回顾：语言模型接收长度为 `sequence_length` 的（可批处理的）整数序列，产生一个大小为 `(sequence_length, vocab_size)` 的矩阵，其中序列的每个元素都是预测该位置之后下一个词元的概率分布。现在我们将编写几个函数，把它变成一个新序列的采样方案。

**Softmax**

按标准约定，语言模型的输出是最终线性层的输出（"logits"），所以我们必须通过 softmax 操作把它变成归一化的概率，即前面式 (10) 见过的那个操作。

**解码（Decoding）**

要从模型生成文本（解码），我们将为模型提供一个前缀词元序列（"prompt"），并让它产生对词表的一个概率分布，用于预测序列中的下一个词元。然后，我们从这个词表项分布中采样，以确定下一个输出词元。

具体而言，解码过程的一步应接收序列 $x_{1\dots t}$，并通过以下方程返回词元 $x_{t+1}$：

$$P(x_{t+1} = i \mid x_{1\dots t}) = \frac{\exp(v_i)}{\sum_j \exp(v_j)}, \quad (21)$$

$$v = \mathrm{TransformerLM}(x_{1\dots t})_t \in \mathbb{R}^{\text{vocab\_size}}, \quad (22)$$

其中 TransformerLM 是我们的模型，它以长度为 `sequence_length` 的序列作为输入，产生一个大小为 `(sequence_length, vocab_size)` 的矩阵，我们取该矩阵的最后一个元素，因为我们想要第 $t$ 个位置的下一词元预测。

这给了我们一个基本的解码器：不断从这些单步条件分布中重复采样（把之前生成的输出词元追加到下一个解码时间步的输入），直到生成序列结束词元 `<|endoftext|>`（或达到用户指定的最大生成词元数）。

**解码技巧**

我们将使用小模型做实验，而小模型有时会生成质量很低的文本。两个简单的解码技巧可以帮助解决这些问题。第一，在**温度缩放（temperature scaling）**中，我们用温度参数 $\tau$ 修改 softmax，新的 softmax 为

$$\mathrm{softmax}(v, \tau)_i = \frac{\exp(v_i/\tau)}{\sum_{j=1}^{\text{vocab\_size}} \exp(v_j/\tau)}. \quad (23)$$

注意 $\tau \to 0$ 时，$v$ 的最大元素将占主导，softmax 的输出变成集中在该最大元素上的独热（one-hot）向量。

第二，另一个技巧是**核采样（nucleus sampling）或称 top-p 采样**，它通过截断低概率词元来修改采样分布。设 $q$ 是我们从（经温度缩放的）softmax 得到的、大小为 vocab_size 的概率分布。以超参数 $p$ 进行核采样时，按以下方程产生下一个词元：

$$P(x_{t+1} = i \mid q) = \begin{cases} \dfrac{q_i}{\sum_{j \in V(p)} q_j} & \text{if } i \in V(p) \\ 0 & \text{otherwise} \end{cases} \quad (24)$$

其中 $V(p)$ 是满足 $\sum_{j \in V(p)} q_j \ge p$ 的最小索引集合。你可以先把概率分布 $q$ 按大小排序，然后选取最大的词表元素直到达到目标水平 $p$，这样就能很容易地计算出这个量。

> **问题（decoding）：解码（3 分）**
>
> **提交要求：** 实现一个从你的语言模型解码的函数。我们建议你支持以下功能：
> - 为用户提供的 prompt 生成续写（即接收某个 $x_{1\dots t}$ 并采样续写，直到遇到 `<|endoftext|>` 词元）。
> - 允许用户控制生成词元的最大数量。
> - 给定期望的温度值，在采样之前对预测的下一词元分布应用 softmax 温度缩放。
> - Top-p 采样（[A. Holtzman et al., 2020]，也称核采样），由用户指定阈值。

---

## 7 实验

现在是时候把所有东西拼起来，在预训练数据集上训练（小型）语言模型了。

### 7.1 如何运行实验与交付要求

理解 Transformer 架构组件背后原理的最佳方式，是亲自动手修改它并运行。实践经验无可替代。

为此，能够快速、一致地实验并记录你所做的事情非常重要。为了快速实验，我们将在一个小规模模型（总共约 1700 万参数）和简单数据集（TinyStories）上运行大量实验。为了一致性，你将以系统化的方式对组件做消融（ablate）并改变超参数；为了留存记录，我们将要求你提交一份实验日志以及每个实验对应的学习曲线。

为了能够提交损失曲线，请确保定期评估验证损失，并同时记录步数和墙上时钟时间（wall-clock time）。你可能会发现 Weights and Biases 之类的日志基础设施很有帮助。

> **问题（experiment_log）：实验日志（3 分）**
>
> 为你的训练和评估代码创建实验跟踪基础设施，使你能够相对于梯度步数和墙上时钟时间跟踪你的实验和损失曲线。
>
> **提交要求：** 实验的日志基础设施代码，以及针对本小节下面各作业问题的实验日志（一份记录你所有尝试的文档）。

### 7.2 TinyStories

我们将从一个非常简单的数据集开始（TinyStories；R. Eldan et al. [1]），模型在它上面训练很快，我们还能观察到一些有趣的行为。获取该数据集的说明见第 1 节。下面是该数据集的一个示例。

> **示例（tinystories_example）：TinyStories 中的一个示例**
>
> Once upon a time there was a little boy named Ben. Ben loved to explore the world around him. He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben was walking through the store when he came across a very special vase. When Ben saw it he was amazed! He said, “Wow, that is a really amazing vase! Can I buy it?” The shopkeeper smiled and said, “Of course you can. You can take it home and show all your friends how amazing it is!” So Ben took the vase home and he was so proud of it! He called his friends over and showed them the amazing vase. All his friends thought the vase was beautiful and couldn’t believe how lucky Ben was. And that’s how Ben found an amazing vase in the store!

（从前有一个小男孩叫 Ben。Ben 喜欢探索周围的世界。他看到了许多神奇的东西，比如商店里陈列的漂亮花瓶。一天，Ben 在商店里闲逛时看到了一个非常特别的花瓶。Ben 看到它时惊呆了！他说："哇，这个花瓶太神奇了！我能买下它吗？"店主微笑着说："当然可以。你可以把它带回家，让所有朋友看看它有多神奇！"于是 Ben 把花瓶带回家，他为此非常自豪！他把朋友们叫过来，给他们看这个神奇的花瓶。他所有的朋友都觉得这个花瓶很漂亮，都不敢相信 Ben 有多幸运。这就是 Ben 在商店里发现神奇花瓶的故事！）

#### 7.2.1 超参数调节

我们会告诉你一些非常基本的初始超参数，并要求你找出另一些效果好的设置。

- **词表大小** 10000。典型的词表大小在几万到几十万之间。你应该改变它，观察词表和模型行为如何变化。
- **上下文长度** 256。像 TinyStories 这样的简单数据集可能不需要很长的序列长度，但对后面 OpenWebText 数据，你可能想改变它。试着改变它，观察它对每次迭代运行时间和最终困惑度的影响。
- **d_model** 512。这比许多小型 Transformer 论文中使用的 768 维略小，但会让训练更快。
- **d_ff** 1344。这大约是 $\frac{8}{3} d_{model}$，同时是 64 的倍数，有利于 GPU 性能。
- **RoPE theta 参数 $\Theta$** 10000。
- **层数和头数** 4 层、16 头。合起来这将得到约 1700 万个非嵌入参数，是一个相当小的 Transformer。
- **处理的词元总数** 327,680,000（你的批量大小 × 总步数 × 上下文长度应大致等于此值）。

你应通过试错来为以下其他超参数找到好的默认值：学习率、学习率 warm-up、其他 AdamW 超参数（$\beta_1, \beta_2, \varepsilon$）和权重衰减。你可以在 D. P. Kingma et al. [22] 中找到这些超参数的一些典型取值。

#### 7.2.2 组装起来

现在你可以把所有东西拼起来：获取一个训练好的 BPE 分词器，对训练数据集分词，然后在你写的训练循环中运行。重要提示：如果你的实现正确且高效，上述超参数在 1 块 B200 GPU 上的运行时间应约为 20–30 分钟。如果你的运行时间远长于此，请检查并确保你的数据加载、检查点或验证损失代码没有成为运行时间的瓶颈，并且你的实现已正确批处理。

#### 7.2.3 调试模型架构的技巧

我们强烈建议熟悉 IDE 内置的调试器（例如 VSCode/Zed），与用 print 语句调试相比它会为你节省时间。如果你用文本编辑器，可以使用 ipdb 之类的工具。调试模型架构时的其他一些良好实践：

- 开发任何神经网络架构时，一个常见的初始步骤是在单个 minibatch 上过拟合。如果你的实现正确，你应该能很快把训练损失压到接近零。
- 在各个模型组件中设置断点，检查中间张量的形状，确保它们符合你的预期。
- 监控激活值、模型权重和梯度的范数，确保它们没有爆炸或消失。

> **问题（learning_rate）：调节学习率（2 B200 小时）（3 分）**
>
> 学习率是最重要的待调节超参数之一。以你训练好的基座模型为基础，回答以下问题：
>
> (a) 对学习率做超参数扫描，报告最终损失（或者记录优化器发散的情况）。
> **提交要求：** 多个学习率对应的学习曲线。解释你的超参数搜索策略。
> **提交要求：** 一个在 TinyStories 上验证损失（逐词元）不高于 1.45 的模型。
>
> > **低资源提示：在 CPU 或 Apple Silicon 上训练几步**
> >
> > 如果你在 cpu 或 mps 上运行，应把处理的词元总数降到 40,000,000，这足以生成相当流畅的文本。你也可以把目标验证损失从 1.45 提高到 2.00。
> >
> > 在 M4 Max 芯片和 36 GB 内存上，用调节好的学习率运行我们的参考实现代码，批量大小 × 总步数 × 上下文长度 = 32×5000×256 = 40,960,000 词元，在 cpu 上耗时 1 小时 22 分钟，在 mps 上耗时 36 分钟。在第 5000 步时，我们达到的验证损失为 1.80。
> >
> > 一些额外的提示：
> > - 使用 $N$ 步训练时，我们建议把余弦学习率衰减调度调整为在第 $N$ 步时恰好完成衰减（即达到最小学习率）。
> > - 使用 mps 时，不要使用 TF32 内核，即不要像在 cuda 设备上那样设置
> >   ```python
> >   torch.set_float32_matmul_precision('high')
> >   ```
> >   我们曾尝试在 mps 上启用 TF32 内核（torch 2.9.0 版本），发现后端有时会使用静默损坏的内核，导致训练不稳定。
> > - 你可以通过用 `torch.compile` JIT 编译模型来加速训练。具体地：
> >   - 在 cpu 上，用 `model = torch.compile(model)` 编译你的模型
> >   - 在 mps 上，可以用 `model = torch.compile(model, backend="aot_eager")` 在一定程度上优化反向传播
> >
> >   截至 torch 2.9.0 版本，mps 尚不支持 Inductor 编译。
>
> (b) 民间经验认为最佳学习率位于"稳定性的边缘"。研究一下学习率发散的临界点与你最佳学习率之间的关系。
> **提交要求：** 一系列递增学习率的学习曲线，其中至少包含一次发散的运行，并分析这与收敛速度的关系。

现在让我们改变批量大小，看看训练会发生什么。批量大小很重要——它们让我们通过更大的矩阵乘法从 GPU 获得更高效率，但我们总是想要大批量吗？让我们运行一些实验来找出答案。

> **问题（batch_size_experiment）：批量大小变化（1 B200 小时）（1 分）**
>
> 把批量大小从 1 一直变到 GPU 内存上限。至少尝试几个中间大小的批量，包括 64 和 128 等典型大小。
> **提交要求：** 不同批量大小运行的学习曲线。如有必要，学习率应重新优化。
> **提交要求：** 几句话讨论你对批量大小及其对训练影响的发现。

有了你的解码器，我们现在可以生成文本了！我们将从模型生成文本并看看它有多好。作为参照，你应该得到至少与下面示例一样好的输出。

> **示例（ts_generate_example）：TinyStories 语言模型的采样输出**
>
> Once upon a time, there was a pretty girl named Lily. She loved to eat gum, especially the big black one. One day, Lily’s mom asked her to help cook dinner. Lily was so excited! She loved to help her mom. Lily’s mom made a big pot of soup for dinner. Lily was so happy and said, “Thank you, Mommy! I love you.” She helped her mom pour the soup into a big bowl. After dinner, Lily’s mom made some yummy soup. Lily loved it! She said, “Thank you, Mommy! This soup is so yummy!” Her mom smiled and said, “I’m glad you like it, Lily.” They finished cooking and continued to cook together. The end.

（从前有一个漂亮的女孩叫 Lily。她喜欢吃口香糖，尤其是大的黑色那种。一天，Lily 的妈妈让她帮忙做晚饭。Lily 非常兴奋！她喜欢帮妈妈干活。Lily 的妈妈做了一大锅汤当晚餐。Lily 非常开心，说："谢谢你，妈妈！我爱你。"她帮妈妈把汤倒进一个大碗里。晚饭后，Lily 的妈妈又做了一些美味的汤。Lily 非常喜欢！她说："谢谢你，妈妈！这个汤太好喝了！"她妈妈微笑着说："我很高兴你喜欢，Lily。"她们做完饭，然后继续一起做饭。完。）

> **低资源提示：在 CPU 或 Apple Silicon 上生成文本**
>
> 如果你使用的是处理 4000 万词元的低资源配置，你应该会看到生成的文本仍像英语，但不如上面流畅。例如，我们在 4000 万词元上训练的 TinyStories 语言模型的采样输出如下：
>
> Once upon a time, there was a little girl named Sue. Sue had a tooth that she loved very much. It was his best head. One day, Sue went for a walk and met a ladybug! They became good friends and played on the path together.
>
> “Hey, Polly! Let’s go out!” said Tim. Sue looked at the sky and saw that it was difficult to find a way to dance shining. She smiled and agreed to help the talking!“
>
> As Sue watched the sky moved, what it was. She

下面是精确的问题描述和我们的要求：

> **问题（generate）：生成文本（1 分）**
>
> 使用你的解码器和训练好的检查点，报告你的模型生成的文本。你可能需要调整解码器参数（温度、top-p 等）以获得流畅的输出。
> **提交要求：** 至少 256 个词元的文本转储（或到第一个 `<|endoftext|>` 词元为止），并附一段简短评论说明该输出的流畅度，以及至少两个影响输出质量好坏的因素。

### 7.3 消融与架构修改

理解 Transformer 的最佳方式是亲手修改它并观察其行为。现在我们将做几个简单的消融和修改。

**消融 1：层归一化**

人们常说层归一化对 Transformer 训练的稳定性很重要。但也许我们想冒险一试。让我们把每个 Transformer 块中的 RMSNorm 都去掉，看看会发生什么。

> **问题（layer_norm_ablation）：移除 RMSNorm 并训练（0.5 B200 小时）（1 分）**
>
> 从你的 Transformer 中移除所有的 RMSNorm 并训练。在之前最优学习率下会发生什么？你能用更低的学习率获得稳定性吗？
> **提交要求：** 移除 RMSNorm 训练时的学习曲线，以及最优学习率下的学习曲线。
> **提交要求：** 几句话评论 RMSNorm 的影响。

现在我们来研究另一种乍看之下很任意的层归一化选择。Pre-norm Transformer 块定义为

$$z = x + \mathrm{MultiHeadSelfAttention}(\mathrm{RMSNorm}(x)), \quad (25)$$

$$y = z + \mathrm{FFN}(\mathrm{RMSNorm}(z)). \quad (26)$$

这是对原始 Transformer 架构的少数"共识性"修改之一，原始架构使用的是 post-norm 方案：

$$z = \mathrm{RMSNorm}(x + \mathrm{MultiHeadSelfAttention}(x)), \quad (27)$$

$$y = \mathrm{RMSNorm}(z + \mathrm{FFN}(z)). \quad (28)$$

让我们改回 post-norm 方案，看看会发生什么。

> **问题（pre_norm_ablation）：实现 post-norm 并训练（0.5 B200 小时）（1 分）**
>
> 把你的 pre-norm Transformer 实现修改为 post-norm 实现。用 post-norm 模型训练，看看会发生什么。
> **提交要求：** post-norm Transformer 的学习曲线，并与 pre-norm 对比。

我们看到层归一化对 Transformer 的行为有重大影响，甚至层归一化的位置也很重要。

**消融 2：位置嵌入**

接下来我们将研究位置嵌入对模型性能的影响。具体而言，我们将比较基座模型（带 RoPE）与完全不含位置嵌入（NoPE）的模型。事实证明，仅解码器 Transformer（即像我们实现的那样带因果掩码的 Transformer）理论上可以在不显式提供位置嵌入的情况下推断出相对或绝对位置信息 [Y.-H. H. Tsai et al., 2019; A. Kazemnejad et al., 2023]。现在我们将实证检验 NoPE 相比 RoPE 的表现如何。

> **问题（no_pos_emb）：实现 NoPE（0.5 B200 小时）（1 分）**
>
> 修改你带 RoPE 的 Transformer 实现，把位置嵌入信息完全移除，看看会发生什么。
> **提交要求：** 比较 RoPE 与 NoPE 性能的学习曲线。

**消融 3：SwiGLU 对比 SiLU**

接下来，我们将遵循 N. Shazeer [20]，通过比较 SwiGLU 前馈网络与使用 SiLU 激活但无门控线性单元（GLU）的前馈网络的性能，来检验前馈网络中门控的重要性：

$$\mathrm{FFN}_{\mathrm{SiLU}}(x) = W_2\,\mathrm{SiLU}(W_1 x). \quad (29)$$

回顾在我们的 SwiGLU 实现中，我们把前馈内层维度设为约 $d_{ff} = \frac{8}{3} d_{model}$（同时确保 $d_{ff} \bmod 64 = 0$，以利用 GPU 张量核心）。在这个消融基线中，你的 $\mathrm{FFN}_{\mathrm{SiLU}}$ 实现应改为设 $d_{ff} = 4 \times d_{model}$，以近似匹配默认 SwiGLU 前馈网络（它有三个而非两个权重矩阵）的参数量。

> **问题（swiglu_ablation）：SwiGLU 对比 SiLU（0.5 B200 小时）（1 分）**
>
> **提交要求：** 比较参数量大致匹配的 SwiGLU 与 SiLU 前馈网络性能的学习曲线。
> **提交要求：** 几句话讨论你的发现。

> **低资源提示：GPU 资源有限的在线学员应在 TinyStories 上测试修改**
>
> 在作业的剩余部分中，我们将转向更大规模、噪声更多的网页数据集（OpenWebText），尝试架构修改并（可选地）向课程排行榜提交。
>
> 在 OpenWebText 上把 LM 训练到流畅需要很长时间，因此我们建议 GPU 访问有限的在线学员继续在 TinyStories 上测试修改（使用验证损失作为评估性能的指标）。

### 7.4 在 OpenWebText 上运行

现在我们将转向一个由网页爬取创建的更标准的预训练数据集。一个小规模的 OpenWebText 样本 [A. Gokaslan et al., 2019] 也以单个文本文件的形式提供：访问方式见第 1 节。

下面是 OpenWebText 的一个示例。注意文本要真实、复杂和多样得多。你可能想浏览一下训练数据集，感受一下网页爬取语料的训练数据是什么样的。

> **示例（owt_example）：OWT 中的一个示例**
>
> Baseball Prospectus director of technology Harry Pavlidis took a risk when he hired Jonathan Judge.
>
> Pavlidis knew that, as Alan Schwarz wrote in The Numbers Game, “no corner of American culture is more precisely counted, more passionately quantified, than performances of baseball players.” With a few clicks here and there, you can find out that Noah Syndergaard’s fastball revolves more than 2,100 times per minute on its way to the plate, that Nelson Cruz had the game’s highest average exit velocity among qualified hitters in 2016 and myriad other tidbits that seem ripped from a video game or science fiction novel. The rising ocean of data has empowered an increasingly important actor in baseball’s culture: the analytical hobbyist.
>
> That empowerment comes with added scrutiny – on the measurements, but also on the people and publications behind them. With Baseball Prospectus, Pavlidis knew all about the backlash that accompanies quantitative imperfection. He also knew the site’s catching metrics needed to be reworked, and that it would take a learned mind – someone who could tackle complex statistical modeling problems – to complete the job.
>
> “He freaks us out.” Harry Pavlidis
>
> Pavlidis had a hunch that Judge “got it” based on the latter’s writing and their interaction at a site-sponsored ballpark event. […]

（Baseball Prospectus 的技术总监 Harry Pavlidis 在聘用 Jonathan Judge 时冒了一次风险。
>
> Pavlidis 知道，正如 Alan Schwarz 在《The Numbers Game》中所写，"美国文化中没有任何一个角落，比棒球运动员的表现被更精确地计数、更狂热地量化了。"随便点几下鼠标，你就能发现 Noah Syndergaard 的快球在飞向本垒板的路上每分钟旋转超过 2,100 次；Nelson Cruz 在 2016 赛季的平均出棒初速是全联盟达标打者中最高的；还有无数其他仿佛从电子游戏或科幻小说中摘出来的趣闻。不断上涨的数据洪流赋予了棒球文化中一个日益重要的角色以力量：分析型业余爱好者。
>
> 这种赋权也带来了更多的审视——不仅针对测量数据，也针对数据背后的人和出版物。在 Baseball Prospectus，Pavlidis 非常清楚伴随量化不完美而来的强烈反弹。他也知道网站的捕手评估指标需要重做，而这需要一个博学的头脑——一个能处理复杂统计建模问题的人——来完成这项工作。
>
> "他让我们 freak out（大吃一惊）。"——Harry Pavlidis
>
> Pavlidis 有一种直觉，认为 Judge"懂了"，依据是后者的文章写作以及他们在一个网站赞助的球场活动中的交流。[……]）

注意：你可能需要为这个实验重新调节超参数，如学习率或批量大小。

> **问题（main_experiment）：在 OWT 上做实验（2 B200 小时）（2 分）**
>
> 在 OpenWebText 上用与 TinyStories 相同的模型架构和总训练迭代数来训练你的语言模型。这个模型表现如何？
> **提交要求：** 你的语言模型在 OpenWebText 上的学习曲线。描述与 TinyStories 相比损失的差异——我们应该如何解读这些损失？
> **提交要求：** OpenWebText LM 生成的文本，格式与 TinyStories 输出相同。这段文本的流畅度如何？为什么即使有与 TinyStories 相同的模型和算力预算，输出质量却更差？

### 7.5 你自己的修改 + 排行榜

恭喜你走到了这一步。你快完成了！现在你将尝试改进 Transformer 架构，并看看你的超参数和架构与班上其他同学相比如何。

**排行榜规则**

除以下规则外没有任何限制：

- **运行时间：** 你的提交在 B200 上最多运行 45 分钟。如果你使用 SLURM 或 Modal，你可能想在提交脚本中强制执行这一点。
- **数据：** 你只能使用我们提供的 OpenWebText 训练数据集。

除此之外，你可以随心所愿地做任何事情。

如果你在寻找实现思路，可以看看这些资源：

- 最先进的开源 LLM 家族，例如 Llama 3 [A. Grattafiori et al., 2024] 或 Qwen 2.5 [A. Yang et al., 2024]。
- NanoGPT speedrun 仓库（github.com/KellerJordan/modded-nanogpt），社区成员在其中发布了用于"速通"小规模语言模型预训练的许多有趣修改。例如，一个可以追溯到原始 Transformer 论文的常见修改，是把输入嵌入和输出嵌入的权重绑定（tie）在一起（见 A. Vaswani et al. [8]（第 3.4 节）和 A. Chowdhery et al. [16]（第 2 节））。如果你尝试权重绑定，可能需要减小嵌入/LM head 初始化的标准差。

在尝试完整的 45 分钟运行之前，你会想在 OpenWebText 的小子集或 TinyStories 上测试这些想法。

需要提醒一句：我们确实注意到，你在这个排行榜上发现的某些有效修改可能无法泛化到更大规模的预训练。我们将在课程的 scaling laws（缩放定律）单元中进一步探讨这个想法。

> **问题（leaderboard）：排行榜（10 B200 小时）（6 分）**
>
> 按照上述排行榜规则训练一个模型，目标是在 0.75 B200 小时内最小化语言模型的验证损失。
> **提交要求：** 记录到的最终验证损失、一条横轴为墙上时钟时间且明显小于 45 分钟的学习曲线，以及你做了什么的描述。我们期望排行榜提交至少超过 5.0 损失的朴素基线。在此提交到排行榜：github.com/stanford-cs336/assignment1-basics-leaderboard。

---

## 参考文献

[1] R. Eldan and Y. Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?." 2023.

[2] A. Gokaslan, V. Cohen, E. Pavlick, and S. Tellex, "OpenWebText corpus." 2019.

[3] R. Sennrich, B. Haddow, and A. Birch, "Neural Machine Translation of Rare Words with Subword Units," in Proc. of ACL, 2016.

[4] C. Wang, K. Cho, and J. Gu, "Neural Machine Translation with Byte-Level Subwords." 2019.

[5] P. Gage, "A new algorithm for data compression," C Users Journal, vol. 12, no. 2, pp. 23–38, Feb. 1994.

[6] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, "Language Models are Unsupervised Multitask Learners." 2019.

[7] A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever, "Improving Language Understanding by Generative Pre-Training." 2018.

[8] A. Vaswani et al., "Attention is All you Need," in Proc. of NeurIPS, 2017.

[9] T. Q. Nguyen and J. Salazar, "Transformers without Tears: Improving the Normalization of Self-Attention," in Proc. of IWSWLT, 2019.

[10] R. Xiong et al., "On Layer Normalization in the Transformer Architecture," in Proc. of ICML, 2020.

[11] J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer Normalization." 2016.

[12] H. Touvron et al., "LLaMA: Open and Efficient Foundation Language Models." 2023.

[13] B. Zhang and R. Sennrich, "Root Mean Square Layer Normalization," in Proc. of NeurIPS, 2019.

[14] A. Grattafiori et al., "The Llama 3 Herd of Models." [Online]. Available: https://arxiv.org/abs/2407.21783

[15] A. Yang et al., "Qwen2.5 Technical Report," arXiv preprint arXiv:2412.15115, 2024.

[16] A. Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways." 2022.

[17] D. Hendrycks and K. Gimpel, "Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units." 2016.

[18] S. Elfwing, E. Uchibe, and K. Doya, "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning." [Online]. Available: https://arxiv.org/abs/1702.03118

[19] Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier, "Language Modeling with Gated Convolutional Networks." [Online]. Available: https://arxiv.org/abs/1612.08083

[20] N. Shazeer, "GLU Variants Improve Transformer." 2020.

[21] J. Su, Y. Lu, S. Pan, B. Wen, and Y. Liu, "RoFormer: Enhanced Transformer with Rotary Position Embedding." 2021.

[22] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in Proc. of ICLR, 2015.

[23] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in Proc. of ICLR, 2019.

[24] T. B. Brown et al., "Language Models are Few-Shot Learners," in Proc. of NeurIPS, 2020.

[25] J. Kaplan et al., "Scaling Laws for Neural Language Models." 2020.

[26] J. Hoffmann et al., "Training Compute-Optimal Large Language Models." 2022.

[27] A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi, "The Curious Case of Neural Text Degeneration," in Proc. of ICLR, 2020.

[28] Y.-H. H. Tsai, S. Bai, M. Yamada, L.-P. Morency, and R. Salakhutdinov, "Transformer Dissection: An Unified Understanding for Transformer`s Attention via the Lens of Kernel," in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), K. Inui, J. Jiang, V. Ng, and X. Wan, Eds., Hong Kong, China: Association for Computational Linguistics, Nov. 2019, pp. 4344–4353. doi: 10.18653/v1/D19-1443.

[29] A. Kazemnejad, I. Padhi, K. Natesan, P. Das, and S. Reddy, "The Impact of Positional Encoding on Length Generalization in Transformers," in Thirty-seventh Conference on Neural Information Processing Systems, 2023. [Online]. Available: https://openreview.net/forum?id=Drrl2gcjzl
