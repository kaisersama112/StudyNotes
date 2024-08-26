## 一、什么是Transformer

​	Transformer 是一种神经网络架构，它从根本上改变了人工智能的方法。Transformer 于 2017 年在开创性论文[“Attention is All You Need”](https://dl.acm.org/doi/10.5555/3295222.3295349)中首次提出，此后成为深度学习模型的首选架构，为 OpenAI 的 **GPT、**Meta 的 **Llama** 和 Google 的 **Gemini** 等文本生成模型提供支持。除了文本，Transformer 还应用于[音频生成](https://huggingface.co/learn/audio-course/en/chapter3/introduction)、[图像识别](https://huggingface.co/learn/computer-vision-course/unit3/vision-transformers/vision-transformers-for-image-classification)、[蛋白质结构预测](https://elifesciences.org/articles/82819)，甚至[玩游戏](https://www.deeplearning.ai/the-batch/reinforcement-learning-plus-transformers-equals-efficiency/)，展示了它在众多领域的多功能性。

​	本质来讲，文本生成式Transformer 模型的工作原理是**下一个单词的预测**：从用户输入的文本提示（Prompt）中预测下一个单词是什么？**Transformers 的核心创新和强大之处在于它们使用自注意力机制**，这使它们能够比以前的架构更有效地处理整个序列并捕获远程依赖关系。

## 二、Transformer 架构

​	每个文本生成Transformer 都由以下三个关键组件组成：Embedding，Transformer Block，Output Probabilities

### 2.1 Embedding（嵌入）

[Embedding]: ./Embedding.md	"嵌入模型.md"

​	文本输入被划分为更小的单元，称为标记，可以是单词或子词。这些标记被转换为称为嵌入向量的数字向量，用于捕获单词的语义。

### 2.2 Transformer Block (变压器块)

​	Transformer Block是处理和转换输入数据的模型。每个区块包括：Attention Mechanism,MLP (Multilayer Perceptron) Layer

#### 2.2.1 Attention Mechanism

​	是Transformer的核心组件，它允许标记与其他标记通信，捕获上下文信息和单词之间的关系。

#### 2.2.2 MLP (Multilayer Perceptron) Layer（多层感知器层）

​	是一个独立对每个令牌进行操作的前馈网络。虽然注意力层的目标是在 Token 之间路由信息，但 MLP 的目标是优化每个 Token 的表示。

### 2.3 Output Probabilities（输出概率）

​	最终的线性层和 softmax 层将处理的嵌入转换为概率，使模型能够预测序列中的下一个标记。

## 三、Embedding（嵌入）

​	假设你想使用Transformer 模型进行文本生成，你添加以下提示`Data visualization empowers users to`,此输入需要转换为模型可以理解和处理的格式，Embedding将文本转换为模型可以使用的数字表达形式，如果要将提示（prompt）转换为Embedding（嵌入），我们需要经历以下几个步骤从而达到目的，我们需要 1） 对输入进行分词，2） 获取分词嵌入，3） 添加位置信息，最后 4） 将分词和位置编码相加以获得最终的嵌入。让我们看看这些步骤中的每一个是如何完成的。

![image-20240826152920855](https://raw.githubusercontent.com/kaisersama112/typora_image/master/image-20240826152920855.png)

[^图1]: 显示如何将输入提示转换为矢量表示。该过程包括 （1） 分词、（2） 分词嵌入、（3） 位置编码和 （4） 最终嵌入。



### 3.1 Token (分词化)

​	**分词是将输入文本分解为更小、更易于管理的片段（称为分词）的过程**。这些标记可以是单词或子词。`“Data”` 和 `“visualization”` 这两个词对应于唯一的标记，而 `empowers` 这个词则分为两个标记。令牌的完整词汇表是在训练模型之前决定的：GPT-2 的词汇表有 `50257` 个唯一的令牌。现在，我们已经将输入文本拆分为具有不同 ID 的标记，我们可以从嵌入中获取它们的向量表示。

### 3.2 Token Embedding (令牌嵌入)

​	GPT-2 Small 将词汇表中的每个标记表示为 768 维向量;向量的维度取决于模型。这些嵌入向量存储在形状为 `（50257,768）` 的矩阵中，包含大约 3900 万个参数！这个广泛的矩阵允许模型为每个标记分配语义含义。

### 3.3 Positional Encoding (位置编码)

​	Embedding 层还对有关输入提示中每个标记位置的信息进行编码。**不同的模型使用不同的方法进行位置编码。**GPT-2 从头开始训练自己的位置编码矩阵，将其直接集成到训练过程中。

## 四、Transformer Block (变压器块)

​	Transformer 处理的核心在于 Transformer 块，这其中**包含了多头自注意力和多层感知机层**。大多数模型都是由多个这样的块一个接一个依次堆叠起来的。标记的表示会通过各个层演变，从第一个块到第十二个块，这样能让模型对每个标记建立起复杂的理解。**这种分层的方式会带来输入的高阶表示。**

### 4.1 Multi-Head Self-Attention（多头自我注意）

​	自我注意机制使模型能够专注于输入序列的相关部分，从而能够捕获数据中的复杂关系和依赖关系。让我们看看这种自我关注是如何逐步计算的。

#### Step 1: Query, Key, and Value Matrices（查询、键和值矩阵）

![image-20240826152742201](https://raw.githubusercontent.com/kaisersama112/typora_image/master/image-20240826152742201.png)

[^图2]: 从原始嵌入计算 Query、Key 和 Value 矩阵

​	每个标记的嵌入向量都转换为三个向量：查询 （Q）、键 （K） 和值 （V）。这些向量是通过将输入嵌入矩阵与学习的 Q、K 和 V 的权重矩阵相乘而得出的。下面是一个 Web 搜索类比，可以帮助我们在这些矩阵背后建立一些直觉：

- [ ] **查询（Q）** 是您在搜索引擎栏中键入的搜索文本。这是您要*“查找更多信息”*的令牌。
- [ ] **键（K）** 是搜索结果窗口中每个网页的标题。它表示查询可以处理的可能令牌。
- [ ] **值（V）** 是显示的网页的实际内容。将适当的搜索词 （Query） 与相关结果 （Key） 匹配后，我们希望获取最相关页面的内容 （Value）。

​	通过使用这些 QKV 值，模型可以计算注意力分数，从而确定每个标记在生成预测时应获得多少关注。

#### Step 2: Masked Self-Attention（掩盖自我注意）

​	掩蔽的自我注意允许模型通过关注输入的相关部分来生成序列，同时阻止访问未来的标记。

![image-20240826153103462](https://raw.githubusercontent.com/kaisersama112/typora_image/master/image-20240826153103462.png)

[^图3]: 使用 Query、Key 和 Value 矩阵计算掩码自我注意。

- [ ] **Attention Score （注意力得分）**：Query 和 Key 矩阵的点积确定每个查询与每个键的对齐方式，从而生成一个反映所有输入标记之间关系的方阵。

- [ ] **Masking （掩码）**：在注意力矩阵的上三角应用掩码，以防止模型获取未来的标记，将这些值设置为负无穷。模型需要学会如何在不“窥视”未来的情况下预测下一个标记。
- [ ] **Softmax（归一化指数函数）**：掩码操作之后，注意力分数通过 Softmax 运算转换为概率，该运算会对每个注意力分数取指数。矩阵的每一行总和为 1，并表明了左侧其他每个标记的相关性。

#### Step 3: Output （输出）

​	该模型使用掩蔽的自我注意分数，并将其与 Value 矩阵相乘，以获得自我注意机制的最终输出。GPT-2 有 `12` 个自我注意头，每个头捕获标记之间的不同关系。这些头的输出是串联的，并通过线性投影传递。



## 五、MLP: Multi-Layer Perceptron（多层感知器）

![image-20240826154146958](https://raw.githubusercontent.com/kaisersama112/typora_image/master/image-20240826154146958.png)

[^图4]: 使用 MLP 层将自注意力表征投射到更高的维度，以增强模型的表征能力。

​	在多个自我注意力头捕获到输入标记之间的不同关系后，连接的输出通过多层感知器 （MLP） 层以增强模型的表示能力。MLP 模块由两个线性变换组成，中间有一个 GELU 激活函数。第一个线性变换将输入的维数从 `768` 增加到 `3072` 四倍。第二次线性变换将维数减小回原始大小 `768`，确保后续层接收到一致维度的输入。与自我注意机制不同，MLP 独立处理令牌，并简单地将它们从一个表示映射到另一个表示。



## 六、Output Probabilities（输出概率）

​	在通过所有 Transformer 模块处理完输入后，输出将通过最终的线性层，为标记预测做好准备。该层将最终表示投影到 `50,257` 维空间中，其中词汇表中的每个标记都有一个对应的值，称为 `logit`。任何词元都可以是下一个词，因此这个过程允许我们简单地根据它们成为下一个词的可能性对这些词元进行排序。然后，我们应用 softmax 函数将 logits 转换为总和 1 的概率分布。这将允许我们根据其可能性对下一个 token 进行采样。

![image-20240826154721111](https://raw.githubusercontent.com/kaisersama112/typora_image/master/image-20240826154721111.png)

[^图5]: 词汇表中的每个标记都根据模型的输出 logit 分配一个概率。这些概率决定了每个标记成为序列中下一个单词的可能性。

​	最后一步是通过从此分布中采样来生成下一个标记`temperature（温度）`超参数在此过程中起着关键作用。从数学上讲，这是一个非常简单的操作：模型输出 logit 简单地除以`temperature（温度）`：

- `temperature = 1`：将 logits 除以 1 对 softmax 输出没有影响。
- `温度 < 1`：较低的温度通过锐化概率分布使模型更具置信度和确定性，从而产生更可预测的输出。
- `温度 > 1`：温度越高，概率分布越柔和，生成的文本就越随机——有些人称之为模型*“创造力*”。

调整温度，看看如何在确定性和多样化输出之间取得平衡！



## 七、Advanced Architectural Features（高级架构功能）

​	有几项高级架构功能可增强 Transformer 模型的性能。虽然它们对模型的整体性能很重要，但对于理解架构的核心概念并不那么重要。**层归一化、Dropout 和 Residual Connections 是 Transformer 模型中的关键组成部分**，尤其是在训练阶段。层归一化可以稳定训练并帮助模型更快地收敛。Dropout 通过随机停用神经元来防止过拟合。残差连接允许梯度直接流经管网，并有助于防止梯度消失问题。

### Layer Normalization（图层归一化）

​	层归一化有助于稳定训练过程并提高收敛性。它的工作原理是规范化特征之间的输入，确保激活的均值和方差一致。这种归一化有助于缓解与内部协变量偏移相关的问题，使模型能够更有效地学习并降低对初始权重的敏感性。层归一化在每个 Transformer 块中应用两次，一次在自注意力机制之前，一次在 MLP 层之前。

### Dropout （正则化）

​	Dropout 是一种正则化技术，通过在训练期间将模型权重的分数随机设置为零来防止神经网络中的过拟合。这鼓励模型学习更强大的特征并减少对特定神经元的依赖，从而帮助网络更好地泛化到新的、看不见的数据。在模型推理期间，dropout 被停用。这实质上意味着我们正在使用经过训练的子网络的系综，从而获得更好的模型性能。

### Residual Connections （残差连接）

​	残差连接于 2015 年首次在 ResNet 模型中引入。这项架构创新通过支持训练非常深入的神经网络，彻底改变了深度学习。实质上，残差连接是绕过一个或多个层的快捷方式，将层的输入添加到其输出中。这有助于缓解梯度消失问题，从而更容易训练多个 Transformer 块相互堆叠的深度网络。在 GPT-2 中，残差连接在每个 Transformer 块中使用两次：一次在 MLP 之前，一次在 MLP 之后，确保梯度更容易流动，并且早期的层在反向传播期间获得足够的更新。

