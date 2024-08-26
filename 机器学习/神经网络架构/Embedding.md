

## 一、Embedding的本质

​	"Embedding" 在字面上的翻译是“嵌入”，但在[机器学习](https://so.csdn.net/so/search?q=机器学习&spm=1001.2101.3001.7020)和自然语言处理的上下文中，我们更倾向于将其理解为一种 “向量化” 或 “向量表示” 的技术，这有助于更准确地描述其在这些领域中的应用和作用。

![img](https://raw.githubusercontent.com/kaisersama112/typora_image/master/7dfe09fadde41fb3de6dd47c338ed25c.gif)

### 1. 机器学习中的Embedding

- **原理：**将离散数据映射为连续变量，捕捉潜在关系。
- **方法：**使用神经网络中的Embedding层，训练得到数据的向量表示。
- **作用：**提升模型性能，增强泛化能力，降低计算成本。

​	在机器学习中，Embedding 主要是指将离散的高维数据（如文字、图片、音频）映射到低纬度的连续向量空间。这个过程会生成由实数构成的向量，用于捕捉原始数据的潜在搞关系和结构。

### 2. NLP中的Embedding

- **原理：**将文本转换为连续向量，基于分布式假设捕捉语义信息。
- **方法：**采用词嵌入技术（如Word2Vec）或复杂模型（如BERT）学习文本表示。
- **作用：**解决词汇鸿沟，支持复杂NLP任务，提供文本的语义理解。

​	在NLP中，Embedding技术（如Word2Vec）将单词或短语映射为向量，使得语义上相似的单词在向量空间中位置相近。这种Embedding对于自然语言处理任务（如文本分类、情感分析、机器翻译）至关重要。

​	Embedding向量不仅仅是对物体进行简单编号或标识，而是通过特征抽象和编码，在尽量保持物体间相似性的前提下，将物体映射到一个高维特征空间中。**Embedding向量能够捕捉到物体之间的相似性和关系**，在映射到高维特征空间后，相似的物体在空间中会聚集在一起，而不同的物体会被分隔开。



## 二、Text Embedding工作原理

​	**文本向量化（Text Embedding）：将文本数据（词、句子、文档）表示成向量的方法。**词向量化将词转为二进制或高维实数向量，句子和文档向量化则将句子或文档转为数值向量，通过平均、神经网络或主题模型实现。

### 1. 词向量化

将单个词转换为数值向量

- [ ] 独热编码（One-Hot Encoding）：为每个词分配一个唯一的二进制向量，其中只有一个位置是1，其余位置是0。
- [ ] 词嵌入（Word Embeddings）：如Word2Vec, GloVe, FastText等，将每个词映射到一个高维实数向量，这些向量在语义上是相关的。

![图片](https://raw.githubusercontent.com/kaisersama112/typora_image/master/846ed61960420402ab1614ad7d6b8ebe.jpeg)

### 2. 句子向量化

将整个句子转换为一个数值向量

- [ ] 简单平均/加权平均：对句子中的词向量进行平均或根据词频进行加权平均。
- [ ] 递归神经网络（RNN）：通过递归地处理句子中的每个词来生成句子表示。
- [ ] 卷积神经网络（CNN）：使用卷积层来捕捉句子中的局部特征，然后生成句子表示。
- [ ] 自注意力机制（如Transformer）：如BERT模型，通过对句子中的每个词进行自注意力计算来生成句子表示。

![图片](https://raw.githubusercontent.com/kaisersama112/typora_image/master/17acae171dfd0d75f9d34eaa96bb7fb7.jpeg)

### 3. BERT句子向量化

文档主题模型（LDA）:通过捕捉文档中的主题分布来生成文档表示。

层次化模型：如Doc2Vec，它扩展了Word2Vec，可以生成整个文档的向量表示。

 文档向量化：将整个文档（如一篇文章或一组句子）转换为一个数值向量。

- 简单平均/加权平均：对文档中的句子向量进行平均或加权平均。

![图片](https://raw.githubusercontent.com/kaisersama112/typora_image/master/6471a490e9812842c30a7e6fb667a24b.png)

### 4. 文档向量化

​	文档向量化目前具有两种主流模式：基于统计方法，基于神经网络方法。

- [ ] 基于统计方法用TF-IDF和N-gram统计生成文本向量

- [ ] 神经网络方法如Word2Vec、GloVe等通过深度学习学习文本向量。

**基于统计方法：**

- TF-IDF：通过统计词频和逆文档频率来生成词向量或文档向量。
- N-gram：基于统计的n个连续词的频率来生成向量。

![图片](https://raw.githubusercontent.com/kaisersama112/typora_image/master/da75f31ad8a264015c78f3a0b960d0e1.png)

**基于神经网络方法：**

词嵌入：

- [ ] Word2Vec：通过预测词的上下文来学习词向量。
  
- [ ] GloVe：通过全局词共现统计来学习词向量。
  
- [ ] FastText：考虑词的n-gram特征来学习词向量。

句子嵌入：

- [ ] RNN：包括LSTM和GRU，可以处理变长句子并生成句子向量。
  
- [ ] Transformer：使用自注意力机制和位置编码来处理句子，生成句子向量。

文档嵌入：

- [ ] Doc2Vec：扩展了Word2Vec，可以生成整个文档的向量表示。
- [ ] BERT：基于Transformer的预训练模型，可以生成句子或短文档的向量表示。

![图片](https://raw.githubusercontent.com/kaisersama112/typora_image/master/e497dd04d13f2a057d8277dafbc19a7b.png)

​	工作原理：**将离散的文字信息（如单词）转换成连续的向量数据。这样，语义相似的词在向量空间中位置相近，并通过高维度捕捉语言的复杂性。**

- [ ] 将离散信息（如单词、符号）转换为分布式连续值数据（向量）。

- [ ] 相似的项目（如语义上相近的单词）在向量空间中被映射到相近的位置。

- [ ] 提供了更多的维度（如1536个维度）来表示人类语言的复杂度。

举例来讲，这里有三句话：

- [ ] “The cat chases the mouse” 猫追逐老鼠。

- [ ] “The kitten hunts rodents” 小猫捕猎老鼠。

- [ ] “I like ham sandwiches” 我喜欢火腿三明治。

​	人类能理解句子1和句子2含义相近，尽管它们只有“The”这个单词相同。但计算机需要Embedding技术来理解这种关系。Embedding将单词转换为向量，使得语义相似的句子在向量空间中位置相近。这样，即使句子1和句子2没有很多共同词汇，计算机也能理解它们的相关性。

​	如果是人类来理解，句子 1 和句子 2 几乎是同样的含义，而句子 3 却完全不同。但我们看到句子 1 和句子 2 只有“The”是相同的，没有其他相同词汇。计算机该如何理解前两个句子的相关性？

​	Embedding将单词转换为向量，使得语义相似的句子在向量空间中位置相近。这样，即使句子1和句子2没有很多共同词汇，计算机也能理解它们的相关性。

![图片](https://raw.githubusercontent.com/kaisersama112/typora_image/master/f927fd4b8fc124cd3e7e51604c27bb6b.jpeg)

## 三、Image Embedding 工作原理

​	卷积神经网络和自编码器都是用于图像向量化的有效工具，前者通过训练提取图像特征并转换为向量，后者则学习图像的压缩编码以生成低维向量表示。

- [ ] 卷积神经网络（CNN）：通过训练卷积神经网络模型，我们可以从原始图像数据中提取特征，并将其表示为向量。例如，使用预训练的模型（如VGG16, ResNet）的特定层作为特征提取器。

- [ ] 自编码器（Autoencoders）：这是一种无监督的神经网络，用于学习输入数据的有效编码。在图像向量化中，自编码器可以学习从图像到低维向量的映射。

![图片](https://raw.githubusercontent.com/kaisersama112/typora_image/master/27edfdc03e56835d9ea6b63431670210.png)



## 四、Vedio Embedding 工作原理

​	OpenAI的Sora将视觉数据转换为图像块（Turning visual data into patches）。

- 视觉块的引入：为了将视觉数据转换成适合生成模型处理的格式，研究者提出了视觉块嵌入编码（visual patches）的概念。这些视觉块是图像或视频的小部分，类似于文本中的词元。
- 处理高维数据：在处理高维视觉数据时（如视频），首先将其压缩到一个低维潜在空间。这样做可以减少数据的复杂性，同时保留足够的信息供模型学习。

![图片](https://raw.githubusercontent.com/kaisersama112/typora_image/master/04a240594c330b80658426f5bdad55b1.jpeg)

​	工作原理：Sora 用visual patches 代表被压缩后的视频向量进行训练，每个patches相当于GPT中的一个token。使用patches，可以对视频、音频、文字进行统一的向量化表示，和大模型中的 tokens 类似，Sora用 patches 表示视频，把视频压缩到低维空间（latent space）后表示为Spacetime patches。

