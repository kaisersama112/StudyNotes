---

---



## 一、Embedding的本质

​	"Embedding" 在字面上的翻译是“嵌入”，但在[机器学习](https://so.csdn.net/so/search?q=机器学习&spm=1001.2101.3001.7020)和自然语言处理的上下文中，我们更倾向于将其理解为一种 “向量化” 或 “向量表示” 的技术，这有助于更准确地描述其在这些领域中的应用和作用。

​	![img](https://raw.githubusercontent.com/kaisersama112/typora_image/master/7dfe09fadde41fb3de6dd47c338ed25c.gif)

1. **机器学习中的Embedding**

- **原理：**将离散数据映射为连续变量，捕捉潜在关系。
- **方法：**使用神经网络中的Embedding层，训练得到数据的向量表示。
- **作用：**提升模型性能，增强泛化能力，降低计算成本。

​	在机器学习中，Embedding 主要是指将离散的高维数据（如文字、图片、音频）映射到低纬度的连续向量空间。这个过程会生成由实数构成的向量，用于捕捉原始数据的潜在搞关系和结构。

2. **NLP中的Embedding**

- **原理：**将文本转换为连续向量，基于分布式假设捕捉语义信息。
- **方法：**采用词嵌入技术（如Word2Vec）或复杂模型（如BERT）学习文本表示。
- **作用：**解决词汇鸿沟，支持复杂NLP任务，提供文本的语义理解。

​	在NLP中，Embedding技术（如Word2Vec）将单词或短语映射为向量，使得语义上相似的单词在向量空间中位置相近。这种Embedding对于自然语言处理任务（如文本分类、情感分析、机器翻译）至关重要。

​	Embedding向量不仅仅是对物体进行简单编号或标识，而是通过特征抽象和编码，在尽量保持物体间相似性的前提下，将物体映射到一个高维特征空间中。**Embedding向量能够捕捉到物体之间的相似性和关系**，在映射到高维特征空间后，相似的物体在空间中会聚集在一起，而不同的物体会被分隔开。



### 1. Text Embedding工作原理

​	**文本向量化（Text Embedding）：将文本数据（词、句子、文档）表示成向量的方法。**词向量化将词转为二进制或高维实数向量，句子和文档向量化则将句子或文档转为数值向量，通过平均、神经网络或主题模型实现。

#### 1. 词向量化

将单个词转换为数值向量

- [ ] 独热编码（One-Hot Encoding）：为每个词分配一个唯一的二进制向量，其中只有一个位置是1，其余位置是0。
- [ ] 词嵌入（Word Embeddings）：如Word2Vec, GloVe, FastText等，将每个词映射到一个高维实数向量，这些向量在语义上是相关的。

![图片](https://raw.githubusercontent.com/kaisersama112/typora_image/master/846ed61960420402ab1614ad7d6b8ebe.jpeg)

#### 2. 句子向量化

将整个句子转换为一个数值向量

- [ ] 简单平均/加权平均：对句子中的词向量进行平均或根据词频进行加权平均。
- [ ] 递归神经网络（RNN）：通过递归地处理句子中的每个词来生成句子表示。
- [ ] 卷积神经网络（CNN）：使用卷积层来捕捉句子中的局部特征，然后生成句子表示。
- [ ] 自注意力机制（如Transformer）：如BERT模型，通过对句子中的每个词进行自注意力计算来生成句子表示。

![图片](https://raw.githubusercontent.com/kaisersama112/typora_image/master/17acae171dfd0d75f9d34eaa96bb7fb7.jpeg)

#### 3. BERT句子向量化

文档主题模型（LDA）:通过捕捉文档中的主题分布来生成文档表示。

层次化模型：如Doc2Vec，它扩展了Word2Vec，可以生成整个文档的向量表示。

 文档向量化：将整个文档（如一篇文章或一组句子）转换为一个数值向量。

- 简单平均/加权平均：对文档中的句子向量进行平均或加权平均。

![图片](https://raw.githubusercontent.com/kaisersama112/typora_image/master/6471a490e9812842c30a7e6fb667a24b.png)

#### 4. 文档向量化

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

### 2. Image Embedding 工作原理

​	卷积神经网络和自编码器都是用于图像向量化的有效工具，前者通过训练提取图像特征并转换为向量，后者则学习图像的压缩编码以生成低维向量表示。

- [ ] 卷积神经网络（CNN）：通过训练卷积神经网络模型，我们可以从原始图像数据中提取特征，并将其表示为向量。例如，使用预训练的模型（如VGG16, ResNet）的特定层作为特征提取器。

- [ ] 自编码器（Autoencoders）：这是一种无监督的神经网络，用于学习输入数据的有效编码。在图像向量化中，自编码器可以学习从图像到低维向量的映射。

![图片](https://raw.githubusercontent.com/kaisersama112/typora_image/master/27edfdc03e56835d9ea6b63431670210.png)



### 3. Vedio Embedding 工作原理

​	OpenAI的Sora将视觉数据转换为图像块（Turning visual data into patches）。

- 视觉块的引入：为了将视觉数据转换成适合生成模型处理的格式，研究者提出了视觉块嵌入编码（visual patches）的概念。这些视觉块是图像或视频的小部分，类似于文本中的词元。
- 处理高维数据：在处理高维视觉数据时（如视频），首先将其压缩到一个低维潜在空间。这样做可以减少数据的复杂性，同时保留足够的信息供模型学习。

![图片](https://raw.githubusercontent.com/kaisersama112/typora_image/master/04a240594c330b80658426f5bdad55b1.jpeg)

​	工作原理：Sora 用visual patches 代表被压缩后的视频向量进行训练，每个patches相当于GPT中的一个token。使用patches，可以对视频、音频、文字进行统一的向量化表示，和大模型中的 tokens 类似，Sora用 patches 表示视频，把视频压缩到低维空间（latent space）后表示为Spacetime patches。

## 二、统计语言模型：N-gram模型

​	什么是所谓的统计语言模型(Language Model)呢？简单来说，统计语言模型就是用来计算句子概率的概率模型。计算句子概率的概率模型很多，n-gram模型便是其中的一种。

​	假设一个长度为m的句子，包含这些词：![(w_1,w_2,w_3,..,w_m)](https://latex.csdn.net/eq?%28w_1%2Cw_2%2Cw_3%2C..%2Cw_m%29)，那么这个句子的概率（也就是这![m](https://latex.csdn.net/eq?m)个词共现的概率)是：

![img](https://raw.githubusercontent.com/kaisersama112/typora_image/master/f99b9f5b6ff94e972a16b2b9fac1d0e4.png)

​	一般来说，语言模型都是为了使得条件概率：![P(w_t|w_1,w_2,..,w_{t-1})](https://latex.csdn.net/eq?P%28w_t%7Cw_1%2Cw_2%2C..%2Cw_%7Bt-1%7D%29)最大化，不过考虑到近因效应，当前词只与距离它比较近的![n](https://latex.csdn.net/eq?n)个词更加相关(一般![n](https://latex.csdn.net/eq?n)不超过5)，而非前面所有的词都有关。

因此上述公式可以近似为：

![P\left(w_{t} \mid w_{1}, w_{2}, \ldots, w_{t-1}\right)=P\left(w_{t} \mid w_{t-1}, w_{t-2} \ldots w_{t-(n-1)}\right)](https://latex.csdn.net/eq?P%5Cleft%28w_%7Bt%7D%20%5Cmid%20w_%7B1%7D%2C%20w_%7B2%7D%2C%20%5Cldots%2C%20w_%7Bt-1%7D%5Cright%29%3DP%5Cleft%28w_%7Bt%7D%20%5Cmid%20w_%7Bt-1%7D%2C%20w_%7Bt-2%7D%20%5Cldots%20w_%7Bt-%28n-1%29%7D%5Cright%29)

上述便是经典的n-gram模型的表示方式

不过，N-gram模型仍有其局限性

- [ ] **首先，由于参数空间的爆炸式增长，它无法处理更长程的context（N>3）**
- [ ] **其次，它没有考虑词与词之间内在的联系性**

​	例如，考虑"the cat is walking in the bedroom"这句话如果我们在训练语料中看到了很多类似“the dog is walking in the bedroom”或是“the cat is running in the bedroom”这样的句子，那么，哪怕我们此前没有见过这句话"the cat is walking in the bedroom"，也可从“cat”和“dog”（“walking”和“running”）之间的相似性，推测出这句话的概率

​	然而Ngram模型做不到这是因为Ngram本质上是将词当做一个个孤立的原子单元(atomic unit)去处理的。这种处理方式对应到数学上的形式是一个个离散的one-hot向量 (除了词典索引的下标对应的方向上是1，其余方向上都是0)
​	例如，对于一个大小为5的词典：{"I", "love", "nature", "language", "processing"}，其中的i、love、nature、language、processing分别对应的one-hot向量为：

| i          | 1    | 0    | 0    | 0    | 0    |
| ---------- | ---- | ---- | ---- | ---- | ---- |
| love       | 0    | 1    | 0    | 0    | 0    |
| nature     | 0    | 0    | 1    | 0    | 0    |
| language   | 0    | 0    | 0    | 1    | 0    |
| processing | 0    | 0    | 0    | 0    | 1    |

显然，one-hot向量的维度等于词典的大小



##  三、神经语言模型：NNLM模型

​	NNLM最初由Bengio在2003年发表的一篇论文《A Neural Probabilistic Language Mode》中提出来的，word2vec便是从其中简化训练而来。

​	假设我们现在有一个**词表![D](https://latex.csdn.net/eq?D)，它的大小为![N](https://latex.csdn.net/eq?N)**（相当于总共有![N](https://latex.csdn.net/eq?N)个词，词用![w](https://latex.csdn.net/eq?w)表示，比如![N](https://latex.csdn.net/eq?N)为一万，则是一万个词），词表里每个词![w](https://latex.csdn.net/eq?w)的维度为![m](https://latex.csdn.net/eq?m)，即![C\epsilon R^{Nm}](https://i-blog.csdnimg.cn/blog_migrate/3d3d4044705996e6f9201efde3839e3e.gif)，如果词的输入是one hot编码，则![N = m](https://latex.csdn.net/eq?N%20%3D%20m)，另外![n](https://latex.csdn.net/eq?n)表示词![w](https://latex.csdn.net/eq?w)的上下文中包含的词数，不超过5。

​	Bengio通过下面的一个三层神经网络来计算![P(w_t | w_{t-1},w_{t-2} \cdots w_{t - n+1})](https://latex.csdn.net/eq?P%28w_t%20%7C%20w_%7Bt-1%7D%2Cw_%7Bt-2%7D%20%5Ccdots%20w_%7Bt%20-%20n&plus;1%7D%29)：

![img](https://raw.githubusercontent.com/kaisersama112/typora_image/master/4a568f85afa867d4dd9c8a644076e33c.png)

- [ ] 首先第一层输入就是前![n-1](https://latex.csdn.net/eq?n-1)个词“![w_{t-n+1},\cdots ,w_{t-2},w_{t-1}](https://latex.csdn.net/eq?w_%7Bt-n&plus;1%7D%2C%5Ccdots%20%2Cw_%7Bt-2%7D%2Cw_%7Bt-1%7D)”去预测第 ![t](https://latex.csdn.net/eq?t) 个词是 ![w_t](https://latex.csdn.net/eq?w_t) 的概率；
- [ ] 然后根据输入的前![n-1](https://latex.csdn.net/eq?n-1)个词，在同一个词汇表![D](https://latex.csdn.net/eq?D)中一一找到它们对应的词向量；
- [ ] 最后把所有词向量直接串联起来成为一个维度为![(n-1)m](https://latex.csdn.net/eq?%28n-1%29m)的向量![x](https://latex.csdn.net/eq?x) 作为接下来三层神经网络的输入 (注意这里的“串联”，其实就是![n-1](https://latex.csdn.net/eq?n-1)个向量按顺序首尾拼接起来形成一个长向量)
- [ ] 隐藏层到输出层之间有大量的矩阵向量计算，在输出层之后还需要做softmax归一化计算 (*使用softmax函数归一化输出之后的值是[0,1]，代表可能的每个词的概率。*)



> 常用于神经网络输出层的激励函数softmax 长什么样子呢？如下图所示
>
> ![img](https://raw.githubusercontent.com/kaisersama112/typora_image/master/2a7c62005eda626ad2e4a22091acf1d3.png)
>
> 从图的样子上看，和普通的全连接方式并无差异，但激励函数的形式却大不一样
>
> ![img](https://raw.githubusercontent.com/kaisersama112/typora_image/master/ba986619d94ebc1c4c7483f020e57f15.png)
>
> ![img](https://raw.githubusercontent.com/kaisersama112/typora_image/master/2a3bbfd718b07c0abaf65696d3a9b0eb.png)
>
> 首先后面一层作为预测分类的输出节点，每一个节点就代表一个分类，如图所示，那么这7个节点就代表着7个分类的模型，任何一个节点的激励函数都是：
>
> ![\sigma_{i}(z)=\frac{e^{z_{i}}}{\sum_{j=1}^{m} e^{z_{i}}}](https://latex.csdn.net/eq?%5Csigma_%7Bi%7D%28z%29%3D%5Cfrac%7Be%5E%7Bz_%7Bi%7D%7D%7D%7B%5Csum_%7Bj%3D1%7D%5E%7Bm%7D%20e%5E%7Bz_%7Bi%7D%7D%7D)
>
> 其中i 就是节点的下标次序，而![z_{i} = w_{i}x + b_{i}](https://i-blog.csdnimg.cn/blog_migrate/eb71626343e23b8a8b031303f748efa4.gif)，也就说这是一个线性分类器的输出作为自然常数e的指数。最有趣的是最后一层有这样的特性：
>
> ![\sum_{i=1}^{J} \sigma_{i}(z)=1](https://latex.csdn.net/eq?\sum_{i%3D1}^{J} \sigma_{i}(z)%3D1)
>
> 也就是说最后一层的每个节点的输出值的加和是1。这种激励函数从物理意义上可以解释为一个样本通过网络进行分类的时候在每个节点上输出的值都是小于等于1的，是它从属于这个分类的概率
>
> 训练数据由训练样本和分类标签组成，如下图所示，假设有7张图，分别为飞机、汽车、轮船、猫、狗、鸟、太阳，则图像的分类标签如下表示：
>
> | 飞机 | 1    | 0    | 0    | 0    | 0    | 0    | 0    |
> | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
> | 汽车 | 0    | 1    | 0    | 0    | 0    | 0    | 0    |
> | 轮船 | 0    | 0    | 1    | 0    | 0    | 0    | 0    |
> | 猫   | 0    | 0    | 0    | 1    | 0    | 0    | 0    |
> | 狗   | 0    | 0    | 0    | 0    | 1    | 0    | 0    |
> | 鸟   | 0    | 0    | 0    | 0    | 0    | 1    | 0    |
> | 太阳 | 0    | 0    | 0    | 0    | 0    | 0    | 1    |
>
> 这种激励函数通常用在神经网络的最后一层作为分类器的输出，有7个节点就可以做7个不同类别的判别，有1000个节点就可以做1000个不同样本类别的判断

神经语言模型构建完成之后，就是训练参数了，这里的参数包括：

- **词向量矩阵C；**
- 神经网络的权重；
- 偏置等参数

​	训练数据就是大堆大堆的语料库。训练结束之后，语言模型得到了：通过“![w_{t-(n-1)},\cdots ,w_{t-2},w_{t-1}](https://latex.csdn.net/eq?w_%7Bt-%28n-1%29%7D%2C%5Ccdots%20%2Cw_%7Bt-2%7D%2Cw_%7Bt-1%7D)”去预测第 ![t](https://latex.csdn.net/eq?t) 个词是 ![w_t](https://latex.csdn.net/eq?w_t) 的概率，但有点意外收获的是词向量“ ![w_{t-(n-1)},\cdots ,w_{t-2},w_{t-1}](https://latex.csdn.net/eq?w_%7Bt-%28n-1%29%7D%2C%5Ccdots%20%2Cw_%7Bt-2%7D%2Cw_%7Bt-1%7D)”也得到了。换言之，词向量是这个语言模型的副产品

​	当然，这个模型的缺点就是速度问题, 因为词汇表往往很大，几十万几百万，训练起来就很耗时，Bengo仅仅训练5个epoch就花了3周，这还是40个CPU并行训练的结果。因此才会有了后续好多的优化工作, word2vec便是其中一个



## 三、什么是词嵌入

​	在nlp当中，最细粒度的是词语，词语组成句子，句子组成段落，文章，文档。

​	各个国家的人们通过各自的语言进行交流，但机器无法直接理解人类的语言，所以需要先把人类的语言“计算机化”，那如何变成计算机可以理解的语言呢？

![img](https://raw.githubusercontent.com/kaisersama112/typora_image/master/03a5e028e17080444f83367a4b6cfb9d.png)

​	我们可以从另外一个角度上考虑。举个例子，对于计算机，它是如何判断一个词的词性，是动词还是名词的呢？

​	我们有一系列样本(x,y)，对于计算机技术机器学习而言，这里的 x 是词语，y 是它们的词性，我们要构建 f(x)->y 的映射：

- [ ] 首先，这个数学模型 f（比如神经网络、SVM）只接受数值型输入；
- [ ] 而 NLP 里的词语，是人类语言的抽象总结，是符号形式的（比如中文、英文、拉丁文等等）;
- [ ] 如此一来，咱们便需要把NLP里的词语转换成数值形式，或者嵌入到一个数学空间里；
- [ ] 我们可以把文本分散嵌入到另一个离散空间，称作分布式表示，又称为词嵌入（word embedding）或词向量
- [ ] 一种简单的词向量是one-hot encoder，其思想跟特征工程里处理类别变量的 one-hot 一样 (如之前所述，本质上是用一个只含一个 1、其他都是 0 的向量来唯一表示词语)

![img](https://raw.githubusercontent.com/kaisersama112/typora_image/master/4f46086cdb738b65e779fb5e82b32248.png)

### Word2Vec

​	当然传统的one-hot 编码仅仅只是将词符号化，不包含任何语义信息，而且词的独热表示（one-hot representation）是高维的，且在高维向量中中只有一个维度描述了词的语义（高到什么程度呢？词典有多大就有多少维，一般至少上万的维度），所以我们需要解决两个问题：**1. 需要赋予词语义信息，2. 降低维度。**

> word2vec是Google研究团队里的Tomas Mikolov等人于2013年的《Distributed Representations ofWords and Phrases and their Compositionality》以及后续的《Efficient Estimation of Word Representations in Vector Space》两篇文章中提出的一种高效训练词向量的模型，基本出发点是**上下文相似的两个词，它们的词向量也应该相似**，比如香蕉和梨在句子中可能经常出现在相同的上下文中，因此这两个词的表示向量应该就比较相似
>

大部分的有监督机器学习模型，都可以归结为：![f(x) \rightarrow y](https://raw.githubusercontent.com/kaisersama112/typora_image/master/eq)

- 在部分nlp问题当中，我们将x 看做一个句子里面的一个词语，y 是这个词语的上下文词语，在这里f便是上文中所谓的『语言模型』（language model），这个语言模型的目的是判断（x,y)这个样本是否符合自然语言的法则，有了语言模型，我们便可以判断出：词语x和词语y放在一起是否是正确的。
- 当然，前面也说了，这个语言模型还得到了一个副产品：词向量矩阵 而对于word2vec 而言，词向量矩阵的意义就不一样了，因为Word2Vec的最终目的不是为了得到一个语言模型，也不是要把 ![f](https://latex.csdn.net/eq?f)训练得多么完美，而是**只关心模型训练完后的副产物：模型参数(这里特指神经网络的权重)，*并将这些参数作为输入 ![x](https://latex.csdn.net/eq?x) 的某种向量化的表示，这个向量便叫做——词向量***

