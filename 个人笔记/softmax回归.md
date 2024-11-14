## 1. 独热编码（one‐hot encoding）

​	注意在分类问题当中 分类问题不与类别之间的自然顺序有关采取措施使:用独热编码（one‐hot encoding）为了估计所有可能类别的条件概率，我们需要一个有多个输出的模型，每个类别对应一个输出。为了解决线性模型的分类问题，我们需要和输出一样多的仿射函数（affine function）。每个输出对应于它自己的仿射函
数。在我们的例子中，由于我们有4个特征和3个可能的输出类别，我们将需要12个标量来表示权重（带下标的w），3个标量来表示偏置（带下标的b）。

下面我们为每个输入计算三个未规范化的预测（logit）：o1、o2和o3。

- `o1 = x1w11 + x2w12 + x3w13 + x4w14 + b1`,
- `o2 = x1w21 + x2w22 + x3w23 + x4w24 +  b2`,
- `o3 = x1w31 + x2w32 + x3w33 + x4w34 + b3`,

我们可以用神经网络图来描述这个计算过程。与线性回归一样，softmax回归也是一个单层神经网络。
由于计算每个输出o1、o2和o3取决于所有输入x1、x2、x3和x4，所以softmax回归的输出层也是全连接层。

## 2. softmax运算

​	现在我们将优化参数以最大化观测数据的概率。为了得到预测结果，我们将设置一个阈值，如选择具有最大概率的标签。
​    我们希望模型的输出yˆj可以视为属于类j的概率，然后选择具有最大输出值的类别argmax_j,y_j作为我们的预测。例如，如果yˆ1、yˆ2和yˆ3分别为0.1、0.8和0.1，那么我们预测的类别是2，在我们的例子中代表“鸡”。然而我们能否将未规范化的预测o直接视作我们感兴趣的输出呢？答案是否定的。

​	因为将线性层的输出直接视为概率时存在一些问题：一方面，我们没有限制这些输出数字的总和为1。另一方面，根据输入的不同，它们可以为负值。要将输出视为概率，我们必须保证在任何数据上的输出都是非负的且总和为1。

​	此外，我们需要一个训练的目标函数，来激励模型精准地估计概率。例如，在分类器输出0.5的所有样本中，我们希望这些样本是刚好有一半实际上属于预测的类别。这个属性叫做校准（calibration）。
​	校准（calibration）:为了确保输出非负,且最终输出的概率值总和为1，对每个未规范化的预测求幂，每个求幂后的结果除以它们的总和。最终对于所有的j总有0 ≤ yˆj ≤ 1。因此，yˆ可以视为一个正确的概率分布。

• softmax运算获取一个向量并将其映射为概率。
• softmax回归适用于分类问题，它使用了softmax运算中输出类别的概率分布。
• 交叉熵是一个衡量两个概率分布之间差异的很好的度量，它测量给定模型编码数据所需的比特数。

## 3. Fashion‐MNIST数据集 

​	MNIST数据集 (LeCun et al., 1998) 是图像分类中广泛使用的数据集之一，但作为基准数据集过于简单。我们将使用类似但更复杂的Fashion‐MNIST数据集 (Xiao et al., 2017)。


```python

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()
```


```python
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)

```


```python
print(mnist_train.data.shape)
print(len(mnist_train))
print(len(mnist_test))
```

    torch.Size([60000, 28, 28])
    60000
    10000



```python
def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

```


```python
def show_iamges(imgs, num_rows, num_cols, titles=None, scale=4):
    figsize = (num_rows * scale, num_cols * scale * 0.1)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, image) in enumerate(zip(axes, imgs)):

        if torch.is_tensor(image):
            ax.imshow(image.numpy())
        else:
            ax.imshow(image)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles is not None:
            ax.set_title(titles[i])
    return axes


x, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_iamges(x.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
```




    array([<Axes: title={'center': 'ankle boot'}>,
           <Axes: title={'center': 't-shirt'}>,
           <Axes: title={'center': 't-shirt'}>,
           <Axes: title={'center': 'dress'}>,
           <Axes: title={'center': 't-shirt'}>,
           <Axes: title={'center': 'pullover'}>,
           <Axes: title={'center': 'sneaker'}>,
           <Axes: title={'center': 'pullover'}>,
           <Axes: title={'center': 'sandal'}>,
           <Axes: title={'center': 'sandal'}>,
           <Axes: title={'center': 't-shirt'}>,
           <Axes: title={'center': 'ankle boot'}>,
           <Axes: title={'center': 'sandal'}>,
           <Axes: title={'center': 'sandal'}>,
           <Axes: title={'center': 'sneaker'}>,
           <Axes: title={'center': 'ankle boot'}>,
           <Axes: title={'center': 'trouser'}>,
           <Axes: title={'center': 't-shirt'}>], dtype=object)




![svg](https://raw.githubusercontent.com/kaisersama112/typora_image/master/output_7_1.svg)
    



```python
# 读取小批量
batch_size = 256


def get_data_loader_workers(num=4):
    return num


train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=get_data_loader_workers())

timer = d2l.Timer()
for x, y in train_iter:
    continue
print(f'{timer.stop():.2f}sec')



```

    3.85sec



```python
# softmax回归

import torch
from IPython import display
from d2l import torch as d2l


def load_data_fashion_mnist(batch_size=128, resize=None):
    trans = [transforms.ToTensor()]
    if resize is not None:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
    return (
        data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=get_data_loader_workers()),
        data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=get_data_loader_workers()))


batch_size = 256
image_resize = 28
train_iter, test_iter = load_data_fashion_mnist(64, resize=image_resize)

num_inputs = image_resize ** 2
num_outputs = 10
"""
在softmax回归中，输出与类别一样多。我们的数据集有10个类别，所以网络输出维度
为10。权重将构成一个784 × 10的矩阵，偏置将构成一个1 × 10的行向量。与线性回归一样，我们将使
用正态分布初始化我们的权重W，偏置初始化为0。
"""
w = torch.normal(0, 0.1, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

```


```python

def softmax(x):
    """
    分母或规范化常数/配分函数
    """
    x_exp = torch.exp(x)
    partition = x_exp.sum(dim=1, keepdim=True)
    return x_exp / partition


def net(x):
    x_reshape = x.reshape((-1, w.shape[0]))
    # print(x_reshape.shape)
    return softmax(torch.matmul(x_reshape, w) + b)


"""
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(y_hat[[0, 1], y])

output:
tensor([0.1000, 0.5000]
交叉熵采用真实标签的预测概率的负对数似然。这里我们通过一个运算符选择所有元素
我们创建一个数据样本y_hat，其中包含2个样本在3个类别的预测概率，以及它们对应的标签y。
有了y，
我们知道在第一个样本中，第一类是正确的预测；
而在第二个样本中，第三类是正确的预测。
然后使用y作为y_hat中概率的索引，我们选择第一个样本中第一个类的概率和第二个样本中第三个类的概率。
"""


def cross_entropy_loss(y_hat, y):
    """
    交叉熵损失
    """
    predicted_probs = y_hat[range(len(y_hat)), y]
    return -torch.log(predicted_probs)


"""
分类精度
给定预测概率分布y_hat，当我们必须输出硬预测（hard prediction）时，我们通常选择预测概率最高的类。
当预测与标签分类y一致时，即是正确的。分类精度即正确预测数量与总预测数量之比。虽然直接优化精度可能很困难（因为精度的计算不可导），但精度通常是我们最关心的性能衡量标准，我们在训练分类器时几乎总会关注它。
为了计算精度，我们执行以下操作。
首先，如果y_hat是矩阵，那么假定第二个维度存储每个类的预测分数。
我们使用argmax获得每行中最大元素的索引来获得预测类别。然后我们将预测类别与真实y元素进行比较。由于等式运算符“==”对数据类型很敏感，因此我们将y_hat的数据类型转换为与y的数据类型一致。结果是一个包含0（错）和1（对）的张量。最后，我们求和会得到正确预测的数量。
"""


def accuracy(y_hat, y):
    """
    模型预测准确率
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    cmp_type = cmp.type(y.dtype)
    return float(cmp_type.sum())


"""
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])

print(y_hat[[0, 1], y])
print(cross_entropy_loss(y_hat, y))
print(accuracy(y_hat, y) / len(y))
"""


class Accumulator:  #@save
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]

        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)
    with torch.no_grad():
        for x, y in data_iter:
            metric.add(accuracy(net(x), y), y.numel())
    return metric[0] / metric[1]


"""
evaluate_accuracy(net, test_iter)
"""


def train_epoch(net, train_iter, loss, updater):
    """
        
    """
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for x, y in train_iter:
        y_hat = net(x)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(x.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    print(f'metric:{metric.data}')
    return metric[0] / metric[2], metric[1] / metric[2]


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(train_metrics)
        print(test_acc)
        animator.add(epoch + 1, train_metrics + (test_acc,))

    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


def sgd(params, lr, batch_size):
    """
    小批量梯度下降
    :param params: 数据集
    :param lr: 学习率
    :param batch_size:  
    :return: 
    """
    # torch.no_grad() 包裹的不会保存梯度
    with torch.no_grad():
        for param in params:
            #更新参数值w 学习率*每个参数导数值
            param -= lr * param.grad / batch_size
            # 梯度清零
            param.grad.zero_()


lr = 0.1


def updater(batch_size):
    return sgd([w, b], lr, batch_size)


num_epochs = 5
train(net=net, train_iter=train_iter, test_iter=test_iter, loss=cross_entropy_loss, num_epochs=num_epochs,
      updater=updater)
```


![svg](https://raw.githubusercontent.com/kaisersama112/typora_image/master/output_10_0.svg)
    



```python
def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        print(x.shape)
        print(y.shape)
        break

    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


predict_ch3(net, test_iter)
```

    torch.Size([96, 1, 28, 28])
    torch.Size([64])




![svg](https://raw.githubusercontent.com/kaisersama112/typora_image/master/output_11_1.svg)
    


## 4. 代码分析
x[0].shape=[28,28] `28*28`的一个图像
y.shape=[1,10] 预测值为`1*10`的tensor
w.shape=[784,10] 

### 1. def softmax(X):
![image-Snipaste_2024-04-12_11-47-42.png](./assets/Snipaste_2024-04-12_11-47-42.png)
回想一下，实现softmax由三个步骤组成：
    1. 对每个项求幂（使用exp）；
    2. 对每一行求和（小批量中每个样本是一行），得到每个样本的规范化常数；
    3. 将每一行除以其规范化常数，确保结果的和为1。
而这个函数就是对这个公式的实现，通过上述公式实现对参数集tensor(x)实现对每个元素进行非负数转换，将一个实数向量转换为概率分布，使得每个元素都在0到1之间，并且所有元素的和为1。

### 2. def net(x):
这个函数定义了输入如何通过网络映射到输出(y=wx+b):注意这里将输入进行了展平处理为一个向量((x.reshape((-1, w.shape[0]))) 这时候的x[0]的形状为`(784)`

### 3. def cross_entropy_loss(y_hat, y):
交叉熵损失函数：用来度量预测值与真实值的差距

### 4. def accuracy(y_hat, y):
分类精度函数：正确预测数量与总预测数量之比

### 5. def evaluate_accuracy(net, data_iter):
计算在指定数据集上模型的精度


### 6. def train_epoch(net, train_iter, loss, updater):
这里定义了一个迭代周期函数，大致步骤如下：
- net函数得到预测值
- loss函数度量损失值
- sgd计算并更新梯度

### 7. def train(net, train_iter, test_iter, loss, num_epochs, updater):
训练函数将会运行多个迭代周期（由num_epochs指定）。在每个迭代周期结束时，利用test_iter访问到的测试数据集对
模型进行评估。并利用Animator类来可视化训练进度。

### 8. def evaluate_accuracy(net, data_iter):
模型评估函数

## 重新审视Softmax的实现
    在前面的例子中，我们计算了模型的输出，然后将此输出送入交叉熵损失。从数学上讲，这是一件完全
合理的事情。然而，从计算角度来看，指数可能会造成数值稳定性问题。
    回想一下，softmax函数![image-Snipaste_2024-04-12_14-41-02.png](./assets/Snipaste_2024-04-12_14-41-02.png)
    其中yˆj是预测的概率分布。oj是未规范化的预测o的第j个元素。如
果ok中的一些数值非常大，那么exp(ok)可能大于数据类型容许的最大数字，即上溢（overflow）。这将使分
母或分子变为inf（无穷大），最后得到的是0、inf或nan（不是数字）的yˆj。在这些情况下，我们无法得到一
个明确定义的交叉熵值。
    解决这个问题的一个技巧是：在继续softmax计算之前，先从所有ok中减去max(ok)。这里可以看到每个ok按
常数进行的移动不会改变softmax的返回值：
![image-Snipaste_2024-04-12_14-41-37.png](./assets/Snipaste_2024-04-12_14-41-37.png)
    在减法和规范化步骤之后，可能有些oj − max(ok)具有较大的负值。由于精度受限，exp(oj − max(ok))将有
接近零的值，即下溢（underflow）。这些值可能会四舍五入为零，使yˆj为零，并且使得log(ˆyj )的值为-inf。反
向传播几步后，我们可能会发现自己面对一屏幕可怕的nan结果。
    尽管我们要计算指数函数，但我们最终在计算交叉熵损失时会取它们的对数。通过将softmax和交叉熵结
合在一起，可以避免反向传播过程中可能会困扰我们的数值稳定性问题。如下面的等式所示，我们避免计
算exp(oj − max(ok))，而可以直接使用oj − max(ok)，因为log(exp(·))被抵消了。
![image-Snipaste_2024-04-12_14-42-31.png](./assets/Snipaste_2024-04-12_14-42-31.png)
    我们也希望保留传统的softmax函数，以备我们需要评估通过模型输出的概率。但是，我们没有将softmax概
率传递到损失函数中，而是在交叉熵损失函数中传递未规范化的预测，并同时计算softmax及其对数，这是
一种类似`LogSumExp技巧`的聪明方式。



```python
# 上面是我们手动实现的softmax ，现在通过torch实现一遍
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal(m.weight, std=0.01)


net.apply(init_weights)
# 损失函数
loss = nn.CrossEntropyLoss(reduction='none')
# 优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 10
# 训练函数直接套用之前实现的
train(net, train_iter, test_iter, loss, num_epochs, trainer)

```


​    ![svg](https://raw.githubusercontent.com/kaisersama112/typora_image/master/output_14_0.svg)
​    


