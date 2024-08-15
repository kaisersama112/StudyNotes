## pth 转换为onnx

一般来讲模型都是采用其他深度学习框架（pytorch,tensorflow）进行训练得到类似于`.pth`类型模型文件，对于移动端是不能直接使用的，我们需要先将模型转换为中间层模型`onnx`然后转换为对应的边缘推理框架支持的模型结构(`.pt`与`.pth`模型具有不同的转换方式)，这里只本项目中用到的`pth`类型文件信息。

**人脸检测网络结构：**

```python
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F

from nets.inception_resnetv1 import InceptionResnetV1
from nets.mobilenet import MobileNetV1

# 骨干网络
class mobilenet(nn.Module):
    def __init__(self, pretrained):
        super(mobilenet, self).__init__()
        self.model = MobileNetV1()
        if pretrained:
            state_dict = load_state_dict_from_url("https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/backbone_weights_of_mobilenetv1.pth", model_dir="model_data",
                                                progress=True)
            self.model.load_state_dict(state_dict)

        del self.model.fc
        del self.model.avg

    def forward(self, x):
        x = self.model.stage1(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        return x

class inception_resnet(nn.Module):
    def __init__(self, pretrained):
        super(inception_resnet, self).__init__()
        self.model = InceptionResnetV1()
        if pretrained:
            state_dict = load_state_dict_from_url("https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/backbone_weights_of_inception_resnetv1.pth", model_dir="model_data",
                                                progress=True)
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        x = self.model.conv2d_1a(x)
        x = self.model.conv2d_2a(x)
        x = self.model.conv2d_2b(x)
        x = self.model.maxpool_3a(x)
        x = self.model.conv2d_3b(x)
        x = self.model.conv2d_4a(x)
        x = self.model.conv2d_4b(x)
        x = self.model.repeat_1(x)
        x = self.model.mixed_6a(x)
        x = self.model.repeat_2(x)
        x = self.model.mixed_7a(x)
        x = self.model.repeat_3(x)
        x = self.model.block8(x)
        return x

     
# 特征提取网络结构
class Facenet(nn.Module):
    def __init__(self, backbone="mobilenet", dropout_keep_prob=0.5, embedding_size=128, num_classes=None, mode="train", pretrained=False):
        super(Facenet, self).__init__()
        if backbone == "mobilenet":
            self.backbone = mobilenet(pretrained)
            flat_shape = 1024
        elif backbone == "inception_resnetv1":
            self.backbone = inception_resnet(pretrained)
            flat_shape = 1792
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1.'.format(backbone))
        self.avg        = nn.AdaptiveAvgPool2d((1,1))
        self.Dropout    = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size,bias=False)
        self.last_bn    = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        if mode == "train":
            self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x, mode = "predict"):
        if mode == 'predict':
            x = self.backbone(x)
            x = self.avg(x)
            x = x.view(x.size(0), -1)
            x = self.Dropout(x)
            x = self.Bottleneck(x)
            x = self.last_bn(x)
            x = F.normalize(x, p=2, dim=1)
            return x
        x = self.backbone(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        
        x = F.normalize(before_normalize, p=2, dim=1)
        cls = self.classifier(before_normalize)
        return x, cls

    def forward_feature(self, x):
        x = self.backbone(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        x = F.normalize(before_normalize, p=2, dim=1)
        return before_normalize, x

    def forward_classifier(self, x):
        x = self.classifier(x)
        return x

```

**模型导出onnx：**

```cmd
import torch
import torch.backends.cudnn as cudnn
from facenet import facenet


class FacenetONNXExporter:
    def __init__(self, model_path, backbone="mobilenet", cuda=False):
        self.model_path = model_path
        self.backbone = backbone
        self.cuda = cuda
        self.net = None

    def generate(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = facenet(backbone=self.backbone, mode="predict").eval()
        self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()
    def export_to_onnx(self, onnx_file_path="facenet_model.onnx"):
        if self.net is None:
            raise ValueError("Model is not generated. Call `generate()` first.")
        model_to_export = self.net.module if isinstance(self.net, torch.nn.DataParallel) else self.net
        # 模型输入尺寸
        input_shape = (3, 112, 112)
        dummy_input = torch.randn(1, *input_shape)
        if self.cuda:
            dummy_input = dummy_input.cuda()
        torch.onnx.export(
            model_to_export,
            dummy_input,
            onnx_file_path,
            verbose=True,
            # 输入层名称
            input_names=["input"],
            # 输出层名称
            output_names=["output"],
            opset_version=11
        )

def exporter_onnx():
    exporter = FacenetONNXExporter(model_path="facenet_mobilenet.pth", backbone="mobilenet",
                                   cuda=torch.cuda.is_available())
    exporter.generate()    
    exporter.export_to_onnx("facenet_model.onnx")

if __name__ == "__main__":
    exporter_onnx()
```



##  模型结构查看

对于onnx模型可通过网址（[Netron](https://netron.app/)）进行在线查看模型输入输出结构以及中间层参数信息：

![image-20240815164931712](https://raw.githubusercontent.com/kaisersama112/typora_image/master/image-20240815164931712.png)

模型输入：tensor: `**float32[1,3,112,112]**`

模型输出：tensor: `**float32[1,128]**`

剪枝优化就是通过对中间层级进行优化，剔除部分中间层，减少计算量，通过牺牲部分精度换取执行效率。量化操作也是同样如此

##  模型转换（onnxToncnn）

通过`onnxtoncnn`进行转换具体命令如下：

`.param`:模型结构

`.bin`： 模型参数

```cmd
onnx2ncnn.exe face_recognition_sface.onnx face_recognition_sface.param face_recognition_sface.bin
```



##  校准表生成

根据验集图像进行创建,验证集文件存在于目录中`imagenet-sample`

根据验证集创建`imagelist.txt`图片路径文件

```cmd
Get-ChildItem -Recurse -File imagenet-sample\* | ForEach-Object { $_.FullName } > imagelist.txt
```



```cmd
ncnn2table.exe face_recognition_sface.param face_recognition_sface.bin imagelist.txt mobilenet.table mean=[127.5,127.5,127.5] norm=[1/128,1/128,1/128] shape=[112,112,3] pixel=BGR thread=8 metho
d=kl
```

## 量化模型

```cmd
ncnn2int8.exe face_recognition_sface.param face_recognition_sface.bin face_recognition_sface_int8.param face_recognition_sface_int8.bin mobilenet.table

```



