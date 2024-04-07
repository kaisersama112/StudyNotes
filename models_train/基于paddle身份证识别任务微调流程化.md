

# ocr（身份证）



## 前置数据标注

使用paddleoc的标注工具实现对身份证数据各个字段进行标注

```cmd
python PPOCRLabel.py --lang ch  --kie True
```

使用软件标注完成后 使用数据集切分脚本划分训练集 测试集 验证集

```cm
python gen_ocr_train_val_test.py --trainValTestRatio 8:2:0 --datasetRootPath ../train_data/drivingData 
```

- （1）SER: 语义实体识别 (Semantic Entity Recognition)，对每一个检测到的文本进行分类，如将其分为姓名，身份证。如下图中的黑色框和红色框。
- （2）RE: 关系抽取 (Relation Extraction)，对每一个检测到的文本进行分类，如将其分为问题 (key) 和答案 (value) 。然后对每一个问题找到对应的答案，相当于完成key-value的匹配过程。如下图中的红色框和黑色框分别代表问题和答案，黄色线代表问题和答案之间的对应关系。

###  1. 关键信息抽取任务流程

PaddleOCR中实现了LayoutXLM等算法（基于Token），同时，在PP-StructureV2中，对LayoutXLM多模态预训练模型的网络结构进行简化，去除了其中的Visual backbone部分，设计了视觉无关的VI-LayoutXLM模型，同时引入符合人类阅读顺序的排序逻辑以及UDML知识蒸馏策略，最终同时提升了关键信息抽取模型的精度与推理速度。

在非End-to-end的KIE方法中，完成关键信息抽取，至少需要**2个步骤**：首先使用OCR模型，完成文字位置与内容的提取，然后使用KIE模型，根据图像、文字位置以及文字内容，提取出其中的关键信息。



### 2.  文本检测模型微调

##### 2-1数据

PaddleOCR中提供的模型大多数为通用模型，在进行文本检测的过程中，相邻文本行的检测一般是根据位置的远近进行区分，如上图，使用PP-OCRv3通用中英文检测模型进行文本检测时，容易将”民族“与“汉”这2个代表不同的字段检测到一起，从而增加后续KIE任务的难度。因此建议在做KIE任务的过程中，首先训练一个针对该文档数据集的检测模型。

在数据标注时，关键信息的标注需要隔开，比上图中的 “民族汉” 3个字相隔较近，此时需要将”民族“与”汉“标注为2个文本检测框，否则会增加后续KIE任务的难度。

对于下游任务，一般来说，`200~300`张的文本训练数据即可保证基本的训练效果，如果没有太多的先验知识，可以先标注 **`200~300`** 张图片，进行后续文本检测模型的训练。

##### 2-2 下载预训练模型

[ch_PP-OCRv4_det_train](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_train.tar)


pretrained_model: https://paddleocr.bj.bcebos.com/pretrained/PPLCNetV3_x0_75_ocr_det.pdparams


##### 2-3 参数配置

配置文件： configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml

##### 2-4 训练

```cmd
python tools/train.py -c ./configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml -o Global.pretrained_model=./pretrain_models/ch_PP-OCRv3_det_distill_train
```

##### 2-5 评估

```cmd
python tools/eval.py -c configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml -o Global.checkpoints="{path/to/weights}/best_accuracy"

```

##### 2-6 推理

```cmd
python tools/infer_det.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml -o Global.infer_img="./train_data/det/val/image_125821041857.jpg" Global.pretrained_model="./output/ch_PP-OCR_V3_det/best_accuracy"

```

##### 2-7 导出

```cmd
python tools/export_model.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml -o Global.pretrained_model="./output/ch_PP-OCR_V3_det/best_accuracy" Global.save_inference_dir="./output/det_db_inference/"

```

##### 2-8 模型继续训练

```cmd
python tools/train.py -c configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml -o Global.pretrained_model=./pretrain_models/MobileNetV3_large_x0_5_pretrained -o Global.checkpoints=output/rec_ppocr_v4/it
er_epoch_60

```



### 3. 文本识别模型微调

相对自然场景，文档图像中的文本内容识别难度一般相对较低（背景相对不太复杂），因此**优先建议**尝试PaddleOCR中提供的PP-OCRv3通用文本识别模型([PP-OCRv3模型库链接](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/models_list.md))。

##### 3-1 数据

训练集&校验集
建议将训练图片放入同一个文件夹，并用一个txt文件（rec_gt_train.txt）记录图片路径和标签，txt文件里的内容如下:

注意： txt文件中默认请将图片路径和图片标签用 \t 分割，如用其他方式分割将造成训练报错。

```cmd
" 图像文件名                 图像标注信息 "

train_data/rec/train/word_001.jpg   简单可依赖
train_data/rec/train/word_002.jpg   用科技让复杂的世界更简单
...
```

最终训练集应有如下文件结构：

```cmd
|-train_data
  |-rec
    |- rec_gt_train.txt
    |- train
        |- word_001.png
        |- word_002.jpg
        |- word_003.jpg
        | ...
```

除上述单张图像为一行格式之外，PaddleOCR也支持对离线增广后的数据进行训练，为了防止相同样本在同一个batch中被多次采样，我们可以将相同标签对应的图片路径写在一行中，以列表的形式给出，在训练中，PaddleOCR会随机选择列表中的一张图片进行训练。对应地，标注文件的格式如下。

```cmd

["11.jpg", "12.jpg"]   简单可依赖
["21.jpg", "22.jpg", "23.jpg"]   用科技让复杂的世界更简单
3.jpg   ocr

```





上述示例标注文件中，"11.jpg"和"12.jpg"的标签相同，都是简单可依赖，在训练的时候，对于该行标注，会随机选择其中的一张图片进行训练。

如果有通用真实场景数据加进来，建议每个epoch中，垂类场景数据与真实场景的数据量保持在1:1左右。

比如：您自己的垂类场景识别数据量为1W，数据标签文件为vertical.txt，收集到的通用场景识别数据量为10W，数据标签文件为general.txt，

那么，可以设置label_file_list和ratio_list参数如下所示。每个epoch中，vertical.txt中会进行全采样（采样比例为1.0），包含1W条数据；general.txt中会按照0.1的采样比例进行采样，包含10W*0.1=1W条数据，最终二者的比例为1:1。

```cmd

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./train_data/
    label_file_list:
    - vertical.txt
    - general.txt
    ratio_list: [1.0, 0.1]

```





字典
需要提供一个自定义字典（{word_dict_name}.txt），使模型在训练时，可以将所有出现的字符映射为字典的索引。

因此字典需要包含所有希望被正确识别的字符，{word_dict_name}.txt需要写成如下格式，并以 utf-8 编码格式保存：

```cmd
l
d
a
d
r
n
```

word_dict.txt 每行有一个单字，将字符与数字索引映射在一起，“and” 将被映射成 [2 5 1]

内置字典

PaddleOCR内置了一部分字典，可以按需使用。

ppocr/utils/ppocr_keys_v1.txt 是一个包含6623个字符的中文字典

ppocr/utils/ic15_dict.txt 是一个包含36个字符的英文字典

ppocr/utils/en_dict.txt 是一个包含96个字符的英文字典


在模型微调的过程中，建议准备至少`5000`张垂类场景的文本识别图像，可以保证基本的模型微调效果。如果希望提升模型的精度与泛化能力，可以合成更多与该场景类似的文本识别数据，从公开数据集中收集通用真实文本识别数据，一并添加到该场景的文本识别训练任务过程中。在训练过程中，建议每个epoch的真实垂类数据、合成数据、通用数据比例在`1:1:1`左右，这可以通过设置不同数据源的采样比例进行控制。如有3个训练文本文件，分别包含1W、2W、5W条数据，那么可以在配置文件中设置数据如下：

```
Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./train_data/
    label_file_list:
    - ./train_data/train_list_1W.txt
    - ./train_data/train_list_2W.txt
    - ./train_data/train_list_5W.txt
    ratio_list: [1.0, 0.5, 0.2]
    ...
```

##### 3-2 下载预训练模型

[ch_PP-OCRv4_rec_train](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar)

##### 3-3 参数配置

```cmd
Global:
  debug: false
  use_gpu: true
  epoch_num: 200
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: ./output/rec_ppocr_v4
  save_epoch_step: 10
  eval_batch_step: [0, 2000]
  cal_metric_during_train: true
  pretrained_model:
  checkpoints:
  save_inference_dir:
  use_visualdl: false
  infer_img: doc/imgs_words/ch/word_1.jpg
  character_dict_path: ppocr/utils/ppocr_keys_v1.txt
  max_text_length: &max_text_length 25
  infer_mode: false
  use_space_char: true
  distributed: true
  save_res_path: ./output/rec/predicts_ppocrv3.txt


Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.001
    warmup_epoch: 5
  regularizer:
    name: L2
    factor: 3.0e-05


Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform:
  Backbone:
    name: PPLCNetV3
    scale: 0.95
  Head:
    name: MultiHead
    head_list:
      - CTCHead:
          Neck:
            name: svtr
            dims: 120
            depth: 2
            hidden_dims: 120
            kernel_size: [1, 3]
            use_guide: True
          Head:
            fc_decay: 0.00001
      - NRTRHead:
          nrtr_dim: 384
          max_text_length: *max_text_length

Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss:
    - NRTRLoss:

PostProcess:  
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc

Train:
  dataset:
    name: MultiScaleDataSet
    ds_width: false
    data_dir: ./train_data/
    ext_op_transform_idx: 1
    label_file_list:
    - ./train_data/train_list.txt
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - RecConAug:
        prob: 0.5
        ext_data_num: 2
        image_shape: [48, 320, 3]
        max_text_length: *max_text_length
    - RecAug:
    - MultiLabelEncode:
        gtc_encode: NRTRLabelEncode
    - KeepKeys:
        keep_keys:
        - image
        - label_ctc
        - label_gtc
        - length
        - valid_ratio
  sampler:
    name: MultiScaleSampler
    scales: [[320, 32], [320, 48], [320, 64]]
    first_bs: &bs 192
    fix_bs: false
    divided_factor: [8, 16] # w, h
    is_training: True
  loader:
    shuffle: true
    batch_size_per_card: *bs
    drop_last: true
    num_workers: 8
Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./train_data
    label_file_list:
    - ./train_data/val_list.txt
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - MultiLabelEncode:
        gtc_encode: NRTRLabelEncode
    - RecResizeImg:
        image_shape: [3, 48, 320]
    - KeepKeys:
        keep_keys:
        - image
        - label_ctc
        - label_gtc
        - length
        - valid_ratio
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 128
    num_workers: 4

```

##### 3-4 训练

```cmd
python tools/train.py -c configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml \
     -o Global.pretrained_model=./pretrain_models/MobileNetV3_large_x0_5_pretrained

```

##### 3-5 评估

```cmd
python tools/eval.py -c configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml -o Global.checkpoints={path/to/weights}/best_accuracy

```

##### 3-6 推理

```cmd
python tools/infer_rec.py -c configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words/ch/word_1.jpg

```

##### 3-7 导出

```cmd
python tools/export_model.py -c configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml -o Global.pretrained_model=./pretrain_models/en_PP-OCRv3_rec_train/best_accuracy  Global.save_inference_dir=./inference/en_PP-OCRv3_rec/

```

### 4.  文本方向分类器微调

##### 4-1 数据准备

训练集&校验集
首先建议将训练图片放入同一个文件夹，并用一个txt文件（cls_gt_train.txt）记录图片路径和标签。

注意： 默认请将图片路径和图片标签用 \t 分割，如用其他方式分割将造成训练报错

0和180分别表示图片的角度为0度和180度

```cmd
" 图像文件名                 图像标注信息 "
train/cls/train/word_001.jpg   0
train/cls/train/word_002.jpg   180
```

最终训练集应有如下文件结构：

```cmd
|-train_data
    |-cls
        |- cls_gt_train.txt
        |- train
            |- word_001.png
            |- word_002.jpg
            |- word_003.jpg
            | ...
```

##### 4-2 下载预训练模型

ch_ppocr_mobile_v2.0_cls_train

##### 4-3 参数配置

将准备好的txt文件和图片文件夹路径分别写入配置文件的 Train/Eval.dataset.label_file_list 和 Train/Eval.dataset.data_dir 字段下，Train/Eval.dataset.data_dir字段下的路径和文件里记载的图片名构成了图片的绝对路径。

##### 4-4 训练

```cmd
python tools/train.py -c configs/cls/cls_mv3.yml
```

##### 4-5 评估

```cmd
python tools/eval.py -c configs/cls/cls_mv3.yml -o Global.checkpoints={path/to/weights}/best_accuracy
```

##### 4-6 推理

```cmd

python tools/infer_cls.py -c configs/cls/cls_mv3.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.
```



### 5.  训练KIE模型

对于识别得到的文字进行关键信息抽取，有2种主要的方法。

（1）直接使用SER，获取关键信息的类别：如身份证场景中，将“姓名“与”张三“分别标记为`name_key`与`name_value`。最终识别得到的类别为`name_value`对应的**文本字段**即为我们所需要的关键信息。

（2）联合SER与RE进行使用：这种方法中，首先使用SER，获取图像文字内容中所有的key与value，然后使用RE方法，对所有的key与value进行配对，找到映射关系，从而完成关键信息的抽取。

#### 2.2.1 SER

以身份证场景为例， 关键信息一般包含`姓名`、`性别`、`民族`等，我们直接将对应的字段标注为特定的类别即可，如下图所示。

[![img](./assets/184526682-8b810397-5a93-4395-93da-37b8b8494c41.png)](https://user-images.githubusercontent.com/14270174/184526682-8b810397-5a93-4395-93da-37b8b8494c41.png)

**注意：**

- 标注过程中，对于无关于KIE关键信息的文本内容，均需要将其标注为`other`类别，相当于背景信息。如在身份证场景中，如果我们不关注性别信息，那么可以将“性别”与“男”这2个字段的类别均标注为`other`。
- 标注过程中，需要以**文本行**为单位进行标注，无需标注单个字符的位置信息。

数据量方面，一般来说，对于比较固定的场景，**50张**左右的训练图片即可达到可以接受的效果，可以使用[PPOCRLabel](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/PPOCRLabel/README_ch.md)完成KIE的标注过程。

模型方面，推荐使用PP-StructureV2中提出的VI-LayoutXLM模型，它基于LayoutXLM模型进行改进，去除其中的视觉特征提取模块，在精度基本无损的情况下，进一步提升了模型推理速度。更多教程请参考：[VI-LayoutXLM算法介绍](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/algorithm_kie_vi_layoutxlm.md)与[KIE关键信息抽取使用教程](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/kie.md)。

##### 模型训练 

```cmd
python tools/train.py -c configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml
```

##### 模型测试

```cmd
python tools/infer_kie_token_ser.py -c configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml -o Architecture.Backbone.checkpoints=./output/ser_vi_layoutxlm_xfund_zh/best_accuracy Global.in
fer_img=./test_image/image_552204578768.jpg
```

##### 模型联合测试 det+SER

```cmd
python tools/infer_kie_token_ser.py -c configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml -o Architecture.Backbone.checkpoints=./output/ser_vi_layoutxlm_xfund_zh/best_accuracy Global.in
fer_img=./test_image/image_552204578768.jpg Global.kie_det_model_dir=output/det_db_inference
```

如果你希望训练自己的数据集，需要修改配置文件中的数据配置、字典文件以及类别数。

以 `configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml` 为例，修改的内容如下所示。

```
Architecture:
  # ...
  Backbone:
    name: LayoutXLMForSer
    pretrained: True
    mode: vi
    # 假设字典中包含n个字段（包含other），由于采用BIO标注，则类别数为2n-1
    num_classes: &num_classes 7

PostProcess:
  name: kieSerTokenLayoutLMPostProcess
  # 修改字典文件的路径为你自定义的数据集的字典路径
  class_path: &class_path train_data/XFUND/class_list_xfun.txt

Train:
  dataset:
    name: SimpleDataSet
    # 修改为你自己的训练数据目录
    data_dir: train_data/XFUND/zh_train/image
    # 修改为你自己的训练数据标签文件
    label_file_list:
      - train_data/XFUND/zh_train/train.json
    ...
  loader:
    # 训练时的单卡batch_size
    batch_size_per_card: 8
    ...

Eval:
  dataset:
    name: SimpleDataSet
    # 修改为你自己的验证数据目录
    data_dir: train_data/XFUND/zh_val/image
    # 修改为你自己的验证数据标签文件
    label_file_list:
      - train_data/XFUND/zh_val/val.json
    ...
  loader:
    # 验证时的单卡batch_size
    batch_size_per_card: 8
```



**注意，预测/评估时的配置文件请务必与训练一致。**

#### 2.2.2 SER + RE

该过程主要包含SER与RE 2个过程。SER阶段主要用于识别出文档图像中的所有key与value，RE阶段主要用于对所有的key与value进行匹配。

以身份证场景为例， 关键信息一般包含`姓名`、`性别`、`民族`等关键信息，在SER阶段，我们需要识别所有的question (key) 与answer (value) 。标注如下所示。每个字段的类别信息（`label`字段）可以是question、answer或者other（与待抽取的关键信息无关的字段）

[![img](./assets/184526785-c3d2d310-cd57-4d31-b933-912716b29856.jpg)](https://user-images.githubusercontent.com/14270174/184526785-c3d2d310-cd57-4d31-b933-912716b29856.jpg)

在RE阶段，需要标注每个字段的的id与连接信息，如下图所示。

[![img](./assets/184528728-626f77eb-fd9f-4709-a7dc-5411cc417dab.jpg)](https://user-images.githubusercontent.com/14270174/184528728-626f77eb-fd9f-4709-a7dc-5411cc417dab.jpg)

每个文本行字段中，需要添加`id`与`linking`字段信息，`id`记录该文本行的唯一标识，同一张图片中的不同文本内容不能重复，`linking`是一个列表，记录了不同文本之间的连接信息。如字段“出生”的id为0，字段“1996年1月11日”的id为1，那么它们均有[[0, 1]]的`linking`标注，表示该id=0与id=1的字段构成key-value的关系（姓名、性别等字段类似，此处不再一一赘述）。

**注意：**

- 标注过程中，如果value是多个字符，那么linking中可以新增一个key-value对，如`[[0, 1], [0, 2]]`

数据量方面，一般来说，对于比较固定的场景，**50张**左右的训练图片即可达到可以接受的效果，可以使用PPOCRLabel完成KIE的标注过程。

模型方面，推荐使用PP-StructureV2中提出的VI-LayoutXLM模型，它基于LayoutXLM模型进行改进，去除其中的视觉特征提取模块，在精度基本无损的情况下，进一步提升了模型推理速度。更多教程请参考：[VI-LayoutXLM算法介绍](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/algorithm_kie_vi_layoutxlm.md)与[KIE关键信息抽取使用教程](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/kie.md)。

#### 2.1 文本检测模型 background模型更换提高精准度

经过实验，ch_det_res18_db_v2.0效果最佳。下载ResNet18_vd的预训练模型：

```text
wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/ResNet18_vd_pretrained.tar
```

\# 解压预训练模型文件，以MobileNetV3为例

```text
tar -xf ./pretrain_models/MobileNetV3_large_x0_5_pretrained.tar -C ./pretrain_models/
```

**4. 配置文件修改：**

```text
Train-dataset-datadir: ./train_data/  # 这里原本写了./train_data/train会报错
Eval-dataset-datadir: ./train_data/  # 这里原本写了./train_data/test会报错
Global-pretrained_model: ./pretrain_models/ResNet18_vd_pretrained
Load_static_weights: True # 这个跟参数形式有关，如果报错就改反值即可。
```

**5. 开始训练：**

```text
python3 tools/train.py -c configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml \
 -o Global.pretrain_weights=./pretrain_models/ResNet18_vd_pretrained
```

**6. 一些可能出现的报错：**

debug的时候没有记录，具体想不起来了。总之就是传参和路径的问题，很好改的。

**7. 训练指标：**

loss应下降到0.1左右

**8. 训练效果比较：**

单张图前向预测：

```text
python tools/infer_det.py -c configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml -o Global.infer_img="./image.jpg" Global.pretrained_model="./output/ch_db_res18/latest" Global.load_static_weights=false
```

det_mv3_db和det_r50_vd_db两个模型，训练loss都下降到0.5左右

运行单张训练样本效果较差，漏检非常多

最后尝试ch_det_res18_db_v2.0，loss下降到0.1左右

运行单张训练样本发现确实学习到了很多训练样本标注的框。

**9. 模型导出：**

```text
Python tools/export_model.py -c configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml -o Global.pretrained_model="./output/ch_db_res18/latest" Global.load_static_weights=False Global.save_inference_dir=./inference/det_db/
```

该命令是将配置文件定义的模型和训练过程中保存的临时参数文件整合并导出包含模型结构的参数文件，作为部署使用的inference参数。

一份inference参数包含三个文件，可以部署在paddleocr库中直接使用。

### 5. 检测数据标注情况：

 后续只进行了识别标注 （后续如果需要继续提升检测模型精度可用）

![image-20240226110023491](./assets/image-20240226110023491.png)
