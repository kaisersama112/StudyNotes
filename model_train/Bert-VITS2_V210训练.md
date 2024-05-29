Bert-vits2更新了版本V210，修正了日/英的bert对齐问题，效果进一步优化；对底模使用的数据进行优化和加量，减少finetune失败以及电音的可能性；日语bert更换了模型，完善了多语言推理。

更多情报请参考Bert-vits2官网：

```text
https://github.com/fishaudio/Bert-VITS2/releases/tag/2.1
```

最近的事情大家也都晓得了，马督工义无反顾带头冲锋，身体力行地实践着横渠四句：为天地立心，为生民立命，为往圣继绝学，为万世开太平。

本次我们基于Bert-vits2的新版本V210，复刻马督工，向他致敬。

### Bert-vits2V210整备数据集

我们知道马督工的风格是语速极快，也没啥肢体语言，语调上也基本没有变化，除了换气，基本上就像机关枪一样无限念稿。当然，这也是因为睡前消息内容密度过大导致的，但作为深度学习训练数据集来说，睡前消息节目的音频素材显然是不合格的。

真正好的高质量数据集应该包含以下几个特征：

音色多样性：数据集应该包含目标说话人的多个语音样本，涵盖他们在不同情感状态、不同语速和不同音高下的说话。这样可以捕捉到目标说话人在不同情境下的声音特征。

音频质量：确保语音样本的音频质量高，没有明显的噪声、失真或其他干扰。音频质量的好坏直接影响到复刻结果的质量。

多样的语音内容：语音样本应该包含不同类型的语音内容，例如单词、短语、句子和段落。这有助于捕捉到目标说话人在不同语境下的音色特征。

语音平衡：确保数据集中包含目标说话人的样本数量相对平衡，以避免训练出偏向某些样本的模型。

覆盖不同音高：收集目标说话人在不同音高和音调下的语音样本。这样可以更好地捕捉到他们声音的变化和音高特征。

语音环境：包含不同环境下的语音样本，例如室内、室外、静音和嘈杂环境等。这样可以使复刻的音色更具鲁棒性，适应不同的环境条件。

长度和多样性：语音样本的长度和多样性也是需要考虑的因素。收集包含不同长度和语音风格的样本，以便更好地捕捉到目标说话人的声音特征。

当然了，完全满足上述特点基本不太可能，这里选择马督工和刘女神的一段采访视频：

```text
https://www.bilibili.com/video/BV1sN411M73g/
```



![img](./assets/v2-ac8e6910eada3e6b819cc00ca26c20c3_720w.webp)

首先将视频进行下载，这里使用you-get:

```text
pip install you-get
```

运行命令：

```text
https://www.bilibili.com/video/BV1sN411M73g/
```

下载成功后，将马督工的声音提取出来。

### Bert-vits2V210训练模型

首先克隆笔者fork自官网的v210项目:

```text
git clone https://github.com/v3ucn/Bert-VITS2_V210.git
```

将素材放入Data/meimei/raw/meimei目录中，注意必须是wav文件。

然后更换新的底模，下载地址：

```text
https://openi.pcl.ac.cn/Stardust_minus/Bert-VITS2/modelmanage/show_model
```

把Bert-VITS2_2.1-Emo底模放入项目的pretrained_models目录。

同时单独把deberta-v2-large-japanese-char-wwm模型放入到项目的bert/deberta-v2-large-japanese-char-wwm目录中。

由于新增了多维情感模型，所以也需要单独下载模型：

```text
https://huggingface.co/facebook/wav2vec2-large-robust/tree/main
```

放入项目的emotional目录：

```text
E:\work\Bert-VITS2-v21_demo\emotional>tree /f  
Folder PATH listing for volume myssd  
Volume serial number is 7CE3-15AE  
E:.  
└───wav2vec2-large-robust-12-ft-emotion-msp-dim  
        .gitattributes  
        config.json  
        LICENSE  
        preprocessor_config.json  
        pytorch_model.bin  
        README.md  
        vocab.json
```

运行脚本，切分素材：

```text
python3 audio_slicer.py
```

随后进行重采样和文本识别：

```text
python3 short_audio_transcribe.py
```

接着进行标注：

```text
python3 preprocess_text.py
```

和V2.0.2不同的是，V2.1需要生成多维情感模型文件：

```text
python3 emo_gen.py
```

相对于原版，新版增加了，针对训练集的spec缓存，可以有效提高训练效率：

```text
python3 spec_gen.py
```

最后生成bert模型可读文件：

```text
python3 bert_gen.py
```

最后开始训练：

```text
python3 train_ms.py
```

### Bert-vits2V210模型推理

模型训练好之后，进入到推理环节，首先修改根目录的config.yml文件：

```text
bert_gen:  
  config_path: config.json  
  device: cuda  
  num_processes: 2  
  use_multi_device: false  
dataset_path: Data\meimei  
mirror: ''  
openi_token: ''  
preprocess_text:  
  clean: true  
  cleaned_path: filelists/cleaned.list  
  config_path: config.json  
  max_val_total: 8  
  train_path: filelists/train.list  
  transcription_path: filelists/short_character_anno.list  
  val_path: filelists/val.list  
  val_per_spk: 5  
resample:  
  in_dir: raw  
  out_dir: raw  
  sampling_rate: 44100  
server:  
  device: cuda  
  models:  
  - config: ./Data/meimei/config.json  
    device: cuda  
    language: ZH  
    model: ./Data/meimei/models/G_0.pth  
    speakers:  
    - length_scale: 1  
      noise_scale: 0.6  
      noise_scale_w: 0.8  
      sdp_ratio: 0.2  
      speaker: "\u79D1\u6BD4"  
    - length_scale: 0.5  
      noise_scale: 0.7  
      noise_scale_w: 0.8  
      sdp_ratio: 0.3  
      speaker: "\u4E94\u6761\u609F"  
    - length_scale: 1.2  
      noise_scale: 0.6  
      noise_scale_w: 0.8  
      sdp_ratio: 0.2  
      speaker: "\u5B89\u500D\u664B\u4E09"  
  - config: ./Data/meimei/config.json  
    device: cuda  
    language: JP  
    model: ./Data/meimei/models/G_0.pth  
    speakers: []  
  port: 7860  
train_ms:  
  base:  
    model_image: "Bert-VITS2_2.1-Emo底模"  
    repo_id: Stardust_minus/Bert-VITS2  
    use_base_model: false  
  config_path: config.json  
  env:  
    MASTER_ADDR: localhost  
    MASTER_PORT: 10086  
    RANK: 0  
    THE_ENV_VAR_YOU_NEED_TO_USE: '1234567'  
    WORLD_SIZE: 1  
  keep_ckpts: 8  
  model: models  
  num_workers: 16  
  spec_cache: true  
translate:  
  app_key: ''  
  secret_key: ''  
webui:  
  config_path: Data/meimei/config.json  
  debug: false  
  device: cuda  
  language_identification_library: langid  
  model: models/G_150.pth  
  port: 7860  
  share: false
```

在后面的webui配置中写入模型文件名:model: models/G_150.pth。

随后启动推理脚本：

```text
python3 webui.py
```

就可以进行推理了：



![img](./assets/v2-4e553bb99a15a2ef20c912434a36d264_720w.webp)

请注意，推理建议使用官方的基于Gradio版本的推理页面，而非FastApi的版本。