<div align="center">
    <h1>
        Chinese-Mixtral-8x7B
    </h1>
</div>

<div align="center">
    <a href="https://github.com/carfly/Chinese-Mixtral-8x7B/pulls">
        <image src="https://img.shields.io/badge/PRs-welcome-brightgreen"></image>
        <image src="https://img.shields.io/badge/License-Apache_2.0-green.svg"></image>
    </a>
</div>

## 🚀 介绍

本项目基于Mistral发布的模型[Mixtral-8x7B](https://mistral.ai/news/mixtral-of-experts/)进行中文扩词表增量预训练，以进一步促进中文自然语言处理社区对稀疏混合专家模型的研究。我们使用了开源的大规模语料进行增量预训练，使模型具备了强大的中文生成和理解能力，同时扩充后的词表显著提高了模型对中文的编解码效率。

项目开源内容：

- 中文Mixtral-8x7B扩词表大模型
- 扩词表增量预训练代码

局限性：Chinese-Mixtral-8x7B仍然可能生成包含事实性错误的误导性回复或包含偏见/歧视的有害内容，请谨慎鉴别和使用生成的内容，请勿将生成的有害内容传播至互联网。若产生不良后果，由传播者自负。

## 📥 模型下载

本项目使用QLoRA进行训练，LoRA权重与合并权重后的模型分别开源，您可以根据自己的需求选择下载：

|             模型名称             | 模型大小  |                                     下载地址                                      |                                                         备注                                                          |
|:----------------------------:|:-----:|:-----------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------:|
|     Chinese-Mixtral-8x7B     | 88GB  |     [🤗HuggingFace](https://huggingface.co/HIT-SCIR/Chinese-Mixtral-8x7B)     |                                                  中文扩词表完整模型，可以直接使用                                                   |
| Chinese-Mixtral-8x7B-adapter | 2.7GB | [🤗HuggingFace](https://huggingface.co/HIT-SCIR/Chinese-Mixtral-8x7B-adapter) | LoRA权重，需要与原版Mixtral-8x7B进行合并才可以使用，合并脚本请参考[这里](https://gist.github.com/ChrisHayduk/1a53463331f52dca205e55982baf9930) |

## 📈 模型评测

### 大模型综合能力评测

我们分别使用以下评测数据集对Chinese-Mixtral-8x7B进行评测：

- C-Eval（中文）：一个全面的中文基础模型评估套件。它包含了13948个多项选择题，涵盖了52个不同的学科和四个难度级别。
- CMMLU（中文）：一个综合性的中文评估基准，专门用于评估语言模型在中文语境下的知识和推理能力，涵盖了从基础学科到高级专业水平的67个主题。
- MMLU（英文）：一个包含57个多选任务的英文评测数据集，涵盖了初等数学、美国历史、计算机科学、法律等，难度覆盖高中水平到专家水平，是目前主流的LLM评测数据集之一。
- HellaSwag（英文）：一个极具挑战的NLI评测数据集，每一个问题都需要对上下文进行深入理解，而不能基于常识进行回答。

根据Mistral发布的[技术报告](https://arxiv.org/pdf/2401.04088.pdf)，Mixtral-8x7B在推理时将激活13B参数。下表为Chinese-Mixtral-8x7B与其他13B规模的中文扩词表模型在各个任务上的评测结果：

|                                              模型名称                                               |      增量训练语料       | C-Eval(5-shot) | CMMLU(5-shot) | MMLU(5-shot) | HellaSwag(5-shot) |
|:-----------------------------------------------------------------------------------------------:|:-----------------:|:--------------:|:-------------:|:------------:|:-----------------:|
|           [IDEA-CCNL/Ziya2-13B-Base](https://huggingface.co/IDEA-CCNL/Ziya2-13B-Base)           |    650B Token     |     59.29      |     60.93     |    59.86     |       58.90       |
| [TigerResearch/tigerbot-13b-base-v3](https://huggingface.co/TigerResearch/tigerbot-13b-base-v3) |    500B Token     |     50.52      |     51.65     |    53.46     |       59.16       |
|    [Linly-AI/Chinese-LLaMA-2-13B-hf](https://huggingface.co/Linly-AI/Chinese-LLaMA-2-13B-hf)    |     11B Token     |     42.57      |     41.95     |    51.32     |       59.05       |
|            [hfl/chinese-llama-2-13b](https://huggingface.co/hfl/chinese-llama-2-13b)            | 约30B Token(120GB) |     41.90      |     42.08     |    51.92     |       59.28       |
|                                  **Chinese-Mixtral-8x7B(本项目)**                                  |     42B Token     |     52.08      |     51.08     |    69.80     |       65.69       |

由于不同版本的评测脚本实现细节有细微差异，为了保证评测结果的一致性和公平性，我们的评测脚本统一使用EleutherAI发布的lm-evaluation-harness，commit hash为[28ec7fa](https://github.com/EleutherAI/lm-evaluation-harness/tree/28ec7fa950346b5a895e85e1f3edd5648168acc4)。

### 中文编解码效率评测

我们的Chinese-Mixtral-8x7B中文编解码效率较原模型提高了43.87%，有利于加速中文文本的推理速度

下表为基于40万字的中文小说对模型中文编解码效率的测试结果：

|         模型名称         | 词表大小  | 40万字中文文本Token量 | 相对压缩率提高 |
|:--------------------:|:-----:|:--------------:|:-------:|
|     Mixtral-8x7B     | 32000 |    506,540     |    -    |
| Chinese-Mixtral-8x7B | 57000 |    284,312     | 43.87%  |

<!-- (506,540 - 284,312) / 506,540 -->

<!-- 下表为基于[SkyPile](https://huggingface.co/datasets/Skywork/SkyPile-150B)数据集对模型中文编解码效率的测试结果：

|         模型名称         | 词表大小  | 1GB中文文本Token量 | 相对压缩率提高 |
|:--------------------:|:-----:|:-------------:|:-------:|
|     Mixtral-8x7B     | 32000 | TODO |    -    |
| Chinese-Mixtral-8x7B | 57000 | TODO |  TODO%   | -->

## ⚙️ 模型细节

<details>
<summary>

### 词表扩充

</summary>

我们使用`sentencepiece`在12G知乎数据和2G悟道数据上训练中文词表。训练词表时每次新增1000个Token，并枚举了新增中文单字的数量。通过Zheng Bo等人提出的[ALP](https://arxiv.org/pdf/2109.07306.pdf)衡量词表对中文的表示能力：

![](./img/alp.png)

为了避免词表过小导致中文压缩率过低，以及词表过大导致embedding层过于稀疏，我们选择了ALP曲线的拐点：新增25000个中文token作为最终Chinese-Mixtral-8x7B的词表。

对于embedding层和lm_head层的扩充部分，我们使用新Token在旧embedding层中的词嵌入平均值对扩充部分进行初始化。

</details>

<details>
<summary>

### 增量预训练

</summary>

Chinese-Mixtral-8x7B基于Mixtral-8x7B，使用QLoRA进行微调。

#### 环境准备

我们建议使用Python 3.10 + torch 2.0.1

```shell
# Pytorch + Transformers
$ pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
$ pip install transformers==4.36.2 datasets evaluate peft accelerate gradio optimum sentencepiece
$ pip install jupyterlab scikit-learn pandas matplotlib tensorboard nltk rouge bitsandbytes fire
# DeepSpeed
$ git clone https://github.com/microsoft/DeepSpeed.git
$ cd DeepSpeed
$ DS_BUILD_FUSED_ADAM=1 pip3 install .
# Flash Attention
$ pip install flash-attn --no-build-isolation
```

#### 数据集下载

我们基于现有的完全开源的数据集训练了Chinese-Mixtral-8x7B，数据集包括：

|                                    数据集名称                                     | 数据集语言 |        备注        |
|:----------------------------------------------------------------------------:|:-----:|:----------------:|
| [Skywork/SkyPile-150B](https://huggingface.co/datasets/Skywork/SkyPile-150B) |  中文   | 仅使用2023+2022年的数据 |
| [DKYoon/SlimPajama-6B](https://huggingface.co/datasets/DKYoon/SlimPajama-6B) |  英文   |        -         |

通过`data/download.py`将数据集下载到`data`中。针对Slimpajama数据集，需要使用`data/parquet2jsonl.py`将原始数据集转换为`jsonl`格式。

下载后的数据集为多个jsonl文件的分片，使用`cat`将多个分片合并为一个jsonl文件。

```shell
$ cat *.jsonl > all.jsonl
```

通过`split`将jsonl切分为train和valid集合。本项目中train和valid的行数比例为999:1。

```shell
$ wc -l all.jsonl                          # 计算数据集总行数
$ split -l <lines> all.jsonl               # 按999:1计算train/valid行数，进行切分
$ mv xaa DKYoon-SlimPajama-6B-train.jsonl  # 重命名
$ mv xab DKYoon-SlimPajama-6B-dev.jsonl
```

#### 数据集预处理

将数据集名称和路径注册到`data/datasets.toml`中：

```toml
[DKYoon-SlimPajama-6B]              # 数据集名称
splits = ["train", "dev"]           # 数据集train/valid集合
root = "{DATA_DIR}/en/{name}"       # 数据集根目录
doc = "{name}-{split}"              # 数据集文件名
encoded = "encoded-{name}-{split}"  # 预处理保存位置
```

使用`data/preprocess_datasets.py`对数据集进行子词切分，从而加快训练速度。

```shell
$ python data/preprocess_datasets.py --ds_name SkyPile-150B-2023 --tokenizer_name_or_path tokenizer/Mixtral-8x7B-v0.1-vocab
$ python data/preprocess_datasets.py --ds_name DKYoon-SlimPajama-6B --tokenizer_name_or_path tokenizer/Mixtral-8x7B-v0.1-vocab
```

在进行子词切分后，可以使用`data/utils.py`查看各个数据集的token总量：

```shell
$ python data/utils.py
```

#### 开始训练

训练启动脚本为`scripts/train.sh`。可以通过修改其中的`TRAIN_DATASETS`修改训练数据集和数据集比例：

```shell
TRAIN_DATASETS=(
    1:SkyPile-150B-2022     # 使用全量SkyPile-150B-2022
    0.1:SkyPile-150B-2023   # 使用SkyPile-150B-2023的10%数据
    1:DKYoon-SlimPajama-6B  # 使用全量DKYoon-SlimPajama-6B
)
```

如果您使用SLURM集群管理系统，可以通过`sbatch`进行提交：

```shell
$ sbatch scripts/train.sh
```

如果没有SLURM或希望通过命令行启动训练，您可以直接提取`scripts/train.sh`中的`torchrun`开始训练。

</details>

<details>
<summary>

### 微调

</summary>

本项目发布的Chinese-Mixtral-8x7B为基座模型，没有经过微调。如果您希望使用Chinese-Mixtral-8x7B进行下游任务微调或SFT，可以参考HuggingFace已给出Mixtral-8x7B的QLoRA微调脚本[HuggingFace的官方示例代码](https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py)。

</details>

## ✒️ 引用

如果您觉得本项目对您的研究有所帮助或使用了本项目的代码，请引用本项目：

```bibtex
@misc{Chinese-Mixtral-8x7B,
    author = {HIT-SCIR-LA}.
    title = {Chinese-Mixtral-8x7B: An Open-Source Universal LLM}
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub repository}
    howpublished = {\url{https://github.com/carfly/Chinese-Mixtral-8x7B}}
}
```

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=carfly/Chinese-Mixtral-8x7B&type=Date)](https://star-history.com/#carfly/Chinese-Mixtral-8x7B&Date)
