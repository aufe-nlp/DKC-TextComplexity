# 多维文本复杂度量化与动态知识修正（DKC）模型

这里是论文 *Multidimensional Text Complexity Quantification and the Design of a Dynamic Knowledge Correction Model* 的数据和代码。可以用来：

- 计算六个评论数据集的文本复杂度，并分成低、中、高三组（论文第 3 节、表 3）；
- 训练 DKC 模型，在评论数据集上做情感分类，并用知识库里已确认的错误案例修正预测（论文第 4 节、表 8）。

## 文件说明

```
DKC-TextComplexity/
├── data/
│   ├── Pub-data/              公开数据集：Chn、Chn_all、weibo、waimai（二分类）
│   └── Self-data/             自建数据集：Bili、Douyin（三分类）
├── chinese-roberta-wwm-ext/   RoBERTa 中文预训练模型的配置和分词器（不含权重）
├── prepare_model.py           下载 RoBERTa 权重
├── dkc/
│   ├── DKC.py                 DKC 模型，用于二分类数据集
│   └── DKC_triclass.py        DKC 模型，用于三分类数据集
└── text_complexity/
    ├── text_complexity_calculator.py   文本复杂度计算
    ├── run_complexity.py               计算六个数据集的复杂度（表 3）
    └── Dictionaries/                   缩写、表情符号、网络热词、停用词词典
```

每个数据集分为训练集（`*_train.tsv`）和测试集（`*_test.tsv`），条数如下：

| 数据集  | 训练集 | 测试集 | 合计  | 标签                   |
| ------- | ------ | ------ | ----- | ---------------------- |
| Chn     | 9280   | 1200   | 10480 | 0 消极，1 积极         |
| Chn_all | 6212   | 1554   | 7766  | 0 消极，1 积极         |
| Weibo   | 12000  | 3000   | 15000 | 0 消极，1 积极         |
| Waimai  | 9589   | 2398   | 11987 | 0 消极，1 积极         |
| Bili    | 4575   | 1144   | 5719  | 0 消极，1 中性，2 积极 |
| Douyin  | 9000   | 2500   | 11500 | 0 消极，1 中性，2 积极 |

文件都是制表符分隔，UTF-8 编码，两列：`label`（标签）和 `text_a`（评论）。

`dkc/` 里的 `knowledge_base*.tsv` 和 `error_predictions*.tsv` 是空文件，运行时会自动写入。

## 准备工作

**1. 安装依赖**

Python 3.8 以上，PyTorch 2.6 以上，以及：

```bash
pip install transformers safetensors pandas numpy scikit-learn jieba tqdm
```

建议用 GPU，用 CPU 也能跑，但会慢很多。

**2. 下载 RoBERTa 权重**

权重约 412 MB，没有放在仓库里。在 `DKC-TextComplexity` 下运行：

```bash
python prepare_model.py
```

它会从 Hugging Face 下载 hfl/chinese-roberta-wwm-ext，核对文件无误后放进 `chinese-roberta-wwm-ext/`。国内下载慢的话可以换镜像：

```bash
python prepare_model.py --endpoint https://hf-mirror.com
```

## 训练和运行 DKC

### 运行命令

二分类数据集（Chn、Chn_all、Weibo、Waimai）用 `DKC.py`，三分类数据集（Bili、Douyin）用 `DKC_triclass.py`：

```bash
cd dkc
python DKC.py --train ../data/Pub-data/weibo_train.tsv --test ../data/Pub-data/weibo_test.tsv
python DKC_triclass.py --train ../data/Self-data/Bili_train.tsv --test ../data/Self-data/Bili_test.tsv
```

不写 `--train` 和 `--test` 时，`DKC.py` 默认用 Chn，`DKC_triclass.py` 默认用 Bili。

### 每运行一次是一轮

- **第 1 轮**：用训练集训练分类网络，训练好的参数保存为 `classifier.pt`（三分类为 `classifier_triclass.pt`）。这时知识库是空的，测试集上的结果就是分类网络自己的预测。预测完，把判错的评论存进知识库 `knowledge_base.tsv`（三分类为 `knowledge_base_triclass.tsv`）。
- **第 2 轮及以后**：再运行同一条命令。程序直接加载第 1 轮保存的分类网络，不再训练；用当前知识库修正预测，再把这一轮新判错的评论存进去。

想跑几轮，就把命令运行几次。

**换数据集，或者想从头再来**，先删掉 `dkc/` 里的 `classifier*.pt`、`knowledge_base*.tsv` 和 `error_predictions*.tsv`。

### 训练设置

和论文表 7 一致：最大长度 512，AdamW 优化器，学习率 5e-5，权重衰减 0.01，35 个 epoch，批大小 32，Dropout 0.4，学习率用 ReduceLROnPlateau 调整，梯度裁剪 1.0。训练时 RoBERTa 的参数不动，只训练后面的分类网络。

### 输出

每轮结束会打印准确率、精确率、召回率和 F1（后三个按各类样本数加权），以及混淆矩阵，同时写入 `dkc/evaluation_results.txt`。每轮符合入库条件的判错评论在 `error_predictions*.tsv` 里，去重后实际存进知识库的全部案例在 `knowledge_base*.tsv` 里。

### 预计结果

第 1 轮知识库为空，结果相当于论文表 8 的 DKC-Base。之后每多跑一轮，知识库里已确认的错误案例越多，能修正的评论也越多，准确率会逐步提高。论文表 8 的准确率如下，可作参考：

|                          | Chn   | Chn_all | Weibo | Waimai | Bili  | Douyin |
| ------------------------ | ----- | ------- | ----- | ------ | ----- | ------ |
| DKC-Base（只用分类网络） | 0.892 | 0.848   | 0.663 | 0.824  | 0.478 | 0.427  |
| DKC                      | 0.974 | 0.935   | 0.969 | 0.994  | 0.945 | 0.988  |

代码没有固定随机种子，每次重新训练分类网络，结果都会有小幅波动。

### DKC 是怎么修正预测的

1. **分类网络**：用 RoBERTa 取出评论在倒数第 1、4、8 层的 [CLS] 向量，拼成一个 2304 维的向量，再经过 CNN、BiLSTM 和两次注意力，得到每个类别的概率。
2. **查知识库**：知识库里存着之前判错的评论、它的正确标签和向量。
   - 如果知识库里有一模一样的评论，就把模型概率和这条案例的标签各取一半，预测改成案例的标签；
   - 如果没有，就找向量最相似的一条案例。相似度超过阈值（二分类 0.90，三分类 0.85）时，按相似度把概率往案例标签的方向调，越相似调得越多。
3. **存入新案例**：一轮预测完，用标签挑出“判错了、而且最大的两个类别概率相差至少 0.15”的评论，去掉重复的和太相似的（余弦相似度大于 0.98），存进知识库，下一轮开始使用。实验中用测试集的标签代替人工核验给出的正确标签。

### 其他说明

- 预处理时会删掉 emoji 表情符号，并合并多余的空格。
- 知识库只保留最近 30 天存进去的案例，连续运行多轮不受影响。
- 显存不够时，可以调小代码里的 `self.batch_size`，但这样就和论文的设置不同了。

## 计算文本复杂度

```bash
cd text_complexity
python run_complexity.py
```

程序把每个数据集的训练集和测试集合在一起，逐条算出每条评论的词语级得分 α 和句子级得分 β，再取平均，得到数据集的 α(w)、β(s) 和 λ(c) = 0.4·α(w) + 0.6·β(s)，最后按 λ(c) 把六个数据集分成低、中、高三组。

六个数据集共约 6 万条评论，用 GPU 需要十几分钟到几十分钟。想先试试能不能跑通，可以加 `--limit 20`，每个数据集只算前 20 条。

### 预计结果

应当与论文表 3 一致或非常接近（第三位小数可能差 0.001）：

| 数据集  | α(w) | β(s) | λ(c) | 复杂度分组 |
| ------- | ----- | ----- | ----- | ---------- |
| Chn     | 0.788 | 0.416 | 0.564 | 低         |
| Chn_all | 0.739 | 0.417 | 0.546 | 低         |
| Weibo   | 0.900 | 0.474 | 0.644 | 中         |
| Waimai  | 0.782 | 0.666 | 0.712 | 高         |
| Bili    | 0.828 | 0.629 | 0.708 | 高         |
| Douyin  | 0.781 | 0.709 | 0.738 | 高         |

运行时会在 `text_complexity/` 下自动生成 `DynamicDictionaries` 文件夹，记录运行中识别出的缩写和字母数字混合词，不影响得分。

## 数据来源

- **Chn**：ChnSentiCorp，酒店、图书和电脑评论（Tan 和 Zhang，2008）。
- **Chn_all、Weibo、Waimai**：ChnSentiCorp_htl_all（酒店评论）、weibo_senti_100k（微博评论，从中随机抽取 15000 条）、waimai_10k（外卖评论），均来自 [ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)。
- **Bili、Douyin**：作者分别从哔哩哔哩和抖音采集的关于人工智能与就业的评论，由五名标注者独立标注、多数投票确定标签。只保留标签和评论文本，不含用户名、用户 ID 等个人信息。

## 开源说明

- **代码**：`dkc/`、`text_complexity/` 和 `prepare_model.py` 以 MIT 许可证开源，可以自由使用、修改和分发，但需保留版权声明。
- **自建数据集**（Bili、Douyin）：仅供学术研究使用，不得用于商业用途。评论内容的版权归原作者和平台所有；数据只保留了标签和评论文本，使用时请勿尝试识别评论者身份。
- **公开数据集**（Chn、Chn_all、Weibo、Waimai）：版权归原发布方所有，使用时请遵守其使用条款，并引用原始来源。
- **RoBERTa 预训练模型**：来自 [hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext)，遵循其 Apache-2.0 许可证。

有问题请联系通讯作者：魏志莹，weizhiying@aufe.edu.cn。
