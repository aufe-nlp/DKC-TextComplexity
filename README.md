# Multidimensional Text Complexity Quantification and the Dynamic Knowledge Correction (DKC) Model

This repository contains the data and code for the paper *Multidimensional Text Complexity Quantification and the Design of a Dynamic Knowledge Correction Model*. It allows you to:

- compute the text complexity of six comment datasets and divide them into low, medium and high groups (Section 3, Table 3);
- train the DKC model for sentiment classification on the comment datasets, and correct its predictions with the error cases already confirmed in the knowledge base (Section 4, Table 8).

## Repository layout

```
DKC-TextComplexity/
├── data/
│   ├── Pub-data/              public datasets: Chn, Chn_all, weibo, waimai (binary)
│   └── Self-data/             self-collected datasets: Bili, Douyin (three-class)
├── chinese-roberta-wwm-ext/   config and tokenizer of the Chinese RoBERTa backbone (no weights)
├── prepare_model.py           downloads the RoBERTa weights
├── dkc/
│   ├── DKC.py                 DKC model for the binary datasets
│   └── DKC_triclass.py        DKC model for the three-class datasets
└── text_complexity/
    ├── text_complexity_calculator.py   text complexity computation
    ├── run_complexity.py               computes the complexity of the six datasets (Table 3)
    └── Dictionaries/                   abbreviation, emoticon, slang and stopword dictionaries
```

Every dataset is split into a training set (`*_train.tsv`) and a test set (`*_test.tsv`), with the following sizes:

| Dataset | Train | Test  | Total  | Labels                     |
| ------- | ----- | ----- | ------ | -------------------------- |
| Chn     | 9280  | 1200  | 10480  | 0 negative, 1 positive     |
| Chn_all | 6212  | 1554  | 7766   | 0 negative, 1 positive     |
| Weibo   | 12000 | 3000  | 15000  | 0 negative, 1 positive     |
| Waimai  | 9589  | 2398  | 11987  | 0 negative, 1 positive     |
| Bili    | 4575  | 1144  | 5719   | 0 negative, 1 neutral, 2 positive |
| Douyin  | 9000  | 2500  | 11500  | 0 negative, 1 neutral, 2 positive |

All files are tab-separated and UTF-8 encoded, with two columns: `label` and `text_a`.

`knowledge_base*.tsv` and `error_predictions*.tsv` in `dkc/` are empty files; they are written automatically at runtime.

## Preparation

**1. Install the dependencies**

Python 3.8 or above, PyTorch 2.6 or above, plus:

```bash
pip install transformers safetensors pandas numpy scikit-learn jieba tqdm
```

A GPU is recommended; CPU also works but is much slower.

**2. Download the RoBERTa weights**

The weights are about 412 MB and are not stored in this repository. Run the following inside `DKC-TextComplexity`:

```bash
python prepare_model.py
```

It downloads hfl/chinese-roberta-wwm-ext from Hugging Face, verifies the file and places it in `chinese-roberta-wwm-ext/`. If the download is slow in mainland China, use a mirror:

```bash
python prepare_model.py --endpoint https://hf-mirror.com
```

## Training and running DKC

### Commands

Use `DKC.py` for the binary datasets (Chn, Chn_all, Weibo, Waimai) and `DKC_triclass.py` for the three-class datasets (Bili, Douyin):

```bash
cd dkc
python DKC.py --train ../data/Pub-data/weibo_train.tsv --test ../data/Pub-data/weibo_test.tsv
python DKC_triclass.py --train ../data/Self-data/Bili_train.tsv --test ../data/Self-data/Bili_test.tsv
```

Without `--train` and `--test`, `DKC.py` uses Chn by default and `DKC_triclass.py` uses Bili by default.

### Each run is one round

- **Round 1**: the classification network is trained on the training set and its parameters are saved as `classifier.pt` (`classifier_triclass.pt` for the three-class case). The knowledge base is still empty, so the results on the test set are the predictions of the network itself. After prediction, the misclassified comments are stored in the knowledge base `knowledge_base.tsv` (`knowledge_base_triclass.tsv` for the three-class case).
- **Round 2 and later**: run the same command again. The program loads the classifier saved in Round 1 instead of retraining; it corrects the predictions with the current knowledge base and then stores the comments misclassified in this round.

Run the command as many times as the number of rounds you want.

**To switch datasets, or to start over**, first delete `classifier*.pt`, `knowledge_base*.tsv` and `error_predictions*.tsv` in `dkc/`.

### Training settings

The settings follow Table 7 of the paper: maximum length 512, AdamW optimizer, learning rate 5e-5, weight decay 0.01, 35 epochs, batch size 32, dropout 0.4, learning rate scheduled by ReduceLROnPlateau, gradient clipping 1.0. The RoBERTa parameters are frozen during training; only the classification network is trained.

### Output

At the end of each round the program prints the accuracy, precision, recall and F1 (the last three weighted by the number of samples of each class), together with the confusion matrix, and writes them to `dkc/evaluation_results.txt`. The misclassified comments that satisfy the criteria for a round are in `error_predictions*.tsv`, and all the cases actually stored in the knowledge base after deduplication are in `knowledge_base*.tsv`.

### Expected results

Round 1 has an empty knowledge base, so its results correspond to DKC-Base in Table 8 of the paper. From then on, each additional round puts more confirmed error cases into the knowledge base, more comments can be corrected, and the accuracy improves gradually. The accuracies reported in Table 8 of the paper are listed below for reference:

|                          | Chn   | Chn_all | Weibo | Waimai | Bili  | Douyin |
| ------------------------ | ----- | ------- | ----- | ------ | ----- | ------ |
| DKC-Base (classifier only) | 0.892 | 0.848   | 0.663 | 0.824  | 0.478 | 0.427  |
| DKC                      | 0.974 | 0.935   | 0.969 | 0.994  | 0.945 | 0.988  |

The code does not fix a random seed, so retraining the classifier gives slightly different results every time.

### How DKC corrects the predictions

1. **Classification network**: the [CLS] vectors of a comment at the 1st, 4th and 8th layers from the top are extracted from RoBERTa and concatenated into a 2304-dimensional vector, which then passes through a CNN, a BiLSTM and two attention layers to produce the probability of each class.
2. **Query the knowledge base**: the knowledge base stores the comments that were misclassified before, together with their correct labels and vectors.
   - If the knowledge base contains exactly the same comment, the model probability and the label of that case are given half the weight each, and the prediction is changed to the label of the case;
   - Otherwise the most similar case is found by vector similarity. When the similarity exceeds a threshold (0.90 for the binary case, 0.85 for the three-class case), the probability is shifted towards the label of the case in proportion to the similarity, so the more similar the case, the stronger the shift.
3. **Store new cases**: after a round of prediction, the comments that are "misclassified and whose two largest class probabilities differ by at least 0.15" are selected with the labels, and the duplicates and the too similar ones (cosine similarity greater than 0.98) are removed before storing, so that they are available from the next round on. In the experiments the labels of the test set are used in place of the correct labels given by manual verification.

### Other notes

- The preprocessing removes emoji and merges redundant spaces.
- The knowledge base only keeps the cases stored within the most recent 30 days; running several rounds in a row is not affected.
- If the GPU memory is insufficient, you can reduce `self.batch_size` in the code, but this deviates from the settings of the paper.

## Computing text complexity

```bash
cd text_complexity
python run_complexity.py
```

The program merges the training set and the test set of each dataset, computes the word-level score α and the sentence-level score β of every comment, and averages them to obtain α(w), β(s) and λ(c) = 0.4·α(w) + 0.6·β(s) for the dataset. Finally it divides the six datasets into low, medium and high groups by λ(c).

The six datasets contain about 60,000 comments in total; this takes from a few tens of minutes on a GPU. To first check that the code runs, add `--limit 20` to compute only the first 20 comments of each dataset.

### Expected results

They should agree with Table 3 of the paper, or be very close to it (the third decimal may differ by 0.001):

| Dataset | α(w)  | β(s)  | λ(c)  | Complexity group |
| ------- | ----- | ----- | ----- | ---------------- |
| Chn     | 0.788 | 0.416 | 0.564 | Low              |
| Chn_all | 0.739 | 0.417 | 0.546 | Low              |
| Weibo   | 0.900 | 0.474 | 0.644 | Medium           |
| Waimai  | 0.782 | 0.666 | 0.712 | High             |
| Bili    | 0.828 | 0.629 | 0.708 | High             |
| Douyin  | 0.781 | 0.709 | 0.738 | High             |

At runtime a `DynamicDictionaries` folder is generated automatically under `text_complexity/`, recording the abbreviations and alphanumeric mixed words identified during the run; it does not affect the scores.

## Data sources

- **Chn**: ChnSentiCorp, reviews of hotels, books and computers (Tan and Zhang, 2008).
- **Chn_all, Weibo, Waimai**: ChnSentiCorp_htl_all (hotel reviews), weibo_senti_100k (Weibo comments, from which 15000 were randomly sampled) and waimai_10k (food-delivery reviews), all from [ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus).
- **Bili, Douyin**: comments on artificial intelligence and employment collected by the authors from Bilibili and Douyin, independently annotated by five annotators with the label determined by majority vote. Only the labels and the comment text are kept, without usernames, user IDs or any other personal information.

## Licensing

- **Code**: `dkc/`, `text_complexity/` and `prepare_model.py` are released under the MIT License; you are free to use, modify and redistribute them, provided that the copyright notice is retained.
- **Self-collected datasets** (Bili, Douyin): for academic research only; commercial use is not permitted. The copyright of the comment content belongs to the original authors and the platforms; only the labels and the comment text are kept, so please do not attempt to identify the commenters.
- **Public datasets** (Chn, Chn_all, Weibo, Waimai): the copyright belongs to the original publishers; please observe their terms of use and cite the original sources.
- **RoBERTa pretrained model**: from [hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext), under its Apache-2.0 license.

For any question, please contact the corresponding author: Zhiying Wei, weizhiying@aufe.edu.cn.
