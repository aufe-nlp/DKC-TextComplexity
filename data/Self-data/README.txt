Self-collected datasets (Bili and Douyin)

Manuscript: "Multidimensional Text Complexity Quantification and the Design of a Dynamic Knowledge Correction Model", submitted to Applied Sciences.

1. Content
These files contain the two self-collected datasets used in the experiments of the manuscript (Section 5.1.1, Table 2). Both datasets contain comments about artificial intelligence and employment: Bili was collected from Bilibili and Douyin was collected from Douyin. The comments were independently annotated by five annotators, and the final label was determined by majority voting.

2. Files
File               Comments   Negative (0)   Neutral (1)   Positive (2)
Bili_train.tsv       4575        1442           1608          1525
Bili_test.tsv        1144         385            378           381
Douyin_train.tsv     9000        3000           3000          3000
Douyin_test.tsv      2500         500           1000          1000

Bili: 5719 comments in total. Douyin: 11500 comments in total. The training and test sets of each dataset share no comment text.

3. Format
Tab-separated values (TSV), UTF-8 encoding, with one header line and two columns:
  label   sentiment label: 0 = negative, 1 = neutral, 2 = positive
  text_a  comment text (Chinese)
The files contain only the label and the comment text. No user names, user IDs or other personal information are included.

4. Use
In the experiments, all models used the same training/test split of each dataset: they were trained on the training set and evaluated on the test set (Section 5.2, Table 8). No separate validation set was used. In the ablation experiments, the test set was used as the comment pool, and its labels were used to simulate feedback (Section 5.3).

The datasets, the text complexity calculation code and the model code will be made publicly available on GitHub upon acceptance of the article.
