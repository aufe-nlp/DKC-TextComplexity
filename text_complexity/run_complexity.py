# -*- coding: utf-8 -*-
"""
计算六个数据集的文本复杂度（论文表3）。

每个数据集把训练集和测试集合在一起，逐条评论计算词语级得分 α 和句子级得分 β；
数据集的 α(w)、β(s) 取全部评论的平均值，λ(c) = 0.4·α(w) + 0.6·β(s)。
最后按 λ(c) 排序，在相邻差距最大的两处断开，分为低、中、高三组。

用法：python run_complexity.py            （建议使用 GPU）
      python run_complexity.py --limit 20 （每个数据集只取前20条，检查能否跑通）
"""
import argparse
import os

import jieba
from tqdm import tqdm

from text_complexity_calculator import TextComplexityCalculator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'data')

# (论文中的名称, 数据文件前缀)
DATASETS = [
    ('Chn', os.path.join('Pub-data', 'Chn')),
    ('Chn_all', os.path.join('Pub-data', 'Chn_all')),
    ('Weibo', os.path.join('Pub-data', 'weibo')),
    ('Waimai', os.path.join('Pub-data', 'waimai')),
    ('Bili', os.path.join('Self-data', 'Bili')),
    ('Douyin', os.path.join('Self-data', 'Douyin')),
]


def load_texts(path):
    """按行读取“标签<TAB>评论”，跳过表头。"""
    with open(path, encoding='utf-8') as f:
        lines = f.read().split('\n')[1:]
    return [line.rstrip('\r').split('\t', 1)[1] for line in lines if line.rstrip('\r')]


def score_comment(calc, text):
    """一条评论的 (α, β)；空评论或计算出错时返回 None。"""
    if not text.strip():
        return None
    try:
        alpha = calc.calculate_irregularity(text)
        word_scores = {w: calc.calculate_word_irregularity(w, text) for w in jieba.lcut(text) if w.strip()}
        beta = calc.calculate_sentence_structure(text, word_scores)
        return alpha, beta
    except Exception as e:
        print(f'跳过一条评论：{e}')
        return None


def main():
    parser = argparse.ArgumentParser(description='计算六个数据集的文本复杂度（表3）')
    parser.add_argument('--limit', type=int, default=0, help='每个数据集只取前N条，0表示全部')
    args = parser.parse_args()

    calc = TextComplexityCalculator()
    if not calc._load_roberta_model():
        raise SystemExit('RoBERTa 加载失败，请先运行上一级目录的 prepare_model.py 准备模型权重')

    results = {}
    for name, prefix in DATASETS:
        texts = load_texts(os.path.join(DATA_DIR, prefix + '_train.tsv')) + load_texts(os.path.join(DATA_DIR, prefix + '_test.tsv'))
        if args.limit:
            texts = texts[:args.limit]
        scores = [s for s in (score_comment(calc, t) for t in tqdm(texts, desc=name)) if s]
        alpha = sum(s[0] for s in scores) / len(scores)
        beta = sum(s[1] for s in scores) / len(scores)
        results[name] = (alpha, beta, calc.calculate_corpus_complexity_linear(alpha, beta))

    # 按 λ(c) 排序，在相邻差距最大的两处断开，分为低、中、高三组
    ordered = sorted(results, key=lambda n: results[n][2])
    gaps = sorted(range(len(ordered) - 1), key=lambda i: results[ordered[i + 1]][2] - results[ordered[i]][2], reverse=True)[:2]
    cut1, cut2 = sorted(gaps)
    group = {n: 'Low' if i <= cut1 else ('Medium' if i <= cut2 else 'High') for i, n in enumerate(ordered)}

    print('\nDataset   alpha(w)  beta(s)  lambda(c)  Group')
    for name, _ in DATASETS:
        a, b, lam = results[name]
        print(f'{name:<9} {a:.3f}     {b:.3f}    {lam:.3f}      {group[name]}')


if __name__ == '__main__':
    main()
