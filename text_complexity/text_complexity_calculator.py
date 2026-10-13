"""
文本复杂度计算模块
包含词语级非规范性、句子级结构性和语料级复杂度的综合评估
我取名为多维特征文本复杂度计算模块
"""

import os
import re
import math
import jieba
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class TextComplexityCalculator:
    def __init__(self, dict_dir="Dictionaries", dynamic_dict_dir="DynamicDictionaries", load_roberta=False):
        """
        初始化文本复杂度计算器
        
        参数:
            dict_dir: 静态词典目录
            dynamic_dict_dir: 动态词典目录
            load_roberta: 是否在初始化时加载RoBERTa模型
        """
        # 词典放在本文件同目录的 Dictionaries 下；RoBERTa 放在上一级目录的 chinese-roberta-wwm-ext 下
        self.module_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.dirname(self.module_dir)
        self.dict_dir = dict_dir if os.path.isabs(dict_dir) else os.path.join(self.module_dir, dict_dir)
        
        # 创建动态词典目录
        self.dynamic_dir = dynamic_dict_dir if os.path.isabs(dynamic_dict_dir) else os.path.join(self.module_dir, dynamic_dict_dir)
        os.makedirs(self.dynamic_dir, exist_ok=True)
        
        # 动态词典文件路径
        self.dynamic_slang_file = os.path.join(self.dynamic_dir, "捕获热词.txt")
        self.dynamic_abbr_file = os.path.join(self.dynamic_dir, "CapturedAbbreviations.txt")
        self.dynamic_alphanum_file = os.path.join(self.dynamic_dir, "CapturedAlphanumerics.txt")
        
        # 加载静态词典
        self.slang_dict = self._load_dict("Slangs.txt")
        self.abbr_dict = self._load_dict("Abbreviations.txt")
        self.emoji_dict = self._load_dict("Emoticons.txt")
        self.stop_dict = self._load_dict("Stop.txt")
        
        # 添加动态词典计数器
        self.new_alphanum_count = 0
        self.new_abbr_count = 0
        
        # 初始化停用词表（如果停用词词典加载失败，使用默认停用词）
        if not self.stop_dict:
            self.stop_dict = {"的", "了", "和", "是", "在", "我", "有", "不", "也", "这", "那", "就", "都", "但", "他", "她", "它", "啊", "呢", "吗", "吧"}
        
        # 权重系数设置
        # 词语级权重
        self.word_weights = {
            "abbr": 0.2,    # λ₁: 缩写词权重
            "emoji": 0.3,   # λ₂: 表情符号权重
            "slang": 0.4,   # λ₃: 网络热词权重
            "alphanum": 0.1 # λ₄: 字母数字混合词权重
        }
        
        # 语料级权重
        self.corpus_weights = {
            "word": 0.4,    # ω₁: 词语级复杂度权重
            "sent": 0.6     # ω₂: 句子级复杂度权重
        }
        
        # 非线性组合模型权重
        self.nonlinear_weights = {
            "word": 0.4,    # η₁: 词语级权重
            "sent": 0.5,    # η₂: 句子级权重
            "interaction": 0.1  # η₃: 交互项权重
        }
        
        # RoBERTa模型相关属性
        self.roberta_model_path = os.path.join(self.base_dir, "chinese-roberta-wwm-ext")
        self.roberta_tokenizer = None
        self.roberta_model = None
        self.roberta_device = None
        self.roberta_load_warning_shown = False  # 新增标志位，控制警告只显示一次
        
        # 如果需要，预加载模型
        if load_roberta:
            self._load_roberta_model()
    
    def _load_roberta_model(self):
        """
        加载RoBERTa模型（如果尚未加载）
        
        返回:
            bool: 模型加载是否成功
        """
        if self.roberta_model is not None:
            return True  # 模型已加载
            
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError:
            print("请先安装必要的库: pip install transformers torch")
            return False
        
        try:
            # 检查是否有可用的GPU
            self.roberta_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"使用设备: {self.roberta_device}")
            
            # 加载中文RoBERTa模型
            print("开始加载RoBERTa模型...")
            # 确保路径格式正确，并检查路径是否存在
            if not os.path.isdir(self.roberta_model_path):
                raise FileNotFoundError(f"RoBERTa模型路径不存在: {self.roberta_model_path}")
            self.roberta_tokenizer = AutoTokenizer.from_pretrained(self.roberta_model_path, local_files_only=True)
            self.roberta_model = AutoModel.from_pretrained(self.roberta_model_path, local_files_only=True)
            
            # 将模型移至GPU（如果可用）
            self.roberta_model = self.roberta_model.to(self.roberta_device)
            print("RoBERTa模型加载完成")
            return True
        except Exception as e:
            print(f"加载RoBERTa模型失败: {e}")
            self.roberta_tokenizer = None
            self.roberta_model = None
            self.roberta_device = None
            return False
    
    def _load_dict(self, filename):
        """加载词典文件"""
        filepath = os.path.join(self.dict_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f if line.strip()]
            except UnicodeDecodeError:
                try:
                    with open(filepath, 'r', encoding='gbk') as f:
                        return [line.strip() for line in f if line.strip()]
                except UnicodeDecodeError:
                    try:
                        with open(filepath, 'r', encoding='gb2312') as f:
                            return [line.strip() for line in f if line.strip()]
                    except Exception as e:
                        print(f"警告: 加载词典文件 {filepath} 失败: {e}")
                        return []
        else:
            print(f"警告: 词典文件 {filepath} 不存在")
            return []
    
    def _save_to_file(self, filepath, items, mode='w'):
        """保存列表到文件"""
        try:
            # 确保文件目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 如果是追加模式且文件不存在，先创建文件
            if mode == 'a' and not os.path.exists(filepath):
                with open(filepath, 'w', encoding='utf-8'):
                    pass
                    
            with open(filepath, mode, encoding='utf-8') as f:
                for item in items:
                    f.write(f"{item}\n")
                    
            # 确保数据立即写入磁盘
            if mode == 'a':
                import time
                time.sleep(0.1)  # 短暂延迟，确保文件操作完成
        except Exception as e:
            print(f"写入文件 {filepath} 失败: {e}")
    
    def _get_file_contents(self, filepath):
        """读取文件内容为列表"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except:
            return []
    
    #-----------------词语级非规范性计算-----------------#
    
    def detect_network_slang(self, word, text=None):
        """
        检测网络热词
        
        参数:
            word: 要检测的词
            text: 原文本(可选)
        
        返回:
            bool: 是否为热词
        """
        # 直接检查词是否在热词词典中
        if word in self.slang_dict:
            return True
        
        return False
    
    def detect_emoji(self, word):
        """
        检测表情符号
        
        参数:
            word: 要检测的词
        
        返回:
            bool: 是否为表情符号
        """
        return word in self.emoji_dict
    
    def detect_word_type(self, word):
        """
        检测词语类型: 缩写词还是字母数字混合词
        
        参数:
            word: 要检测的词
        
        返回:
            (is_abbr, is_alphanum): 是否缩写词, 是否字母数字混合词
        """
        # 1. 判断是否包含数字或特殊符号与字母混合
        has_digit = bool(re.search(r'[0-9]', word))
        has_special_char_with_alphanum = bool(re.search(r'[A-Za-z].*[-_+.].*[0-9]|[0-9].*[-_+.].*[A-Za-z]', word))
        
        if has_digit or has_special_char_with_alphanum:
            # 属于字母数字混合词
            is_alphanum = True
            is_abbr = False
            
            # 记录到字母数字混合词文档
            existing_words = self._get_file_contents(self.dynamic_alphanum_file)
            if word not in existing_words:
                self._save_to_file(self.dynamic_alphanum_file, [word], mode='a')
                #print(f"[动态词典] 捕获到新字母数字混合词: {word}")
                self.new_alphanum_count += 1  # 增加计数器
        else:
            # 判断是否匹配缩写词库
            is_in_static_dict = word.lower() in [a.lower() for a in self.abbr_dict]
            
            # 判断是否符合缩写词规则
            is_abbr_by_rule = (re.match(r'^[A-Za-z]{1,5}$', word) or 
                            re.match(r'^([A-Za-z]+\.)+[A-Za-z]?$', word) or
                            (word.isupper() and len(word) <= 10))
            
            # 判断是否为缩写词
            is_abbr = is_in_static_dict or is_abbr_by_rule
            
            # 只有不在静态词典中、但符合缩写词规则的词才添加到动态词典
            if is_abbr_by_rule and not is_in_static_dict:
                # 记录到缩写词文档
                existing_words = self._get_file_contents(self.dynamic_abbr_file)
                # 既检查原始形式是否存在，也检查小写形式是否存在，更精确地避免重复
                if word not in existing_words and word.lower() not in [w.lower() for w in existing_words]:
                    # 确保文件存在
                    if not os.path.exists(self.dynamic_abbr_file):
                        with open(self.dynamic_abbr_file, 'w', encoding='utf-8'):
                            pass
                    self._save_to_file(self.dynamic_abbr_file, [word], mode='a')
                    #print(f"[动态词典] 捕获到新缩写词: {word}")
                    self.new_abbr_count += 1  # 增加计数器
            
            # 既不在缩写词库中也不符合缩写词规则
            is_alphanum = False
        
        return is_abbr, is_alphanum
    
    def calculate_word_irregularity(self, word, text=None):
        """
        计算词语级非规范性得分 α(w)
        
        使用非线性变换(k=4)：
        α(w) = 1 - exp(-4*(I_abb(w) + I_emo(w) + I_slg(w) + I_aln(w)))
        
        参数:
            word: 要计算的词
            text: 原文本(可选)
        
        返回:
            α值，范围[0,1]
        """
        # 检测各个指标
        is_slang = self.detect_network_slang(word, text)
        has_emoji = self.detect_emoji(word)
        is_abbr, is_alphanum = self.detect_word_type(word)
        
        # 指示函数取值
        I_abb = 1 if is_abbr else 0
        I_emo = 1 if has_emoji else 0
        I_slg = 1 if is_slang else 0
        I_aln = 1 if is_alphanum else 0
        
        # 非线性变换(k=4)
        α = 1 - math.exp(-4 * (I_abb + I_emo + I_slg + I_aln))
        
        return 1 - min(1.0, α)  # 确保α值不超过1
        
    def calculate_irregularity(self, text):
        """
        计算整篇文本的词语级非规范性得分
        基于优化后的非线性变换(k=4)
        
        参数:
            text: 要分析的文本
            
        返回:
            float: 整篇文本的词语级非规范性平均得分
        """
        if not text or not isinstance(text, str):
            return 0.0
            
        # 分词并过滤停用词
        words = jieba.lcut(text)
        words = [w for w in words if w not in self.stop_dict]
        
        if not words:
            return 0.0
        
        # 计算每个词的非规范性得分并求平均
        total_score = 0.0
        word_count = len(words)
        
        for word in words:
            # 根据四个指标计算
            i_abb = 1 if word.lower() in [a.lower() for a in self.abbr_dict] or re.match(r'^[A-Za-z]{1,5}$', word) else 0
            i_emo = 1 if word in self.emoji_dict else 0
            i_slg = 1 if word in self.slang_dict else 0
            i_aln = 1 if bool(re.search(r'[a-zA-Z]', word)) and bool(re.search(r'[0-9\u4e00-\u9fff]', word)) else 0
            
            # 使用非线性变换(k=4)
            irregularity = 1 - math.exp(-4 * (i_abb + i_emo + i_slg + i_aln))
            total_score += irregularity
        
        # 返回平均得分
        return 1 - (total_score / word_count) if word_count > 0 else 0.0
    
    #-----------------句子级结构性计算-----------------#
    
    def detect_abnormal_punctuation(self, text):
        """
        检测异常标点密度
        
        参数:
            text: 待分析的文本
        
        返回:
            abnormal_ratio: 异常标点占比
        """
        # 匹配连续标点符号(如!!!, ???, 。。。等)
        consecutive_puncts = re.findall(r'([!?.,，。？！；;：:]{2,})', text)
        
        # 计算异常标点总数
        abnormal_count = sum(len(punct) for punct in consecutive_puncts)
        
        # 计算文本总字符数
        total_chars = len(text)
        
        # 计算异常标点占比
        if total_chars > 0:
            return abnormal_count / total_chars
        return 0
    
    def calculate_informal_ratio(self, segmented_text, word_scores):
        """
        计算口语化词占比
        
        参数:
            segmented_text: 分词后的文本列表
            word_scores: 词语级非规范性得分字典
        
        返回:
            inf_ratio: 口语化词占比
        """
        # 认为α(w) ≥ 0.3的词为口语化词
        informal_threshold = 0.3
        
        # 统计口语化词数量
        informal_count = sum(1 for word in segmented_text if word in word_scores and word_scores[word] >= informal_threshold)
        
        # 计算口语化词占比
        if len(segmented_text) > 0:
            return informal_count / len(segmented_text)
        return 0
    
    def calculate_semantic_ambiguity(self, text):
        """
        计算语义模糊度 (基于句子自相似度) - 传统方法
        
        参数:
            text: 待分析的文本
        
        返回:
            ambiguity: 语义模糊度
        """
        # 如果文本太短，返回0
        if len(text) < 2:
            return 0
        
        # 将句子分为不同片段 (根据标点符号或每隔5个词)
        segments = re.split(r'[.。!！?？;；]', text)
        segments = [s.strip() for s in segments if s.strip()]
        
        # 如果分段后只有一个片段，则直接返回0
        if len(segments) <= 1:
            return 0
        
        # 使用TF-IDF向量化句子片段
        vectorizer = TfidfVectorizer(
            tokenizer=jieba.lcut,
            stop_words=self.stop_dict,
            min_df=1
        )
        
        try:
            # 转换为TF-IDF向量
            segments_tfidf = vectorizer.fit_transform(segments)
            
            # 计算片段间的余弦相似度
            similarity_matrix = cosine_similarity(segments_tfidf)
            
            # 计算平均余弦相似度 (不包括自己与自己的相似度)
            n = len(segments)
            total_similarity = 0
            count = 0
            
            for i in range(n):
                for j in range(i+1, n):
                    total_similarity += similarity_matrix[i, j]
                    count += 1
            
            avg_similarity = total_similarity / count if count > 0 else 0
            
            # 语义模糊度 = 1 - 平均相似度
            return 1 - avg_similarity
            
        except Exception as e:
            print(f"计算语义模糊度时出错: {e}")
            return 0.5  # 出错时返回中等模糊度
    
    def calculate_semantic_ambiguity_roberta(self, text):
        """
        使用RoBERTa计算语义模糊度
        
        参数:
            text: 待分析的文本
        
        返回:
            ambiguity: 语义模糊度，范围为[0,1]
        """
        # 如果文本太短，返回0
        if len(text) < 2:
            return 0
        
        # 尝试加载模型（如果尚未加载）
        if not self._load_roberta_model():
            if not self.roberta_load_warning_shown:
                print("无法使用RoBERTa模型，将使用传统方法计算语义模糊度")
                self.roberta_load_warning_shown = True
            return self.calculate_semantic_ambiguity(text)  # 回退到传统方法
        
        # 分割句子
        segments = re.split(r'[.。!！?？;；]', text)
        segments = [s.strip() for s in segments if len(s.strip()) > 3]  # 只保留有意义的片段
        
        if len(segments) <= 1:
            return 0  # 单片段无法计算模糊度
        
        try:
            import torch
            import numpy as np
            
            # 获取每个片段的嵌入表示
            segment_embeddings = []
            for segment in segments:
                inputs = self.roberta_tokenizer(segment, return_tensors="pt", padding=True, truncation=True, max_length=128)
                # 将输入移至GPU或CPU
                inputs = {k: v.to(self.roberta_device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.roberta_model(**inputs)
                
                # 使用[CLS]标记的输出作为整个句子的表示
                # 将结果移回CPU并转为numpy
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                segment_embeddings.append(embeddings)
            
            # 计算片段间的相似度
            similarities = []
            for i in range(len(segment_embeddings)):
                for j in range(i+1, len(segment_embeddings)):
                    similarity = cosine_similarity(segment_embeddings[i], segment_embeddings[j])[0][0]
                    similarities.append(similarity)
            
            # 计算平均相似度
            avg_similarity = np.mean(similarities) if similarities else 0
            
            # 语义模糊度 = 1 - 相似度（越不相似越模糊）
            ambiguity = 1 - avg_similarity
            
            # 确保结果在[0,1]范围内
            return max(0, min(1, ambiguity))
            
        except Exception as e:
            print(f"RoBERTa模型处理出错: {e}，将使用备用TF-IDF方法")
            return self.calculate_semantic_ambiguity(text)  # 回退到传统方法
    
    def calculate_sentence_structure(self, text, word_scores):
        """
        计算句子级结构性 β(s)
        
        采用逆向融合增强法(IFEM)：结合多特征融合、对比度增强和复杂度取反
        
        参数:
            text: 待分析的文本
            word_scores: 词语级非规范性得分字典
        
        返回:
            β值，范围[0,1]
        """
        # 使用逆向融合增强法计算句子级复杂度
        return self.calculate_sentence_structure_fusion_contrast(text, word_scores)
    
    def calculate_sentence_structure_fusion_contrast(self, text, word_scores, corpus_stats={'mean': 0.523, 'std': 0.343}):
        """
        逆向融合增强法(IFEM)：结合多特征融合、对比度增强和复杂度取反
        
        核心思想:
        1. 使用非线性组合计算初始复杂度
        2. 应用对比度增强处理，使分数分布更加均衡
        3. 引入多特征自适应权重，根据特征间的一致性调整
        4. 通过取反操作(1-score)将规范度转换为非规范度
        
        参数:
            text: 文本内容
            word_scores: 词语得分词典
            corpus_stats: 语料库统计值，默认均值0.523，标准差0.343
        
        返回:
            复杂度得分，范围[0,1]
        """
        # 分词
        segmented_text = jieba.lcut(text)
        
        # 计算多种特征
        J_punc = self.detect_abnormal_punctuation(text)
        J_inf = self.calculate_informal_ratio(segmented_text, word_scores)
        J_amb = self.calculate_semantic_ambiguity_roberta(text)
        J_len = min(1.0, len(segmented_text) / 15.0)
        
        # 特征间协同和抑制
        synergy = math.log(1 + J_inf * J_amb * 10) * 0.3
        inhibition = 0.2 * (1 - math.exp(-3 * abs(J_inf - J_amb)))
        
        # 特征一致性权重（当特征值接近时增加权重）
        consistency_weight = 0.3 * (1 - math.sqrt(
            (J_punc - J_inf)**2 + (J_inf - J_amb)**2 + (J_amb - J_len)**2) / 2)
        
        # 计算初始复杂度分数（基于非线性组合）
        base_score = 0.1 * J_punc + 0.4 * J_inf + 0.2 * J_amb + 0.1 * J_len + synergy - inhibition + consistency_weight
        base_score = min(1.0, max(0.0, base_score))
        
        # 应用对比度增强处理
        mean, std = corpus_stats['mean'], corpus_stats['std']
        z_score = (base_score - mean) / std
        enhanced_score = 1 / (1 + math.exp(-z_score * 2))  # sigmoid函数增强对比度
        
        # 复杂度取反：将规范度转换为非规范度
        inverse_score = 1.0 - enhanced_score
        
        return round(inverse_score, 3)
    
    #-----------------语料级复杂度计算-----------------#
    
    def calculate_corpus_complexity_linear(self, avg_word_score, avg_sent_score):
        """
        计算语料级复杂度 (线性组合模型)
        
        γ(C) = ω₁·E_w[α(w)] + ω₂·E_s[β(s)]
        
        参数:
            avg_word_score: 平均词语级非规范性得分
            avg_sent_score: 平均句子级结构性得分
        
        返回:
            γ值，范围[0,1]
        """
        γ = (self.corpus_weights["word"] * avg_word_score + 
             self.corpus_weights["sent"] * avg_sent_score)
        
        return min(1.0, γ)  # 确保γ值不超过1
    
    def calculate_corpus_complexity_nonlinear(self, avg_word_score, avg_sent_score):
        """
        计算语料级复杂度 (非线性组合模型)
        
        γ(C) = η₁·α + η₂·β + η₃·α·β
        
        参数:
            avg_word_score: 平均词语级非规范性得分
            avg_sent_score: 平均句子级结构性得分
        
        返回:
            γ值，范围[0,1]
        """
        γ = (self.nonlinear_weights["word"] * avg_word_score + 
             self.nonlinear_weights["sent"] * avg_sent_score + 
             self.nonlinear_weights["interaction"] * avg_word_score * avg_sent_score)
        
        return min(1.0, γ)  # 确保γ值不超过1
    
    #-----------------热词检测与捕获-----------------#
    
    def capture_high_frequency_words(self, all_comments, top_n=100):
        """
        从所有评论中捕获高频词，并保存到捕获热词.txt
        
        参数:
            all_comments: 所有评论文本列表
            top_n: 捕获的高频词数量
        
        返回:
            list: 捕获的高频词列表
        """
        # 分词并统计词频
        word_counter = Counter()
        for comment in all_comments:
            if isinstance(comment, str) and comment.strip():
                words = jieba.lcut(comment)
                for word in words:
                    if len(word) >= 2 and word not in self.stop_dict:
                        word_counter[word] += 1
        
        # 提取前top_n个高频词
        all_top_words = [word for word, _ in word_counter.most_common(top_n)]
        
        #加载静态热词词库
        static_slang_words = self.slang_dict  # 直接使用已加载的热词词典
        
        # 过滤掉已经存在于静态词库的词语
        new_top_words = [word for word in all_top_words if word not in static_slang_words]
        
        # 覆盖写入到捕获热词.txt
        self._save_to_file(self.dynamic_slang_file, new_top_words)
        print(f"高频词分析: 共捕获{len(all_top_words)}个高频词，其中{len(new_top_words)}个新词已保存至: {self.dynamic_slang_file}")
        
        return new_top_words
    
    #-----------------综合计算与分析-----------------#
    
    def analyze_text(self, text):
        """
        分析单个文本的复杂度
        
        参数:
            text: 待分析的文本
        
        返回:
            字典: 包含词语级、句子级和语料级复杂度分析结果
        """
        
        # 初始化结果字典
        result = {
            "text": text,
            "word_level": {},
            "sentence_level": {},
            "corpus_level": {}
        }
        
        # 词语级分析
        words = jieba.lcut(text)
        word_scores = {}
        
        for word in words:
            if word.strip():
                α = self.calculate_word_irregularity(word, text)
                word_scores[word] = α
        
        # 计算词语级平均得分
        avg_word_score = sum(word_scores.values()) / len(word_scores) if word_scores else 0
        
        # 句子级分析
        β = self.calculate_sentence_structure(text, word_scores)
        
        # 语料级分析
        γ_linear = self.calculate_corpus_complexity_linear(avg_word_score, β)
        γ_nonlinear = self.calculate_corpus_complexity_nonlinear(avg_word_score, β)
        
        # 填充结果
        result["word_level"] = {
            "average_score": round(avg_word_score, 3),
            "word_scores": {word: round(score, 3) for word, score in word_scores.items() if score > 0}
        }
        
        result["sentence_level"] = {
            "structure_score": round(β, 3),
            "punctuation_density": round(self.detect_abnormal_punctuation(text), 3),
            "informal_ratio": round(self.calculate_informal_ratio(words, word_scores), 3),
            "semantic_ambiguity": round(self.calculate_semantic_ambiguity_roberta(text), 3)
        }
        
        result["corpus_level"] = {
            "linear_complexity": round(γ_linear, 3),
            "nonlinear_complexity": round(γ_nonlinear, 3)
        }
        
        return result
    
    def analyze_corpus_from_csv(self, csv_file, text_column="评论内容", top_n=100):
        """
        分析CSV文件中的语料复杂度
        
        参数:
            csv_file: CSV文件路径
            text_column: 包含文本的列名
            top_n: 捕获的高频词数量
        
        返回:
            DataFrame: 包含复杂度分析结果的DataFrame
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file, encoding='utf-8')
            
            # 确保text_column存在
            if text_column not in df.columns:
                print(f"错误: 列 '{text_column}' 在CSV文件中不存在")
                return None
            
            # 提取所有评论列表
            comments = df[text_column].dropna().tolist()
            
            # 捕获高频词
            high_freq_words = self.capture_high_frequency_words(comments, top_n=top_n)
            print(f"已捕获{len(high_freq_words)}个高频词: {', '.join(high_freq_words[:20])}...")
            
            # 对每条评论进行分析
            results = []
            for i, comment in enumerate(df[text_column]):
                if isinstance(comment, str) and comment.strip():
                    result = self.analyze_text(comment)
                    results.append({
                        "index": i,
                        "text": comment,
                        "word_complexity": result["word_level"]["average_score"],
                        "sentence_complexity": result["sentence_level"]["structure_score"],
                        "linear_complexity": result["corpus_level"]["linear_complexity"],
                        "nonlinear_complexity": result["corpus_level"]["nonlinear_complexity"],
                        "contains_popular_slang": any(word in high_freq_words for word in jieba.lcut(comment))
                    })
            
            # 转换为DataFrame
            results_df = pd.DataFrame(results)
            
            # 计算语料库整体的平均值
            avg_results = {
                "平均词语级复杂度": results_df["word_complexity"].mean(),
                "平均句子级复杂度": results_df["sentence_complexity"].mean(),
                "平均线性组合复杂度": results_df["linear_complexity"].mean(),
                "平均非线性组合复杂度": results_df["nonlinear_complexity"].mean(),
                "包含高频词的评论比例": results_df["contains_popular_slang"].mean() if "contains_popular_slang" in results_df else 0
            }
            
            print("语料平均复杂度:")
            for metric, value in avg_results.items():
                print(f"{metric}: {value:.3f}")
            
            # 显示动态词典捕获统计
            if self.new_alphanum_count > 0:
                print(f"[动态词典] 共捕获到{self.new_alphanum_count}个新字母数字混合词")
                self.new_alphanum_count = 0  # 重置计数器   
            
            # 显示动态词典捕获统计
            if self.new_abbr_count > 0:
                print(f"[动态词典] 共捕获到{self.new_abbr_count}个新缩写词")
                self.new_abbr_count = 0  # 重置计数器   
            
            return results_df
            
        except Exception as e:
            print(f"分析CSV文件时出错: {e}")
            return None
    
    def analyze_corpus_from_excel(self, excel_file, text_column="text", top_n=100):
        """
        分析Excel文件中的语料复杂度
        
        参数:
            excel_file: Excel文件路径
            text_column: 包含文本的列名
            top_n: 捕获的高频词数量
        
        返回:
            DataFrame: 包含复杂度分析结果的DataFrame
        """
        try:
            # 读取Excel文件
            df = pd.read_excel(excel_file)
            
            # 确保text_column存在
            if text_column not in df.columns:
                print(f"错误: 列 '{text_column}' 在Excel文件中不存在")
                return None
            
            print(f"成功读取Excel文件: {excel_file}")
            print(f"总评论数: {len(df)}")
            
            # 提取所有评论列表
            comments = df[text_column].dropna().tolist()
            
            # 捕获高频词
            high_freq_words = self.capture_high_frequency_words(comments, top_n=top_n)
            print(f"已捕获{len(high_freq_words)}个高频词: {', '.join(high_freq_words[:20])}...")
            
            # 对每条评论进行分析
            results = []
            for i, comment in tqdm(enumerate(df[text_column]), total=len(df[text_column]), desc="分析评论"):
                if isinstance(comment, str) and comment.strip():
                    result = self.analyze_text(comment)
                    results.append({
                        "index": i,
                        "text": comment,
                        "word_complexity": result["word_level"]["average_score"],
                        "sentence_complexity": result["sentence_level"]["structure_score"],
                        "linear_complexity": result["corpus_level"]["linear_complexity"],
                        "nonlinear_complexity": result["corpus_level"]["nonlinear_complexity"],
                        "contains_popular_slang": any(word in high_freq_words for word in jieba.lcut(comment))
                    })
            
            # 转换为DataFrame并计算平均值
            results_df = pd.DataFrame(results)
            
            # 计算平均值
            avg_results = {
                "平均词语级复杂度": results_df["word_complexity"].mean(),
                "平均句子级复杂度": results_df["sentence_complexity"].mean(),
                "线性组合复杂度": results_df["linear_complexity"].mean(),
                "非线性组合复杂度": results_df["nonlinear_complexity"].mean(),
                "包含高频词的评论比例": results_df["contains_popular_slang"].mean() if "contains_popular_slang" in results_df else 0
            }
            
            print("语料平均复杂度:")
            for metric, value in avg_results.items():
                print(f"{metric}: {value:.3f}")
                
            # 显示动态词典捕获统计
            if self.new_alphanum_count > 0:
                print(f"[动态词典] 共捕获到{self.new_alphanum_count}个新字母数字混合词")
                self.new_alphanum_count = 0  # 重置计数器   
            
            # 显示动态词典捕获统计
            if self.new_abbr_count > 0:
                print(f"[动态词典] 共捕获到{self.new_abbr_count}个新缩写词")
                self.new_abbr_count = 0  # 重置计数器
            
            return results_df
            
        except Exception as e:
            print(f"分析Excel文件时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def analyze_file(self, file_path, text_column="text", top_n=100):
        """
        根据文件扩展名自动选择适当的分析方法
        
        参数:
            file_path: 文件路径
            text_column: 包含文本的列名
            top_n: 捕获的高频词数量
            
        返回:
            DataFrame: 包含复杂度分析结果的DataFrame
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            return self.analyze_corpus_from_csv(file_path, text_column, top_n)
        elif file_ext in ['.xlsx', '.xls']:
            return self.analyze_corpus_from_excel(file_path, text_column, top_n)
        else:
            print(f"不支持的文件类型: {file_ext}")
            return None


if __name__ == "__main__":
    # 初始化计算器
    calculator = TextComplexityCalculator(load_roberta=True)
    
    # 获取数据目录
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data")
    
    # 示例：分析不同数据集
    excel_file = os.path.join(data_dir, "worktext.xlsx")
    
    if os.path.exists(excel_file):
        print(f"\n分析文件: {excel_file}")
        results = calculator.analyze_file(excel_file, text_column="text", top_n=100)
        if results is not None:
            # 保存分析结果
            results_file = os.path.join(data_dir, "测试可删除.xlsx")
            results.to_excel(results_file, index=False)
            print(f"分析结果已保存至: {results_file}")
    else:
        print(f"文件不存在: {excel_file}")
        
        # 列出Data目录下的文件供用户选择
        print("\nData目录下可用的文件:")
        for file in os.listdir(data_dir):
            if file.endswith(('.xlsx', '.xls', '.csv')):
                print(f"- {file}")
