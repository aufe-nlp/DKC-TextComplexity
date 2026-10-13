import sys
import os
import random
import hashlib
import traceback
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoModel, AutoTokenizer, AutoConfig

# 设置pandas显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# 获取当前脚本目录与代码开源根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))  # 代码根目录：dkc 文件夹的上一级
# 将项目根目录添加到Python路径
sys.path.append(project_root)

import re
import string
import warnings
warnings.filterwarnings('ignore')

class KnowledgeBase:
    def __init__(
        self,
        encoder=None,
        time_window_days=30,
        max_kb_size_per_class=20000,
        near_dup_threshold=0.98,
        score_lambda=0.4,
        score_mu=0.4,
        score_nu=0.2,
    ):
        """
        初始化知识库

        Args:
            encoder: RoBERTa编码器实例，用于生成文本的语义向量表示
        """
        self.encoder = encoder             # RoBERTa编码器
        self.text_to_label = {}           # 文本到标签的映射
        self.text_to_embedding = {}       # 文本到向量的映射
        self.kb_file = os.path.join(current_dir, 'knowledge_base.tsv')  # 知识库文件路径
        self._enhance_log_shown = False
        self.time_window_days = int(time_window_days)
        self.max_kb_size_per_class = int(max_kb_size_per_class)
        self.near_dup_threshold = float(near_dup_threshold)
        self.score_lambda = float(score_lambda)
        self.score_mu = float(score_mu)
        self.score_nu = float(score_nu)

    @staticmethod
    def _normalize_prob(prob):
        p = np.array(prob, dtype=float).reshape(-1)
        p = np.clip(p, 1e-12, 1.0)
        s = p.sum()
        if s <= 0:
            return np.array([0.5, 0.5], dtype=float)
        return p / s

    @staticmethod
    def _parse_timestamp(ts):
        if pd.isna(ts):
            return None
        try:
            return pd.to_datetime(ts)
        except Exception:
            return None

    def load(self):
        """加载知识库，如果文件不存在或为空则创建新文件"""
        try:
            if os.path.exists(self.kb_file):
                # 检查文件是否为空
                if os.path.getsize(self.kb_file) == 0:
                    # 如果文件为空，创建带有列名的空DataFrame
                    empty_df = pd.DataFrame(columns=['label', 'text', 'embedding', 'timestamp', 'text_hash', 'score'])
                    empty_df.to_csv(self.kb_file, sep='\t', index=False, encoding='utf-8')
                    print("\n创建了新的知识库文件")
                    return

                print("\n加载知识库...")
                kb_df = pd.read_csv(self.kb_file, sep='\t')

                kb_df.columns = [str(c).strip().lower() for c in kb_df.columns]
                if 'timestamp' not in kb_df.columns:
                    kb_df['timestamp'] = pd.Timestamp.now().isoformat()
                if 'text_hash' not in kb_df.columns:
                    kb_df['text_hash'] = kb_df['text'].astype(str).apply(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()) if 'text' in kb_df.columns else ''
                if 'score' not in kb_df.columns:
                    kb_df['score'] = 0.0

                # 时间窗过滤（默认最近30天）
                now_ts = pd.Timestamp.now()
                min_ts = now_ts - pd.Timedelta(days=self.time_window_days)
                kb_df['timestamp_dt'] = pd.to_datetime(kb_df['timestamp'], errors='coerce')
                kb_df = kb_df[kb_df['timestamp_dt'].fillna(now_ts) >= min_ts].copy()
                if 'timestamp_dt' in kb_df.columns:
                    kb_df = kb_df.drop(columns=['timestamp_dt'])

                # 检查并修正表头
                correct_columns = ['label', 'text', 'embedding', 'timestamp', 'text_hash', 'score']

                # 如果表头完全不匹配，尝试推断表头
                if not any(col in kb_df.columns for col in correct_columns):
                    if len(kb_df.columns) >= 3:  # 假设至少有3列
                        kb_df.columns = correct_columns[:len(kb_df.columns)]
                        # 保存修正的表头
                        kb_df.to_csv(self.kb_file, sep='\t', index=False)
                        print("\n知识库文件表头修复完成")

                # 加载文本和标签
                for _, row in kb_df.iterrows():
                    try:
                        text = row['text']
                        label = row['label']
                        self.text_to_label[text] = label

                        # 如果存在embedding列，加载向量表示
                        if 'embedding' in kb_df.columns:
                            try:
                                # 将字符串表示的向量转换回NumPy数组
                                embedding_str = row['embedding']
                                if isinstance(embedding_str, str):
                                    # 解析字符串形式的向量
                                    embedding_list = eval(embedding_str)
                                    emb_np = np.array(embedding_list, dtype=float).reshape(-1)
                                    norm = np.linalg.norm(emb_np)
                                    if norm > 0:
                                        emb_np = emb_np / norm
                                    self.text_to_embedding[text] = emb_np
                            except Exception as e:
                                print(f"加载向量表示出错: {e}")
                                # 如果加载失败且有编码器，重新生成向量
                                if self.encoder is not None:
                                    emb_np = self.encoder.get_text_vector(text).cpu().numpy().reshape(-1)
                                    norm = np.linalg.norm(emb_np)
                                    if norm > 0:
                                        emb_np = emb_np / norm
                                    self.text_to_embedding[text] = emb_np

                    except Exception as e:
                        print(f"知识库表头错误，缺少必要列: {e}")
                        print(f"当前表头: {kb_df.columns.tolist()}")
                        break

                # 如果有编码器但没有向量，为所有文本生成向量
                if self.encoder is not None and 'embedding' not in kb_df.columns:
                    print("为知识库生成RoBERTa语义向量...")
                    self._generate_embeddings()

                pos_count = sum(val == 1 for val in self.text_to_label.values())
                neg_count = sum(val == 0 for val in self.text_to_label.values())

                print(f"知识库加载完成:")
                print(f"- 总条目: {len(self.text_to_label)}条")
                print(f"- 积极样本: {pos_count}条")
                print(f"- 消极样本: {neg_count}条")
                print(f"- 包含向量表示: {len(self.text_to_embedding)}条")

            else:
                # 如果文件不存在，创建带有列名的空DataFrame
                empty_df = pd.DataFrame(columns=['label', 'text', 'embedding', 'timestamp', 'text_hash', 'score'])
                empty_df.to_csv(self.kb_file, sep='\t', index=False, encoding='utf-8')
                print("\n创建了新的知识库文件")

        except Exception as e:
            print(f"加载知识库出错: {e}")
            traceback.print_exc()

    def _generate_embeddings(self):
        """为知识库中的所有文本生成RoBERTa向量表示"""
        if self.encoder is None or not self.text_to_label:
            return

        for text in self.text_to_label.keys():
            if text not in self.text_to_embedding:
                try:
                    # 获取RoBERTa向量表示
                    embedding = self.encoder.get_text_vector(text).cpu().numpy().reshape(-1)
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    self.text_to_embedding[text] = embedding
                except Exception as e:
                    print(f"生成文本'{text[:20]}...'的向量表示出错: {e}")

    def enhance_prediction(self, text, prob, kb_df=None, similarity_threshold=0.90, weight_max=0.8, fusion_mode="targeted"):
        try:
            base_prob = self._normalize_prob(prob)

            # 先检查是否存在完全匹配的文本--第一级匹配
            if text in self.text_to_label: # 检查是否完全匹配
                kb_label = self.text_to_label[text] # 获取知识库中的标签

                onehot = np.zeros_like(base_prob)
                if 0 <= int(kb_label) < len(onehot):
                    onehot[int(kb_label)] = 1.0
                enhanced_prob = 0.5 * base_prob + 0.5 * onehot

                if not self._enhance_log_shown:
                    print("知识库增强已启用（后续增强日志已省略）")
                    self._enhance_log_shown = True
                return enhanced_prob.tolist()

            # 如果没有完全匹配且有编码器，尝试语义相似度匹配--第二级匹配
            if self.encoder is not None and self.text_to_embedding:
                # 获取输入文本的向量表示
                text_embedding = self.encoder.get_text_vector(text).cpu().numpy()

                # 找到最相似的文本
                max_similarity = -1
                most_similar_text = None

                # 遍历知识库中所有文本的向量表示
                for kb_text, kb_embedding in self.text_to_embedding.items():
                    # 计算余弦相似度
                    similarity = self._cosine_similarity(text_embedding, kb_embedding)

                    # 保留最高相似度及其文本
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_text = kb_text

                # 设定相似度阈值（二分类默认0.90），只有高于此值才进行增强
                if max_similarity > similarity_threshold and most_similar_text is not None:
                    kb_label = self.text_to_label[most_similar_text]

                    # 动态计算权重，相似度越高权重越大
                    weight = (max_similarity - similarity_threshold) / (1 - similarity_threshold)  # 归一化权重
                    weight = min(weight_max, max(0.0, weight))   # 设置上限，避免知识库完全覆盖模型预测，且不为负

                    # 向量式靶向增强（与全文分布融合在 one-hot 条件下等价）
                    enhanced_prob = base_prob.copy()
                    if fusion_mode == "targeted":
                        if 0 <= int(kb_label) < len(enhanced_prob):
                            enhanced_prob = enhanced_prob * (1 - weight)
                            enhanced_prob[int(kb_label)] += weight
                    else:
                        onehot = np.zeros_like(enhanced_prob)
                        if 0 <= int(kb_label) < len(onehot):
                            onehot[int(kb_label)] = 1.0
                        enhanced_prob = (1 - weight) * enhanced_prob + weight * onehot

                    enhanced_prob = self._normalize_prob(enhanced_prob)

                    if not self._enhance_log_shown:
                        print("知识库增强已启用（后续增强日志已省略）")
                        self._enhance_log_shown = True
                    return enhanced_prob.tolist()

        except Exception as e:
            print(f"知识库增强出错: {e}")
            traceback.print_exc()

        # 如果没有匹配或出错，返回原始预测
        return self._normalize_prob(prob).tolist()

    def _cosine_similarity(self, vec1, vec2):
        """计算两个向量之间的余弦相似度"""
        if isinstance(vec1, torch.Tensor):
            vec1 = vec1.cpu().numpy()

        if isinstance(vec2, torch.Tensor):
            vec2 = vec2.cpu().numpy()

        vec1 = vec1.reshape(-1)  # 将形状从(1,2304)变为(2304,)
        vec2 = vec2.reshape(-1)  # 将形状从(1,2304)变为(2304,)

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def update(self, errors_df):
        """基于高置信错误样本更新知识库（时间窗+去重+容量控制）"""
        try:
            if errors_df is None or len(errors_df) == 0:
                print("\n没有新的高置信错误样本，跳过知识库更新")
                return

            now_ts = pd.Timestamp.now()
            min_ts = now_ts - pd.Timedelta(days=self.time_window_days)

            if os.path.exists(self.kb_file) and os.path.getsize(self.kb_file) > 0:
                try:
                    existing_df = pd.read_csv(self.kb_file, sep='\t', encoding='utf-8-sig')
                except Exception:
                    existing_df = pd.read_csv(self.kb_file, sep='\t', encoding='utf-8')
            else:
                existing_df = pd.DataFrame(columns=['label', 'text', 'embedding', 'timestamp', 'text_hash', 'score'])

            existing_df.columns = [str(c).strip().lower() for c in existing_df.columns]
            for col in ['label', 'text', 'embedding', 'timestamp', 'text_hash', 'score']:
                if col not in existing_df.columns:
                    existing_df[col] = np.nan

            existing_df['timestamp_dt'] = pd.to_datetime(existing_df['timestamp'], errors='coerce')
            existing_df = existing_df[existing_df['timestamp_dt'].fillna(now_ts) >= min_ts].copy()

            existing_hashes = set(existing_df['text_hash'].dropna().astype(str).tolist())
            if 'text' in existing_df.columns:
                for t in existing_df['text'].dropna().astype(str):
                    existing_hashes.add(hashlib.md5(t.encode('utf-8')).hexdigest())

            # 重建内存字典
            self.text_to_label = {}
            self.text_to_embedding = {}
            all_embeddings = []
            for _, row in existing_df.iterrows():
                text = str(row.get('text', ''))
                if not text:
                    continue
                try:
                    label = int(row.get('label'))
                except Exception:
                    continue
                self.text_to_label[text] = label
                emb_raw = row.get('embedding')
                if isinstance(emb_raw, str) and emb_raw.strip():
                    try:
                        emb_np = np.array(eval(emb_raw), dtype=float).reshape(-1)
                        norm = np.linalg.norm(emb_np)
                        if norm > 0:
                            emb_np = emb_np / norm
                        self.text_to_embedding[text] = emb_np
                        all_embeddings.append(emb_np)
                    except Exception:
                        pass

            errors_df = errors_df.copy()
            errors_df.columns = [str(c).strip().lower() for c in errors_df.columns]
            for col in ['text', 'label', 'margin', 'entropy', 'error_gap', 'timestamp']:
                if col not in errors_df.columns:
                    errors_df[col] = np.nan

            new_rows = []
            for _, row in errors_df.iterrows():
                text = str(row.get('text', '')).strip()
                if not text:
                    continue

                try:
                    label = int(row.get('label'))
                except Exception:
                    continue

                ts = self._parse_timestamp(row.get('timestamp'))
                if ts is None:
                    ts = now_ts
                if ts < min_ts:
                    continue

                text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                if text_hash in existing_hashes:
                    continue

                if self.encoder is None:
                    continue

                try:
                    emb_np = self.encoder.get_text_vector(text).detach().cpu().numpy().reshape(-1)
                    norm = np.linalg.norm(emb_np)
                    if norm > 0:
                        emb_np = emb_np / norm
                except Exception as e:
                    print(f"生成向量表示出错，跳过该条: {e}")
                    continue

                is_near_dup = False
                for kb_vec in all_embeddings:
                    if self._cosine_similarity(emb_np, kb_vec) > self.near_dup_threshold:
                        is_near_dup = True
                        break
                if is_near_dup:
                    continue

                margin = float(row.get('margin', 0.0) if not pd.isna(row.get('margin', np.nan)) else 0.0)
                entropy = float(row.get('entropy', 0.0) if not pd.isna(row.get('entropy', np.nan)) else 0.0)
                error_gap = float(row.get('error_gap', margin) if not pd.isna(row.get('error_gap', np.nan)) else margin)
                recency = max(0.0, 1.0 - ((now_ts - ts).total_seconds() / (self.time_window_days * 86400.0)))
                score = self.score_lambda * (1.0 - entropy) + self.score_mu * error_gap + self.score_nu * recency

                new_rows.append({
                    'label': label,
                    'text': text,
                    'embedding': emb_np.tolist(),
                    'timestamp': ts.isoformat(),
                    'text_hash': text_hash,
                    'score': float(score),
                })

                existing_hashes.add(text_hash)
                all_embeddings.append(emb_np)
                self.text_to_label[text] = label
                self.text_to_embedding[text] = emb_np

            if not new_rows:
                print("\n没有新的知识需要追加到知识库")
                return

            new_df = pd.DataFrame(new_rows)
            updated_df = pd.concat([existing_df.drop(columns=['timestamp_dt'], errors='ignore'), new_df], ignore_index=True)

            updated_df['timestamp_dt'] = pd.to_datetime(updated_df['timestamp'], errors='coerce').fillna(now_ts)
            updated_df['score'] = pd.to_numeric(updated_df['score'], errors='coerce').fillna(0.0)

            # 每类容量上限 K_max（默认 2e4）
            updated_df = updated_df.sort_values(['label', 'score', 'timestamp_dt'], ascending=[True, False, False])
            updated_df = updated_df.groupby('label', group_keys=False).head(self.max_kb_size_per_class).copy()
            updated_df = updated_df.drop(columns=['timestamp_dt'], errors='ignore')

            updated_df.to_csv(self.kb_file, sep='\t', index=False, encoding='utf-8')
            print(f"\n已追加{len(new_rows)}条新知识到知识库，当前知识库共有{len(updated_df)}条（已启用时间窗/高置信筛选/去重/容量控制）")

        except Exception as e:
            print(f"更新知识库时出错: {e}")
            traceback.print_exc()

    def apply_knowledge(self, x):
        
        # 保持向量不变，仅供兼容旧代码接口
        return x

class SentimentCNNBiLSTM(nn.Module):
    def __init__(self, input_size=768*3, hidden_size=256, num_filters=256, sent_guidance_alpha=0.1):
        super().__init__()
        self.sent_guidance_alpha = sent_guidance_alpha

        # 初始化知识库
        self.knowledge_base = KnowledgeBase()
        self.knowledge_base.load()

        # 多维度特征处理层
        self.feature_projection = nn.Linear(input_size, hidden_size)

        # 两层CNN
        self.conv1 = nn.Sequential(
            nn.Conv1d(hidden_size, num_filters, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),  # 添加批归一化
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(num_filters, num_filters, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),  # 添加批归一化
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        )

        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
            dropout=0.2  # 添加LSTM内部dropout
        )

        # 层次化注意力机制
        # 词级注意力
        self.word_attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        # 句级注意力
        self.sentence_attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        # 输出层
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),  # 增加dropout率
            nn.Linear(hidden_size, 2)  # 二分类：积极/消极
        )

    def forward(self, x):
        # 使用知识库进行预处理
        x = self.knowledge_base.apply_knowledge(x)

        # 多维度特征投影
        projected = self.feature_projection(x)

        # CNN特征提取
        cnn_input = projected.transpose(1, 2)
        conv1_out = self.conv1(cnn_input)
        conv2_out = self.conv2(conv1_out)

        # BiLSTM处理
        lstm_input = conv2_out.transpose(1, 2)
        lstm_out, _ = self.lstm(lstm_input)

        # 层次化注意力机制
        # 1. 词级注意力: 对序列中每个token分配权重
        word_attn_weights = self.word_attention(lstm_out)
        word_context = torch.sum(word_attn_weights * lstm_out, dim=1)

        # 2. 句级注意力: 基于词级信息再次处理
        # 将word_context扩展成与lstm_out相同的形状
        word_context_expanded = word_context.unsqueeze(1).expand(-1, lstm_out.size(1), -1)
        # 组合词级与原始表示
        combined_repr = lstm_out + word_context_expanded * self.sent_guidance_alpha  # 可调注入权重α
        # 再次应用注意力
        sent_attn_weights = self.sentence_attention(combined_repr)
        sent_context = torch.sum(sent_attn_weights * combined_repr, dim=1)

        # 输出层
        output = self.output_layers(sent_context)

        return output

class RoBERTaEncoder:
    def __init__(self, model_path=None, max_length=512):
        """
        初始化RoBERTa编码器
        """
        # 默认使用仓库内的本地模型路径（相对当前脚本）
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(project_root, 'chinese-roberta-wwm-ext')

        print(f"\n加载RoBERTa模型: {model_path}")

        # 加载模型配置
        self.config = AutoConfig.from_pretrained(model_path)

        # 在配置中设置output_hidden_states为True
        self.config.output_hidden_states = True

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 加载模型，启用隐藏状态输出
        self.model = AutoModel.from_pretrained(
            model_path,
            config=self.config,
        )

        # 默认先冻结（可由外部切换）
        self.model.eval()

        # 设置最大长度
        self.max_length = max_length

        # 获取输出维度
        self.embedding_dim = self.config.hidden_size

        # 需要提取的层索引
        self.target_layers = [-1, -4, -8]
        print(f"RoBERTa模型加载完成，使用多层特征提取: {['最后一层' if i==-1 else f'倒数第{abs(i)}层' for i in self.target_layers]}")
        self._embedding_log_shown = False

        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def set_trainable(self, trainable=False):
        for p in self.model.parameters():
            p.requires_grad = trainable
        if trainable:
            self.model.train()
        else:
            self.model.eval()


    def encode_text(self, text, return_tensors="pt"):
        """将文本（单条或批量）编码为token IDs"""
        if text is None:
            return None

        if isinstance(text, str):
            if not text.strip():
                return None
        elif isinstance(text, list):
            if len(text) == 0:
                return None
            text = [str(t) for t in text]
        else:
            return None

        # 使用tokenizer处理文本
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors
        )

        # 将所有输入张量移动到与模型相同的设备上
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return inputs

    def get_embeddings(self, text, layers_to_use="all"):
        
        if not self._embedding_log_shown:
            print("进入get_embeddings方法，开始批量处理文本（后续每条日志已省略）")
            self._embedding_log_shown = True
        if not text or not isinstance(text, str):
            # 返回零向量
            return torch.zeros(1, self.embedding_dim * len(self.target_layers), device=self.device)

        # 编码文本
        inputs = self.encode_text(text)
        if inputs is None:
            return torch.zeros(1, self.embedding_dim * len(self.target_layers), device=self.device)

        # 获取模型输出，无需梯度计算
        with torch.no_grad():
            outputs = self.model(**inputs)

            # 获取所有隐藏层状态
            hidden_states = outputs.hidden_states

            # 提取目标层的表示
            if layers_to_use == "all":
                # 使用预定义的目标层
                selected_layers = [hidden_states[i if i >= 0 else len(hidden_states) + i]
                                for i in self.target_layers]
            elif layers_to_use == "last":
                # 只使用最后一层
                selected_layers = [hidden_states[-1]]
            elif isinstance(layers_to_use, list):
                # 使用指定的层
                selected_layers = [hidden_states[i if i >= 0 else len(hidden_states) + i]
                                for i in layers_to_use]
            else:
                raise ValueError(f"不支持的layers_to_use值: {layers_to_use}")

            # 提取[CLS]令牌的表示（句子级别表示）
            cls_embeddings = [layer[:, 0] for layer in selected_layers]  # 每个形状: [batch_size, hidden_size]

            # 融合多层表示
            # 方法1: 拼接
            concatenated = torch.cat(cls_embeddings, dim=1)  # [batch_size, hidden_size * num_layers]
            return concatenated.to(self.device)  # 确保返回的向量在正确的设备上

    def get_text_vector(self, text):
        """获取文本的向量表示"""

        return self.get_embeddings(text)

    def get_batch_embeddings(self, texts, layers_to_use="all", requires_grad=False):
        """批量获取文本向量；训练时可开启梯度回传到编码器"""
        if not texts:
            return torch.zeros(0, self.embedding_dim * len(self.target_layers), device=self.device)

        inputs = self.encode_text(texts)
        if inputs is None:
            return torch.zeros(0, self.embedding_dim * len(self.target_layers), device=self.device)

        with torch.set_grad_enabled(requires_grad):
            outputs = self.model(**inputs)
            hidden_states = outputs.hidden_states

            if layers_to_use == "all":
                selected_layers = [hidden_states[i if i >= 0 else len(hidden_states) + i]
                                   for i in self.target_layers]
            elif layers_to_use == "last":
                selected_layers = [hidden_states[-1]]
            elif isinstance(layers_to_use, list):
                selected_layers = [hidden_states[i if i >= 0 else len(hidden_states) + i]
                                   for i in layers_to_use]
            else:
                raise ValueError(f"不支持的layers_to_use值: {layers_to_use}")

            cls_embeddings = [layer[:, 0] for layer in selected_layers]
            concatenated = torch.cat(cls_embeddings, dim=1)
            return concatenated.to(self.device)

class IntegratedAnalyzer:
    def __init__(self):
        # 初始化初始数据
        self.data = pd.DataFrame()

        # 初始化RoBERTa编码器和模型
        self.encoder = RoBERTaEncoder()  # 使用默认路径
        self.model = None
        self.unfreeze_encoder = False

        # 检查知识库状态
        self.kb_exists = os.path.exists(os.path.join(current_dir, 'knowledge_base.tsv'))

        # 训练参数
        self.batch_size = 32
        self.lr = 5e-5
        self.epochs = 35
        self.warmup_steps = 0.1
        self.weight_decay = 0.01
        self.grad_clip_norm = 1.0

        # 可调控参数
        self.sent_guidance_alpha = 0.10
        self.similarity_threshold = 0.90
        self.kb_weight_max = 0.80
        self.kb_fusion_mode = "targeted"

        # 知识演化约束
        self.high_conf_margin = 0.15
        self.kb_time_window_days = 30
        self.kb_max_size_per_class = 20000
        self.kb_near_dup_threshold = 0.98
        self.kb_score_lambda = 0.4
        self.kb_score_mu = 0.4
        self.kb_score_nu = 0.2

        self.encoder.set_trainable(self.unfreeze_encoder)

        print("初始化集成分析器，使用RoBERTa模型")
        print("多层语义融合 + 层次化注意力机制 + 无偏知识库增强")

    def preprocess_text(self, text):
        """文本预处理"""
        if not isinstance(text, str) or not text.strip():
            return ""

        # 基础清理: 去除多余空格和特殊符号
        text = re.sub(r'\s+', ' ', text)  # 合并多个空格
        text = text.strip()

        # 去除特殊表情符号等
        emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # 表情
                           u"\U0001F300-\U0001F5FF"  # 符号和象形文字
                           u"\U0001F680-\U0001F6FF"  # 交通和地图
                           u"\U0001F700-\U0001F77F"  # 符号
                           u"\U0001F780-\U0001F7FF"  # 几何形状
                           u"\U0001F800-\U0001F8FF"  # 补充符号
                           u"\U0001F900-\U0001F9FF"  # 补充符号II
                           u"\U0001FA00-\U0001FA6F"  # 扩展符号
                           u"\U0001FA70-\U0001FAFF"  # 其他符号
                           u"\U00002702-\U000027B0"  # Dingbats
                           "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)

        return text

    def analyze(self, train_file, test_file):
        try:
            print("\n" + "="*50)
            print("RoBERTa-CNN-BiLSTM情感分析模型")
            print("="*50)

            # 检查知识库状态
            kb_file = os.path.join(current_dir, 'knowledge_base.tsv')
            if os.path.exists(kb_file) and os.path.getsize(kb_file) > 0:
                kb_df = pd.read_csv(kb_file, sep='\t')
                pos_count = sum(kb_df['label'] == 1)
                neg_count = sum(kb_df['label'] == 0)
                print("\n知识库状态:")
                print(f"- 总条目: {len(kb_df)}条")
                print(f"- 积极样本: {pos_count}条")
                print(f"- 消极样本: {neg_count}条")
            else:
                print("\n未找到已有知识库，将从头开始学习")

            print("\n第一步：数据预处理...")

            # 初始化模型
            # 使用RoBERTa得到的向量尺寸是768*3(多层融合)
            self.model = SentimentCNNBiLSTM(input_size=self.encoder.embedding_dim * len(self.encoder.target_layers), sent_guidance_alpha=self.sent_guidance_alpha)

            # 初始化模型的知识库并传入RoBERTa编码器
            self.model.knowledge_base = KnowledgeBase(
                encoder=self.encoder,
                time_window_days=self.kb_time_window_days,
                max_kb_size_per_class=self.kb_max_size_per_class,
                near_dup_threshold=self.kb_near_dup_threshold,
                score_lambda=self.kb_score_lambda,
                score_mu=self.kb_score_mu,
                score_nu=self.kb_score_nu,
            )
            self.model.knowledge_base.load()

            # 加载TSV文件
            train_data = pd.read_csv(train_file, sep='\t')
            test_data = pd.read_csv(test_file, sep='\t')

            # 处理训练数据
            train_texts = []
            train_labels = []
            for text, label in zip(train_data['text_a'], train_data['label']):
                processed = self.preprocess_text(str(text))
                if processed:  # 如果不为空
                    train_texts.append(processed)
                    train_labels.append(int(label))

            print(f"处理得到 {len(train_texts)} 条训练数据")

            # 训练准备
            print("\n第二步：训练准备...")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"使用设备: {device}")

            # 将RoBERTa模型移到设备上
            self.encoder.model.to(device)

            # 训练模型：只在第一轮训练并保存分类网络；之后各轮直接加载，模型参数不再更新
            weights_file = os.path.join(current_dir, 'classifier.pt')
            if os.path.exists(weights_file):
                print(f"\n第三步：加载已训练的分类网络 {weights_file}，本轮不再训练，只使用并更新知识库")
                self.model.load_state_dict(torch.load(weights_file, map_location=device))
                self.model = self.model.to(device)
            else:
                print("\n第三步：训练RoBERTa-CNN-BiLSTM模型...")
                self._train_model(train_texts, train_labels, num_epochs=self.epochs, batch_size=self.batch_size)
                torch.save(self.model.state_dict(), weights_file)
                print(f"分类网络已保存到 {weights_file}，之后各轮直接加载")

            # 测试模型
            print("\n第四步：模型评估...")
            test_texts = []
            test_labels = []
            for text, label in zip(test_data['text_a'], test_data['label']):
                processed = self.preprocess_text(str(text))
                if processed:
                    test_texts.append(processed)
                    test_labels.append(int(label))

            # 在测试集上评估模型
            predictions, pred_probs = self._evaluate_model(test_texts, batch_size=self.batch_size)
            self._calculate_metrics(test_labels, predictions)

            # 只记录测试集的错误预测
            self._save_error_predictions(predictions, pred_probs, test_labels, test_texts)

            # 更新知识库
            self._update_knowledge_base()

        except Exception as e:
            print(f"分析过程出错: {e}")
            traceback.print_exc()

    def _train_model(self, texts, labels, num_epochs=35, batch_size=32):
        # 将模型设为训练模式
        self.model.train()
        if self.unfreeze_encoder:
            self.encoder.model.train()
        else:
            self.encoder.model.eval()

        # 获取设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        # 初始化损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        params = list(self.model.parameters())
        if self.unfreeze_encoder:
            params += list(self.encoder.model.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

        # 学习率调度器
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

        total_batches = (len(texts) + batch_size - 1) // batch_size
        progress_step = max(1, total_batches // 10)
        print(f"\n开始训练RoBERTa-CNN-BiLSTM模型，共{len(texts)}条数据")
        print("使用多层RoBERTa语义表示 + CNN提取局部特征 + BiLSTM处理全局上下文 + 层次化注意力")
        print("-" * 50)

        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            total = 0

            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("处理批次：", end="", flush=True)

            # 训练阶段
            for batch_idx in range(0, len(texts), batch_size):
                if (batch_idx // batch_size) % progress_step == 0:
                    print("▭", end="", flush=True)

                # 获取当前批次的文本和标签
                batch_texts = texts[batch_idx:batch_idx+batch_size]
                batch_labels = torch.LongTensor(labels[batch_idx:batch_idx+batch_size]).to(device)

                # 联合训练：实时编码并回传梯度到RoBERTa
                batch_tensor = self.encoder.get_batch_embeddings(
                    batch_texts,
                    layers_to_use="all",
                    requires_grad=self.unfreeze_encoder,
                )

                # 确保形状是 [batch, seq_len, dim]
                if len(batch_tensor.shape) == 2:  # 如果是 [batch, dim]
                    batch_tensor = batch_tensor.unsqueeze(1)  # 变为 [batch, 1, dim]

                # 前向传播和反向传播
                optimizer.zero_grad()
                outputs = self.model(batch_tensor)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=self.grad_clip_norm)  # 梯度裁剪
                optimizer.step()

                # 计算指标
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

            # 计算训练指标
            train_accuracy = correct / total
            avg_loss = total_loss / total_batches

            print(f"\n当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"训练损失: {avg_loss:.4f}")
            print(f"训练准确率: {train_accuracy:.4f}")

            # 根据准确率调整学习率
            scheduler.step(train_accuracy)

        print("\n训练完成！")

    def _evaluate_model(self, test_texts, batch_size=16):

        try:
            # 设置模型为评估模式
            self.model.eval()

            # 获取设备
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 初始化知识库（传入RoBERTa编码器）和预测结果列表
            kb = KnowledgeBase(
                encoder=self.encoder,
                time_window_days=self.kb_time_window_days,
                max_kb_size_per_class=self.kb_max_size_per_class,
                near_dup_threshold=self.kb_near_dup_threshold,
                score_lambda=self.kb_score_lambda,
                score_mu=self.kb_score_mu,
                score_nu=self.kb_score_nu,
            )
            predictions = []
            prediction_probs = []

            # 加载知识库
            kb_file = os.path.join(current_dir, 'knowledge_base.tsv')
            self.kb_exists = os.path.exists(kb_file) and os.path.getsize(kb_file) > 0

            # 让知识库类自行加载和处理
            kb.load()

            if self.kb_exists:
                print("使用RoBERTa语义向量进行知识库增强预测")
            else:
                print("无知识库增强，仅使用模型预测")

            print("评估进度：", end="", flush=True)
            total_batches = (len(test_texts) + batch_size - 1) // batch_size
            progress_step = max(1, total_batches // 10)

            # 禁用梯度计算，提高推理效率
            with torch.no_grad():
                for batch_idx in range(0, len(test_texts), batch_size):
                    if (batch_idx // batch_size) % progress_step == 0:
                        print("△", end="", flush=True)

                    # 获取当前批次文本
                    batch_texts = test_texts[batch_idx:batch_idx+batch_size]

                    # 评估阶段实时编码（不回传梯度）
                    batch_tensor = self.encoder.get_batch_embeddings(
                        batch_texts,
                        layers_to_use="all",
                        requires_grad=False,
                    )

                    # 确保形状是 [batch, seq_len, dim]
                    if len(batch_tensor.shape) == 2:  # 如果是 [batch, dim]
                        batch_tensor = batch_tensor.unsqueeze(1)  # 变为 [batch, 1, dim]

                    # 模型推理
                    outputs = self.model(batch_tensor)

                    # 转换为概率
                    probs = F.softmax(outputs, dim=1)

                    # 知识库增强 - 无偏内容整合
                    for i, text in enumerate(batch_texts):
                        prob = probs[i].cpu().tolist()  # 先移到CPU再转换为列表

                        # 如果存在知识库，则进行知识库增强
                        if self.kb_exists:
                            # 使用无偏知识库增强方法（修改后的API不再需要传入kb_df）
                            enhanced_prob = kb.enhance_prediction(text, prob, similarity_threshold=self.similarity_threshold, weight_max=self.kb_weight_max, fusion_mode=self.kb_fusion_mode)
                            pred_label = 1 if enhanced_prob[1] > 0.5 else 0
                            final_prob = enhanced_prob
                        else:
                            # 如果不存在知识库，直接使用模型预测
                            pred_label = 1 if prob[1] > 0.5 else 0
                            final_prob = prob

                        predictions.append(pred_label)
                        prediction_probs.append(final_prob)

            print()
            return predictions, prediction_probs

        except Exception as e:
            print(f"评估过程出错: {e}")
            traceback.print_exc()
            return [], []

    def _calculate_metrics(self, true_labels, predictions):
        """
        计算评估指标：准确率，以及按各类样本数加权的精确率、召回率和F1值（Weighted-F1）
        """
        try:
            # 计算整体评估指标
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, average='weighted')
            recall = recall_score(true_labels, predictions, average='weighted')
            f1 = f1_score(true_labels, predictions, average='weighted')

            print("\n" + "="*50)
            print("模型评估结果:")
            print("="*50)
            print(f"准确率 (Accuracy): {accuracy:.4f}")
            print(f"精确率 (Precision): {precision:.4f}")
            print(f"召回率 (Recall): {recall:.4f}")
            print(f"F1值 (F1-Score): {f1:.4f}")

            # 打印混淆矩阵
            print("\n混淆矩阵:")
            print(f"真实\\预测    消极(0)  积极(1)")

            for i in range(2):
                row = [sum((pred == j) and (true == i) for pred, true in zip(predictions, true_labels)) for j in range(2)]
                label = "消极(0)" if i == 0 else "积极(1)"
                print(f"{label}      {row[0]:^8d} {row[1]:^8d}")

            # 记录评估结果到文件
            with open(os.path.join(current_dir, 'evaluation_results.txt'), 'w', encoding='utf-8') as f:
                f.write(f"准确率 (Accuracy): {accuracy:.4f}\n")
                f.write(f"精确率 (Precision): {precision:.4f}\n")
                f.write(f"召回率 (Recall): {recall:.4f}\n")
                f.write(f"F1值 (F1-Score): {f1:.4f}\n")

            print("\n评估结果已保存到evaluation_results.txt文件")
        except Exception as e:
            print(f"\n计算评估指标出错: {e}")
            traceback.print_exc()

    def _save_error_predictions(self, predictions, pred_probs, true_labels, texts):
        """保存高置信错误预测（margin>=δ）用于知识库演化"""
        errors = {
            'predicted_label': [],
            'label': [],
            'text': [],
            'margin': [],
            'entropy': [],
            'error_gap': [],
            'timestamp': [],
        }

        error_count = 0
        for pred, prob, true, text in zip(predictions, pred_probs, true_labels, texts):
            p = np.array(prob, dtype=float).reshape(-1)
            p = np.clip(p, 1e-12, 1.0)
            p = p / p.sum()

            top2 = np.sort(p)[-2:]
            margin = float(top2[-1] - top2[-2]) if len(top2) == 2 else 0.0
            entropy = float(-(p * np.log(p)).sum() / np.log(len(p)))
            true_prob = float(p[int(true)]) if 0 <= int(true) < len(p) else 0.0
            error_gap = float(np.max(p) - true_prob)

            if pred != true and margin >= self.high_conf_margin:
                error_count += 1
                errors['predicted_label'].append(int(pred))
                errors['label'].append(int(true))
                errors['text'].append(' '.join(text) if isinstance(text, list) else text)
                errors['margin'].append(margin)
                errors['entropy'].append(entropy)
                errors['error_gap'].append(error_gap)
                errors['timestamp'].append(pd.Timestamp.now().isoformat())

        error_file = os.path.join(current_dir, 'error_predictions.tsv')
        error_df = pd.DataFrame(errors)
        error_df.to_csv(error_file, sep='\t', index=False, encoding='utf-8')
        print(f"\n已保存{error_count}条高置信错误预测记录（margin >= {self.high_conf_margin:.2f}）")

        if error_count > 0:
            print("\n错误预测内容随机预览（5条）:")
            preview_n = min(5, len(error_df))
            print(error_df.sample(n=preview_n, random_state=None).to_string(index=False))

    def _update_knowledge_base(self):
        """更新知识库，追加新的错误预测"""
        try:
            # 使用与保存错误预测相同的路径
            error_file = os.path.join(current_dir, 'error_predictions.tsv')
            if os.path.exists(error_file):
                print("\n读取错误预测文件...")
                errors_df = pd.read_csv(error_file, sep='\t')

                print(f"发现 {len(errors_df)} 条高置信错误预测")

                # 更新知识库
                self.model.knowledge_base.update(errors_df)

        except Exception as e:
            print(f"更新知识库时出错: {e}")
            traceback.print_exc()

    def evaluate_with_stats(self, test_texts, batch_size=16):
        """评估并统计KB触发率与分布变化
        返回: predictions, {kb_trigger_rate_total, kb_trigger_rate_semantic, avg_l1_shift}
        """
        try:
            self.model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            kb = KnowledgeBase(
                encoder=self.encoder,
                time_window_days=self.kb_time_window_days,
                max_kb_size_per_class=self.kb_max_size_per_class,
                near_dup_threshold=self.kb_near_dup_threshold,
                score_lambda=self.kb_score_lambda,
                score_mu=self.kb_score_mu,
                score_nu=self.kb_score_nu,
            )
            preds = []

            kb_file = os.path.join(current_dir, 'knowledge_base.tsv')
            self.kb_exists = os.path.exists(kb_file) and os.path.getsize(kb_file) > 0
            kb.load()

            n_total = len(test_texts)
            trig_total = 0
            trig_sem = 0
            l1_sum = 0.0
            eps = 1e-8
            high_theta = 0.9999

            with torch.no_grad():
                total_batches = (len(test_texts) + batch_size - 1) // batch_size
                for batch_idx in range(0, len(test_texts), batch_size):
                    batch_texts = test_texts[batch_idx:batch_idx+batch_size]

                    batch_tensor = self.encoder.get_batch_embeddings(
                        batch_texts,
                        layers_to_use="all",
                        requires_grad=False,
                    )
                    if len(batch_tensor.shape) == 2:
                        batch_tensor = batch_tensor.unsqueeze(1)

                    outputs = self.model(batch_tensor)
                    probs = F.softmax(outputs, dim=1)

                    for i, text in enumerate(batch_texts):
                        p = probs[i].detach().cpu().numpy().tolist()
                        if self.kb_exists:
                            p_enh = kb.enhance_prediction(text, p, similarity_threshold=self.similarity_threshold, weight_max=self.kb_weight_max, fusion_mode=self.kb_fusion_mode)
                            p_enh_high = kb.enhance_prediction(text, p, similarity_threshold=high_theta, weight_max=self.kb_weight_max, fusion_mode=self.kb_fusion_mode)
                            changed_total = any(abs(a-b) > eps for a, b in zip(p_enh, p))
                            changed_high = any(abs(a-b) > eps for a, b in zip(p_enh_high, p))
                            if changed_total:
                                trig_total += 1
                                l1_sum += sum(abs(a-b) for a, b in zip(p_enh, p))
                                if not changed_high:
                                    trig_sem += 1
                            pred_label = 1 if p_enh[1] > 0.5 else 0
                        else:
                            pred_label = 1 if p[1] > 0.5 else 0
                        preds.append(pred_label)

            stats = {
                'kb_trigger_rate_total': (trig_total / n_total) if n_total > 0 else 0.0,
                'kb_trigger_rate_semantic': (trig_sem / n_total) if n_total > 0 else 0.0,
                'avg_l1_shift': (l1_sum / max(trig_total, 1)) if self.kb_exists else 0.0,
            }
            return preds, stats
        except Exception as e:
            print(f"评估+统计出错: {e}")
            traceback.print_exc()
            return [], {'kb_trigger_rate_total': 0.0, 'kb_trigger_rate_semantic': 0.0, 'avg_l1_shift': 0.0}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="DKC模型：每运行一次为一轮")
    parser.add_argument("--train", default=os.path.join(project_root, 'data', 'Pub-data', 'Chn_train.tsv'), help="训练集路径")
    parser.add_argument("--test", default=os.path.join(project_root, 'data', 'Pub-data', 'Chn_test.tsv'), help="测试集（评论池）路径")
    args = parser.parse_args()

    try:
        # 创建分析器并运行
        analyzer = IntegratedAnalyzer()
        analyzer.analyze(args.train, args.test)

    except Exception as e:
        print(f"运行错误: {e}")
        traceback.print_exc()
