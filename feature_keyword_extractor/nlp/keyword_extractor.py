from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_STOPWORDS = {
    "的",
    "了",
    "和",
    "在",
    "是",
    "与",
    "或",
    "及",
    "以及",
    "进行",
    "通过",
    "需要",
    "如果",
    "则",
}

DEFAULT_GENERIC_WORDS = {
    "用户",
    "系统",
    "功能",
    "页面",
    "按钮",
    "模块",
    "支持",
    "可以",
    "需要",
    "相关",
    "信息",
    "内容",
    "安全事件",
    "诈骗电话",
    "紧急电话",
    "恶意应用",
    "存储空间不足",
    "实名",
    "清除实名",
    "二要素",
    "人脸实名",
    "冻结",
    "解冻",
    "风控",
    "注销",
    "自助注销",
    "演示机",
    "门店机",
    "设备管理",
    "信任设备",
    "删除设备",
    "Pad",
    "折叠屏",
    "登录",
    "登录态",
    "开机引导",
    "密码",
    "改密",
    "找回密码",
    "换绑",
    "修改手机",
    "家人共享",
    "家庭群组",
    "孩子账号",
    "云服务",
    "查找设备",
    "双重验证",
    "免校验",
    "学生认证",
    "V7.",
    "V6.",
    "EX6.",
    "版本",
    "灰度",
    "上线",
    "黑名单",
    # 通用业务无关词
    "优化",
    "增加",
    "事件",
    "保护",
    "减少",
    "及时",
    "通知",
    "确认",
    "使用",
    "符合",
    "加入",
    "同意",
    "标题",
    "卡片",
    "提升",
    "提醒",
    "流程"
}


@dataclass(frozen=True)
class KeywordScore:
    term: str
    score: float


class KeywordExtractor:
    def __init__(
        self,
        top_k: int = 10,
        stopwords: Optional[Set[str]] = None,
        generic_words: Optional[Set[str]] = None,
        domain_terms: Optional[Set[str]] = None,
    ):
        self.top_k = top_k
        self.stopwords = set(DEFAULT_STOPWORDS if stopwords is None else stopwords)
        self.generic_words = set(DEFAULT_GENERIC_WORDS if generic_words is None else generic_words)
        self.domain_terms = set(domain_terms or set())
        for term in self.domain_terms:
            jieba.add_word(term, freq=10_000)

    def extract(self, texts: Iterable[str]) -> List[KeywordScore]:
        docs = [text for text in texts if text and text.strip()]
        if not docs:
            return []

        tokenized_docs = [self._tokens(text) for text in docs]
        all_tokens = [token for doc in tokenized_docs for token in doc]
        if not all_tokens:
            return []

        term_frequency = Counter(all_tokens)
        max_tf = max(term_frequency.values())
        tfidf_scores = self._tfidf_scores(tokenized_docs)

        scored = []
        for term, count in term_frequency.items():
            tf_norm = count / max_tf
            tfidf = tfidf_scores.get(term, 0.0)
            length_weight = math.log(len(term) + 1)
            score = 0.55 * tfidf + 0.15 * tf_norm + 0.30 * length_weight
            scored.append(KeywordScore(term=term, score=round(score, 6)))

        scored.sort(key=lambda item: (-item.score, item.term))
        return scored[: self.top_k]

    def _tokens(self, text: str) -> List[str]:
        clean_text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9._+\-#]+", " ", text)
        tokens = []
        for token in jieba.lcut(clean_text):
            token = token.strip()
            if not token or token.isspace():
                continue
            if token in self.domain_terms:
                tokens.append(token)
                continue
            if len(token) <= 1:
                continue
            if token in self.stopwords or token in self.generic_words:
                continue
            tokens.append(token)
        return tokens

    @staticmethod
    def _tfidf_scores(tokenized_docs: List[List[str]]) -> dict[str, float]:
        try:
            vectorizer = TfidfVectorizer(tokenizer=lambda text: text.split(), token_pattern=None)
            matrix = vectorizer.fit_transform([" ".join(tokens) for tokens in tokenized_docs])
            names = vectorizer.get_feature_names_out()
            sums = matrix.sum(axis=0).A1
            max_score = max(sums) if len(sums) else 1.0
            return {name: float(score / max_score) for name, score in zip(names, sums)}
        except ValueError:
            return {}
