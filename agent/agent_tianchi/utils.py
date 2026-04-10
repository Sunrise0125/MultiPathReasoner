# utils.py
import re
import hashlib
from typing import List, Tuple, Any

def safe_text(text: str, max_len:int = 1000) -> str:
    """简单清洗 HTML / 控制长度以降低触发内容审查的风险"""
    if not text:
        return ""
    # remove tags
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_len]

def normalize_query(q: Any) -> str:
    """保证 query 为简单字符串"""
    if q is None:
        return ""
    if isinstance(q, str):
        return q.strip()
    if isinstance(q, (list, tuple)):
        return " ".join(map(str, q))
    if isinstance(q, dict):
        # common keys
        for k in ("query", "q", "text"):
            if k in q:
                return str(q[k]).strip()
        return " ".join(f"{k}:{v}" for k,v in q.items())
    return str(q)

# lightweight similarity: token-set Jaccard, plus hash-based dedup
def token_set_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    sa = set(re.findall(r'\w+', a.lower()))
    sb = set(re.findall(r'\w+', b.lower()))
    if not sa or not sb:
        return 0.0
    inter = sa & sb
    union = sa | sb
    return len(inter) / len(union)

def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
