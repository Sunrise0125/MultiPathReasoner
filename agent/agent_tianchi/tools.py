# tools.py
import os
import requests
from utils import safe_text
from typing import Optional
from dotenv import load_dotenv
load_dotenv()
SERPER_KEY = os.getenv("SERPER_API_KEY")

def search_serper(query: str, max_snippets: int = 3, timeout: int = 10) -> str:
        
    """
    Use Serper (google.serper.dev) simple snippet fetcher.
    Returns concatenated snippets (cleaned).
    If SERPER_KEY not set or fails, returns empty string.
    """
    if not SERPER_KEY:
        return ""
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"}
    payload = {"q": query}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        j = r.json()
        organics = j.get("organic", [])
        texts = []
        for item in organics[:max_snippets]:
            snippet = item.get("snippet") or item.get("title") or ""
            texts.append(safe_text(snippet, max_len=800))
        return "\n".join(texts)
    except Exception as e:
        print("search_serper error:", e)
        return ""

# Provide a very small fallback "websearch" using Bing/Google scraping would be against TOS.
# In practice, ensure SERPER_KEY is present for reliable results.

if __name__ == "__main__":
    print("=== Serper 搜索测试 ===")

    test_query = "NASA Space Shuttle Enterprise naming year"
    print(f"\n测试查询: {test_query}")

    result = search_serper(test_query)

    if not result:
        print("\n❌ 未获取到结果，请检查：")
        print("1. 是否设置了 SERPER_API_KEY")
        print("2. .env 是否加载成功")
        print("3. API Key 是否有效")
    else:
        print("\n✅ 搜索结果：\n")
        print(result)