import re

def extract_query(generation: str) -> str | None:
    match = re.search(r'<query>(.*?)</query>', generation, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_thinking(generation: str) -> str | None:
    match = re.search(r'<think>(.*?)</think>', generation, re.DOTALL)
    return match.group(1).strip() if match else None

def is_title_leaked(query: str, title: str) -> bool:
    return title.lower().strip() in query.lower()