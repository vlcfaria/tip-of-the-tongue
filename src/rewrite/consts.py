def get_retriever_prompt(retriever_type: str) -> str:
    if retriever_type == "BM25":
        return "Output a space-separated list of search keywords. For the most distinctive concepts, include 2-3 key synonyms or attribute variants since BM25 cannot expand vocabulary on its own. Describe the unknown entity through its properties (genre, setting, period, themes), never by guessing its name."
    elif retriever_type == "SPLADE":
        return "Write a keyword-focused natural query using distinctive anchor terms and context. SPLADE handles semantic expansion internally so you don't need exhaustive synonyms. Never guess the entity name."
    elif retriever_type == "DENSE":
        return "Write a descriptive natural language query capturing the most distinctive attributes the user remembers — themes, setting, period, emotional tone, relationships, format. Semantic richness matters more than exact terms. Never guess the entity name."
    
def get_system_prompt() -> str:
    return "You are an expert Information Retrieval Query Rewriter."

def get_user_prompt(tot_query: str, retriever_type: str) -> str:
    return f"""\
Your task is to translate an ambiguous "Tip-of-the-Tongue" user query into an optimal search string for a {retriever_type} retrieval system.

User Query: {tot_query}

Instructions for {retriever_type}:
{get_retriever_prompt(retriever_type)}

CRITICAL CONSTRAINTS:
1. DO NOT try to guess the exact name of the target entity (e.g., the specific movie name or book title). If you guess wrong, it will ruin the retrieval.
2. Instead, focus on extracting the core concepts, themes, actions, and constraints (like dates or genres) from the user's query.
3. Clean up any conversational noise ("I'm trying to remember...", "Does anyone know...").
4. Add 2-3 synonyms for the most distinctive concepts only. Avoid over-expanding generic terms like "movie" or "book".

In your <think> tags, reason about which parts of the user's query are the most important anchors, and what safe expansions would help {retriever_type}.
After you are done thinking, wrap your final rewritten query in <query></query> tags.
"""

RETRIEVERS = ['BM25', 'SPLADE', 'DENSE']