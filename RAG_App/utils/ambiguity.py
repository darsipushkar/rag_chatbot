import re

AMBIGUOUS_PRONOUNS = [
    r"\bit\b",
    r"\bthis\b",
    r"\bthat\b",
    r"\bthey\b",
    r"\bthese\b",
    r"\bthose\b"
]

def is_ambiguous(query: str, chat_history: bool) -> bool:
    
    query = query.lower().strip()

    if len(query.split()) <= 3 and not chat_history:
        return True

    if not chat_history:
        for pattern in AMBIGUOUS_PRONOUNS:
            if re.search(pattern, query):
                return True

    return False
