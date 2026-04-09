import json
from typing import List, Dict, Tuple

from langchain_openai import ChatOpenAI


# Reuse a small, low-temperature model for deterministic control logic.
resolver_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def resolve_question(
    history: List[Dict[str, str]],
    question: str,
    max_history_messages: int = 6,
) -> Tuple[str, str | None]:
    """
    Resolve the user's question into an explicit, context-aware question.

    This helper is intentionally narrow and production-oriented:
    - It does NOT encode any domain-specific rules.
    - It relies on the LLM only to:
        * detect whether the current question introduces a new explicit topic
        * otherwise, infer whether it is a follow-up that should inherit the
          most recent explicit topic from history
    - It always returns:
        * resolved_question: the explicit question to send to the main LLM
        * explicit_topic: the explicit topic string if one is detected in the
          current question, otherwise None
    """

    # Trim history to the most recent messages to keep prompts small and fast.
    recent_history = history[-max_history_messages:] if history else []

    payload = {
        "history": recent_history,
        "question": question,
    }

    prompt = f"""
You are a small control helper for a production chatbot.

Your job:
- Look at the previous chat history and the current user question.
- Decide whether the current question introduces a NEW explicit topic
  (for example: "What is cheque bounce fee?", "Explain overdraft fee") or not.
- If it introduces a new explicit topic, extract that topic as a short string.
- If it does NOT introduce a new explicit topic, but clearly refers to the
  most recent explicit topic discussed in the history (e.g. "what is it?",
  "how much does it cost?"), then construct an explicit, resolved question
  that mentions that topic by name.

Return ONLY a JSON object with this exact shape:
{{
  "resolved_question": "<string>",
  "explicit_topic": "<string or null>"
}}

Rules:
- "explicit_topic" MUST be a short phrase naming the topic (e.g. "cheque bounce fee"),
  or null if the current question does not introduce a NEW topic.
- "resolved_question" MUST always be a well-formed question that can be sent
  directly to a larger model for answering.
- If you cannot confidently resolve the referent for a vague question and the
  history does not provide a single clear topic, keep "resolved_question" equal
  to the original question and set "explicit_topic" to null.

Here is the input, as JSON: 
{json.dumps(payload, ensure_ascii=False)}
"""

    try:
        response = resolver_llm.invoke(prompt)
        raw = response.content.strip()
        data = json.loads(raw)

        resolved_question = data.get("resolved_question") or question
        explicit_topic = data.get("explicit_topic")

        # Normalise explicit_topic to None when it's empty or "null"-like.
        if isinstance(explicit_topic, str):
            explicit_topic = explicit_topic.strip() or None

        return resolved_question, explicit_topic

    except Exception:
        # In case of any parsing or model issues, fall back gracefully to the
        # original question so the main flow is never broken.
        return question, None