from sqlalchemy.orm import Session
from .models import ChatMessage

def get_chat_history(db: Session, session_id: str) -> str:
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
        .all()
    )

    history = ""
    for msg in messages:
        history += f"{msg.role}: {msg.content}\n"

    return history
