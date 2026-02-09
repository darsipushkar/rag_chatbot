
from sqlalchemy.orm import Session
from .models import ChatMessage
from typing import List, Dict

def get_chat_history(db: Session, session_id: str) -> List[Dict[str, str]]:
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
        .all()
    )

    history = []

    for msg in messages:
        history.append({
            "role": msg.role,        
            "content": msg.content   
        })

    return history
