from fastapi import APIRouter,Depends
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import ChatMessage,ChatSession
from uuid import UUID
import uuid
from ..ai.retrieval import search_similar_chunks
from ..ai.llm_chat import chat_llm
from ..chat_history import get_chat_history
#from ..utils.ambiguity import is_ambiguous



router = APIRouter(prefix="/chat",tags=["Chat"])

@router.post("/")
async def Chat(query: str, session_id: str | None = None, db: Session = Depends(get_db)):

    # chat_history=False

    if session_id is None:
        new_session =ChatSession()
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        session_id=str(new_session.id)
    # else:
    #     chat_history=True
    
    # if is_ambiguous(query, chat_history):
    #     return {
    #         "session_id": session_id,
    #         "answer": "Could you please clarify what you are referring to?"
    #     }
    
    user_message=ChatMessage(session_id=UUID(session_id),role="user",content=query)
    db.add(user_message)
    db.commit()

    chunks=search_similar_chunks(db,query)

    context="\n".join(chunks)

    history=get_chat_history(db,session_id)

    answer=chat_llm(context,history,query)

    ai_message=ChatMessage(session_id=UUID(session_id),role="assistant",content=answer)

    db.add(ai_message)
    db.commit()

    return{
        "session_id":session_id,
        "answer":answer
    }



    
      