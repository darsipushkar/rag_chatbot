from fastapi import APIRouter,Depends
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import ChatMessage,ChatSession
from uuid import UUID
from ..ai.retrieval import search_similar_chunks
from ..ai.llm_chat import chat_llm
from ..chat_history import get_chat_history
from fastapi import HTTPException


router = APIRouter(prefix="/chat",tags=["Chat"])

@router.post("/")
async def Chat(query: str, session_id: str | None = None, document_id: str | None = None, db: Session = Depends(get_db)):

    
    if session_id is None:
        if document_id is None:
            raise HTTPException(status_code=400,detail="document_id is required to start a new chat session")

        new_session = ChatSession(document_id=UUID(document_id))
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        session_id = str(new_session.id)

    else:
        session = db.query(ChatSession).filter(ChatSession.id == UUID(session_id)).first()
        if not session:
            raise HTTPException(status_code=404, detail="Invalid session_id")
        document_id = session.document_id


    
    user_message=ChatMessage(session_id=UUID(session_id),role="user",content=query)
    db.add(user_message)
    db.commit()

    chunks=search_similar_chunks(db=db,query=query,document_id=document_id)
    print(chunks)

    context="\n".join(chunks)

    history=get_chat_history(db,session_id)
    print(history)

    answer=chat_llm(context,history,query)

    ai_message=ChatMessage(session_id=UUID(session_id),role="assistant",content=answer)

    db.add(ai_message)
    db.commit()

    return{
        "session_id":session_id,
        "answer":answer
    }



    
      