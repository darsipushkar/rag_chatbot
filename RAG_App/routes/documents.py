from fastapi import APIRouter,UploadFile,File,Depends
from sqlalchemy.orm import Session
from ..database import get_db
from ..utils.pdf import extract_text_from_pdf
from ..utils.chunker import chunked_text
from ..ai.embeddings import get_embeddings
from sqlalchemy import text
import json
from ..models import Document


router = APIRouter(prefix="/documents",tags=["Documents"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...),db: Session = Depends(get_db)):
    
    file_content = await file.read()
    extracted_text = extract_text_from_pdf(file_content)
    
    chunks = chunked_text(extracted_text)

    new_doc=Document(filename=file.filename)
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)

    document_id=new_doc.id

    for i,chunk in enumerate(chunks):
        embedding=get_embeddings(chunk)

        db.execute(text("""INSERT INTO document_chunks (document_id,content,embedding,metadata) VALUES(:document_id,:content,:embedding,:metadata)"""),{
            "document_id":document_id,
            "content":chunk,
            "embedding":embedding,
            "metadata":json.dumps({"chunk_index":i})
        })

    db.commit()

    return{
        "filename":file.filename,
        "Characters_extracted":len(extracted_text),
        "total_chunks":len(chunks),
        "document_id":str(document_id)
    }
    