from sqlalchemy import text
from ..ai.embeddings import get_embeddings
 
def search_similar_chunks(db,query:str,top_chunks:int=5):
 
    query_embedding = get_embeddings(query)
 
    sql_string=f""" SELECT content
                 FROM document_chunks
                 ORDER BY embedding <-> (:query_embedding)::vector
                 LIMIT {int(top_chunks)}
                 """
   
    sql=text(sql_string)
 
    result=db.execute(sql,{
         "query_embedding": query_embedding,
         "top_chunks":top_chunks
        }
    )
 
    rows=result.fetchall()
    return [row[0] for row in rows]