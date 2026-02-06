from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings=OpenAIEmbeddings(model="text-embedding-3-small")

def get_embeddings(text: str)->list[float]:
    return embeddings.embed_query(text)

