from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings=OpenAIEmbeddings(model="text-embedding-3-small")

def get_embeddings(text: str)->list[float]:
    return embeddings.embed_query(text)

# import os 
# from openai import OpenAI 

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 
# def get_embeddings(text:str)->list[float]: 
#     response = client.embeddings.create(model="text-embedding-3-small",input=text) 
#     return response.data[0].embedding