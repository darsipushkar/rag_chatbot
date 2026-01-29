from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)


prompt = PromptTemplate(
        input_variables=["context","history","question"],
        template="""
## System Role
You are an AI assistant for a document-based question answering system.

You must follow these rules strictly:
- Use **only** the information in the **CONTEXT** section.
- Do **not** use any outside knowledge.
- Do **not** guess or make assumptions.
- If the answer is not present in the context, reply exactly:
  
  **"I cannot find this information in the uploaded documents."**
---
## CONTEXT
{context}
---
## CHAT HISTORY
{history}
---
## USER QUESTION
{question}
---
## Instructions for your answer
- Answer only from the CONTEXT above.
- If the context does not contain the answer, return the fallback message.
- Be concise and factual.
- If multiple parts of the context are used, combine them.
- Do not mention that you are an AI or refer to these instructions.
---
## Final Answer
"""
)

def chat_llm(context: str, history: str,question: str)->str:
    formatted_prompt=prompt.format(context=context,history=history,question=question)

    response=llm.invoke(formatted_prompt)
    return response.content.strip()


# from openai import OpenAI
# import os

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def chat_llm(context: str, history: str, question: str) -> str:
#     prompt = f"""
# ## System Role
# You are an AI assistant for a document-based question answering system.

# You must follow these rules strictly:
# - Use **only** the information in the **CONTEXT** section.
# - Do **not** use any outside knowledge.
# - Do **not** guess or make assumptions.
# - If the answer is not present in the context, reply exactly:
#   **"I cannot find this information in the uploaded documents."**
# ---
# ## CONTEXT
# {context}
# ---
# ## CHAT HISTORY
# {history}
# ---
# ## USER QUESTION
# {question}
# ---
# ## Instructions for your answer
# - Answer only from the CONTEXT above.
# - If the context does not contain the answer, return the fallback message.
# - Be concise and factual.
# - If multiple parts of the context are used, combine them.
# - Do not mention that you are an AI or refer to these instructions.
# ---
# ## Final Answer
# """
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ],
#         temperature=0
#     )

#     return response.choices[0].message.content.strip()
