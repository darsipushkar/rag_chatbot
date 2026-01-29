from fastapi import FastAPI
from .database import engine
from .database import Base
from .routes import documents,chat

app=FastAPI()

Base.metadata.create_all(bind=engine)


app.include_router(documents.router)
app.include_router(chat.router)