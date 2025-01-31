from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic_models import QueryInput, QueryResponse 
from langchain_utils import get_rag_chain
from db_utils import insert_application_logs, get_chat_history 
from chroma_utils import index_document_to_chroma
import os
import uuid
import logging

logging.basicConfig(filename='app.log', level=logging.INFO)

root = os.getcwd()
file_path = f"{root}/trademarkia.txt"

@asynccontextmanager
async def lifespan(app: FastAPI):
    success = index_document_to_chroma(file_path,1)
    print(success)
    yield

app = FastAPI()


@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id or str(uuid.uuid4())
    chat_history = get_chat_history(session_id)
    rag_chain = get_rag_chain(query_input.model.value)

    answer = rag_chain.invoke({
        "input": query_input.question,
        "chat_history": chat_history
    })["answer"]

    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model.value)


