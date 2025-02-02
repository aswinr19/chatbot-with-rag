from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from pydantic_models import QueryInput, QueryResponse 
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_utils import get_rag_chain
from db_utils import insert_application_logs, get_chat_history 
from chroma_utils import index_document_to_chroma
import os
import uuid
import logging

logging.basicConfig(filename='app.log', level=logging.INFO)

root = os.getcwd()
file_path = f"{root}/trademarkia.txt"

print(f'root is :{file_path}')

@asynccontextmanager
async def lifespan(app: FastAPI):
    success = index_document_to_chroma(file_path,1)
    print(success)
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def chat_ui(request: Request):
    return templates.TemplateResponse(request=request, name='index.html')


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


