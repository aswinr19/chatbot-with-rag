import os
import requests
from typing import Annotated
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Field, Session, SQLModel, create_engine, select
from pydantic import BaseModel
from datetime import datetime


OLLAMA_SERVER_URL = "http://localhost:11434"
sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

class Chat(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    human: str | None = Field(default=None)
    bot: str | None = Field(default=None)
    created_at: datetime | None = Field(default=datetime.utcnow) 

class Query(BaseModel):
    prompt: str
    model: str

class RaggedModel:
    def __init__(self, model):
        self.local_llm = model
        self.llm = ChatOllama(model=self.local_llm, temperature=0)

    def load_model(self):
        self.root = os.getcwd()
        self.loader = TextLoader(f"{self.root}/src/trademarkia.txt")
        self.docs = self.loader.load()
        
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.splits = self.text_splitter.split_documents(self.docs)
        
        self.vectorstore = SKLearnVectorStore.from_documents(
            documents=self.splits,
            embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
            )
        self.retriever = self.vectorstore.as_retriever(k=3)
       
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a virtual assistant called CHATTORNEY for the trade mark registering law firm LegalForce which operates under the name Trademarkia. You are tasked with helping people register for trademarks. You should talk very politely with the customer. And you should help the customer register trademarks, explain about pricing of trademarking and assist in simple tasks. The different information regarding the different services, pricing, and features of trademarkia are all given in the context. You should always anser from the context only. If you dont know the answer from the context, you should answer that you can't help with this and should ask the customer to if he want to pass it to a human expert. Use the following pieces of retrieved context to augment your own knowledge."),
            ("human", "Context: {context}"),
            ("human", "Question: {input}"),
            ("human", "Please provide an short answer with 2-4 sentences and is relevent to the question from the retrieved context, but don't mention the context. Always be polite when answering.")
        ])
        
        self.question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, self.question_answer_chain)
        
    def ask_question(self,question):
        result = self.rag_chain.invoke({"input": question})
        return result['answer']
  
    def close(self):
        pass


connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)

def create_db_and_tables():
    try:
        SQLModel.metadata.create_all(engine)
    except Exception as e:
        print(f"Error creating tables: {e}")

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]

model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(sqlite_file_name):
        create_db_and_tables()

    model["chat_model"] = RaggedModel(model="llama3.2:1b")
    model["chat_model"].load_model()
    
    yield

    model.clear()

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
async def health_check(request: Request):
    return templates.TemplateResponse(request=request, name='index.html')

@app.post("/generate")
async def generate_response(query: Query, session: SessionDep):
    try:
        chat_model: RaggedModel = model["chat_model"]
        response = chat_model.ask_question(question=query.prompt)
        print(f"{query.prompt}: {response}")

        chat_data = Chat(human=query.prompt, bot=response, created_at=datetime.now()) 
        session.add(chat_data)
        session.commit()
        session.refresh(chat_data)

        return { "generated_text": response }
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error while trying to communicate to ollama {str(e)}")


@app.get("/all-chats")
async def get_chats(session: SessionDep):
    chats = session.exec(select(Chat).offset(0).limit(5)).all()

    print(chats)

    return chats

