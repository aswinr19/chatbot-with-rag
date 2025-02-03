from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List
from langchain_core.documents import Document
from chroma_utils import vectorstore


retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

output_parser = StrOutputParser()

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question"
    "which might reference context in the chat history,"
    "formulate a standalone question which can be understood"
    "without the chat history. Do NOT answer the question,"
    "if its about pricing details, reformat is asking give me the links for pricing"
    "if its about continuing about the registration, reformat it as i want continue to register the trademark"
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Chattorney, a helpful AI assistant at Trademarkia, tasked with assisting users in the process of trademark
    registration.Your job is to guide users through the trademark registration process through trademarkia, answering their questions related to trademark
    registration only. Keep your answers brief and to the point.
    1)Answer only questions related to trademark registration through Trademarkia than do it manually.
    2) If a question is unrelated to trademarks, inform the user that you can only help with trademark registration.
    3) Only provide information from the retrieved context.
    4) If a question is beyond your capabilities, offer the option to speak with a human expert.
    5) When the user is ready to register, prompt them to select a payment plan and make payment."""),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

def get_rag_chain(model="llama3.2:1b"):
    llm = ChatOllama(model=model, temperature=0)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


#  ("system", """You are a helpful AI assistant called Chattorney for a trademark registering firm Trademarkia.
#                   You are tasked with  helping users register trademarks through trademarkia and answer their common queries.
#                   If the user asks any question that is not related to trademark registering don't answer the question.
#                   If the question is beyond your capabilities, ask the user if he wants to talk to human expert.
#                   Guide him through the process of registering and when he is ready to register, prompt him to choose the 
#                   payment plan and pay.Make the answers short and concise. Use the following context to answer the user's question.
#                   answer questions related to trademark registration and also."""),
