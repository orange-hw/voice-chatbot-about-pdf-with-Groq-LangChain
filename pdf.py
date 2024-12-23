import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embeddings = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# embeddings = HuggingFaceEmbeddings(model_name="All-MiniLM-L6-v2")

api_key = "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"


llm = ChatGroq(groq_api_key=api_key, model_name="Gemma-7b-It")

loader = PyPDFLoader('attention.pdf')
docs = loader.load()
documents = []
documents.extend(docs)

# Embeddings and text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
st.write(f"Split into {len(docs)} chunks.")

db = FAISS.from_documents(documents=docs,embedding=embeddings)
retriever = db.as_retriever()     

# Adding chat history
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

# input = "what is attention mechanism?"

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Prompt Template
system_prompt = (
    "You are a question-answering assistant. Use the provided retrieved context to respond to the user's query. If the answer is not available, acknowledge it clearly. Limit your response to three concise sentences."
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


with_message_history = RunnableWithMessageHistory(
    rag_chain,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

user_input = "what is attention?"

response = with_message_history.invoke(
    {"input": user_input},
)

print(response["answer"])

