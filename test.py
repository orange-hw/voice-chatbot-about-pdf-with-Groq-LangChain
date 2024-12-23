
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Step 1: Configure embeddings and LLM
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
# PDF_DIRECTORY = os.path.join("./file/")
pdf_path = "./attention.pdf"

INDEX_FILE = "faiss_index"

def initialize_embedding_model():
    print("Initializing embeddings...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# def initialize_llm():
#     print("Initializing Groq LLM...")
#     api_key = "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"
#     return ChatGroq(model="llama-3.1-70b-versatile", api_key=api_key, max_tokens=500)

# Step 2: Load PDFs and preprocess documents
# def load_and_preprocess_pdfs(pdf_path):
#     print("Loading and preprocessing PDFs...")
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     documents = []

#     for file_name in os.listdir(pdf_directory):
#         if file_name.endswith(".pdf"):
#             pdf_path = os.path.join(pdf_directory, file_name)
#             loader = PyPDFLoader(pdf_path)
#             raw_text = loader.load()
#             split_texts = text_splitter.split_documents(raw_text)
#             documents.extend(split_texts)

#     return documents

def load_and_preprocess_pdfs(pdf_path):
    print("Loading and preprocessing PDFs...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = []
    loader = PyPDFLoader(pdf_path)
    raw_text = loader.load()
    split_texts = text_splitter.split_documents(raw_text)
    documents.extend(split_texts)

    return documents

# Step 3: Create FAISS vectorstore
def create_or_load_faiss_index(documents, embedding_model):
    if os.path.exists(INDEX_FILE):
        print("Loading existing FAISS index...")
        return FAISS.load_local(INDEX_FILE, embedding_model,allow_dangerous_deserialization=True)
    else:
        print("Creating new FAISS index...")
        faiss_index = FAISS.from_documents(documents, embedding_model)
        faiss_index.save_local(INDEX_FILE)
        return faiss_index

# Step 4: Define semantic search and prompting
def semantic_search(query, faiss_index):
    print("Performing semantic search...")
    results = faiss_index.similarity_search(query, k=5)

    # Combine results into a single context
    context = "\n".join([doc.page_content for doc in results])
    
    return context

# Main Execution
def main():
    embedding_model = initialize_embedding_model()

    documents = load_and_preprocess_pdfs(pdf_path)
    faiss_index = create_or_load_faiss_index(documents, embedding_model)

    while True:
        query = input("what is attention?")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        answer = semantic_search(query, faiss_index)
        print("Answer:", answer)

if __name__ == "__main__":
    main()
