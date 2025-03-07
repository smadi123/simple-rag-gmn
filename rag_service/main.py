from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# LlamaIndex imports (for document loading and indexing)
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# LangChain imports (for LLM and embeddings)
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama import  OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

app = FastAPI()

class Query(BaseModel):
    question: str
    model_name: str  # Still accept model name

class QueryResponse(BaseModel):
    answer: str
    source: str

DATA_DIR = "/data/uploads"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# --- LangChain Setup (moved outside load_index) ---
llm = ChatOllama(model="command-r7b-arabic", base_url=OLLAMA_BASE_URL)  # Default LLM
embedding_model = OllamaEmbeddings(model="granite-embedding:278m", base_url=OLLAMA_BASE_URL)

# --- Keep track of whether files have been processed ---
files_processed = False

# --- Modified load_index function ---
def load_index():
    """Loads documents and creates a LlamaIndex VectorStoreIndex.
       Creates an empty index if no files are found.
    """
    global files_processed # Access the global variable
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # --- Check if any PDF files exist ---
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        # --- Return None if no PDF files are found ---
        return None

    documents = SimpleDirectoryReader(DATA_DIR, required_exts=[".pdf"]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    files_processed = True  # Set to True after successful processing
    return index

# --- Don't load the index here on startup ---
# index = load_index()

# --- RAG Prompt Template ---
rag_prompt_template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Answer:
"""
rag_prompt = PromptTemplate.from_template(rag_prompt_template)

# --- Fallback Prompt Template ---
fallback_prompt_template = """You are a helpful AI assistant for the (كلية القيادة والأركان المشتركة ).
    Your role is to:
    - Provide clear and concise answers in Arabic language
    - Use formal Arabic (الفصحى) in your responses
    - Be direct and to the point
    - Be respectful and professional
    - Focus on accuracy and clarity
    - When uncertain, acknowledge limitations
    - Do not use emojis or informal language
    - Maintain a consistent tone throughout the conversation.

    Question: {question}
    Answer:
"""
fallback_prompt = PromptTemplate.from_template(fallback_prompt_template)

@app.post("/query", response_model=QueryResponse)
async def query(query_data: Query):
    global files_processed # Access the global variable
    try:
        # --- Load the index here (on demand) ---
        index = load_index()
        if index is not None:  # Only proceed if an index was created
            # Convert LlamaIndex index to LangChain Chroma (in-memory)
            from langchain_community.vectorstores import Chroma
            from llama_index.core import StorageContext, load_index_from_storage
            from llama_index.core import Settings

            Settings.llm = llm
            Settings.embed_model = embedding_model

            storage_context = StorageContext.from_defaults() # Use in memory
            storage_context.docstore.add_documents(index.docstore.docs.values())
            vectorstore = Chroma(
                embedding_function=embedding_model,
                persist_directory=None,  # In-memory for this example
                storage_context=storage_context
            )

            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            # --- 1. Try RAG ---
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOllama(model=query_data.model_name, base_url=OLLAMA_BASE_URL),
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": rag_prompt}
            )
            rag_result = qa_chain.invoke({"query": query_data.question})

            if rag_result["source_documents"]:
                return QueryResponse(answer=rag_result["result"], source="knowledge_base")

        # --- 2. Fallback to LLM (always available) ---
        fallback_chain = fallback_prompt | ChatOllama(model=query_data.model_name, base_url=OLLAMA_BASE_URL)
        fallback_result = fallback_chain.invoke({"question": query_data.question})
        return QueryResponse(answer=str(fallback_result), source="llm")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))