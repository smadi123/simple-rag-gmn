# rag_service/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter

app = FastAPI()

class Query(BaseModel):
    question: str
    model_name: str

class QueryResponse(BaseModel):
    answer: str
    source: str

DATA_DIR = "/data/uploads"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

Settings.llm = Ollama(model="command-r7b-arabic", base_url=OLLAMA_BASE_URL)
Settings.embed_model = OllamaEmbedding(model_name="granite-embedding:278m", base_url=OLLAMA_BASE_URL)
Settings.node_parser = SentenceSplitter.from_defaults(
    chunk_size=1024,
    chunk_overlap=20
)

def load_index() -> VectorStoreIndex:
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    documents = SimpleDirectoryReader(DATA_DIR, required_exts=[".pdf"]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index

index = load_index()
query_engine = index.as_query_engine(similarity_top_k=3, streaming=False)

@app.post("/query", response_model=QueryResponse)
async def query(query_data: Query):
    try:
        response = query_engine.query(query_data.question)
        if response.source_nodes:
           return QueryResponse(answer=str(response), source="knowledge_base")

        llm = Ollama(model=query_data.model_name, base_url=OLLAMA_BASE_URL)
        prompt = f"""You are a helpful AI assistant for the (كلية القيادة والأركان المشتركة ).
                 Your role is to:
                 - Provide clear and concise answers in Arabic language
                 - Use formal Arabic (الفصحى) in your responses
                 - Be direct and to the point
                 - Be respectful and professional
                 - Focus on accuracy and clarity
                 - When uncertain, acknowledge limitations
                 - Do not use emojis or informal language
                 - Maintain a consistent tone throughout the conversation.

                 Question: {query_data.question}"""

        response = llm.complete(prompt)
        return QueryResponse(answer=str(response), source="llm")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))