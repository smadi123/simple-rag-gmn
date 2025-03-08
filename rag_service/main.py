from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import asyncio
import json  # Import the json module

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# LangChain imports
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama import  OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

class Query(BaseModel):
    question: str
    model_name: str

# --- No QueryResponse model needed for streaming ---

DATA_DIR = "/data/uploads"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

llm = ChatOllama(model="command-r7b-arabic", base_url=OLLAMA_BASE_URL)
embedding_model = OllamaEmbeddings(model="granite-embedding:278m", base_url=OLLAMA_BASE_URL)

files_processed = False

def load_index():
    global files_processed
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        return None
    documents = SimpleDirectoryReader(DATA_DIR, required_exts=[".pdf"]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    files_processed = True
    return index

rag_prompt_template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Answer:
"""
rag_prompt = PromptTemplate.from_template(rag_prompt_template)

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

# --- Streaming Callback Handler ---
class StreamingLLMCallbackHandler(BaseCallbackHandler):
    def __init__(self, queue):
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.queue.put_nowait(token)

    def on_llm_end(self, response, **kwargs) -> None:
        self.queue.put_nowait(None)

    def on_llm_error(self, error, **kwargs) -> None:
        self.queue.put_nowait(None)


@app.post("/query")
async def query(request: Request, query_data: Query):
    global files_processed
    index = load_index()

    async def generate():
        queue = asyncio.Queue()
        handler = StreamingLLMCallbackHandler(queue)

        try:
            if index is not None:
                from langchain_community.vectorstores import Chroma
                from llama_index.core import StorageContext
                from llama_index.core import Settings

                Settings.llm = llm
                Settings.embed_model = embedding_model

                storage_context = StorageContext.from_defaults()
                storage_context.docstore.add_documents(index.docstore.docs.values())
                vectorstore = Chroma(
                    embedding_function=embedding_model,
                    persist_directory=None,
                    storage_context=storage_context
                )
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatOllama(model=query_data.model_name, base_url=OLLAMA_BASE_URL, callbacks=[handler]),
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": rag_prompt},
                )

                # Run the chain in a separate thread
                loop = asyncio.get_event_loop()
                task = loop.run_in_executor(None, qa_chain.invoke, {"query": query_data.question})

                source = "knowledge_base" #Presume it is from knowledge base
                while True:
                    token = await queue.get()
                    if token is None:
                        break
                    # --- Correctly encode as JSON ---
                    data = {"content": token, "source": source}
                    yield f"data: {json.dumps(data)}\n\n"
                await task
                # Check source docs After chain is finished
                if not task.result()["source_documents"]:
                    source = "llm" # update if not from knowledge_base
                    # --- Correctly encode as JSON ---
                    data = {"content": "", "source": source}  # Empty content, just update source
                    yield f"data: {json.dumps(data)}\n\n"
                    return #End

            # Fallback (or if no index)
            model = ChatOllama(model=query_data.model_name, base_url=OLLAMA_BASE_URL, callbacks=[handler])
            chain =  fallback_prompt | model | StrOutputParser()

            loop = asyncio.get_event_loop()
            task = loop.run_in_executor(None, chain.invoke, {"question": query_data.question})

            while True:
                token = await queue.get()
                if token is None:
                    break
                # --- Correctly encode as JSON ---
                data = {"content": token, "source": "llm"}
                yield f"data: {json.dumps(data)}\n\n"
            await task

        except Exception as e:
            # --- Correctly encode error as JSON ---
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")