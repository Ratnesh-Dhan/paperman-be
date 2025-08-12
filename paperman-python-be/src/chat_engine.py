# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.ollama import Ollama
from llama_index.core import load_index_from_storage, Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.storage.kvstore.simple_kvstore import SimpleKVStore
from llama_index.core import VectorStoreIndex
from typing import AsyncGenerator
import time
import requests, json

class ChatEngine:
    def __init__(self):
        # Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # self.model = "mistral"
        self.model = "phi3:3.8b"


        # storage_context = StorageContext.from_defaults(persist_dir="vector_store")
        # index = load_index_from_storage(storage_context)
        # self.retriever = index.as_retriever()
        # self.chat_engine = index.as_chat_engine(streaming=True)

        persist_dir = r"C:\Users\NDT Lab\Software\SCIENTIFIC-RAG\paperman-be\paperman-python-be\vector_store"
        docstore = SimpleDocumentStore.from_persist_dir(persist_dir=persist_dir)
        vector_store = SimpleDocumentStore.from_persist_dir(persist_dir)
        # vector_store = SimpleVectorStore.from_persist_dir(persist_dir)
        kvstore = SimpleKVStore.from_persist_path(persist_dir)

        storage_context = StorageContext.from_defaults(
            docstore=docstore,
            vector_store=vector_store,
            kvstore=kvstore,
        )

        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )

        self.retriever = index.as_retriever(similarity_top_k=3)
        self.chat_engine = index.as_chat_engine(streaming=True)

    async def chat_with_llama_index(self, query: str) -> AsyncGenerator[str, None]:
        try:
            t0 = time.time()
            response = self.chat_engine.stream_chat(query)
            t1 = time.time()
            print(f"[Timing] Retrieval + LLM start took: {t1 - t0:.2f} sec")

            print("Streaming response:")
            for token in response.response_gen:
                t2 = time.time()
                print(f"[{t2 - t1:.2f}s] {token}", end="", flush=True)
                # print(token, end="", flush=True) # This is used to print the token in the same line
                yield token
        except Exception as e:
            print(e)
            yield "Error: " + str(e)                   

    def stream_ollama(self, prompt:str):
        buffer = ""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                },
                stream=True,
            )
            for line in response.iter_lines():
                if line:
                    try:
                        # yield line.decode("utf-8")
                        data = json.loads(line.decode("utf-8"))
                        token = data.get("response", "")
                        buffer += token
                        while " " in buffer:
                            word, buffer = buffer.split(" ", 1)
                            yield word + " "
                    except json.JSONDecodeError:
                        continue
            if buffer:
                yield buffer
        except Exception as e:
            print(f"Error in stream_ollama {e}")
            return False
    
    async def chat(self, query: str) -> AsyncGenerator[str, None]:
        try:
            # Step 1: Retrieve context using LlamaIndex
            t0 = time.time()
            nodes = self.retriever.retrieve(query)
            t1 = time.time()
            print(f"[Timing] Retrieval + LLM start took: {t1 - t0:.2f} sec")
            context_str = "\n".join([node.text for node in nodes])

            # Step 2: Build prompt
            prompt = f"""Your name is **Paperman**. You are a helpful and knowledgeable research assistant.

            Use the following context to answer the user's question as clearly and helpfully as possible and if not told dont make the answers too long.

            Context:
            {context_str}

            Question:
            {query}
            """
            # Step 3: Stream directly from Ollama and yield word-by-word
            buffer = ""
            for chunk in self.stream_ollama(prompt):
                t2 = time.time()
                try:
                    data = json.loads(chunk.replace("data : ", "").replace("'", '"').rstrip())
                    print(f"Data: {data}")
                    response_text = data.get("response", "")
                    print(f"Response: {response_text}")
                    # print(f"[{t2 - t1:.2f}s] {response_text}", end="", flush=True)
                except Exception:
                    print(f"[{t2 - t1:.2f}s]", end="", flush=True)
                # print(f"[{t2 - t1:.2f}s] {chunk}", end="", flush=True)
                buffer += chunk
                while " " in buffer:
                    word, buffer = buffer.split(" ", 1)
                    yield word + " "
            
            # Yield the last part if anything's left
            if buffer:
                yield buffer

        except Exception as e:
            print(e)
            yield "Error: " + str(e)



        # Settings.llm = Ollama(
        #     model=self.model,
        #     request_timeout=120.0,
        #     context_window=8000,
        # )
        # Settings.chunk_size = 512
        # Settings.chunk_overlap = 50
