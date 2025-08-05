from llama_index.core import load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.storage.storage_context import StorageContext
from typing import AsyncGenerator
from llama_index.core.base.response.schema import RESPONSE_TYPE
# from llama_index.vector_stores.faiss import FaissVectorStore
# import faiss

class RAGEngine:
    def __init__(self):
        # documents = SimpleDirectoryReader("papers").load_data()

        # Embeddings
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
         
        # LLM
        # llm = Ollama(model="phi3:3.8b")
        Settings.llm = Ollama(
            model="phi3:3.8b",
            # model="phi3:mini",
            request_timeout=120.0,
            # Manually set the context window to limit memory usage
            context_window=8000,
        )
        # For passing documents in smaller chunks & overlapping
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50

        # # FAISS setup
        # index_flat = faiss.IndexFlatL2(384)
        # vector_store = FaissVectorStore(faiss_index=index_flat)
        # storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Service + Index
        # index = VectorStoreIndex.from_documents(
        #     documents,
        #     storage_context=storage_context,
        # )
        storage_context = StorageContext.from_defaults(persist_dir="vector_store")
        index = load_index_from_storage(storage_context)
        
        self.query_engine = index.as_query_engine(streaming=True, similarity_top_k=5)

    # def query(self, query: str) -> str:
    #     response = self.query_engine.query(query)
    #     return str(response)

    async def query(self, query: str) -> AsyncGenerator[str, None]:
        try:
            response = self.query_engine.query(query)

            print("Streaming response:")
            for token in response.response_gen:
                print(token, end="", flush=True) # This is used to print the token in the same line
                yield token
        except Exception as e:
            print(e)
            yield "Error: " + str(e)