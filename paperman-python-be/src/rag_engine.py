from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

class RAGEngine:
    def __init__(self):
        documents = SimpleDirectoryReader("papers").load_data()

        # Embeddings
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
         
        # LLM
        # llm = Ollama(model="phi3:3.8b")
        Settings.llm = Ollama(
            model="phi3:3.8b",
            request_timeout=60.0,
            # Manually set the context window to limit memory usage
            context_window=8000,
        )
        # For passing documents in smaller chunks & overlapping
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50

        # FAISS setup
        index_flat = faiss.IndexFlatL2(384)
        vector_store = FaissVectorStore(faiss_index=index_flat)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Service + Index
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )
        self.query_engine = index.as_query_engine(similarity_top_k=5)

    def query(self, query: str) -> str:
        response = self.query_engine.query(query)
        return str(response)