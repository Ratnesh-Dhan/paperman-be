from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

class RAGEngine:
    def __init__(self):
        documents = SimpleDirectoryReader("papers").load_data()

        # Embeddings
        embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        # LLM
        llm = Ollama(model="phi3:3.8b")
        settings = Settings(embed_model=embedding_model, llm=llm)

        # Chunking
        parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

        # FAISS setup
        index_flat = faiss.IndexFlatL2(384)
        vector_store = FaissVectorStore(faiss_index=index_flat)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Service + Index
        service_context = ServiceContext.from_defaults(embed_model=embedding_model, llm=llm)
        self.index = VectorStoreIndex.from_documents(
            documents,
            service_context=settings,
            storage_context=storage_context,
            transformations=[parser]
        )
        self.query_engine = self.index.as_query_engine(similarity_top_k=5)

    def query(self, query: str) -> str:
        response = self.query_engine.query(query)
        return str(response)