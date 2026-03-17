from src.services.extractor import Extractor
from src.services.nodes import Nodes
import os
import faiss

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)

from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class Ingestion:
    def __init__(self):
        self.extractor = Extractor()
        self.nodes = Nodes()
        
        self.database = "papers"
        self.persist_dir = "vector_store"

        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            embed_batch_size=64,
            normalize=True
        )
        Settings.chunk_size = 800
        Settings.chunk_overlap = 200
        self.embedding_dim = 384

    def run(self):
        try:
            print("Loading documents...")
            loader = SimpleDirectoryReader(
                input_dir=self.database,
                recursive=True,
                required_exts=[".pdf"]
            )
            documents = loader.load_data()      
            print("Creating FAISS index...")
            faiss_index = faiss.IndexFlatIP(self.embedding_dim)

            vector_store = FaissVectorStore(faiss_index=faiss_index)

            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )

            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
            )

            print("Persisting index...")
            os.makedirs(self.persist_dir, exist_ok=True)
            index.storage_context.persist(persist_dir=self.persist_dir)
            faiss.write_index(faiss_index,
            os.path.join(self.persist_dir, "vector_index.faiss"))
            print("Ingestion completed successfully.")

        except Exception as e:
            print(f"Error during ingestion: {e}")


if __name__ == "__main__":
    ingestion = Ingestion()
    ingestion.run()