import os
import faiss

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)

from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class Ingestion:
    def __init__(self):
        self.database = "../papers"
        self.persist_dir = "../vector_store3"

        # Embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

        # Chunk settings
        self.chunk_size = 800
        self.chunk_overlap = 200

        # bge-small-en-v1.5 embedding dimension
        self.embedding_dim = 384

    def run(self):
        try:
            print("Loading documents...")
            loader = SimpleDirectoryReader(
                input_dir=self.database,
                recursive=True
            )
            documents = loader.load_data()

            print("Splitting into nodes...")
            splitter = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            nodes = splitter.get_nodes_from_documents(documents)

            # Metadata enrichment
            for node in nodes:
                node.metadata["source"] = node.metadata.get("file_name", "unknown")
                node.metadata["page_number"] = node.metadata.get("page_label", "N/A")

            print("Creating FAISS index...")
            faiss_index = faiss.IndexFlatL2(self.embedding_dim)

            vector_store = FaissVectorStore(faiss_index=faiss_index)

            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )

            print("Building VectorStoreIndex...")
            index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                embed_model=self.embed_model
            )

            print("Persisting index...")
            os.makedirs(self.persist_dir, exist_ok=True)
            index.storage_context.persist(persist_dir=self.persist_dir)

            print("Ingestion completed successfully.")

        except Exception as e:
            print(f"Error during ingestion: {e}")


if __name__ == "__main__":
    ingestion = Ingestion()
    ingestion.run()