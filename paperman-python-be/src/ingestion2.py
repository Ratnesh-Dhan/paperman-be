from src.services.extractor import Extractor
from src.services.nodes import Nodes
import os

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings
)
from src.database.qdrantClient import client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class Ingestion2:
    def __init__(self, database):
        self.extractor = Extractor()
        self.nodes = Nodes()
        
        self.database = database
        self.persist_dir = "vector_store"

        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            embed_batch_size=64,
            normalize=True
        )
        Settings.chunk_size = 800
        Settings.chunk_overlap = 200
        self.embedding_dim = 384

        # Qdrant client
        # self.client = QdrantClient(url="http://localhost:6333")

    def run(self):
        all_nodes = []
        try:
            print("Loading documents...")
            for file in os.listdir(self.database):
                if file.endswith(".pdf"):
                    path = os.path.join(self.database, file)

                    pages = self.extractor.extract_pdf_text(path)
                    sections = self.extractor.split_into_sections(pages)
                    nodes = self.nodes.create_nodes(sections)
                    for n in nodes:
                        if "section" not in n.metadata: 
                            n.metadata["section"] = "UNKNOWN"
                    all_nodes.extend(nodes)

            print("Creating QDRANT index...")
            vector_store = QdrantVectorStore(
                client=client,
                collection_name="papers"
            )

            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )

            VectorStoreIndex(
                all_nodes,
                storage_context=storage_context,
            )
            print("Ingestion completed successfully.")

        except Exception as e:
            raise e