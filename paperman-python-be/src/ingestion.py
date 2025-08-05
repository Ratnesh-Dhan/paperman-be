from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

class Ingestion:
    def __init__(self):
        self.index = None

    def ingest(self):
        documents = SimpleDirectoryReader("../../papers").load_data()
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.llm = Ollama(
            model="phi3:3.8b",
            # model="phi3:mini",
            request_timeout=60.0,
            context_window=8000,
        )
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
        # Settings.verbose = True
        self.index = VectorStoreIndex.from_documents(documents) 
        self.index.storage_context.persist(persist_dir="../vector_store")

if __name__ == "__main__":
    ingestion = Ingestion()
    ingestion.ingest()
