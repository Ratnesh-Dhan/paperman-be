from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.extractors import TitleExtractor
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.kvstore.simple_kvstore import SimpleKVStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.llms.ollama import Ollama
import os

from llama_index.vector_stores.faiss import FaissVectorStore

class Ingestion:
    def __init__(self, my_llm):
        self.llm = my_llm
        self.index = None
        self.persist_dir="../vector_store"
        self.database = "../../papers"
        self.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.chunk_size = 256
        self.overlap = 30

    def ingestion(self):
        # This is optimized version of the above
        Settings.embed_model = self.embed_model
        documents = SimpleDirectoryReader(self.database).load_data()
        parser = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.overlap)
        nodes = parser.get_nodes_from_documents(documents)

        # This computes and attaches embeddings to each node
        for node in nodes:
            node.embedding = Settings.embed_model.get_text_embedding(node.text)
            self.index = VectorStoreIndex(nodes)
            self.index.storage_context.persist(persist_dir=self.persist_dir)

    def ingestion_pipline(self):
        try:
            print(os.listdir(self.database))
            persist_dir = "../vector_store2"
            os.makedirs(persist_dir, exist_ok=True)

            # Define paths
            vector_path = os.path.join(persist_dir, "default__vector_store.json")
            kv_path = os.path.join(persist_dir, "kvstore.json")
            docstore_path = os.path.join(persist_dir, "docstore.json")

            # Load or initialize vector store
            if os.path.exists(vector_path):
                vector_store = SimpleVectorStore.from_persist_path(vector_path)
            else:
                vector_store = SimpleVectorStore()

            # Load or initialize kvstore
            if os.path.exists(kv_path):
                kvstore = SimpleKVStore.from_persist_path(kv_path)
            else:
                kvstore = SimpleKVStore()

            # Load or initialize docstore
            if os.path.exists(docstore_path):
                docstore = SimpleDocumentStore.from_persist_path(docstore_path)
            else:
                docstore = SimpleDocumentStore()

            # Ingestion cache to avoid re-processing
            cache = IngestionCache(cache=kvstore, collection="ingestion_cache")

            # Setting up local llm
            llm = Ollama(model=self.llm, request_timeout=120.0)

            # Set up the ingestion pipeline
            pipeline = IngestionPipeline(
                transformations=[
                    SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.overlap),
                    TitleExtractor(llm=llm),
                    self.embed_model
                ],
                vector_store=vector_store,
                docstore=docstore,
                cache=cache,
            )

            # Load files from directory
            documents = SimpleDirectoryReader(self.database).load_data()

            # Run ingestion pipeline
            pipeline.run(documents)

            # Save stores to disk
            pipeline.persist(persist_dir)

        except Exception as e:
            print(f"Error in ingestion pipeline: {e}")

    def proper_ingestion_pipeline(self):
        try:
            loader = SimpleDirectoryReader(
                input_dir=self.database,
                recursive=True
            )
            documents = loader.load_data()
            splitter = SentenceSplitter(chunk_size=800, chunk_overlap=200)
            nodes = splitter.get_nodes_from_documents(documents)
            # Metadata Enrichment ( tagging each chunk with souce )
            for node in nodes:
                node.metadata['source'] = node.metadata.get("file_name", "unknown")
                node.metadata['page_number'] = node.metadata.get("page_label", "N/A")

            # Embedding + vector store
            embeding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

            # Faiss persistance store
            faiss_store = FaissVectorStore.from_documents(
                nodes,
                embed_model=embeding_model,
                index_path="../vector_store3/vector_index.faiss"
            )
            faiss_store.save("vector_index.faiss")
        except Exception as e:
            print(f"Error in proper_ingestion_pipeline: {e}")


if __name__ == "__main__":
    ingestion = Ingestion(my_llm="phi3:3.8b")
    ingestion.proper_ingestion_pipeline()
