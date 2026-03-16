from llama_index.core import load_index_from_storage, Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
# from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from typing import AsyncGenerator
from llama_index.core.prompts import RichPromptTemplate
import faiss

class RAGEngine:
    def __init__(self):

        rich_promt = RichPromptTemplate("""
                "Youre name is 'paperman' and you assistant in scientific laboratory.\n"
                "Use the following context to answer the question as accurately as possible.\n"
                "If you don't know the answer, say you don't have the context for that.\n\n"
                "Context:\n{context_str}\n\n"
                "Question: {query_str}\n\n"
                "Answer:"
            """
        )
        # Embeddings
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
         
        # LLM
        # llm = Ollama(model="phi3:3.8b")
        Settings.llm = Ollama(
            model="mistral:latest",
            # model="phi3:mini",
            request_timeout=120.0,
            # Manually set the context window to limit memory usage
            context_window=8000,
        )
        # For passing documents in smaller chunks & overlapping
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50


        # storage_context = StorageContext.from_defaults(persist_dir="vector_store3")
        # index = load_index_from_storage(storage_context)
        persist_dir = "src/vector_store"
        # faiss_index = faiss.read_index(f"{persist_dir}/vector_index.faiss")

        # vector_store = FaissVectorStore.from_persist_dir(persist_dir)
        storage_context = StorageContext.from_defaults(
            persist_dir=persist_dir,
            # vector_store=vector_store
        )

        index = load_index_from_storage(storage_context)
        
        self.query_engine = index.as_query_engine(streaming=True, similarity_top_k=5, text_qa_template=rich_promt)

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



# from llama_index.core import load_index_from_storage

# storage_context = StorageContext.from_defaults(
#     persist_dir="../vector_store"
# )

# index = load_index_from_storage(storage_context)