from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, load_index_from_storage, StorageContext, Settings
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from src.database.qdrantClient import client
from typing import AsyncGenerator
import requests, json

class ChatEngine:
    def __init__(self):
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            normalize=True
        )
        self.model = "phi3:mini"
        # persist_dir = "src/vector_store"
        # faiss_index = faiss.read_index(f"{persist_dir}/vector_index.faiss")
        # vector_store = FaissVectorStore(faiss_index=faiss_index)
        # storage_context = StorageContext.from_defaults(
        #     vector_store=vector_store,
        #     persist_dir=persist_dir
        # )
        # index = load_index_from_storage(storage_context)
        # self.retriever = index.as_retriever(similarity_top_k=3,
        # filters={"section": "RESULTS"})             

        vector_store = QdrantVectorStore(
            client=client,
            collection_name='papers'
        )
        index = VectorStoreIndex.from_vector_store(vector_store)
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key='section', value="RESULTS")]
        )
        self.retriever = index.as_retriever(
            similarity_top_k=3,
            filters=filters
        )


    def stream_ollama(self, prompt:str):
        buffer = "\n+==========================+"
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
                    data = json.loads(line.decode("utf-8"))
                    yield data.get("response", "")

            if buffer:
                yield buffer
        except Exception as e:
            print(f"Error in stream_ollama {e}")
            return False
    
    async def chat(self, query: str) -> AsyncGenerator[str, None]:
        try:
            # Step 1: Retrieve context using LlamaIndex
            nodes = self.retriever.retrieve(query)
            print("Retrieved nodes:", len(nodes))
            # For ingestion.py
            # context_str = "\n".join([node.node.get_content() for node in nodes])
            
            #For ingestion2.py
            context_str = "\n\n".join([
            f"[Page {n.node.metadata.get('page')} | Section: {n.node.metadata.get('section')}]\n{n.node.get_content()}"
                for n in nodes
                ])
            # Step 2: Build prompt
            prompt = f"""
            1> Your name is **Paperman**. You are a helpful and knowledgeable research assistant.
            2> If you cant find answers then clerly state that you do not have the context for it.
            3> Use the following context to answer the user's question as clearly and helpfully as possible and do not make the answers too long.

            Context:
            {context_str}

            Question:
            {query}
            """
            # Step 3: Stream directly from Ollama and yield word-by-word
            buffer = ""
            
            for chunk in self.stream_ollama(prompt):
                yield chunk
                # buffer += chunk
                # while " " in buffer:
                #     word, buffer = buffer.split(" ", 1)
                #     yield word + " "
                
            
            # Yield the last part if anything's left
            if buffer:
                yield buffer

        except Exception as e:
            print(e)
            yield "Error: " + str(e)

