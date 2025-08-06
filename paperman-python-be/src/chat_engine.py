from llama_index.core import load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.storage.storage_context import StorageContext
from typing import AsyncGenerator


class ChatEngine:
    def __init__(self):
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
         
        Settings.llm = Ollama(
            model="phi3:3.8b",
            request_timeout=120.0,
            context_window=8000,
        )
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50


        storage_context = StorageContext.from_defaults(persist_dir="vector_store")
        index = load_index_from_storage(storage_context)
        self.chat_engine = index.as_chat_engine()

    async def chat(self, query: str) -> AsyncGenerator[str, None]:
        try:
            response = self.chat_engine.stream_chat(query)

            print("Streaming response:")
            for token in response.response_gen:
                print(token, end="", flush=True) # This is used to print the token in the same line
                yield token
        except Exception as e:
            print(e)
            yield "Error: " + str(e)
