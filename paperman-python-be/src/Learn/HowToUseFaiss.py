# import faiss
# from llama_index.core import VectorStoreIndex, StorageContext
# from llama_index.vector_stores.faiss import FaissVectorStore

# # Create FAISS index
# dimension = 384  # bge-small-en-v1.5 embedding size
# faiss_index = faiss.IndexFlatL2(dimension)

# vector_store = FaissVectorStore(faiss_index=faiss_index)

# storage_context = StorageContext.from_defaults(
#     vector_store=vector_store
# )

# index = VectorStoreIndex(
#     nodes,
#     storage_context=storage_context,
#     embed_model=embeding_model
# )

# # Persist properly
# index.storage_context.persist(persist_dir="../vector_store3")