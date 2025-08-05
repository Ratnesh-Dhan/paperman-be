# uvicorn main:app --host 0.0.0.0 --port 8000 --reload

from src.rag_engine import RAGEngine
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

app = FastAPI()
RAGEngine = RAGEngine()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

@app.post("/query")
async def query_endpoint(data: Query):
    try:
        print("We are inside the query endpoint")
        print("Query: ", data.query)

        # async def generate() -> AsyncGenerator[str, None]:
        #     for chunk in RAGEngine.query(data.query):
        #         print(chunk, end="", flush=True) 
        #         yield chunk

        # return StreamingResponse(generate(), media_type="text/plain")
        # return StreamingResponse(generate(), media_type="event-stream")
        return StreamingResponse(RAGEngine.query(data.query), media_type="event-stream")
    except Exception as e:
        print(e)
        return {"error": str(e)}
    
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) # reload=True is used to reload the server when the code is changed
