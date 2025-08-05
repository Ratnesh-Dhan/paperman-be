# uvicorn main:app --host 0.0.0.0 --port 8000

from src.rag_engine import RAGEngine
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

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
        print(data)
        result = RAGEngine.query(data.query)
        return {"answer": result}
    except Exception as e:
        print(e)
        return {"error": str(e)}