# uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# from src.rag_engine import RAGEngine
from src.ingestion2 import Ingestion2
from src.chat_engine import ChatEngine
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn, os

Ingestion = Ingestion2(os.path.join("src", "papers"))

app = FastAPI()
# RAGEngine = RAGEngine()
CHATEngine = None
def get_chat_engine():
    global CHATEngine
    if CHATEngine is None:
        CHATEngine = ChatEngine()
    return CHATEngine

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

@app.get("/ingest")
async def ingest_endpoint():
    try:
        status = Ingestion.run()
        global CHATEngine
        CHATEngine = ChatEngine()
        return {"success": status}
    except Exception as e:
        print(f"EXCEPTION DURING INGESTION --> {e}")
        return {"error": str(e)}

@app.post("/chat")
async def chat_endpoint(data: Query):
    try:
        CHATEngine = get_chat_engine()
        print("we are inside chat endpoint.")
        print("Query: ", data.query)
        return StreamingResponse(CHATEngine.chat(data.query), media_type="text/event-stream")
    except Exception as e:
        print(e)
        print("Error here")
        return {"error": str(e)}


# @app.post("/query")
# async def query_endpoint(data: Query):
#     try:
#         print("We are inside the query endpoint")
#         print("Query: ", data.query)
#         return StreamingResponse(RAGEngine.query(data.query), media_type="event-stream")
#     except Exception as e:
#         print(e)
#         return {"error": str(e)}
    
@app.get("/test")
def test_endpoint():
    try:
        return {"message": "Python server is running"}
    except Exception as e:
        exception_name = type(e).__name__
        print("Error in testing : ",exception_name)

@app.get("/")
def test_run_endpoint():
    return {"Python server is running smoothly on server 8000"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=False, log_level="info")
    # uvicorn.run("main:app", host="localhost", port=8000, reload=True, log_level="info")