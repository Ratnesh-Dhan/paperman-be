# uvicorn main:app --host 0.0.0.0 --port 8000 --reload

from src.rag_engine import RAGEngine
from src.chat_engine import ChatEngine
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI()
RAGEngine = RAGEngine()
CHATEngine = ChatEngine()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(data: Query):
    try:
        print("we are inside chat endpoint.")
        print("Query: ", data.query)
        return StreamingResponse(CHATEngine.chat(data.query), media_type="event-stream")
    except Exception as e:
        print(e)
        print("Error here")
        return {"error": str(e)}

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
    
@app.get("/test")
def test_endpoint():
    try:
        return {"message": "Python server is running"}
    except Exception as e:
        print(f"Error in testing: {e}")
        return {"error": "Internal server error"}
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) # reload=True is used to reload the server when the code is changed
