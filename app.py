from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from Workflow.workflow import workflow
from router.ask import router as ask
from router.whisper import router as whisper

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize your workflow graph
graph_instance = workflow()
graph = graph_instance.get_graph()

# Define request body schema
class AskRequest(BaseModel):
    session_id: str
    message: str

# Root route
@app.get("/")
def home():
    return {"message": "Hello, World!"}

# @app.include_router(ask)
# @app.include_router(whisper)

# Ask route
@app.post("/ask")
async def ask_agent(request: AskRequest):
    session_id = request.session_id
    user_message = request.message

    if not session_id or not user_message:
        raise HTTPException(status_code=400, detail="Missing session_id or message")

    config = {"configurable": {"session_id": session_id, "thread_id": session_id}}

    try:
        result = graph.invoke({"messages": [HumanMessage(content=user_message)]}, config=config)
        messages = result.get("messages", [])
        response_text = messages[-1].content if messages else "No response generated."
        return {"response": response_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
