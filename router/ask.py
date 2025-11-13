from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from Workflow.workflow import workflow

router = APIRouter( prefix="/api" )

graph_instance = workflow()
graph = graph_instance.get_graph()


class AskRequest(BaseModel):
    session_id: str
    message: str


@router.post("/ask")
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