import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool 
from src.Agents.agent import workflow 
from psycopg.rows import dict_row
app = FastAPI(title="Not Jarvis Production API")

# Setup the pool WITHOUT opening it yet
DB_URI = os.getenv("DATABASE_URL")


# Update your pool configuration here
connection_pool = AsyncConnectionPool(
    conninfo=DB_URI, 
    max_size=10, 
    open=False,
    timeout=30,  # Connection timeout
    max_idle=300,  # 5 minutes max idle time
    kwargs={
        "autocommit": True, 
        "row_factory": dict_row,
        "prepare_threshold": None  # <--- ADD THIS LINE TO FIX THE ERROR
    },
    check=AsyncConnectionPool.check_connection  # Health check on acquire
)
class ChatRequest(BaseModel):
    user_goal: str
    thread_id: str

@app.on_event("startup")
async def startup():
    try:
        # 1. Explicitly open the pool
        await connection_pool.open()
        
        # 2. Use wait() to ensure connection is live before proceeding
        # This prevents the 30s hang if the password is wrong
        checkpointer = AsyncPostgresSaver(connection_pool)
        await checkpointer.setup()
        
        app.state.checkpointer = checkpointer
        print("INFO: Supabase Connected & Checkpointer Ready.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not connect to Supabase: {e}")
        # Terminate early so you can fix the password
        os._exit(1)

@app.on_event("shutdown")
async def shutdown():
    # Clean up the pool when the server stops
    await connection_pool.close()

# main.py

# main.py

@app.post("/not-jarvis/stream")
async def stream_agent(request: ChatRequest):
    async def event_generator():
        import asyncio
        checkpointer = app.state.checkpointer
        app_instance = workflow.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": request.thread_id}}
        
        async for event in app_instance.astream(
            {"user_goal": request.user_goal}, 
            config=config, 
            stream_mode="updates" 
        ):
            # Iterate through the nodes that ran in this step
            for node_name, values in event.items():
                # Check if this node produced a message for the user
                if "reception_output" in values:
                    output = values.get("reception_output")
                    if output and output.strip():  # Skip empty or whitespace-only outputs
                        # Yield the data packet immediately
                        yield f"data: {output}\n\n"
                        # Small delay to ensure packet is sent
                        await asyncio.sleep(0.1)
        
        # Signal end of stream (optional but helps clients)
        await asyncio.sleep(0.1)  # Ensure last packet is sent
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")