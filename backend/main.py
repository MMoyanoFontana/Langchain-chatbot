from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import uvicorn
from typing import AsyncIterator
from langchain_core.messages import HumanMessage
from app.db import init_db
from app.routers.catalog import router as catalog_router
from app.routers.users import router as users_router

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users_router)
app.include_router(catalog_router)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, streaming=True)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


async def stream_llm(prompt: str) -> AsyncIterator[bytes]:
    async for chunk in llm.astream([HumanMessage(content=prompt)]):
        if chunk.content:
            yield chunk.content.encode("utf-8")


@app.get("/chat")
async def chat(prompt: str):
    return StreamingResponse(stream_llm(prompt), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
