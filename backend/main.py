from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import uvicorn
from typing import AsyncIterator
from langchain_core.messages import HumanMessage

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, streaming=True)


async def stream_llm(prompt: str) -> AsyncIterator[bytes]:
    async for chunk in llm.astream([HumanMessage(content=prompt)]):
        if chunk.content:
            yield chunk.content.encode("utf-8")


@app.get("/chat")
async def chat(prompt: str):
    return StreamingResponse(stream_llm(prompt), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
