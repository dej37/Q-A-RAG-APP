from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import get_chain

app = FastAPI(title = " PDF Q/A API")
chain = get_chain()

class Question(BaseModel):
    question: str


@app.post("/ask")

def ask(q: Question):
    response = chain.invoke(q.question)
    return ({"answer": response})

@app.get("/health")
def health():
    return {"status": "api is running"}