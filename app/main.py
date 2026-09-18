from fastapi import FastAPI

app = FastAPI(
    title="Evidence Grounded Hybrid RAG",
    version="0.0.1",
    description="Hybrid retrieval RAG with citation verification."    
)



@app.get("/health")

def health_check() -> dict[str,str]:
    return {"status": "ok"}
