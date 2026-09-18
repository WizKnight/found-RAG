from fastapi import FastAPI

from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Evidence-grounded hybrid RAG system with dense retrieval, "
        "BM25, cross-encoder reranking, and citation verification."    
)



@app.get("/health")

def health_check() -> dict[str,str]:
    return {"status": "ok",
            "environment": settings.app_env,
            }
