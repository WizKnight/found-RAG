from app.core.config import get_settings


def test_settings_load():
    settings = get_settings()

    assert settings.app_name == "Evidence Grounded Hybrid RAG"
    assert settings.app_env == "development"
    assert settings.api_port == 8000
    assert settings.dense_top_k == 20
    assert settings.bm25_top_k == 20
    assert settings.rerank_top_k == 8