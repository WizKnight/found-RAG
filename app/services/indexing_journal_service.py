
from functools import lru_cache

from app.core.config import get_settings
from app.storage.indexing_journal import IndexingJournal


@lru_cache
def get_indexing_journal() -> IndexingJournal:
    settings = get_settings()
    return IndexingJournal(settings.indexing_journal_path)
