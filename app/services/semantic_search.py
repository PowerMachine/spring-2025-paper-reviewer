from typing import List, Dict, Any
import sqlite3
from app.config import settings

class SemanticSearchService:
    def __init__(self):
        self.db_path = settings.DATABASE_URL.replace('sqlite:///', '')

    def search_by_semantic(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # SQL 시맨틱 검색 쿼리 실행
        cursor.execute("""
            SELECT p.id, p.title, p.abstract, p.authors, p.publication_date,
                   p.venue, p.citations, p.url, p.embedding
            FROM papers p
            WHERE p.embedding IS NOT NULL
            ORDER BY similarity(p.embedding, ?) DESC
            LIMIT ?
        """, (query, limit))

        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'title': row[1],
                'abstract': row[2],
                'authors': row[3],
                'publication_date': row[4],
                'venue': row[5],
                'citations': row[6],
                'url': row[7]
            })

        conn.close()
        return results