import os
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Settings(BaseModel):
    # Database (하이브리드 지원: SQLite + PostgreSQL)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./paper_reviewer.db")

    # PostgreSQL 설정 (선택적)
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "paper_reviewer")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")

    # Redis (개발 중에는 비활성화)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Paper Reviewer System")
    VERSION: str = os.getenv("VERSION", "1.0.0")

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # OpenAI API Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

    # AI/ML Settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_PAPER_RESULTS: int = int(os.getenv("MAX_PAPER_RESULTS", "10"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

    # External APIs
    ARXIV_API_BASE: str = "http://export.arxiv.org/api/query"
    SEMANTIC_SCHOLAR_API: str = "https://api.semanticscholar.org/graph/v1"

    class Config:
        # 환경변수에서 값을 가져오도록 설정
        case_sensitive = True

# 설정 인스턴스 생성
settings = Settings()