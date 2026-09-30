from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime

# Paper 스키마
class PaperBase(BaseModel):
    """논문 기본 스키마"""
    title: str
    authors: List[str]
    abstract: str
    conference: str
    year: int
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    impact_score: float = 0.0
    introduction_text: Optional[str] = None
    method_text: Optional[str] = None
    result_text: Optional[str] = None
    conclusion_text: Optional[str] = None
    methodology_type: Optional[str] = None
    dataset_used: Optional[List[str]] = None
    evaluation_metrics: Optional[List[str]] = None

class PaperCreate(PaperBase):
    """논문 생성 스키마"""
    pass

class PaperUpdate(BaseModel):
    """논문 업데이트 스키마"""
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    conference: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    impact_score: Optional[float] = None
    introduction_text: Optional[str] = None
    method_text: Optional[str] = None
    result_text: Optional[str] = None
    conclusion_text: Optional[str] = None
    methodology_type: Optional[str] = None
    dataset_used: Optional[List[str]] = None
    evaluation_metrics: Optional[List[str]] = None

class PaperResponse(PaperBase):
    """논문 응답 스키마"""
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class PaperEmbeddingBase(BaseModel):
    """논문 임베딩 기본 스키마"""
    paper_id: int
    abstract_vector: Optional[List[float]] = None
    introduction_vector: Optional[List[float]] = None
    method_vector: Optional[List[float]] = None
    result_vector: Optional[List[float]] = None
    conclusion_vector: Optional[List[float]] = None

class PaperEmbeddingCreate(PaperEmbeddingBase):
    """논문 임베딩 생성 스키마"""
    pass

class PaperEmbeddingResponse(PaperEmbeddingBase):
    """논문 임베딩 응답 스키마"""
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class PaperWithSimilarity(PaperResponse):
    similarity_score: float = Field(..., description="유사도 점수")

# Embedding 스키마
class EmbeddingCreate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    paper_id: int
    embedding: List[float]
    model_name: str
    embedding_type: str = "abstract"

class EmbeddingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    paper_id: int
    model_name: str
    embedding_type: str
    created_at: datetime

# Query 스키마
class QueryRequest(BaseModel):
    query_text: str = Field(..., description="검색 질의")
    max_results: int = Field(default=3, description="최대 결과 수")
    similarity_threshold: float = Field(default=0.7, description="유사도 임계값")

class QueryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    query_text: str
    similar_papers: Optional[Dict[str, Any]] = None
    timestamp: datetime
    response_time: Optional[float] = None

# Review 스키마
class ReviewContent(BaseModel):
    summary: str = Field(..., description="논문 요약")
    strengths: List[str] = Field(..., description="강점 목록")
    weaknesses: List[str] = Field(..., description="약점 목록")
    questions: List[str] = Field(..., description="질문 목록")
    score: int = Field(..., ge=1, le=10, description="점수 (1-10)")
    recommendation: str = Field(..., description="추천 사항")
    detailed_analysis: str = Field(..., description="상세 분석")

class ReviewCreate(BaseModel):
    paper_id: int
    query_id: int
    review_content: ReviewContent
    similarity_score: float
    ai_model_used: str = "gpt-4"

class ReviewResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    paper_id: int
    query_id: int
    review_content: ReviewContent
    similarity_score: float
    ai_model_used: str
    created_at: datetime

# Search Result 스키마
class SearchResult(BaseModel):
    papers: List[PaperWithSimilarity] = Field(..., description="검색된 논문 목록")
    total_count: int = Field(..., description="전체 검색 결과 수")
    query_time: float = Field(..., description="검색 소요 시간")
    query_id: int = Field(..., description="쿼리 ID")

# Complete Review Result 스키마
class CompleteReviewResult(BaseModel):
    query: QueryResponse
    papers: List[PaperWithSimilarity]
    reviews: List[ReviewResponse]
    total_processing_time: float

# Statistics 스키마
class PaperStats(BaseModel):
    total_papers: int
    total_embeddings: int
    total_queries: int
    total_reviews: int
    avg_similarity_score: float
    most_popular_conferences: List[Dict[str, Any]]

# Health Check 스키마
class HealthCheck(BaseModel):
    """헬스 체크 응답 스키마"""
    status: str
    database: str

class PaperEvaluationBase(BaseModel):
    """논문 평가 기본 스키마"""
    paper_id: int
    reference_paper_ids: List[int]
    similarity_scores: Dict[str, float]
    evaluation_result: Dict[str, Any]
    methodology_comparison: Optional[str] = None
    result_comparison: Optional[str] = None
    novelty_analysis: Optional[str] = None

class PaperEvaluationCreate(PaperEvaluationBase):
    """논문 평가 생성 스키마"""
    pass

class PaperEvaluationResponse(PaperEvaluationBase):
    """논문 평가 응답 스키마"""
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True