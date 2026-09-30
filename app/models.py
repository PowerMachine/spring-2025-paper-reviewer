from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
from datetime import datetime

class Paper(Base):
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    authors = Column(JSON)  # 저자 리스트
    abstract = Column(String)
    conference = Column(String, index=True)
    year = Column(Integer, index=True)
    field = Column(String, index=True)  # 연구 분야
    keywords = Column(JSON)    # 키워드 리스트
    doi = Column(String, unique=True, index=True)
    arxiv_id = Column(String, unique=True, index=True)
    url = Column(String)
    impact_score = Column(Float, default=0.0)

    # 섹션별 텍스트
    introduction_text = Column(String)
    method_text = Column(String)
    result_text = Column(String)
    conclusion_text = Column(String)

    # 메타데이터 기반 필터링을 위한 필드
    methodology_type = Column(String, index=True)
    dataset_used = Column(JSON)  # 데이터셋 리스트
    evaluation_metrics = Column(JSON)  # 평가 지표 리스트

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 관계 설정
    embedding = relationship("PaperEmbedding", back_populates="paper", uselist=False)
    evaluations = relationship("PaperEvaluation", back_populates="paper")

class ReferencePaper(Base):
    """PDF 업로드로 추가된 기준 논문들 (RAG 비교 대상)"""
    __tablename__ = "reference_papers"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    authors = Column(JSON)  # 저자 리스트
    abstract = Column(String)
    conference = Column(String, index=True)
    year = Column(Integer, index=True)
    field = Column(String, index=True)  # 연구 분야
    keywords = Column(JSON)    # 키워드 리스트
    doi = Column(String, unique=True, index=True)
    arxiv_id = Column(String, unique=True, index=True)
    url = Column(String)
    impact_score = Column(Float, default=0.8)  # 기준 논문은 기본적으로 높은 점수

    # 섹션별 텍스트
    introduction_text = Column(String)
    method_text = Column(String)
    result_text = Column(String)
    conclusion_text = Column(String)

    # 메타데이터 기반 필터링을 위한 필드
    methodology_type = Column(String, index=True)
    dataset_used = Column(JSON)  # 데이터셋 리스트
    evaluation_metrics = Column(JSON)  # 평가 지표 리스트

    # 논문 품질 지표
    citation_count = Column(Integer, default=0)
    acceptance_status = Column(String, default="accepted")  # accepted, published 등

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 관계 설정
    embedding = relationship("ReferencePaperEmbedding", back_populates="reference_paper", uselist=False)

class PaperEmbedding(Base):
    __tablename__ = "paper_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(Integer, ForeignKey("papers.id"))

    # 섹션별 임베딩 벡터
    abstract_vector = Column(JSON)  # 초록 임베딩 벡터
    introduction_vector = Column(JSON)  # 서론 임베딩 벡터
    method_vector = Column(JSON)  # 방법론 임베딩 벡터
    result_vector = Column(JSON)  # 결과 임베딩 벡터
    conclusion_vector = Column(JSON)  # 결론 임베딩 벡터

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 관계 설정
    paper = relationship("Paper", back_populates="embedding")

class ReferencePaperEmbedding(Base):
    """기준 논문들의 임베딩"""
    __tablename__ = "reference_paper_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    reference_paper_id = Column(Integer, ForeignKey("reference_papers.id"))

    # 섹션별 임베딩 벡터
    abstract_vector = Column(JSON)  # 초록 임베딩 벡터
    introduction_vector = Column(JSON)  # 서론 임베딩 벡터
    method_vector = Column(JSON)  # 방법론 임베딩 벡터
    result_vector = Column(JSON)  # 결과 임베딩 벡터
    conclusion_vector = Column(JSON)  # 결론 임베딩 벡터

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 관계 설정
    reference_paper = relationship("ReferencePaper", back_populates="embedding")

class PaperEvaluation(Base):
    __tablename__ = "paper_evaluations"

    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(Integer, ForeignKey("papers.id"))

    # 참조 논문 ID와 유사도 점수
    reference_paper_ids = Column(JSON)  # 참조 논문 ID 리스트
    similarity_scores = Column(JSON)  # 섹션별 유사도 점수

    # 평가 결과
    evaluation_result = Column(JSON)  # 전체 평가 결과
    methodology_comparison = Column(String)  # 방법론 비교 분석
    result_comparison = Column(String)  # 결과 비교 분석
    novelty_analysis = Column(String)  # 참신성 분석

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 관계 설정
    paper = relationship("Paper", back_populates="evaluations")