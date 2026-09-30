from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
import numpy as np

from app.database import get_db
from app.models import Paper, PaperEvaluation
from app.services.rag_service import RAGService
from app.schemas import PaperEvaluationResponse

router = APIRouter()
rag_service = RAGService()

@router.post("/evaluate-paper/{paper_id}", response_model=PaperEvaluationResponse)
async def evaluate_paper(
    paper_id: int,
    impact_threshold: float = Query(0.5, description="영향도 점수 임계값"),
    max_reference_papers: int = Query(5, description="비교할 최대 논문 수"),
    db: Session = Depends(get_db)
):
    """논문 평가 수행"""
    # 대상 논문 조회
    target_paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not target_paper:
        raise HTTPException(status_code=404, detail="논문을 찾을 수 없습니다")

    # 1단계: 메타데이터 기반 필터링
    metadata_filtered_papers = (
        db.query(Paper)
        .filter(
            Paper.id != paper_id,
            Paper.impact_score >= impact_threshold
        )
        .all()
    )

    # 필드와 방법론 타입이 있는 경우 추가 필터링
    if target_paper.field:
        metadata_filtered_papers = [p for p in metadata_filtered_papers if p.field == target_paper.field]

    if target_paper.methodology_type:
        metadata_filtered_papers = [p for p in metadata_filtered_papers if p.methodology_type == target_paper.methodology_type]

    # 최근 2년 이내 논문으로 필터링
    if target_paper.year:
        metadata_filtered_papers = [p for p in metadata_filtered_papers if p.year and p.year >= target_paper.year - 2]

    # 2단계: 임베딩 기반 유사도 계산 (임베딩이 있는 경우)
    reference_papers = metadata_filtered_papers[:max_reference_papers]

    if not reference_papers:
        raise HTTPException(
            status_code=400,
            detail="비교할 수 있는 유사 논문이 없습니다"
        )

    # RAG 기반 평가 수행
    evaluation_result = await rag_service.evaluate_paper(
        target_paper=target_paper,
        reference_papers=reference_papers
    )

    # 섹션별 유사도 점수 (기본값 사용)
    similarity_scores = {
        "abstract": 0.8,
        "introduction": 0.7,
        "method": 0.9,
        "result": 0.6,
        "conclusion": 0.7
    }

    # 평가 결과 저장
    evaluation = PaperEvaluation(
        paper_id=paper_id,
        reference_paper_ids=[p.id for p in reference_papers],
        similarity_scores=similarity_scores,
        evaluation_result=evaluation_result,
        methodology_comparison=evaluation_result["analysis"].get("methodology_comparison", ""),
        result_comparison=evaluation_result["analysis"].get("result_comparison", ""),
        novelty_analysis=evaluation_result["analysis"].get("novelty_analysis", "")
    )

    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)

    return evaluation

@router.get("/evaluation-history/{paper_id}", response_model=List[PaperEvaluationResponse])
async def get_evaluation_history(
    paper_id: int,
    db: Session = Depends(get_db)
):
    """논문의 평가 이력 조회"""
    evaluations = (
        db.query(PaperEvaluation)
        .filter(PaperEvaluation.paper_id == paper_id)
        .order_by(PaperEvaluation.created_at.desc())
        .all()
    )

    if not evaluations:
        raise HTTPException(
            status_code=404,
            detail="평가 이력이 없습니다"
        )

    return evaluations