#!/usr/bin/env python3
"""
논문 관련 API 엔드포인트
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
from app.database import get_db
from app.models import Paper, PaperEmbedding, PaperEvaluation
from app.schemas import (
    PaperCreate,
    PaperUpdate,
    PaperResponse,
    PaperEmbeddingCreate,
    PaperEmbeddingResponse,
    PaperEvaluationCreate,
    PaperEvaluationResponse
)
from app.services.rag_paper_review_service import RAGPaperReviewService

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=PaperResponse)
def create_paper(paper: PaperCreate, db: Session = Depends(get_db)):
    """새로운 논문 생성"""
    # 중복 체크 (제목 기준)
    existing_paper = db.query(Paper).filter(Paper.title == paper.title).first()
    if existing_paper:
        raise HTTPException(status_code=400, detail="동일한 제목의 논문이 이미 존재합니다.")

    db_paper = Paper(**paper.model_dump())
    db.add(db_paper)
    db.commit()
    db.refresh(db_paper)
    return db_paper

@router.get("/", response_model=List[PaperResponse])
def get_papers(
    skip: int = 0,
    limit: int = 100,
    conference: Optional[str] = None,
    year: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """논문 목록 조회"""
    query = db.query(Paper)

    if conference:
        query = query.filter(Paper.conference == conference)
    if year:
        query = query.filter(Paper.year == year)

    papers = query.offset(skip).limit(limit).all()
    return papers

@router.get("/{paper_id}", response_model=PaperResponse)
def get_paper(paper_id: int, db: Session = Depends(get_db)):
    """특정 논문 조회"""
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if paper is None:
        raise HTTPException(status_code=404, detail="논문을 찾을 수 없습니다.")
    return paper

@router.put("/{paper_id}", response_model=PaperResponse)
def update_paper(paper_id: int, paper: PaperUpdate, db: Session = Depends(get_db)):
    """논문 정보 업데이트"""
    db_paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if db_paper is None:
        raise HTTPException(status_code=404, detail="논문을 찾을 수 없습니다.")

    # 업데이트할 필드만 추출
    update_data = paper.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_paper, field, value)

    db.commit()
    db.refresh(db_paper)
    return db_paper

@router.delete("/{paper_id}")
def delete_paper(paper_id: int, db: Session = Depends(get_db)):
    """논문 삭제"""
    db_paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if db_paper is None:
        raise HTTPException(status_code=404, detail="논문을 찾을 수 없습니다.")

    db.delete(db_paper)
    db.commit()
    return {"message": "논문이 삭제되었습니다."}

# 간단한 리뷰 엔드포인트들 추가
@router.post("/{paper_id}/review")
async def create_paper_review(paper_id: int, db: Session = Depends(get_db)):
    """논문 리뷰 생성"""
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if paper is None:
        raise HTTPException(status_code=404, detail="논문을 찾을 수 없습니다.")

    try:
        # 실제 LLM 서비스 사용
        from app.services.llm_service import get_llm_service
        llm_service = get_llm_service()

        # LLM 서비스를 통한 리뷰 생성
        review_result = await llm_service.generate_paper_review(paper, db=db)

        if "error" in review_result:
            # OpenAI API 사용 불가 시 간단한 리뷰 서비스 대체
            from app.services.simple_review_service import SimpleReviewService
            simple_service = SimpleReviewService()
            review_result = simple_service.generate_review(
                paper.title,
                paper.abstract or "초록 없음",
                "AI/ML"  # 기본 분야
            )
            logger.warning(f"OpenAI API 사용 불가로 Simple Review Service 사용")

        return {
            "status": "success",
            "message": "리뷰가 생성되었습니다.",
            "paper_title": paper.title,
            "note": "AI를 이용하여 분석한 논문 리뷰입니다.",
            "review": {
                "summary": review_result.get("summary", "요약 없음"),
                "score": review_result.get("overall_score", 0),
                "recommendation": review_result.get("recommendation", "판정 없음"),
                "overall_score": review_result.get("overall_score", 0),
                "confidence": review_result.get("confidence", 0),
                "review_content": review_result.get("review_content", "리뷰 내용 없음"),
                "raw_response": review_result.get("raw_response", "원본 응답 없음"),
                "detailed_comments": review_result.get("review_content", "상세 코멘트 없음"),
                "strengths": review_result.get("strengths", []),
                "weaknesses": review_result.get("weaknesses", [])
            }
        }

    except Exception as e:
        logger.error(f"AI 리뷰 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"AI 리뷰 생성 중 오류가 발생했습니다: {str(e)}")

@router.post("/{paper_id}/rag-review")
async def create_rag_review(paper_id: int, db: Session = Depends(get_db)):
    """RAG 리뷰 생성"""
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if paper is None:
        raise HTTPException(status_code=404, detail="논문을 찾을 수 없습니다.")

    try:
        # 실제 RAG 서비스 사용
        rag_service = RAGPaperReviewService()

        # RAG 기반 리뷰 생성
        result = await rag_service.generate_rag_review(db, paper)

        return {
            "status": "success",
            "message": "RAG 리뷰가 생성되었습니다.",
            "review_result": result
        }

    except Exception as e:
        logger.error(f"RAG 리뷰 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"RAG 리뷰 생성 중 오류가 발생했습니다: {str(e)}")

@router.post("/{paper_id}/embedding", response_model=PaperEmbeddingResponse)
def create_paper_embedding(
    paper_id: int,
    embedding: PaperEmbeddingCreate,
    db: Session = Depends(get_db)
):
    """논문 임베딩 생성"""
    # 논문 존재 여부 확인
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if paper is None:
        raise HTTPException(status_code=404, detail="논문을 찾을 수 없습니다.")

    # 기존 임베딩 확인
    existing_embedding = db.query(PaperEmbedding).filter(PaperEmbedding.paper_id == paper_id).first()
    if existing_embedding:
        raise HTTPException(status_code=400, detail="이미 임베딩이 존재합니다.")

    db_embedding = PaperEmbedding(**embedding.model_dump())
    db.add(db_embedding)
    db.commit()
    db.refresh(db_embedding)
    return db_embedding

@router.get("/{paper_id}/embedding", response_model=PaperEmbeddingResponse)
def get_paper_embedding(paper_id: int, db: Session = Depends(get_db)):
    """논문 임베딩 조회"""
    embedding = db.query(PaperEmbedding).filter(PaperEmbedding.paper_id == paper_id).first()
    if embedding is None:
        raise HTTPException(status_code=404, detail="임베딩을 찾을 수 없습니다.")
    return embedding

@router.post("/{paper_id}/evaluation", response_model=PaperEvaluationResponse)
def create_paper_evaluation(
    paper_id: int,
    evaluation: PaperEvaluationCreate,
    db: Session = Depends(get_db)
):
    """논문 평가 생성"""
    # 논문 존재 여부 확인
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if paper is None:
        raise HTTPException(status_code=404, detail="논문을 찾을 수 없습니다.")

    db_evaluation = PaperEvaluation(**evaluation.model_dump())
    db.add(db_evaluation)
    db.commit()
    db.refresh(db_evaluation)
    return db_evaluation

@router.get("/{paper_id}/evaluation", response_model=List[PaperEvaluationResponse])
def get_paper_evaluations(paper_id: int, db: Session = Depends(get_db)):
    """논문 평가 목록 조회"""
    evaluations = db.query(PaperEvaluation).filter(PaperEvaluation.paper_id == paper_id).all()
    return evaluations

@router.post("/sample-data")
async def generate_sample_data(db: Session = Depends(get_db)):
    """샘플 논문 데이터를 생성합니다."""
    try:
        # 샘플 논문 데이터
        sample_papers = [
            {
                "title": "Attention Is All You Need",
                "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
                "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
                "conference": "NIPS",
                "year": 2017,
                "doi": "10.5555/3295222.3295349"
            },
            {
                "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"],
                "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
                "conference": "NAACL",
                "year": 2019,
                "doi": "10.18653/v1/N19-1423"
            },
            {
                "title": "Deep Residual Learning for Image Recognition",
                "authors": ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren"],
                "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.",
                "conference": "CVPR",
                "year": 2016,
                "doi": "10.1109/CVPR.2016.90"
            }
        ]

        created_papers = []
        for paper_data in sample_papers:
            # 중복 체크
            existing = db.query(Paper).filter(Paper.title == paper_data["title"]).first()
            if not existing:
                paper = Paper(**paper_data)
                db.add(paper)
                db.commit()
                db.refresh(paper)
                created_papers.append(paper)

        return {
            "status": "success",
            "message": f"{len(created_papers)}개 샘플 논문을 성공적으로 생성했습니다.",
            "total_created": len(created_papers),
            "papers": [{"id": p.id, "title": p.title} for p in created_papers]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"샘플 데이터 생성 중 오류가 발생했습니다: {str(e)}"
        )