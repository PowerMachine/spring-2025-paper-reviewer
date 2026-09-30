from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_, text
import logging

from app.models import Paper, PaperEmbedding, PaperEvaluation
from app.schemas import PaperCreate, PaperUpdate

logger = logging.getLogger(__name__)

# Paper CRUD
def get_paper(db: Session, paper_id: int) -> Optional[Paper]:
    """논문 ID로 논문을 조회합니다."""
    return db.query(Paper).filter(Paper.id == paper_id).first()

def get_paper_by_doi(db: Session, doi: str) -> Optional[Paper]:
    """DOI로 논문을 조회합니다."""
    return db.query(Paper).filter(Paper.doi == doi).first()

def get_paper_by_arxiv_id(db: Session, arxiv_id: str) -> Optional[Paper]:
    """ArXiv ID로 논문을 조회합니다."""
    return db.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()

def get_papers(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    conference: Optional[str] = None,
    year: Optional[int] = None,
    keyword: Optional[str] = None
) -> List[Paper]:
    """논문 목록을 조회합니다."""
    query = db.query(Paper)

    if conference:
        query = query.filter(Paper.conference.ilike(f"%{conference}%"))

    if year:
        query = query.filter(Paper.year == year)

    if keyword:
        query = query.filter(
            or_(
                Paper.title.ilike(f"%{keyword}%"),
                Paper.abstract.ilike(f"%{keyword}%")
            )
        )

    return query.order_by(desc(Paper.created_at)).offset(skip).limit(limit).all()

def get_all_papers(db: Session) -> List[Paper]:
    """모든 논문을 조회합니다."""
    return db.query(Paper).all()

def get_paper_by_id(db: Session, paper_id: int) -> Optional[Paper]:
    """논문 ID로 논문을 조회합니다."""
    return db.query(Paper).filter(Paper.id == paper_id).first()

def create_paper(db: Session, paper: PaperCreate) -> Paper:
    """새 논문을 생성합니다."""
    db_paper = Paper(**paper.model_dump())
    db.add(db_paper)
    db.commit()
    db.refresh(db_paper)
    return db_paper

def update_paper(db: Session, paper_id: int, paper: PaperUpdate) -> Optional[Paper]:
    """논문을 업데이트합니다."""
    db_paper = get_paper_by_id(db, paper_id)
    if db_paper:
        for key, value in paper.model_dump(exclude_unset=True).items():
            setattr(db_paper, key, value)
        db.commit()
        db.refresh(db_paper)
    return db_paper

def delete_paper(db: Session, paper_id: int) -> bool:
    """논문을 삭제합니다."""
    db_paper = get_paper_by_id(db, paper_id)
    if db_paper:
        db.delete(db_paper)
        db.commit()
        return True
    return False

# Embedding CRUD
def get_paper_embedding(db: Session, paper_id: int) -> Optional[PaperEmbedding]:
    """논문의 임베딩을 조회합니다."""
    return db.query(PaperEmbedding).filter(PaperEmbedding.paper_id == paper_id).first()

def get_all_embeddings(
    db: Session,
    skip: int = 0,
    limit: int = 1000
) -> List[PaperEmbedding]:
    """모든 임베딩을 조회합니다."""
    return db.query(PaperEmbedding).offset(skip).limit(limit).all()

def get_papers_with_embeddings(
    db: Session,
    skip: int = 0,
    limit: int = 100
) -> List[Paper]:
    """임베딩이 있는 논문들을 조회합니다."""
    return db.query(Paper).join(PaperEmbedding).offset(skip).limit(limit).all()

# Evaluation CRUD
def get_paper_evaluations(db: Session, paper_id: int) -> List[PaperEvaluation]:
    """논문의 평가 목록을 조회합니다."""
    return db.query(PaperEvaluation).filter(PaperEvaluation.paper_id == paper_id).all()

def get_evaluation(db: Session, evaluation_id: int) -> Optional[PaperEvaluation]:
    """평가 ID로 평가를 조회합니다."""
    return db.query(PaperEvaluation).filter(PaperEvaluation.id == evaluation_id).first()

# Statistics and Analytics
def get_paper_statistics(db: Session) -> Dict[str, Any]:
    """논문 관련 통계를 조회합니다."""
    total_papers = db.query(Paper).count()
    total_embeddings = db.query(PaperEmbedding).count()
    total_evaluations = db.query(PaperEvaluation).count()

    # 연도별 논문 수
    papers_by_year = db.query(
        Paper.year,
        func.count(Paper.id).label('count')
    ).group_by(Paper.year).order_by(Paper.year).all()

    # 컨퍼런스별 논문 수
    papers_by_conference = db.query(
        Paper.conference,
        func.count(Paper.id).label('count')
    ).group_by(Paper.conference).order_by(desc('count')).limit(10).all()

    return {
        "total_papers": total_papers,
        "total_embeddings": total_embeddings,
        "total_evaluations": total_evaluations,
        "papers_by_year": [{"year": year, "count": count} for year, count in papers_by_year],
        "papers_by_conference": [{"conference": conf, "count": count} for conf, count in papers_by_conference]
    }

def search_papers_by_text(
    db: Session,
    search_text: str,
    limit: int = 20
) -> List[Paper]:
    """텍스트로 논문을 검색합니다."""
    return db.query(Paper).filter(
        or_(
            Paper.title.ilike(f"%{search_text}%"),
            Paper.abstract.ilike(f"%{search_text}%")
        )
    ).limit(limit).all()