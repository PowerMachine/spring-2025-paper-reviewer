from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any
import os
import logging
import json

from app.database import get_db
from app.services.pdf_processor import PDFProcessor, ParsedPaper
from app.services.embedding_service import EmbeddingService
from app.services.duplicate_checker import DuplicateChecker
from app.models import Paper, PaperEmbedding, ReferencePaper, ReferencePaperEmbedding
from app.schemas import PaperCreate

logger = logging.getLogger(__name__)
router = APIRouter()

# 업로드 디렉토리 생성
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload-paper", response_model=Dict[str, Any])
async def upload_paper(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """PDF 논문 파일 업로드 및 처리"""

    # 파일 검증
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    if file.size and file.size > 10 * 1024 * 1024:  # 10MB 제한
        raise HTTPException(status_code=400, detail="파일 크기는 10MB 이하여야 합니다.")

    try:
        # 1. PDF 파일 읽기
        pdf_content = await file.read()

        # 2. PDF 처리
        pdf_processor = PDFProcessor()
        parsed_paper = await pdf_processor.process_pdf(pdf_content)

        # 3. 중복 체크
        duplicate_checker = DuplicateChecker()
        is_duplicate, existing_paper = duplicate_checker.check_duplicate(
            db=db,
            title=parsed_paper.title,
            authors=parsed_paper.authors,
            doi=None,  # PDF에서는 DOI 추출이 어려움
            arxiv_id=None  # PDF에서는 ArXiv ID 추출이 어려움
        )

        if is_duplicate:
            logger.info(f"중복 논문 발견: {parsed_paper.title} (기존 ID: {existing_paper.id})")
            return {
                "status": "duplicate",
                "message": "이미 데이터베이스에 존재하는 논문입니다.",
                "existing_paper_id": existing_paper.id,
                "existing_paper_title": existing_paper.title,
                "extracted_info": {
                    "title": parsed_paper.title,
                    "authors": parsed_paper.authors,
                    "abstract": parsed_paper.abstract[:200] + "..." if parsed_paper.abstract and len(parsed_paper.abstract) > 200 else parsed_paper.abstract,
                    "year": parsed_paper.year,
                    "conference": parsed_paper.conference,
                }
            }

        # 4. 임베딩 생성
        embedding_service = EmbeddingService()

        # 임베딩할 텍스트 준비 (제목 + 초록 + 내용 일부)
        embedding_text = ""
        if parsed_paper.title:
            embedding_text += f"Title: {parsed_paper.title}\n"
        if parsed_paper.abstract:
            embedding_text += f"Abstract: {parsed_paper.abstract}\n"
        if parsed_paper.content:
            # 내용의 첫 1000자만 사용
            content_preview = parsed_paper.content[:1000]
            embedding_text += f"Content: {content_preview}"

        embedding_vector = await embedding_service.create_embedding(embedding_text)

        # 4. 데이터베이스에 저장
        # authors를 리스트로 변환
        authors_list = []
        if parsed_paper.authors:
            # 쉼표나 세미콜론으로 분리하여 리스트로 변환
            authors_str = parsed_paper.authors.replace(';', ',')
            authors_list = [author.strip() for author in authors_str.split(',') if author.strip()]

        if not authors_list:
            authors_list = ["Unknown"]

        # Paper 객체 직접 생성 (모든 필드 포함)
        db_paper = Paper(
            title=parsed_paper.title or f"Uploaded Paper - {file.filename}",
            authors=json.dumps(authors_list),  # JSON 문자열로 저장
            abstract=parsed_paper.abstract or "No abstract extracted",
            conference=parsed_paper.conference or "Unknown",
            year=parsed_paper.year or 2024,
            field=None,  # 추후 분류 가능
            keywords=json.dumps([]),  # 빈 키워드 리스트
            doi=None,
            arxiv_id=None,
            url=None,
            impact_score=0.0,
            introduction_text=None,
            method_text=None,
            result_text=None,
            conclusion_text=None,
            methodology_type=None,
            dataset_used=json.dumps([]),
            evaluation_metrics=json.dumps([])
        )

        db.add(db_paper)
        db.commit()
        db.refresh(db_paper)

        # 임베딩 저장 (PaperEmbedding 모델에 맞게 수정)
        if embedding_vector:
            paper_embedding = PaperEmbedding(
                paper_id=db_paper.id,
                abstract_vector=json.dumps(embedding_vector)  # JSON 문자열로 저장
            )
            db.add(paper_embedding)
            db.commit()

        logger.info(f"PDF 논문 업로드 완료: {file.filename} -> Paper ID: {db_paper.id}")

        return {
            "status": "success",
            "message": "논문이 성공적으로 업로드되고 처리되었습니다.",
            "paper_id": db_paper.id,
            "extracted_info": {
                "title": parsed_paper.title,
                "authors": parsed_paper.authors,
                "abstract": parsed_paper.abstract[:200] + "..." if parsed_paper.abstract and len(parsed_paper.abstract) > 200 else parsed_paper.abstract,
                "year": parsed_paper.year,
                "conference": parsed_paper.conference,
                "content_length": len(parsed_paper.content) if parsed_paper.content else 0,
                "has_embedding": embedding_vector is not None
            }
        }

    except Exception as e:
        logger.error(f"PDF 업로드 처리 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF 처리 중 오류가 발생했습니다: {str(e)}")

@router.get("/upload-status/{paper_id}")
async def get_upload_status(paper_id: int, db: Session = Depends(get_db)):
    """업로드된 논문 상태 확인"""

    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="논문을 찾을 수 없습니다.")

    embedding = db.query(PaperEmbedding).filter(PaperEmbedding.paper_id == paper_id).first()

    return {
        "paper_id": paper.id,
        "title": paper.title,
        "authors": json.loads(paper.authors) if paper.authors else [],
        "abstract": paper.abstract,
        "year": paper.year,
        "conference": paper.conference,
        "has_embedding": embedding is not None,
        "created_at": paper.created_at
    }

@router.post("/upload-reference-paper", response_model=Dict[str, Any])
async def upload_reference_paper(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """PDF 기준 논문 파일 업로드 및 처리 (RAG 비교 대상용)"""

    # 파일 검증
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    if file.size and file.size > 10 * 1024 * 1024:  # 10MB 제한
        raise HTTPException(status_code=400, detail="파일 크기는 10MB 이하여야 합니다.")

    try:
        # 1. PDF 파일 읽기
        pdf_content = await file.read()

        # 2. PDF 처리
        pdf_processor = PDFProcessor()
        parsed_paper = await pdf_processor.process_pdf(pdf_content)

        # 3. 기준 논문 중복 체크
        existing_ref_paper = db.query(ReferencePaper).filter(
            ReferencePaper.title == parsed_paper.title
        ).first()

        if existing_ref_paper:
            logger.info(f"중복 기준 논문 발견: {parsed_paper.title} (기존 ID: {existing_ref_paper.id})")
            return {
                "status": "duplicate",
                "message": "이미 기준 논문으로 등록된 논문입니다.",
                "existing_paper_id": existing_ref_paper.id,
                "existing_paper_title": existing_ref_paper.title,
            }

        # 4. 임베딩 생성
        embedding_service = EmbeddingService()

        # 임베딩할 텍스트 준비
        embedding_text = ""
        if parsed_paper.title:
            embedding_text += f"Title: {parsed_paper.title}\n"
        if parsed_paper.abstract:
            embedding_text += f"Abstract: {parsed_paper.abstract}\n"
        if parsed_paper.content:
            content_preview = parsed_paper.content[:1000]
            embedding_text += f"Content: {content_preview}"

        embedding_vector = await embedding_service.create_embedding(embedding_text)

        # 5. 기준 논문으로 데이터베이스에 저장
        authors_list = []
        if parsed_paper.authors:
            authors_str = parsed_paper.authors.replace(';', ',')
            authors_list = [author.strip() for author in authors_str.split(',') if author.strip()]

        if not authors_list:
            authors_list = ["Unknown"]

        # ReferencePaper 객체 생성
        db_ref_paper = ReferencePaper(
            title=parsed_paper.title or f"Reference Paper - {file.filename}",
            authors=json.dumps(authors_list),
            abstract=parsed_paper.abstract or "No abstract extracted",
            conference=parsed_paper.conference or "Unknown",
            year=parsed_paper.year or 2024,
            field=None,
            keywords=json.dumps([]),
            doi=None,
            arxiv_id=None,
            url=None,
            impact_score=0.8,  # 기준 논문은 높은 점수
            introduction_text=None,
            method_text=None,
            result_text=None,
            conclusion_text=None,
            methodology_type=None,
            dataset_used=json.dumps([]),
            evaluation_metrics=json.dumps([]),
            citation_count=0,
            acceptance_status="accepted"
        )

        db.add(db_ref_paper)
        db.commit()
        db.refresh(db_ref_paper)

        # 기준 논문 임베딩 저장
        if embedding_vector:
            ref_paper_embedding = ReferencePaperEmbedding(
                reference_paper_id=db_ref_paper.id,
                abstract_vector=json.dumps(embedding_vector)
            )
            db.add(ref_paper_embedding)
            db.commit()

        logger.info(f"기준 논문 업로드 완료: {file.filename} -> Reference Paper ID: {db_ref_paper.id}")

        return {
            "status": "success",
            "message": "기준 논문이 성공적으로 업로드되고 처리되었습니다.",
            "reference_paper_id": db_ref_paper.id,
            "extracted_info": {
                "title": parsed_paper.title,
                "authors": parsed_paper.authors,
                "abstract": parsed_paper.abstract[:200] + "..." if parsed_paper.abstract and len(parsed_paper.abstract) > 200 else parsed_paper.abstract,
                "year": parsed_paper.year,
                "conference": parsed_paper.conference,
                "content_length": len(parsed_paper.content) if parsed_paper.content else 0,
                "has_embedding": embedding_vector is not None
            }
        }

    except Exception as e:
        logger.error(f"기준 논문 업로드 처리 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"기준 논문 처리 중 오류가 발생했습니다: {str(e)}")

@router.post("/upload-multiple-papers", response_model=Dict[str, Any])
async def upload_multiple_papers(
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """여러 PDF 논문 파일 업로드 및 처리"""
    results = []
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            results.append({"filename": file.filename, "status": "fail", "reason": "PDF 파일만 업로드 가능"})
            continue
        try:
            pdf_content = await file.read()
            pdf_processor = PDFProcessor()
            parsed_paper = await pdf_processor.process_pdf(pdf_content)

            # 중복 체크
            duplicate_checker = DuplicateChecker()
            is_duplicate, existing_paper = duplicate_checker.check_duplicate(
                db=db,
                title=parsed_paper.title,
                authors=parsed_paper.authors
            )

            if is_duplicate:
                results.append({
                    "filename": file.filename,
                    "status": "duplicate",
                    "reason": f"이미 존재하는 논문 (ID: {existing_paper.id})",
                    "existing_paper_id": existing_paper.id
                })
                continue

            authors_list = []
            if parsed_paper.authors:
                authors_str = parsed_paper.authors.replace(';', ',')
                authors_list = [author.strip() for author in authors_str.split(',') if author.strip()]
            if not authors_list:
                authors_list = ["Unknown"]

            # Paper 객체 직접 생성
            db_paper = Paper(
                title=parsed_paper.title or f"Uploaded Paper - {file.filename}",
                authors=json.dumps(authors_list),
                abstract=parsed_paper.abstract or "No abstract extracted",
                conference=parsed_paper.conference or "Unknown",
                year=parsed_paper.year or 2024,
                field=None,
                keywords=json.dumps([]),
                doi=None,
                arxiv_id=None,
                url=None,
                impact_score=0.0,
                introduction_text=parsed_paper.content[:2000] if parsed_paper.content else None,
                method_text=None,
                result_text=None,
                conclusion_text=None,
                methodology_type=None,
                dataset_used=json.dumps([]),
                evaluation_metrics=json.dumps([])
            )

            db.add(db_paper)
            db.commit()
            db.refresh(db_paper)

            results.append({"filename": file.filename, "status": "success", "paper_id": db_paper.id})

        except Exception as e:
            logger.error(f"파일 {file.filename} 처리 실패: {str(e)}")
            results.append({"filename": file.filename, "status": "fail", "reason": str(e)})

    return {"results": results}