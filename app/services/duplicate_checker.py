import re
import logging
from typing import Optional, Tuple
from sqlalchemy.orm import Session
from difflib import SequenceMatcher
import json

from app.models import Paper

logger = logging.getLogger(__name__)

class DuplicateChecker:
    """논문 중복 체크 서비스"""

    def __init__(self):
        self.title_similarity_threshold = 0.85  # 제목 유사도 임계값
        self.author_similarity_threshold = 0.7   # 저자 유사도 임계값

    def check_duplicate(self, db: Session, title: str, authors: str = None,
                       doi: str = None, arxiv_id: str = None) -> Tuple[bool, Optional[Paper]]:
        """
        논문 중복 여부를 체크합니다.

        Returns:
            Tuple[bool, Optional[Paper]]: (중복여부, 중복된 논문 객체)
        """

        # 1. DOI 기반 중복 체크 (가장 확실한 방법)
        if doi:
            existing_paper = db.query(Paper).filter(Paper.doi == doi).first()
            if existing_paper:
                logger.info(f"DOI 중복 발견: {doi}")
                return True, existing_paper

        # 2. ArXiv ID 기반 중복 체크
        if arxiv_id:
            existing_paper = db.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()
            if existing_paper:
                logger.info(f"ArXiv ID 중복 발견: {arxiv_id}")
                return True, existing_paper

        # 3. 제목 기반 중복 체크
        if title:
            # 정확한 제목 매칭
            existing_paper = db.query(Paper).filter(Paper.title == title).first()
            if existing_paper:
                logger.info(f"정확한 제목 중복 발견: {title}")
                return True, existing_paper

            # 유사한 제목 매칭
            similar_paper = self._find_similar_title(db, title)
            if similar_paper:
                # 저자 정보도 비교하여 더 정확한 판단
                if authors and self._compare_authors(authors, similar_paper):
                    logger.info(f"유사한 제목 + 저자 중복 발견: {title}")
                    return True, similar_paper

        return False, None

    def _find_similar_title(self, db: Session, title: str) -> Optional[Paper]:
        """유사한 제목을 가진 논문을 찾습니다."""

        # 제목 정규화
        normalized_title = self._normalize_title(title)

        # 모든 논문의 제목과 비교
        all_papers = db.query(Paper).all()

        for paper in all_papers:
            if not paper.title:
                continue

            normalized_existing_title = self._normalize_title(paper.title)
            similarity = SequenceMatcher(None, normalized_title, normalized_existing_title).ratio()

            if similarity >= self.title_similarity_threshold:
                logger.info(f"유사한 제목 발견 (유사도: {similarity:.2f}): {paper.title}")
                return paper

        return None

    def _normalize_title(self, title: str) -> str:
        """제목을 정규화합니다."""
        if not title:
            return ""

        # 소문자 변환
        normalized = title.lower()

        # 특수문자 제거 (공백은 유지)
        normalized = re.sub(r'[^\w\s]', '', normalized)

        # 연속된 공백을 하나로
        normalized = re.sub(r'\s+', ' ', normalized)

        # 앞뒤 공백 제거
        normalized = normalized.strip()

        return normalized

    def _compare_authors(self, authors1: str, paper2: Paper) -> bool:
        """저자 정보를 비교합니다."""
        if not authors1 or not paper2.authors:
            return False

        try:
            # paper2의 저자 정보 파싱 (JSON 형태일 수 있음)
            if isinstance(paper2.authors, str):
                try:
                    authors2_list = json.loads(paper2.authors)
                    if isinstance(authors2_list, list):
                        authors2 = ', '.join(authors2_list)
                    else:
                        authors2 = str(authors2_list)
                except json.JSONDecodeError:
                    authors2 = paper2.authors
            else:
                authors2 = str(paper2.authors)

            # 저자명 정규화 및 비교
            normalized_authors1 = self._normalize_authors(authors1)
            normalized_authors2 = self._normalize_authors(authors2)

            # 공통 저자 수 계산
            common_authors = len(set(normalized_authors1) & set(normalized_authors2))
            total_authors = len(set(normalized_authors1) | set(normalized_authors2))

            if total_authors == 0:
                return False

            similarity = common_authors / total_authors

            logger.info(f"저자 유사도: {similarity:.2f} (공통: {common_authors}, 전체: {total_authors})")

            return similarity >= self.author_similarity_threshold

        except Exception as e:
            logger.error(f"저자 비교 중 오류: {e}")
            return False

    def _normalize_authors(self, authors_str: str) -> set:
        """저자 문자열을 정규화하여 저자명 집합을 반환합니다."""
        if not authors_str:
            return set()

        # 쉼표, 세미콜론, 'and' 등으로 분리
        authors_str = re.sub(r'\s+and\s+', ',', authors_str, flags=re.IGNORECASE)
        authors_str = re.sub(r'[;]', ',', authors_str)

        authors = []
        for author in authors_str.split(','):
            author = author.strip()
            if author:
                # 상첨자 제거 (†, ‡, *, ⇤ 등)
                author = re.sub(r'[†‡*⇤¹²³⁴⁵⁶⁷⁸⁹⁰]+', '', author)
                # 소문자 변환 및 정리
                author = author.lower().strip()
                if len(author) > 2:  # 너무 짧은 이름 제외
                    authors.append(author)

        return set(authors)

    def get_duplicate_summary(self, db: Session) -> dict:
        """중복 논문 현황을 요약합니다."""

        all_papers = db.query(Paper).all()
        total_papers = len(all_papers)

        # 제목 기반 중복 그룹 찾기
        title_groups = {}
        for paper in all_papers:
            if not paper.title:
                continue
            normalized_title = self._normalize_title(paper.title)
            if normalized_title not in title_groups:
                title_groups[normalized_title] = []
            title_groups[normalized_title].append(paper)

        # 중복된 그룹만 필터링
        duplicate_groups = {k: v for k, v in title_groups.items() if len(v) > 1}

        duplicate_count = sum(len(group) - 1 for group in duplicate_groups.values())

        return {
            "total_papers": total_papers,
            "duplicate_groups": len(duplicate_groups),
            "duplicate_papers": duplicate_count,
            "unique_papers": total_papers - duplicate_count,
            "duplicate_rate": (duplicate_count / total_papers * 100) if total_papers > 0 else 0
        }