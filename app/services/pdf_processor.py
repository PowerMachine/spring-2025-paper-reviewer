import io
import re
from typing import Dict, Optional, List
import PyPDF2
import pdfplumber
from dataclasses import dataclass

@dataclass
class ParsedPaper:
    title: Optional[str] = None
    authors: Optional[str] = None
    abstract: Optional[str] = None
    keywords: Optional[str] = None
    conference: Optional[str] = None
    year: Optional[int] = None
    content: str = ""
    raw_text: str = ""

class PDFProcessor:
    def __init__(self):
        self.title_patterns = [
            r'^([A-Z][^.!?]*(?:[.!?](?![A-Z]))?)(?:\n|$)',
            r'^(.{10,100})\n',
            r'Title[:\s]*(.+?)(?:\n|Author)',
        ]

        self.author_patterns = [
            # 일반적인 저자 패턴들
            r'Authors?[:\s]*(.+?)(?:\n\n|\n[A-Z])',
            r'By[:\s]*(.+?)(?:\n\n|\n[A-Z])',

            # 논문에서 흔한 패턴: 제목 다음 줄에 저자들이 나오는 경우
            r'(?:^|\n)([A-Z][a-z]+ [A-Z][a-z]+(?:[¹²³⁴⁵⁶⁷⁸⁹⁰*†‡§¶#]+)?(?:, [A-Z][a-z]+ [A-Z][a-z]+(?:[¹²³⁴⁵⁶⁷⁸⁹⁰*†‡§¶#]+)?)*)',

            # 상첨자가 있는 저자 패턴 (¹, ², *, † 등)
            r'([A-Z][a-z]+ [A-Z][a-z]+[¹²³⁴⁵⁶⁷⁸⁹⁰*†‡§¶#]*(?:, [A-Z][a-z]+ [A-Z][a-z]+[¹²³⁴⁵⁶⁷⁸⁹⁰*†‡§¶#]*)*)',

            # 이메일이나 소속 앞에 나오는 저자들
            r'^(.+?)(?:\n.*@|\n.*University|\n.*Institute|\n.*Department)',

            # 첫 번째 줄이 제목이고 두 번째 줄이 저자인 경우
            r'(?:^[^\n]+\n)([A-Z][^\n]+?)(?:\n[¹²³⁴⁵⁶⁷⁸⁹⁰*†‡§¶#]|\n[a-z]|\n$)',
        ]

        self.abstract_patterns = [
            r'Abstract[:\s]*(.+?)(?:\n\n|Keywords?|Introduction|1\.)',
            r'ABSTRACT[:\s]*(.+?)(?:\n\n|KEYWORDS?|INTRODUCTION|1\.)',
        ]

    async def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """PDF에서 텍스트 추출"""
        try:
            # PyPDF2로 기본 텍스트 추출 시도
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

            # 텍스트가 제대로 추출되지 않으면 pdfplumber 사용
            if len(text.strip()) < 100:
                pdf_file.seek(0)
                with pdfplumber.open(pdf_file) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"

            return text.strip()

        except Exception as e:
            raise Exception(f"PDF 텍스트 추출 실패: {str(e)}")

    def parse_metadata(self, text: str) -> ParsedPaper:
        """텍스트에서 논문 메타데이터 추출"""
        paper = ParsedPaper()
        paper.raw_text = text
        paper.content = text

        # 제목 추출
        paper.title = self._extract_title(text)

        # 저자 추출
        paper.authors = self._extract_authors(text)

        # 초록 추출
        paper.abstract = self._extract_abstract(text)

        # 키워드 추출
        paper.keywords = self._extract_keywords(text)

        # 연도 추출
        paper.year = self._extract_year(text)

        # 학회/저널명 추출
        paper.conference = self._extract_conference(text)

        return paper

    def _extract_title(self, text: str) -> Optional[str]:
        """제목 추출"""
        lines = text.split('\n')

        # 첫 번째 줄이 제목일 가능성이 높음
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 10 and len(line) < 200:
                # 특수 문자나 숫자로만 이루어진 줄 제외
                if re.search(r'[a-zA-Z]', line) and not line.lower().startswith(('page', 'doi:', 'arxiv:')):
                    return line

        return None

    def _extract_authors(self, text: str) -> Optional[str]:
        """저자 추출"""
        lines = text.split('\n')

        # 처음 몇 줄에서 직접 저자 패턴을 찾기
        for i, line in enumerate(lines[:15]):  # 처음 15줄 확인
            line = line.strip()

            # 저자 패턴 확인 (더 정확한 판단)
            if self._is_author_line(line):
                authors = self._clean_authors(line)
                if authors:
                    return authors

        # 기존 패턴들도 시도
        for pattern in self.author_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                authors = self._clean_authors(match.group(1))
                if authors:
                    return authors

        return None

    def _is_author_line(self, line: str) -> bool:
        """해당 줄이 저자 정보인지 판단"""
        line = line.strip()

        # 너무 짧거나 긴 줄은 제외
        if len(line) < 5 or len(line) > 300:
            return False

        # 명확히 저자가 아닌 것들 제외
        exclude_patterns = [
            r'^Abstract$',
            r'^Introduction$',
            r'^Keywords?:',
            r'^\d+\.',  # 섹션 번호
            r'^Page \d+',
            r'^DOI:',
            r'^arXiv:',
            r'University$',  # 소속만 있는 줄
            r'Department$',  # 부서만 있는 줄
            r'Institute$',   # 연구소만 있는 줄
        ]

        for pattern in exclude_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return False

        # 저자 패턴 확인 (더 정확한 패턴들)
        author_indicators = [
            # 상첨자가 있는 이름들 (†, ‡, *, ⇤ 등)
            r'[A-Z][a-z]+ [A-Z][a-z]+[†‡*⇤¹²³⁴⁵⁶⁷⁸⁹⁰]+',
            # 쉼표로 구분된 여러 이름 (상첨자 포함)
            r'[A-Z][a-z]+ [A-Z][a-z]+[†‡*⇤¹²³⁴⁵⁶⁷⁸⁹⁰]*,.*[A-Z][a-z]+ [A-Z][a-z]+',
            # 기본 이름 패턴 (First Last 형태)
            r'[A-Z][a-z]+ [A-Z][a-z]+',
        ]

        # 저자 패턴이 있는지 확인
        for pattern in author_indicators:
            if re.search(pattern, line):
                # 추가 검증: 최소 2개 이상의 이름이 있거나, 상첨자가 있어야 함
                name_count = len(re.findall(r'[A-Z][a-z]+ [A-Z][a-z]+', line))
                has_superscript = bool(re.search(r'[†‡*⇤¹²³⁴⁵⁶⁷⁸⁹⁰]', line))

                if name_count >= 2 or has_superscript:
                    return True

        return False

    def _clean_authors(self, authors_text: str) -> Optional[str]:
        """저자 텍스트 정리"""
        if not authors_text:
            return None

        authors = authors_text.strip()

        # 줄바꿈 제거
        authors = re.sub(r'\n+', ' ', authors)

        # 이메일이나 소속 정보 제거
        authors = re.sub(r'\s*@[^\s,]+', '', authors)
        authors = re.sub(r'\s*\([^)]*University[^)]*\)', '', authors, flags=re.IGNORECASE)
        authors = re.sub(r'\s*\([^)]*Institute[^)]*\)', '', authors, flags=re.IGNORECASE)
        authors = re.sub(r'\s*\([^)]*Department[^)]*\)', '', authors, flags=re.IGNORECASE)

        # 상첨자 숫자나 기호는 유지 (저자 구분에 중요)
        # 하지만 너무 긴 상첨자 설명은 제거
        authors = re.sub(r'\s*\([^)]{50,}\)', '', authors)

        # 연속된 공백 정리
        authors = re.sub(r'\s+', ' ', authors)

        # 길이 제한
        authors = authors[:300]

        # 최소한의 검증: 적어도 하나의 이름 패턴이 있는지 확인
        if re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', authors):
            return authors.strip()

        return None

    def _extract_abstract(self, text: str) -> Optional[str]:
        """초록 추출"""
        for pattern in self.abstract_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # 줄바꿈 정리
                abstract = re.sub(r'\n+', ' ', abstract)
                abstract = re.sub(r'\s+', ' ', abstract)
                return abstract[:1000]  # 길이 제한

        return None

    def _extract_keywords(self, text: str) -> Optional[str]:
        """키워드 추출"""
        keyword_match = re.search(r'Keywords?[:\s]*(.+?)(?:\n\n|Introduction|1\.)', text, re.IGNORECASE | re.DOTALL)
        if keyword_match:
            keywords = keyword_match.group(1).strip()
            keywords = re.sub(r'\n+', ' ', keywords)
            return keywords[:200]

        return None

    def _extract_year(self, text: str) -> Optional[int]:
        """연도 추출"""
        # 최근 논문들의 연도 범위
        year_matches = re.findall(r'\b(20[0-2]\d)\b', text)
        if year_matches:
            # 가장 자주 나오는 연도 선택
            year_counts = {}
            for year in year_matches:
                year_counts[year] = year_counts.get(year, 0) + 1
            most_common_year = max(year_counts, key=year_counts.get)
            return int(most_common_year)

        return None

    def _extract_conference(self, text: str) -> Optional[str]:
        """학회/저널명 추출"""
        # 일반적인 학회 패턴들
        conference_patterns = [
            r'Proceedings of (.+?)(?:\n|,|\d{4})',
            r'Published in (.+?)(?:\n|,|\d{4})',
            r'Conference on (.+?)(?:\n|,|\d{4})',
            r'Journal of (.+?)(?:\n|,|\d{4})',
        ]

        for pattern in conference_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                conference = match.group(1).strip()
                return conference[:100]

        return None

    async def process_pdf(self, pdf_content: bytes) -> ParsedPaper:
        """PDF 전체 처리 파이프라인"""
        try:
            # 1. 텍스트 추출
            text = await self.extract_text_from_pdf(pdf_content)

            # 디버깅을 위한 로그 출력
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"PDF 텍스트 추출 완료. 길이: {len(text)}")
            logger.info(f"PDF 텍스트 첫 500자: {text[:500]}")

            # 2. 메타데이터 파싱
            paper = self.parse_metadata(text)

            # 추출 결과 로그
            logger.info(f"제목 추출: {paper.title}")
            logger.info(f"저자 추출: {paper.authors}")
            logger.info(f"초록 추출: {paper.abstract[:100] if paper.abstract else None}...")

            return paper

        except Exception as e:
            raise Exception(f"PDF 처리 실패: {str(e)}")