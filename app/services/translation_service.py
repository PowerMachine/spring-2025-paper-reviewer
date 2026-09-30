import re
from typing import Dict, Optional
from openai import AsyncOpenAI
from app.config import settings

class TranslationService:
    """번역 서비스 클래스"""

    def __init__(self):
        self.settings = settings

        # 기본 용어 매핑 테이블 (빠른 응답을 위한 캐시)
        self.term_mapping = {
            # AI/ML 관련
            "인공지능": "artificial intelligence",
            "머신러닝": "machine learning",
            "기계학습": "machine learning",
            "딥러닝": "deep learning",
            "심층학습": "deep learning",
            "신경망": "neural network",
            "뉴럴네트워크": "neural network",
            "자연어처리": "natural language processing",
            "컴퓨터비전": "computer vision",
            "강화학습": "reinforcement learning",
            "지도학습": "supervised learning",
            "비지도학습": "unsupervised learning",
            "전이학습": "transfer learning",
            "생성모델": "generative model",
            "판별모델": "discriminative model",
            "트랜스포머": "transformer",
            "어텐션": "attention",
            "임베딩": "embedding",

            # 데이터 과학
            "데이터마이닝": "data mining",
            "빅데이터": "big data",
            "데이터분석": "data analysis",
            "데이터과학": "data science",
            "통계학습": "statistical learning",
            "예측모델": "predictive model",
            "클러스터링": "clustering",
            "분류": "classification",
            "회귀": "regression",

            # 컴퓨터 과학
            "알고리즘": "algorithm",
            "자료구조": "data structure",
            "데이터베이스": "database",
            "소프트웨어공학": "software engineering",
            "웹개발": "web development",
            "모바일앱": "mobile app",
            "클라우드컴퓨팅": "cloud computing",
            "분산시스템": "distributed system",
            "네트워크": "network",
            "보안": "security",
            "암호화": "encryption",
            "블록체인": "blockchain",
            "양자컴퓨팅": "quantum computing",

            # 기타 기술 분야
            "로봇공학": "robotics",
            "사물인터넷": "internet of things",
            "가상현실": "virtual reality",
            "증강현실": "augmented reality",
            "게임개발": "game development",
            "바이오인포매틱스": "bioinformatics",
            "의료정보학": "medical informatics",
            "핀테크": "fintech",
            "헬스케어": "healthcare",

            # 학문 분야
            "수학": "mathematics",
            "통계학": "statistics",
            "물리학": "physics",
            "화학": "chemistry",
            "생물학": "biology",
            "의학": "medicine",
            "경제학": "economics",
            "심리학": "psychology",
            "언어학": "linguistics"
        }

        # OpenAI 클라이언트 초기화
        if self.settings.OPENAI_API_KEY:
            self.openai_client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
            self.openai_available = True
        else:
            self.openai_client = None
            self.openai_available = False

    def is_korean(self, text: str) -> bool:
        """텍스트에 한글이 포함되어 있는지 확인"""
        korean_pattern = re.compile(r'[가-힣]')
        return bool(korean_pattern.search(text))

    def translate_with_mapping(self, korean_text: str) -> Optional[str]:
        """매핑 테이블을 사용한 빠른 번역"""
        # 정확한 매치 확인
        korean_lower = korean_text.lower().strip()
        if korean_lower in self.term_mapping:
            return self.term_mapping[korean_lower]

        # 부분 매치 확인 (단어가 포함된 경우)
        for korean_term, english_term in self.term_mapping.items():
            if korean_term in korean_lower:
                return korean_lower.replace(korean_term, english_term)

        return None

    async def translate_with_openai(self, korean_text: str) -> Optional[str]:
        """OpenAI를 사용한 번역"""
        if not self.openai_available or not self.openai_client:
            return None

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional translator specializing in academic and technical terms. "
                            "Translate Korean text to English, focusing on academic keywords and technical terms. "
                            "Return only the English translation without any explanation."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Translate this Korean text to English: {korean_text}"
                    }
                ],
                max_tokens=100,
                temperature=0.1
            )

            translation = response.choices[0].message.content.strip()
            return translation

        except Exception as e:
            print(f"OpenAI 번역 실패: {e}")
            return None

    async def translate_search_query(self, query_text: str) -> str:
        """검색 쿼리를 번역 (필요한 경우)"""
        # 영어만 포함된 경우 번역하지 않음
        if not self.is_korean(query_text):
            return query_text

        print(f"한글 키워드 감지, 번역 시도: {query_text}")

        # 1. 먼저 매핑 테이블로 빠른 번역 시도
        mapped_translation = self.translate_with_mapping(query_text)
        if mapped_translation:
            print(f"매핑 테이블 번역 성공: {query_text} -> {mapped_translation}")
            return mapped_translation

        # 2. 매핑 테이블에 없으면 OpenAI로 번역 시도
        if self.openai_available:
            openai_translation = await self.translate_with_openai(query_text)
            if openai_translation:
                print(f"OpenAI 번역 성공: {query_text} -> {openai_translation}")
                return openai_translation

        # 3. 번역 실패 시 원본 반환
        print(f"번역 실패, 원본 키워드 사용: {query_text}")
        return query_text

    def add_term_mapping(self, korean_term: str, english_term: str):
        """새로운 용어 매핑 추가"""
        self.term_mapping[korean_term.lower()] = english_term.lower()
        print(f"새 용어 매핑 추가: {korean_term} -> {english_term}")

# 싱글톤 인스턴스
_translation_service = None

def get_translation_service() -> TranslationService:
    """번역 서비스 인스턴스를 반환합니다."""
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService()
    return _translation_service