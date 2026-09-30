"""
LLM 서비스 - OpenAI GPT-4를 활용한 논문 분석 및 리뷰 생성
"""

import openai
from typing import List, Dict, Any, Optional
import logging
from sqlalchemy.orm import Session
import os
import json

from app.config import settings
from app.models import Paper, PaperEmbedding

logger = logging.getLogger(__name__)

# 전역 변수로 LLM 서비스 인스턴스 저장
_llm_service_instance = None

def get_llm_service():
    """LLM 서비스 인스턴스를 Lazy 방식으로 반환"""
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = LLMService()
    return _llm_service_instance

class LLMService:
    def __init__(self):
        """LLM 서비스 초기화"""
        self.client = None
        self._initialized = False
        self._api_key = None
        # 초기화 시도
        self.initialize()

    def initialize(self):
        """OpenAI 클라이언트 초기화"""
        if self._initialized:
            return True

        try:
            # 환경변수에서 API 키 확인
            api_key = os.environ.get('OPENAI_API_KEY') or getattr(settings, 'OPENAI_API_KEY', None)

            if not api_key or api_key.strip() == "":
                logger.warning("OpenAI API 키가 설정되지 않았습니다. 환경변수 OPENAI_API_KEY를 설정해주세요.")
                return False

            # API 키 저장
            self._api_key = api_key.strip()

            # OpenAI 클라이언트 생성
            self.client = openai.OpenAI(api_key=self._api_key)

            self._initialized = True
            logger.info("OpenAI 클라이언트 초기화 완료")
            return True

        except Exception as e:
            logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
            self.client = None
            self._initialized = False
            return False

    def is_available(self):
        """LLM 서비스 사용 가능 여부 확인"""
        return self._initialized and self.client is not None

    async def generate_paper_review(
        self,
        paper: Paper,
        query_text: str = "",
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """논문에 대한 NeurIPS 스타일 리뷰 생성"""

        # 필요시 초기화
        if not self.is_available():
            if not self.initialize():
                return {
                    "error": "OpenAI API가 초기화되지 않았습니다. API 키를 확인해주세요."
                }

        if not self.client:
            return {
                "error": "OpenAI API가 초기화되지 않았습니다. API 키를 확인해주세요."
            }

        try:
            # 논문 정보 정리
            authors_str = ", ".join(paper.authors) if paper.authors else "저자 미상"

            # 프롬프트 생성
            prompt = self._get_review_prompt(paper)

            # OpenAI API 호출 (설정된 모델 사용)
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 최고 수준의 AI/ML 연구자이며, NeurIPS, ICML, ICLR 등 최고 학회의 논문 리뷰어입니다. 전문적이고 건설적인 논문 리뷰를 작성해주세요."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.7
            )

            review_content = response.choices[0].message.content

            # JSON 파싱 시도
            try:
                if review_content.strip().startswith('```json'):
                    review_content = review_content.strip()[7:-3]
                elif review_content.strip().startswith('```'):
                    # 다른 형태의 코드 블록 제거
                    review_content = review_content.strip()[3:-3]

                review_dict = json.loads(review_content)
                logger.info("JSON 파싱 성공")
            except Exception as json_error:
                logger.warning(f"JSON 파싱 실패, 텍스트 파싱 시도: {json_error}")
                # JSON 파싱 실패 시 텍스트에서 정보 추출
                review_dict = self._parse_review_response(review_content)

            review_dict.update({
                "paper_id": paper.id,
                "paper_title": paper.title,
                "paper_authors": authors_str,
                "paper_conference": paper.conference,
                "paper_year": paper.year,
                "raw_response": review_content
            })

            logger.info(f"논문 '{paper.title}' 리뷰 생성 완료")

            # 리뷰 생성 후 사용자 논문 삭제 (일회용) - 비활성화
            # if db:
            #     await self._cleanup_user_paper(db, paper)

            return review_dict

        except Exception as e:
            logger.error(f"논문 리뷰 생성 실패: {e}")
            return {
                "error": f"리뷰 생성 중 오류가 발생했습니다: {str(e)}"
            }

    def _get_review_prompt(self, paper: Paper) -> str:
        """리뷰 프롬프트 생성"""
        authors_str = ", ".join(paper.authors) if paper.authors else "저자 미상"

        return f"""당신은 NeurIPS, ICML, ICLR 등 최고 수준 AI 학회의 전문 리뷰어입니다. 다음 논문에 대해 상세하고 전문적인 리뷰를 작성해주세요.

논문 정보:
제목: {paper.title}
저자: {authors_str}
학회: {paper.conference}
연도: {paper.year}
초록: {paper.abstract}

반드시 아래 JSON 형식으로만 응답해주세요. JSON 외의 다른 텍스트는 포함하지 마세요:

{{
    "summary": "논문의 핵심 내용과 기여도를 3-4문장으로 요약",
    "strengths": [
        "구체적인 강점 1: 기술적 혁신성이나 새로운 접근법",
        "구체적인 강점 2: 실험 설계의 우수성이나 성능 향상",
        "구체적인 강점 3: 논문 품질이나 재현성 측면"
    ],
    "weaknesses": [
        "구체적인 약점 1: 실험의 한계나 이론적 부족함",
        "구체적인 약점 2: 비교 실험 부족이나 설명 부족",
        "구체적인 약점 3: 일반화 가능성이나 실용성 한계"
    ],
    "overall_score": 5,
    "confidence": 4,
    "recommendation": "Accept",
    "review_content": "**요약:** 이 논문은 [분야]에서 [문제]를 해결하기 위해 [방법]을 제안합니다.\\n\\n**기술적 기여도:**\\n- 구체적인 기술적 혁신점\\n- 기존 방법과의 차별점\\n- 이론적/실험적 근거\\n\\n**실험 평가:**\\n- 실험 설계의 적절성\\n- 데이터셋과 평가 지표\\n- 성능 향상 정도와 의미\\n- 기존 방법들과의 비교\\n\\n**논문 품질:**\\n- 논문 작성의 명확성\\n- 관련 연구 조사의 충실성\\n- 재현 가능성\\n\\n**전체 평가:**\\n논문의 학술적 가치, 실용적 의미, 향후 연구 방향에 대한 종합적 평가\\n\\n**리뷰어 권고사항:**\\n수용/거부 이유와 개선 방향 제시"
}}

중요:
- overall_score는 1-6 중 하나 (1=강력거부, 2=거부, 3=약한거부, 4=약한수용, 5=수용, 6=강력수용)
- confidence는 1-5 중 하나 (1=매우낮음, 5=매우높음)
- recommendation은 "Accept", "Weak Accept", "Weak Reject", "Reject" 중 하나
- 반드시 유효한 JSON 형식으로만 응답하세요."""

    def _parse_review_response(self, review_content: str) -> Dict[str, Any]:
        """LLM 응답을 구조화된 데이터로 파싱"""

        try:
            # 1단계: JSON 파싱 시도 (우선)
            import json
            import re

            # JSON 부분 추출 시도
            json_match = re.search(r'\{.*\}', review_content, re.DOTALL)
            if json_match:
                try:
                    json_data = json.loads(json_match.group())
                    logger.info(f"JSON 파싱 성공: 점수={json_data.get('overall_score')}, 신뢰도={json_data.get('confidence')}")

                    # JSON 파싱 성공시 해당 데이터 사용
                    parsed = {
                        "summary": json_data.get("summary", ""),
                        "strengths": json_data.get("strengths", []),
                        "weaknesses": json_data.get("weaknesses", []),
                        "overall_score": json_data.get("overall_score", 5),
                        "confidence": json_data.get("confidence", 3),
                        "recommendation": json_data.get("recommendation", "Borderline"),
                        "review_content": json_data.get("review_content", review_content)
                    }
                    return parsed
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON 파싱 실패, 텍스트 파싱으로 전환: {e}")

            # 2단계: 텍스트 파싱 (JSON 파싱 실패시)
            logger.info("텍스트 파싱 모드로 진행")
            parsed = {
                "summary": "",
                "strengths": [],
                "weaknesses": [],
                "overall_score": 5,
                "confidence": 3,
                "recommendation": "Borderline",
                "review_content": review_content
            }

            # 텍스트를 줄 단위로 분석
            lines = review_content.split('\n')
            current_section = ""
            summary_lines = []
            strengths = []
            weaknesses = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 1. 요약 부분 찾기 (첫 번째 문단이나 기술적 기여도 앞부분)
                if any(keyword in line for keyword in ["이 논문은", "본 연구는", "제안합니다", "해결하기 위해"]):
                    if len(summary_lines) < 3:  # 요약은 최대 3줄
                        summary_lines.append(line)

                # 2. 강점 찾기
                if any(keyword in line for keyword in ["기술적 혁신", "실험 설계", "성능 향상", "새로운 접근", "체계적", "효과적"]):
                    if "- " in line:
                        strengths.append(line.replace("- ", "").strip())
                    else:
                        strengths.append(line.strip())

                # 3. 약점 찾기
                if any(keyword in line for keyword in ["한계", "부족", "우려", "문제", "제한", "개선"]):
                    if "- " in line:
                        weaknesses.append(line.replace("- ", "").strip())
                    else:
                        weaknesses.append(line.strip())

                # 4. 점수 추출 - 더 정확한 패턴 매칭
                if any(keyword in line for keyword in ["점수", "전체 점수", "overall_score", "score"]):
                    import re
                    # 다양한 패턴 시도
                    patterns = [
                        r'(\d+)/(\d+)',  # "5/6" 형태
                        r'점수.*?(\d+)',  # "전체 점수: 5" 형태
                        r'score.*?(\d+)',  # "overall_score: 5" 형태
                        r'(\d+)점',       # "5점" 형태
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, line.lower())
                        if match:
                            score = int(match.group(1))
                            # 1-6 범위로 제한
                            parsed["overall_score"] = max(1, min(6, score))
                            break

                # 5. 신뢰도 추출 - 더 정확한 패턴 매칭
                if any(keyword in line for keyword in ["신뢰도", "confidence", "확신"]):
                    import re
                    patterns = [
                        r'(\d+)/(\d+)',     # "4/5" 형태
                        r'신뢰도.*?(\d+)',   # "신뢰도: 4" 형태
                        r'confidence.*?(\d+)', # "confidence: 4" 형태
                        r'(\d+)점',          # "4점" 형태
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, line.lower())
                        if match:
                            confidence = int(match.group(1))
                            # 1-5 범위로 제한
                            parsed["confidence"] = max(1, min(5, confidence))
                            break

                # 6. 추천사항 추출
                if any(keyword in line for keyword in ["수용", "거부", "Accept", "Reject"]):
                    if "수용" in line:
                        parsed["recommendation"] = "Accept"
                    elif "거부" in line:
                        parsed["recommendation"] = "Reject"

            # 요약 조합
            if summary_lines:
                parsed["summary"] = " ".join(summary_lines)
            else:
                # 요약이 없으면 첫 번째 문단 사용
                first_paragraph = review_content.split('\n\n')[0] if '\n\n' in review_content else review_content.split('\n')[0]
                parsed["summary"] = first_paragraph[:200] + "..." if len(first_paragraph) > 200 else first_paragraph

            # 강점이 없으면 기본값 추가
            if not strengths:
                if "효과적" in review_content or "우수한" in review_content:
                    strengths.append("제안된 방법론이 효과적임")
                if "체계적" in review_content:
                    strengths.append("체계적인 연구 접근")
                if "성능" in review_content and "향상" in review_content:
                    strengths.append("성능 향상을 달성함")

            # 약점이 없으면 기본값 추가
            if not weaknesses:
                if "다양성" in review_content and "부족" in review_content:
                    weaknesses.append("데이터셋 다양성 부족")
                if "일반화" in review_content:
                    weaknesses.append("일반화 가능성에 대한 우려")
                if "이론적" in review_content and ("부족" in review_content or "한계" in review_content):
                    weaknesses.append("이론적 분석 보강 필요")

            parsed["strengths"] = strengths[:5]  # 최대 5개
            parsed["weaknesses"] = weaknesses[:5]  # 최대 5개

            logger.info(f"텍스트 파싱 완료: 요약={len(parsed['summary'])}, 강점={len(parsed['strengths'])}, 약점={len(parsed['weaknesses'])}")

            return parsed

        except Exception as e:
            logger.error(f"리뷰 파싱 실패: {e}")
            return {
                "summary": "리뷰 파싱 중 오류가 발생했습니다.",
                "strengths": ["파싱 오류로 인한 강점 추출 실패"],
                "weaknesses": ["파싱 오류로 인한 약점 추출 실패"],
                "overall_score": 3,
                "confidence": 2,
                "recommendation": "파싱 오류",
                "review_content": review_content
            }



    async def generate_general_analysis(self, query_text: str) -> Dict[str, Any]:
        """데이터베이스에 관련 논문이 없을 때 일반적인 분석 제공"""

        # 필요시 초기화
        if not self.is_available():
            if not self.initialize():
                return {
                    "analysis": f"📚 '{query_text}' 관련 정보\n\n❌ OpenAI API 키가 설정되지 않아 AI 분석을 제공할 수 없습니다.\n\n💡 더 상세한 분석을 위해서는 OpenAI API 키 설정이 필요합니다."
                }

        if not self.client:
            return {
                "analysis": f"📚 '{query_text}' 관련 정보\n\n❌ OpenAI API가 초기화되지 않았습니다.\n\n💡 API 키를 확인해주세요."
            }

        try:
            prompt = f"""
## 🔍 사용자 질의: "{query_text}"

사용자가 '{query_text}'에 대해 질문했습니다. 데이터베이스에서 관련 논문을 찾지 못했지만, 이 주제에 대한 유용한 정보를 제공해주세요.

다음 형식에 맞춰 포괄적이고 유익한 분석을 작성해주세요:

### 🎯 1. 주제 개요
'{query_text}'이 무엇인지, 왜 중요한 연구 분야인지 간단명료하게 설명해주세요.

### 🔬 2. 주요 연구 영역
이 분야의 핵심 연구 영역들과 세부 주제들을 나열해주세요:
• 연구 영역 1: 간단한 설명
• 연구 영역 2: 간단한 설명
• 연구 영역 3: 간단한 설명

### 📈 3. 최신 동향 및 트렌드
이 분야의 최근 연구 동향, 주목받는 기술이나 방법론을 설명해주세요.

### 🛠️ 4. 주요 방법론 및 기술
이 분야에서 자주 사용되는 핵심 기술이나 방법론들:
• 방법론 1: 용도와 특징
• 방법론 2: 용도와 특징
• 방법론 3: 용도와 특징

### 🌍 5. 실용적 응용 분야
이 기술이나 연구가 실제로 어떤 분야에서 활용되는지:
• 응용 분야 1: 구체적 활용 사례
• 응용 분야 2: 구체적 활용 사례
• 응용 분야 3: 구체적 활용 사례

### 🔮 6. 향후 전망
이 분야의 미래 발전 방향과 해결해야 할 과제들

### 📚 7. 추천 학습 자료
이 주제를 더 깊이 공부하고 싶다면:
• 추천 키워드: 검색할 때 유용한 키워드들
• 관련 학회/저널: 주요 학술 발표 장소들
• 참고할 만한 연구자나 기관

---

⭐ **참고:**
- 전문적이지만 이해하기 쉽게 작성해주세요
- 구체적인 예시를 포함해주세요
- 최신 정보를 바탕으로 작성해주세요
"""

            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 AI/ML 및 컴퓨터 과학 분야의 전문가로서, 다양한 기술 주제에 대해 포괄적이고 정확한 정보를 제공하는 역할을 합니다. 최신 동향과 실용적인 관점을 포함하여 답변해주세요."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=settings.OPENAI_MAX_TOKENS,
                temperature=0.4  # 일반 분석은 조금 더 창의적으로
            )

            analysis = response.choices[0].message.content

            return {
                "query": query_text,
                "analysis": analysis,
                "type": "general_analysis"
            }

        except Exception as e:
            logger.error(f"일반 분석 생성 실패: {e}")
            return {
                "error": f"일반 분석 생성 중 오류가 발생했습니다: {str(e)}"
            }

    async def _cleanup_user_paper(self, db: Session, target_paper: Paper):
        """
        AI 리뷰 완료 후 사용자 논문을 삭제합니다 (일회용).

        Args:
            db: 데이터베이스 세션
            target_paper: 삭제할 사용자 논문
        """
        try:
            # 1. 논문의 임베딩 먼저 삭제
            paper_embedding = db.query(PaperEmbedding).filter(
                PaperEmbedding.paper_id == target_paper.id
            ).first()

            if paper_embedding:
                db.delete(paper_embedding)
                logger.info(f"사용자 논문 임베딩 삭제 완료: paper_id={target_paper.id}")

            # 2. 논문 삭제
            db.delete(target_paper)
            db.commit()

            logger.info(f"사용자 논문 삭제 완료 (일회용): paper_id={target_paper.id}, title='{target_paper.title}'")

        except Exception as e:
            logger.error(f"사용자 논문 삭제 실패: paper_id={target_paper.id}, error={e}")
            db.rollback()
            # 삭제 실패해도 리뷰는 정상적으로 반환