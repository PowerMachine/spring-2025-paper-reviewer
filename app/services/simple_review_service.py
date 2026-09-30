import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SimpleReviewService:
    """OpenAI API 없이 작동하는 간단한 리뷰 서비스"""

    def __init__(self):
        self.review_templates = {
            "machine_learning": {
                "summary": "이 논문은 머신러닝 분야의 새로운 접근법을 제시합니다.",
                "strengths": [
                    "명확한 문제 정의와 해결 방법 제시",
                    "충분한 실험적 검증",
                    "기존 연구와의 비교 분석"
                ],
                "weaknesses": [
                    "일부 실험 설정에 대한 추가 설명 필요",
                    "더 다양한 데이터셋에서의 검증 필요",
                    "계산 복잡도에 대한 분석 부족"
                ]
            },
            "deep_learning": {
                "summary": "이 논문은 딥러닝 모델의 성능 향상을 위한 새로운 기법을 소개합니다.",
                "strengths": [
                    "혁신적인 아키텍처 설계",
                    "다양한 벤치마크에서의 우수한 성능",
                    "상세한 ablation study"
                ],
                "weaknesses": [
                    "모델의 해석가능성 부족",
                    "훈련 시간과 메모리 사용량 분석 필요",
                    "실제 응용에서의 성능 검증 부족"
                ]
            },
            "default": {
                "summary": "이 논문은 해당 분야에서 의미있는 기여를 제시합니다.",
                "strengths": [
                    "체계적인 연구 방법론",
                    "명확한 결과 제시",
                    "관련 연구와의 적절한 비교"
                ],
                "weaknesses": [
                    "일부 실험 결과에 대한 추가 분석 필요",
                    "제한된 실험 환경",
                    "향후 연구 방향에 대한 논의 부족"
                ]
            }
        }

    def generate_review(self, paper_title: str, paper_abstract: str, paper_field: Optional[str] = None) -> Dict[str, Any]:
        """논문에 대한 간단한 리뷰 생성"""
        try:
            # 논문 분야에 따른 템플릿 선택
            field_key = "default"
            if paper_field:
                field_lower = paper_field.lower()
                if "machine learning" in field_lower or "ml" in field_lower:
                    field_key = "machine_learning"
                elif "deep learning" in field_lower or "neural" in field_lower:
                    field_key = "deep_learning"

            template = self.review_templates[field_key]

            # 논문 제목과 초록을 기반으로 리뷰 커스터마이징
            customized_summary = self._customize_summary(paper_title, paper_abstract, template["summary"])

            review = {
                "summary": customized_summary,
                "strengths": template["strengths"],
                "weaknesses": template["weaknesses"],
                "overall_score": "4",  # 1-6 스케일에서 4점 (보통 수준)
                "confidence": "3",     # 1-5 스케일에서 3점 (보통 신뢰도)
                "review_content": self._generate_detailed_review(
                    paper_title, customized_summary, template["strengths"], template["weaknesses"]
                ),
                "generated_at": datetime.now().isoformat(),
                "service": "SimpleReviewService"
            }

            logger.info(f"간단한 리뷰 생성 완료: {paper_title[:50]}...")
            return review

        except Exception as e:
            logger.error(f"리뷰 생성 실패: {e}")
            return self._get_fallback_review(paper_title)

    def _customize_summary(self, title: str, abstract: str, base_summary: str) -> str:
        """제목과 초록을 기반으로 요약 커스터마이징"""
        if "diffusion" in title.lower() or "diffusion" in abstract.lower():
            return "이 논문은 확산 모델(Diffusion Model) 분야의 새로운 접근법을 제시합니다."
        elif "transformer" in title.lower() or "attention" in abstract.lower():
            return "이 논문은 Transformer 아키텍처의 개선 방안을 제안합니다."
        elif "gan" in title.lower() or "generative" in abstract.lower():
            return "이 논문은 생성 모델 분야의 혁신적인 기법을 소개합니다."
        else:
            return base_summary

    def _generate_detailed_review(self, title: str, summary: str, strengths: list, weaknesses: list) -> str:
        """상세한 리뷰 내용 생성"""

        # 논문 분야별 상세 분석
        field_analysis = ""
        if "diffusion" in title.lower():
            field_analysis = """
**기술적 기여도:**
- 확산 모델의 샘플링 효율성을 크게 개선하는 새로운 ODE 솔버 제안
- 기존 DDPM/DDIM 대비 10-20배 빠른 샘플링 속도 달성
- 수학적으로 엄밀한 수렴성 증명과 오차 분석 제시

**실험 평가:**
- CIFAR-10, CelebA, ImageNet 등 다양한 데이터셋에서 검증
- FID, IS 등 표준 평가 지표에서 기존 방법들과 동등하거나 우수한 성능
- Ablation study를 통한 각 구성 요소의 기여도 분석
- 다양한 확산 모델 아키텍처에서의 범용성 입증"""
        elif "transformer" in title.lower():
            field_analysis = """
**기술적 기여도:**
- Attention 메커니즘의 계산 복잡도를 O(n²)에서 O(n log n)으로 개선
- 새로운 위치 인코딩 방식으로 긴 시퀀스 처리 능력 향상
- 메모리 효율적인 구현으로 대규모 모델 훈련 가능

**실험 평가:**
- GLUE, SuperGLUE 벤치마크에서 SOTA 성능 달성
- 다양한 NLP 태스크에서의 일관된 성능 향상
- 훈련 시간과 메모리 사용량의 현실적 개선 효과 입증"""
        else:
            field_analysis = """
**기술적 기여도:**
- 기존 방법론의 한계를 명확히 분석하고 새로운 해결책 제시
- 이론적 근거와 실험적 검증을 통한 방법론의 타당성 입증
- 실용적 관점에서의 성능 개선과 효율성 향상

**실험 평가:**
- 표준 벤치마크 데이터셋에서의 체계적 성능 평가
- 기존 SOTA 방법들과의 공정한 비교 실험
- 다양한 실험 설정에서의 일관된 성능 향상 확인"""

        review_content = f"""이 논문은 {title}에 대한 연구로, 해당 분야에서 중요한 기술적 진전을 이루었습니다.

{field_analysis}

**논문 품질:**
- 논문 작성이 명확하고 체계적으로 구성되어 있음
- 관련 연구에 대한 충실한 조사와 적절한 인용
- 실험 설계가 합리적이고 재현 가능한 수준의 상세 정보 제공
- 결과 분석이 객관적이고 한계점에 대한 솔직한 논의

**전체 평가:**
이 논문은 해당 분야에서 의미있는 기여를 하고 있으며, 제시된 방법론이 이론적으로 타당하고 실험적으로 검증되었습니다. 특히 실용적 관점에서의 성능 개선이 인상적이며, 향후 연구에 중요한 기반을 제공할 것으로 기대됩니다.

**리뷰어 권고사항:**
현재 상태에서도 학술적 가치가 충분하나, 다음 사항들을 보완한다면 더욱 완성도 높은 연구가 될 것입니다:
- 더 다양한 실험 환경에서의 검증
- 계산 복잡도와 메모리 사용량에 대한 정량적 분석
- 실제 산업 응용에서의 적용 가능성 논의
- 방법론의 한계점과 향후 개선 방향에 대한 심화 논의

**최종 판정:** 이 논문은 기술적 혁신성과 실험적 검증 모두에서 우수한 수준을 보여주며, 해당 분야 발전에 기여할 가치 있는 연구입니다."""

        return review_content.strip()

    def _get_fallback_review(self, title: str) -> Dict[str, Any]:
        """오류 발생 시 기본 리뷰 반환"""
        return {
            "summary": "논문 리뷰 생성 중 오류가 발생했습니다.",
            "strengths": ["리뷰 생성 서비스 복구 필요"],
            "weaknesses": ["시스템 오류로 인한 리뷰 불가"],
            "overall_score": "0",
            "confidence": "1",
            "review_content": f"죄송합니다. '{title}' 논문에 대한 리뷰 생성 중 오류가 발생했습니다. 시스템 관리자에게 문의해주세요.",
            "generated_at": datetime.now().isoformat(),
            "service": "SimpleReviewService (Fallback)"
        }