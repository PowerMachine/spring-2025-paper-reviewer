from typing import List, Dict, Any
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from app.models import Paper
import numpy as np

class RAGService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.7
        )

        # 평가 프롬프트 템플릿
        self.evaluation_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""당신은 NeurIPS 논문 리뷰어입니다.
            주어진 논문을 유사한 기존 논문들과 비교하여 평가해주세요.

            평가 기준:
            1. 독창성 (Originality): 새로운 아이디어나 접근 방식의 제시
            2. 방법론 (Methodology): 연구 방법의 적절성과 엄격성
            3. 신뢰성 (Reliability): 실험 결과의 신뢰성과 재현 가능성
            4. 의미성 (Significance): 연구의 학문적/실용적 기여도
            5. 명확성 (Clarity): 논문의 구조와 표현의 명확성

            각 기준에 대해 1-5점으로 평가하고, 그 근거를 제시해주세요.

            다음 항목들에 대해 상세히 분석해주세요:
            1. 방법론 비교: 제안된 방법론이 기존 방법론과 어떻게 다른지
            2. 결과 비교: 실험 결과가 기존 연구와 비교하여 어떤 장단점이 있는지
            3. 참신성 분석: 연구의 새로운 기여점과 한계점

            또한 논문의 강점, 약점, 개선점을 구체적으로 분석해주세요.
            마지막으로 최종 추천(accept/reject)과 그 이유를 제시해주세요."""),
            HumanMessage(content="""평가할 논문:
            제목: {title}
            저자: {authors}
            초록: {abstract}
            방법론: {methodology}
            결과: {results}

            비교 대상 논문들:
            {reference_papers}

            섹션별 유사도 점수:
            {similarity_scores}""")
        ])

    async def evaluate_paper(self, target_paper: Paper, reference_papers: List[Paper]) -> Dict[str, Any]:
        """논문 평가 수행"""
        # 참조 논문 정보 포맷팅
        ref_papers_text = "\n\n".join([
            f"논문 {i+1}:\n"
            f"제목: {p.title}\n"
            f"저자: {', '.join(p.authors) if p.authors else 'Unknown'}\n"
            f"초록: {p.abstract or 'No abstract'}\n"
            f"방법론: {p.method_text or 'No methodology'}\n"
            f"결과: {p.result_text or 'No results'}"
            for i, p in enumerate(reference_papers)
        ])

        # 섹션별 유사도 점수 포맷팅 (임시로 기본값 사용)
        similarity_scores = {
            "초록": 0.8,
            "서론": 0.7,
            "방법론": 0.9,
            "결과": 0.6,
            "결론": 0.7
        }

        # 프롬프트 생성
        prompt = self.evaluation_prompt.format_messages(
            title=target_paper.title,
            authors=", ".join(target_paper.authors) if target_paper.authors else "Unknown",
            abstract=target_paper.abstract or "No abstract",
            methodology=target_paper.method_text or "No methodology",
            results=target_paper.result_text or "No results",
            reference_papers=ref_papers_text,
            similarity_scores=similarity_scores
        )

        # LLM을 통한 평가 수행
        try:
            evaluation = await self.llm.agenerate([prompt])
            evaluation_text = evaluation.generations[0][0].text
        except Exception as e:
            # OpenAI API 오류 시 기본 평가 반환
            evaluation_text = f"평가 중 오류 발생: {str(e)}"

        # 평가 결과 파싱
        scores = self._parse_scores(evaluation_text)
        analysis = self._parse_analysis(evaluation_text)

        return {
            "evaluation_text": evaluation_text,
            "scores": scores,
            "analysis": analysis
        }

    def _parse_scores(self, text: str) -> Dict[str, float]:
        """평가 텍스트에서 점수 추출"""
        score_mapping = {
            "독창성": "originality",
            "방법론": "methodology",
            "신뢰성": "reliability",
            "의미성": "significance",
            "명확성": "clarity"
        }

        scores = {}
        for kr, en in score_mapping.items():
            # 정규표현식으로 점수 추출
            import re
            pattern = rf"{kr}.*?(\d+)점"
            match = re.search(pattern, text)
            if match:
                scores[en] = float(match.group(1))
            else:
                scores[en] = 3.0  # 기본값

        return scores

    def _parse_analysis(self, text: str) -> Dict[str, List[str]]:
        """평가 텍스트에서 분석 내용 추출"""
        analysis = {
            "strengths": [],
            "weaknesses": [],
            "improvements": [],
            "recommendation": "",
            "methodology_comparison": "",
            "result_comparison": "",
            "novelty_analysis": ""
        }

        # 강점, 약점, 개선점 추출
        sections = {
            "강점": "strengths",
            "약점": "weaknesses",
            "개선점": "improvements"
        }

        for kr, en in sections.items():
            pattern = rf"{kr}.*?[:：](.*?)(?=\n\n|\Z)"
            import re
            match = re.search(pattern, text, re.DOTALL)
            if match:
                points = [p.strip() for p in match.group(1).split("\n") if p.strip()]
                analysis[en] = points

        # 방법론 비교, 결과 비교, 참신성 분석 추출
        analysis_sections = {
            "방법론 비교": "methodology_comparison",
            "결과 비교": "result_comparison",
            "참신성 분석": "novelty_analysis"
        }

        for kr, en in analysis_sections.items():
            pattern = rf"{kr}.*?[:：](.*?)(?=\n\n|\Z)"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                analysis[en] = match.group(1).strip()

        # 최종 추천 추출
        rec_pattern = r"최종\s*추천.*?[:：]\s*(accept|reject)"
        rec_match = re.search(rec_pattern, text, re.IGNORECASE)
        if rec_match:
            analysis["recommendation"] = rec_match.group(1).lower()
        else:
            analysis["recommendation"] = "accept"  # 기본값

        return analysis
