from typing import Dict, Any
import openai
from app.config import settings

class ReviewService:
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        openai.api_key = self.api_key

    def generate_review(self, paper: Dict[str, Any]) -> Dict[str, str]:
        prompt = f"""
        다음 논문에 대한 전문적인 리뷰를 작성해주세요:

        제목: {paper['title']}
        저자: {paper['authors']}
        초록: {paper['abstract']}

        다음 형식으로 리뷰를 작성해주세요:
        1. 주요 기여점
        2. 방법론 평가
        3. 실험 결과 분석
        4. 개선 가능한 점
        5. 종합 평가
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 AI 연구 논문 리뷰어입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        review = response.choices[0].message.content

        return {
            'paper_id': paper['id'],
            'review': review
        }