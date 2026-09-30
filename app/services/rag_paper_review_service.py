from typing import List, Dict, Any, Tuple, Optional
import json
import logging
from sqlalchemy.orm import Session
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import numpy as np

from app.models import Paper, PaperEmbedding, ReferencePaper, ReferencePaperEmbedding
from app.services.embedding_service import EmbeddingService
from app.config import settings

logger = logging.getLogger(__name__)

class RAGPaperReviewService:
    """RAG 기반 논문 리뷰 서비스

    프로세스:
    1. A 논문 입력
    2. DB에서 A 논문과 도메인이 유사한 논문 3개(B, C, D) 추출 (벡터 임베딩 + 코사인 유사도)
    3. B, C, D는 accept된 고품질 논문들, A는 아직 accept되지 않은 논문
    4. LangChain으로 A논문과 B,C,D 논문을 함께 프롬프트에 넣고 분석
    5. 분석 결과 출력
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.embedding_service.initialize()  # 임베딩 서비스 초기화
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=settings.OPENAI_API_KEY
        )

        # RAG 리뷰 프롬프트 템플릿
        self.review_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""당신은 NeurIPS 논문 리뷰어입니다.

            주어진 논문(A)을 유사한 기존 accept된 논문들(B, C, D)과 비교하여 평가해주세요.

            **평가 기준:**
            1. **독창성 (Originality)**: 새로운 아이디어나 접근 방식의 제시 (1-10점)
            2. **방법론 (Methodology)**: 연구 방법의 적절성과 엄격성 (1-10점)
            3. **신뢰성 (Reliability)**: 실험 결과의 신뢰성과 재현 가능성 (1-10점)
            4. **의미성 (Significance)**: 연구의 학문적/실용적 기여도 (1-10점)
            5. **명확성 (Clarity)**: 논문의 구조와 표현의 명확성 (1-10점)

            **분석 항목:**
            1. **방법론 비교**: 제안된 방법론이 기존 방법론(B,C,D)과 어떻게 다른지
            2. **결과 비교**: 실험 결과가 기존 연구와 비교하여 어떤 장단점이 있는지
            3. **참신성 분석**: 연구의 새로운 기여점과 한계점
            4. **강점**: 논문의 주요 강점들
            5. **약점**: 논문의 주요 약점들
            6. **개선점**: 구체적인 개선 제안

            **최종 결론:**
            - 최종 추천: Accept/Weak Accept/Weak Reject/Reject
            - 신뢰도: 1-5점 (리뷰어의 확신 정도)
            - 한 줄 요약

            참고 논문들(B,C,D)은 이미 accept된 고품질 논문들이므로, 이들과의 비교를 통해 논문 A의 수준을 평가하세요."""),

            HumanMessage(content="""
**평가 대상 논문 (A):**
제목: {target_title}
저자: {target_authors}
초록: {target_abstract}
방법론: {target_methodology}
결과: {target_results}

**참고 논문들 (Accept된 고품질 논문):**

**논문 B (유사도: {similarity_b:.3f}):**
제목: {ref_title_b}
저자: {ref_authors_b}
초록: {ref_abstract_b}
방법론: {ref_methodology_b}
결과: {ref_results_b}

**논문 C (유사도: {similarity_c:.3f}):**
제목: {ref_title_c}
저자: {ref_authors_c}
초록: {ref_abstract_c}
방법론: {ref_methodology_c}
결과: {ref_results_c}

**논문 D (유사도: {similarity_d:.3f}):**
제목: {ref_title_d}
저자: {ref_authors_d}
초록: {ref_abstract_d}
방법론: {ref_methodology_d}
결과: {ref_results_d}

위의 정보를 바탕으로 논문 A에 대한 상세한 리뷰를 작성해주세요.
""")
        ])

    async def find_similar_papers(self, db: Session, target_paper: Paper, top_k: int = 3) -> List[Tuple[ReferencePaper, float]]:
        """
        대상 논문과 유사한 기준 논문들을 찾습니다.

        Args:
            db: 데이터베이스 세션
            target_paper: 평가 대상 논문 (사용자 업로드)
            top_k: 반환할 유사 논문 개수

        Returns:
            [(기준논문, 유사도), ...] 형태의 리스트
        """
        try:
            # 1. 대상 논문의 임베딩 생성 또는 조회
            target_embedding = await self._get_or_create_embedding(db, target_paper)
            if not target_embedding:
                logger.error(f"논문 {target_paper.id}의 임베딩을 생성할 수 없습니다")
                return []

            # 2. 대상 논문의 내용으로 분야 추정
            target_content = f"{target_paper.title or ''} {target_paper.abstract or ''}".lower()
            estimated_field = self._estimate_field(target_content)
            logger.info(f"대상 논문 추정 분야: {estimated_field}")

            # 3. 기준 논문들 조회 - 분야별 우선순위 적용
            if estimated_field:
                # 동일 분야 논문 우선 조회
                candidate_papers = db.query(ReferencePaper).filter(
                    ReferencePaper.field == estimated_field,
                    ReferencePaper.impact_score >= 0.7
                ).all()
                logger.info(f"동일 분야 ({estimated_field}) 고품질 논문: {len(candidate_papers)}개")

                # 동일 분야 논문이 부족하면 다른 분야도 포함
                if len(candidate_papers) < top_k:
                    other_papers = db.query(ReferencePaper).filter(
                        ReferencePaper.field != estimated_field,
                        ReferencePaper.impact_score >= 0.7
                    ).all()
                    candidate_papers.extend(other_papers)
                    logger.info(f"다른 분야 포함하여 총 {len(candidate_papers)}개")
            else:
                # 분야 추정 실패시 기존 로직
                candidate_papers = db.query(ReferencePaper).filter(
                    ReferencePaper.impact_score >= 0.7
                ).all()
                logger.info(f"전체 고품질 기준 논문: {len(candidate_papers)}개")

            if len(candidate_papers) < top_k:
                # 고품질 기준 논문이 부족하면 기준을 낮춤
                additional_papers = db.query(ReferencePaper).filter(
                    ReferencePaper.impact_score >= 0.5
                ).all()
                candidate_papers.extend([p for p in additional_papers if p not in candidate_papers])
                logger.info(f"중품질 기준 논문 포함하여 총 {len(candidate_papers)}개 발견")

            if len(candidate_papers) == 0:
                # 그래도 없으면 모든 기준 논문 대상
                candidate_papers = db.query(ReferencePaper).all()
                logger.info(f"모든 기준 논문 대상으로 확장하여 총 {len(candidate_papers)}개 발견")

            # 4. 각 후보 논문과의 유사도 계산
            similarities = []
            for ref_paper in candidate_papers:
                ref_embedding = await self._get_or_create_reference_embedding(db, ref_paper)
                if ref_embedding:
                    similarity = self.embedding_service.calculate_similarity(
                        target_embedding, ref_embedding
                    )

                    # 동일 분야 논문에 가중치 부여
                    if estimated_field and ref_paper.field == estimated_field:
                        similarity *= 1.1  # 10% 가중치
                        similarity = min(similarity, 1.0)  # 최대 1.0으로 제한
                        logger.info(f"동일분야 가중치 적용: {ref_paper.title[:30]}... -> {similarity:.3f}")

                    similarities.append((ref_paper, similarity))
                    logger.info(f"유사도 계산 완료: {ref_paper.title[:30]}... -> {similarity:.3f}")

            # 5. 유사도 순으로 정렬하여 상위 k개 반환
            similarities.sort(key=lambda x: x[1], reverse=True)
            result = similarities[:top_k]

            logger.info(f"최종 선택된 유사 논문 {len(result)}개:")
            for i, (paper, sim) in enumerate(result):
                logger.info(f"  {i+1}. {paper.title[:50]}... (유사도: {sim:.3f}, 분야: {paper.field})")

            return result

        except Exception as e:
            logger.error(f"유사 기준 논문 검색 실패: {e}")
            return []

    def _estimate_field(self, content: str) -> Optional[str]:
        """논문 내용을 분석해서 분야를 추정합니다."""
        try:
            diffusion_keywords = ['diffusion', 'denoising', 'generative', 'sampling', 'ddpm', 'ddim', 'stable diffusion', 'latent diffusion']
            vla_keywords = ['vision-language', 'action', 'robotics', 'multimodal', 'embodied', 'manipulation', 'vlm', 'vla']

            diffusion_score = sum(1 for keyword in diffusion_keywords if keyword in content)
            vla_score = sum(1 for keyword in vla_keywords if keyword in content)

            if diffusion_score > vla_score and diffusion_score > 0:
                return "Computer Vision/Machine Learning"
            elif vla_score > diffusion_score and vla_score > 0:
                return "Vision-Language-Action"

            return None

        except Exception as e:
            logger.error(f"분야 추정 실패: {e}")
            return None

    async def _get_or_create_embedding(self, db: Session, paper: Paper) -> Optional[List[float]]:
        """사용자 논문의 임베딩을 조회하거나 생성합니다."""
        try:
            # 기존 임베딩 조회
            existing_embedding = db.query(PaperEmbedding).filter(
                PaperEmbedding.paper_id == paper.id
            ).first()

            if existing_embedding and existing_embedding.abstract_vector:
                # JSON 문자열을 리스트로 변환
                if isinstance(existing_embedding.abstract_vector, str):
                    return json.loads(existing_embedding.abstract_vector)
                return existing_embedding.abstract_vector

            # 임베딩이 없으면 새로 생성
            content = f"{paper.title}\n{paper.abstract or ''}"
            embedding_vector = await self.embedding_service.create_embedding(content)

            if embedding_vector:
                # 데이터베이스에 저장
                if existing_embedding:
                    existing_embedding.abstract_vector = json.dumps(embedding_vector)
                else:
                    new_embedding = PaperEmbedding(
                        paper_id=paper.id,
                        abstract_vector=json.dumps(embedding_vector)
                    )
                    db.add(new_embedding)

                db.commit()
                return embedding_vector

            return None

        except Exception as e:
            logger.error(f"사용자 논문 임베딩 처리 실패 (paper_id: {paper.id}): {e}")
            return None

    async def _get_or_create_reference_embedding(self, db: Session, ref_paper: ReferencePaper) -> Optional[List[float]]:
        """기준 논문의 임베딩을 조회하거나 생성합니다."""
        try:
            # 기존 임베딩 조회
            existing_embedding = db.query(ReferencePaperEmbedding).filter(
                ReferencePaperEmbedding.reference_paper_id == ref_paper.id
            ).first()

            if existing_embedding and existing_embedding.abstract_vector:
                # JSON 문자열을 리스트로 변환
                if isinstance(existing_embedding.abstract_vector, str):
                    return json.loads(existing_embedding.abstract_vector)
                return existing_embedding.abstract_vector

            # 임베딩이 없으면 새로 생성
            content = f"{ref_paper.title}\n{ref_paper.abstract or ''}"
            embedding_vector = await self.embedding_service.create_embedding(content)

            if embedding_vector:
                # 데이터베이스에 저장
                if existing_embedding:
                    existing_embedding.abstract_vector = json.dumps(embedding_vector)
                else:
                    new_embedding = ReferencePaperEmbedding(
                        reference_paper_id=ref_paper.id,
                        abstract_vector=json.dumps(embedding_vector)
                    )
                    db.add(new_embedding)

                db.commit()
                return embedding_vector

            return None

        except Exception as e:
            logger.error(f"기준 논문 임베딩 처리 실패 (reference_paper_id: {ref_paper.id}): {e}")
            return None

    async def generate_rag_review(self, db: Session, target_paper: Paper) -> Dict[str, Any]:
        """
        RAG 기반 논문 리뷰를 생성합니다.

        Args:
            db: 데이터베이스 세션
            target_paper: 평가 대상 논문

        Returns:
            리뷰 결과 딕셔너리
        """
        try:
            # 1. 유사한 논문 3개 찾기
            similar_papers = await self.find_similar_papers(db, target_paper, top_k=3)

            if len(similar_papers) < 3:
                logger.warning(f"유사 논문이 {len(similar_papers)}개만 발견됨")
                # 부족한 경우 더미 논문으로 채움
                while len(similar_papers) < 3:
                    similar_papers.append((None, 0.0))

            # 2. 프롬프트 데이터 준비
            def safe_get(paper, attr, default="정보 없음"):
                if paper is None:
                    return default
                value = getattr(paper, attr, None)
                if attr == 'authors' and value:
                    if isinstance(value, str):
                        try:
                            authors_list = json.loads(value)
                            return ', '.join(authors_list) if isinstance(authors_list, list) else str(value)
                        except:
                            return str(value)
                    return str(value)
                return str(value) if value else default

            # 대상 논문 정보
            target_authors = safe_get(target_paper, 'authors', '저자 정보 없음')

            # 참고 논문들 정보
            ref_papers = similar_papers[:3]

            prompt_data = {
                'target_title': target_paper.title or '제목 없음',
                'target_authors': target_authors,
                'target_abstract': target_paper.abstract or '초록 없음',
                'target_methodology': target_paper.method_text or '방법론 정보 없음',
                'target_results': target_paper.result_text or '결과 정보 없음',

                'similarity_b': ref_papers[0][1] if ref_papers[0][0] else 0.0,
                'ref_title_b': safe_get(ref_papers[0][0], 'title'),
                'ref_authors_b': safe_get(ref_papers[0][0], 'authors'),
                'ref_abstract_b': safe_get(ref_papers[0][0], 'abstract'),
                'ref_methodology_b': safe_get(ref_papers[0][0], 'method_text'),
                'ref_results_b': safe_get(ref_papers[0][0], 'result_text'),

                'similarity_c': ref_papers[1][1] if ref_papers[1][0] else 0.0,
                'ref_title_c': safe_get(ref_papers[1][0], 'title'),
                'ref_authors_c': safe_get(ref_papers[1][0], 'authors'),
                'ref_abstract_c': safe_get(ref_papers[1][0], 'abstract'),
                'ref_methodology_c': safe_get(ref_papers[1][0], 'method_text'),
                'ref_results_c': safe_get(ref_papers[1][0], 'result_text'),

                'similarity_d': ref_papers[2][1] if ref_papers[2][0] else 0.0,
                'ref_title_d': safe_get(ref_papers[2][0], 'title'),
                'ref_authors_d': safe_get(ref_papers[2][0], 'authors'),
                'ref_abstract_d': safe_get(ref_papers[2][0], 'abstract'),
                'ref_methodology_d': safe_get(ref_papers[2][0], 'method_text'),
                'ref_results_d': safe_get(ref_papers[2][0], 'result_text'),
            }

            # 3. LangChain으로 리뷰 생성
            messages = self.review_prompt.format_messages(**prompt_data)
            response = await self.llm.ainvoke(messages)

            # 4. 결과 구성
            # 유효한 유사도 점수만 추출 (None이 아닌 논문들)
            valid_similarities = [p[1] for p in similar_papers if p[0] is not None]
            min_similarity = min(valid_similarities) if valid_similarities else 0.0

            result = {
                'status': 'success',
                'target_paper': {
                    'id': target_paper.id,
                    'title': target_paper.title,
                    'authors': target_authors
                },
                'reference_papers': [
                    {
                        'id': paper[0].id if paper[0] else None,
                        'title': safe_get(paper[0], 'title'),
                        'similarity': paper[1]
                    } for paper in similar_papers[:3]
                ],
                'review_content': response.content,
                'methodology': 'RAG with vector similarity',
                'model_used': 'gpt-4o-mini',
                'similarity_threshold': min_similarity,
                'found_similar_papers': len(valid_similarities)
            }

            logger.info(f"RAG 리뷰 생성 완료: 논문 {target_paper.id}")

            # 5. 사용자 논문 삭제 (일회용) - 비활성화
            # await self._cleanup_user_paper(db, target_paper)

            return result

        except Exception as e:
            logger.error(f"RAG 리뷰 생성 실패: {e}")
            return {
                'status': 'error',
                'message': f'RAG 리뷰 생성 중 오류 발생: {str(e)}',
                'target_paper': {
                    'id': target_paper.id,
                    'title': target_paper.title
                }
            }

    def parse_review_scores(self, review_content: str) -> Dict[str, Any]:
        """리뷰 내용에서 점수를 파싱합니다."""
        scores = {}
        try:
            # 간단한 정규식으로 점수 추출 (실제로는 더 정교한 파싱 필요)
            import re

            # 독창성, 방법론, 신뢰성, 의미성, 명확성 점수 찾기
            criteria = ['독창성', '방법론', '신뢰성', '의미성', '명확성']
            for criterion in criteria:
                pattern = rf'{criterion}.*?(\d+)점'
                match = re.search(pattern, review_content)
                if match:
                    scores[criterion] = int(match.group(1))

            # 최종 추천 찾기
            if 'Accept' in review_content:
                if 'Weak Accept' in review_content:
                    scores['recommendation'] = 'Weak Accept'
                else:
                    scores['recommendation'] = 'Accept'
            elif 'Reject' in review_content:
                if 'Weak Reject' in review_content:
                    scores['recommendation'] = 'Weak Reject'
                else:
                    scores['recommendation'] = 'Reject'
            else:
                scores['recommendation'] = 'Unknown'

            return scores

        except Exception as e:
            logger.error(f"점수 파싱 실패: {e}")
            return {}

    async def _cleanup_user_paper(self, db: Session, target_paper: Paper):
        """
        RAG 리뷰 완료 후 사용자 논문을 삭제합니다 (일회용).

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