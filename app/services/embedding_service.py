from typing import List, Tuple, Optional
import numpy as np
from sqlalchemy.orm import Session
import logging
from datetime import datetime
import openai

from app.config import settings
from app.models import Paper, PaperEmbedding
from app import crud

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.client = None
        self.model = "text-embedding-ada-002"  # OpenAI 임베딩 모델
        self._initialized = False

    def initialize(self):
        """임베딩 서비스를 초기화합니다."""
        if self._initialized:
            return True

        try:
            # OpenAI 클라이언트 초기화
            api_key = settings.OPENAI_API_KEY
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
                self._initialized = True
                logger.info("OpenAI 임베딩 서비스 초기화 완료")
                return True
            else:
                logger.warning("OpenAI API 키가 없어 임베딩 서비스를 가상 모드로 초기화합니다")
                self.model = "virtual_model"
                self._initialized = True
                return False

        except Exception as e:
            logger.error(f"임베딩 서비스 초기화 실패: {e}")
            self.model = "virtual_model"
            self._initialized = True
            return False

    def get_model(self):
        """모델 인스턴스를 반환합니다."""
        if self.model is None:
            self.initialize()
        return self.model

    async def create_embedding(self, text: str) -> List[float]:
        """텍스트를 임베딩 벡터로 변환"""
        # 필요시 초기화
        if not self._initialized:
            self.initialize()

        try:
            # OpenAI 클라이언트가 있는 경우 실제 임베딩 생성
            if self.client:
                # 텍스트 길이 제한 (OpenAI 토큰 제한)
                if len(text) > 8000:
                    text = text[:8000]

                response = self.client.embeddings.create(
                    model=self.model,
                    input=text
                )

                embedding = response.data[0].embedding
                logger.info(f"임베딩 생성 완료: 차원={len(embedding)}")

                return embedding
            else:
                # 가상 모드: 더미 임베딩 반환
                logger.warning("OpenAI 클라이언트가 없어 더미 임베딩을 반환합니다")
                return [0.0] * 1536  # ada-002 모델의 차원

        except Exception as e:
            logger.error(f"임베딩 생성 실패: {str(e)}")
            # 실패 시 더미 벡터 반환
            return [0.0] * 1536  # ada-002 모델의 차원

    async def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트를 배치로 임베딩"""
        # 필요시 초기화
        if not self._initialized:
            self.initialize()

        try:
            if self.client:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )

                embeddings = [item.embedding for item in response.data]
                logger.info(f"배치 임베딩 생성 완료: {len(embeddings)}개")

                return embeddings
            else:
                # 가상 모드: 더미 임베딩들 반환
                logger.warning("OpenAI 클라이언트가 없어 더미 임베딩들을 반환합니다")
                return [[0.0] * 1536 for _ in texts]

        except Exception as e:
            logger.error(f"배치 임베딩 생성 실패: {str(e)}")
            # 실패 시 더미 임베딩들 반환
            return [[0.0] * 1536 for _ in texts]

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """두 임베딩 벡터 간의 코사인 유사도 계산"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # 코사인 유사도 계산
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            logger.error(f"유사도 계산 실패: {str(e)}")
            return 0.0

    def find_similar_papers(self, query_embedding: List[float], paper_embeddings: List[tuple], top_k: int = 5) -> List[tuple]:
        """쿼리 임베딩과 유사한 논문들을 찾아 반환

        Args:
            query_embedding: 검색 쿼리의 임베딩
            paper_embeddings: [(paper_id, embedding), ...] 형태의 논문 임베딩 리스트
            top_k: 반환할 상위 논문 개수

        Returns:
            [(paper_id, similarity_score), ...] 형태의 유사한 논문 리스트
        """
        similarities = []

        for paper_id, embedding in paper_embeddings:
            similarity = self.calculate_similarity(query_embedding, embedding)
            similarities.append((paper_id, similarity))

        # 유사도 점수로 정렬하여 상위 k개 반환
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    async def create_paper_embedding(self, db: Session, paper_id: int, content_type: str = "abstract"):
        """논문에 대한 임베딩을 생성하고 저장합니다."""
        try:
            # 논문 조회
            paper = crud.get_paper(db=db, paper_id=paper_id)
            if not paper:
                logger.error(f"논문을 찾을 수 없습니다: {paper_id}")
                return None

            # 임베딩할 텍스트 선택
            if content_type == "abstract":
                text = paper.abstract or ""
            elif content_type == "title":
                text = paper.title or ""
            elif content_type == "full":
                text = f"{paper.title} {paper.abstract}".strip()
            else:
                text = paper.abstract or ""

            if not text:
                logger.warning(f"논문 {paper_id}에 대해 임베딩할 텍스트가 없습니다")
                return None

            # 기존 임베딩 확인
            existing_embedding = crud.get_paper_embedding(
                db=db, paper_id=paper_id, content_type=content_type
            )

            # 임베딩 생성
            embedding_vector = await self.create_embedding(text)

            if existing_embedding:
                # 기존 임베딩 업데이트
                existing_embedding.embedding_vector = embedding_vector
                existing_embedding.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(existing_embedding)
                logger.info(f"논문 {paper_id} 임베딩 업데이트 완료")
                return existing_embedding
            else:
                # 새 임베딩 생성
                new_embedding = PaperEmbedding(
                    paper_id=paper_id,
                    content_type=content_type,
                    embedding_vector=embedding_vector,
                    dimension=len(embedding_vector),
                    model_name=settings.EMBEDDING_MODEL
                )
                db.add(new_embedding)
                db.commit()
                db.refresh(new_embedding)
                logger.info(f"논문 {paper_id} 임베딩 생성 완료")
                return new_embedding

        except Exception as e:
            logger.error(f"논문 임베딩 생성 실패 (paper_id: {paper_id}): {e}")
            db.rollback()
            return None

    async def batch_create_embeddings(
        self,
        db: Session,
        paper_ids: List[int],
        content_type: str = "abstract"
    ):
        """여러 논문에 대해 배치로 임베딩을 생성합니다."""
        success_count = 0
        total_count = len(paper_ids)

        logger.info(f"배치 임베딩 생성 시작: {total_count}개 논문")

        for paper_id in paper_ids:
            try:
                result = await self.create_paper_embedding(
                    db=db, paper_id=paper_id, content_type=content_type
                )
                if result:
                    success_count += 1
            except Exception as e:
                logger.error(f"논문 {paper_id} 임베딩 생성 실패: {e}")
                continue

        logger.info(f"배치 임베딩 생성 완료: {success_count}/{total_count}")
        return success_count

    async def find_similar_papers(
        self,
        db: Session,
        query_text: str,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Tuple]:
        """텍스트 쿼리로 유사한 논문들을 찾습니다."""
        # 필요시 초기화
        if not self._initialized:
            self.initialize()

        try:
            # 1. 쿼리 텍스트를 임베딩으로 변환
            query_embedding = await self.create_embedding(query_text)

            # 2. 데이터베이스에서 모든 논문 임베딩 조회
            paper_embeddings = crud.get_all_paper_embeddings(db=db)

            if not paper_embeddings:
                logger.warning("데이터베이스에 임베딩이 없습니다")
                return []

            # 3. 유사도 계산
            similarities = []
            for embedding_record in paper_embeddings:
                similarity = self.calculate_similarity(
                    query_embedding,
                    embedding_record.embedding_vector
                )
                if similarity >= similarity_threshold:
                    # 논문 정보 조회
                    paper = crud.get_paper(db=db, paper_id=embedding_record.paper_id)
                    if paper:
                        similarities.append((paper, similarity))

            # 4. 유사도 점수로 정렬하여 상위 결과 반환
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:limit]

        except Exception as e:
            logger.error(f"유사 논문 검색 실패: {str(e)}")
            return []

# 전역 임베딩 서비스 인스턴스
embedding_service = EmbeddingService()