import arxiv
import requests
import time
from typing import List, Dict, Optional, Any
from sqlalchemy.orm import Session
import logging
from datetime import datetime, timedelta

from app.models import Paper
from app.schemas import PaperCreate
from app.config import settings
from app.services.duplicate_checker import DuplicateChecker

logger = logging.getLogger(__name__)

class PaperCrawler:
    def __init__(self):
        self.arxiv_client = arxiv.Client()
        self.semantic_scholar_base = settings.SEMANTIC_SCHOLAR_API
        self.duplicate_checker = DuplicateChecker()

    async def crawl_arxiv_papers(
        self,
        db: Session,
        query: str = "machine learning",
        max_results: int = 100,
        start_date: Optional[datetime] = None
    ) -> List[Paper]:
        """ArXiv에서 논문을 크롤링합니다."""

        try:
            # ArXiv 검색 생성
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )

            crawled_papers = []

            for paper in search.results():
                try:
                    # 중복 체크 (개선된 방식)
                    arxiv_id = paper.entry_id.split('/')[-1]
                    authors_str = ', '.join([author.name for author in paper.authors])

                    is_duplicate, existing_paper = self.duplicate_checker.check_duplicate(
                        db=db,
                        title=paper.title,
                        authors=authors_str,
                        doi=paper.doi,
                        arxiv_id=arxiv_id
                    )

                    if is_duplicate:
                        logger.info(f"중복 논문 스킵: {paper.title} (기존 ID: {existing_paper.id})")
                        continue

                    # 날짜 필터링
                    if start_date and paper.published < start_date:
                        continue

                    # Paper 객체 생성
                    new_paper = Paper(
                        title=paper.title,
                        authors=[author.name for author in paper.authors],
                        conference="arXiv",
                        year=paper.published.year,
                        abstract=paper.summary,
                        keywords=self._extract_keywords_from_categories(paper.categories),
                        arxiv_id=arxiv_id,
                        url=paper.entry_id,
                        doi=paper.doi
                    )

                    db.add(new_paper)
                    db.commit()
                    db.refresh(new_paper)

                    crawled_papers.append(new_paper)
                    logger.info(f"ArXiv 논문 저장 완료: {new_paper.title}")

                    # API 제한을 위한 지연
                    time.sleep(0.5)

                except Exception as e:
                    logger.error(f"ArXiv 논문 처리 실패: {e}")
                    db.rollback()
                    continue

            logger.info(f"ArXiv 크롤링 완료: {len(crawled_papers)}개 수집")
            return crawled_papers

        except Exception as e:
            logger.error(f"ArXiv 크롤링 실패: {e}")
            return []

    def _extract_keywords_from_categories(self, categories: List[str]) -> List[str]:
        """ArXiv 카테고리를 키워드로 변환합니다."""
        category_mapping = {
            'cs.AI': 'Artificial Intelligence',
            'cs.LG': 'Machine Learning',
            'cs.CV': 'Computer Vision',
            'cs.CL': 'Natural Language Processing',
            'cs.IR': 'Information Retrieval',
            'cs.DB': 'Database',
            'cs.DS': 'Data Structures',
            'stat.ML': 'Statistical Machine Learning',
        }

        keywords = []
        for category in categories:
            if category in category_mapping:
                keywords.append(category_mapping[category])
            else:
                keywords.append(category)

        return keywords

    async def crawl_semantic_scholar_papers(
        self,
        db: Session,
        query: str = "machine learning",
        max_results: int = 100
    ) -> List[Paper]:
        """Semantic Scholar API에서 논문을 크롤링합니다."""

        try:
            url = f"{self.semantic_scholar_base}/paper/search"
            headers = {
                'User-Agent': 'Paper Reviewer System v1.0'
            }

            params = {
                'query': query,
                'limit': min(max_results, 100),  # API 제한
                'fields': 'paperId,title,authors,year,abstract,venue,citationCount,url,externalIds'
            }

            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()
            crawled_papers = []

            for paper_data in data.get('data', []):
                try:
                    # 중복 체크 (개선된 방식)
                    doi = None
                    external_ids = paper_data.get('externalIds', {})
                    if external_ids and 'DOI' in external_ids:
                        doi = external_ids['DOI']

                    # 저자 정보 추출
                    authors_str = ""
                    if paper_data.get('authors'):
                        authors_str = ', '.join([author['name'] for author in paper_data['authors']])

                    is_duplicate, existing_paper = self.duplicate_checker.check_duplicate(
                        db=db,
                        title=paper_data['title'],
                        authors=authors_str,
                        doi=doi
                    )

                    if is_duplicate:
                        logger.info(f"중복 논문 스킵: {paper_data['title']} (기존 ID: {existing_paper.id})")
                        continue

                    # Paper 객체 생성 (저자 정보는 이미 추출됨)
                    authors = []
                    if paper_data.get('authors'):
                        authors = [author['name'] for author in paper_data['authors']]

                    new_paper = Paper(
                        title=paper_data['title'],
                        authors=authors,
                        conference=paper_data.get('venue', 'Unknown'),
                        year=paper_data.get('year'),
                        abstract=paper_data.get('abstract', ''),
                        doi=doi,
                        url=paper_data.get('url', ''),
                        citation_count=paper_data.get('citationCount', 0)
                    )

                    db.add(new_paper)
                    db.commit()
                    db.refresh(new_paper)

                    crawled_papers.append(new_paper)
                    logger.info(f"Semantic Scholar 논문 저장 완료: {new_paper.title}")

                    # API 제한을 위한 지연
                    time.sleep(0.5)

                except Exception as e:
                    logger.error(f"Semantic Scholar 논문 처리 실패: {e}")
                    db.rollback()
                    continue

            logger.info(f"Semantic Scholar 크롤링 완료: {len(crawled_papers)}개 수집")
            return crawled_papers

        except Exception as e:
            logger.error(f"Semantic Scholar 크롤링 실패: {e}")
            return []

    async def crawl_sample_papers(self, db: Session) -> List[Paper]:
        """샘플 논문 데이터를 생성합니다."""

        sample_papers_data = [
            {
                "title": "Attention Is All You Need",
                "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
                "conference": "NEURIPS",
                "year": 2017,
                "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
                "keywords": ["transformer", "attention", "neural networks", "NLP"],
                "doi": "10.5555/3295222.3295349"
            },
            {
                "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"],
                "conference": "NAACL",
                "year": 2019,
                "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
                "keywords": ["BERT", "transformer", "pre-training", "NLP", "bidirectional"],
                "doi": "10.18653/v1/N19-1423"
            },
            {
                "title": "Deep Residual Learning for Image Recognition",
                "authors": ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren"],
                "conference": "CVPR",
                "year": 2016,
                "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.",
                "keywords": ["ResNet", "deep learning", "computer vision", "residual learning"],
                "doi": "10.1109/CVPR.2016.90"
            },
            {
                "title": "Generative Adversarial Networks",
                "authors": ["Ian J. Goodfellow", "Jean Pouget-Abadie", "Mehdi Mirza"],
                "conference": "NEURIPS",
                "year": 2014,
                "abstract": "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.",
                "keywords": ["GAN", "generative models", "adversarial training", "deep learning"],
                "doi": "10.5555/2969033.2969125"
            },
            {
                "title": "You Only Look Once: Unified, Real-Time Object Detection",
                "authors": ["Joseph Redmon", "Santosh Divvala", "Ross Girshick"],
                "conference": "CVPR",
                "year": 2016,
                "abstract": "We present YOLO, a new approach to object detection. Prior work on object detection repurposes classifiers to perform detection. Instead, we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities.",
                "keywords": ["YOLO", "object detection", "computer vision", "real-time"],
                "doi": "10.1109/CVPR.2016.91"
            }
        ]

        created_papers = []

        for paper_data in sample_papers_data:
            try:
                # 중복 체크 (개선된 방식)
                authors_str = ', '.join(paper_data["authors"]) if paper_data.get("authors") else ""

                is_duplicate, existing_paper = self.duplicate_checker.check_duplicate(
                    db=db,
                    title=paper_data["title"],
                    authors=authors_str,
                    doi=paper_data.get("doi")
                )

                if is_duplicate:
                    logger.info(f"중복 샘플 논문 스킵: {paper_data['title']} (기존 ID: {existing_paper.id})")
                    continue

                # Paper 객체 생성
                new_paper = Paper(**paper_data)

                db.add(new_paper)
                db.commit()
                db.refresh(new_paper)

                created_papers.append(new_paper)
                logger.info(f"샘플 논문 생성 완료: {new_paper.title}")

            except Exception as e:
                logger.error(f"샘플 논문 생성 실패: {e}")
                db.rollback()
                continue

        logger.info(f"샘플 논문 생성 완료: {len(created_papers)}개")
        return created_papers

    async def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """ArXiv ID로 특정 논문을 가져옵니다."""

        try:
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(search.results())

            return {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "conference": "arXiv",
                "year": paper.published.year,
                "abstract": paper.summary,
                "keywords": self._extract_keywords_from_categories(paper.categories),
                "arxiv_id": arxiv_id,
                "url": paper.entry_id,
                "doi": paper.doi
            }

        except Exception as e:
            logger.error(f"ArXiv 논문 조회 실패 ({arxiv_id}): {e}")
            return None

# 전역 크롤러 인스턴스
paper_crawler = PaperCrawler()