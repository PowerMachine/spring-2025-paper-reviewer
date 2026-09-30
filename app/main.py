from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging
import os

from app.config import settings
from app.database import get_db, engine, Base
from app.api import papers
from app.schemas import HealthCheck

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 데이터베이스 테이블 생성
Base.metadata.create_all(bind=engine)

# FastAPI 앱 생성
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI 기반 논문 검색 및 리뷰 시스템",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# 절대경로로 정적 파일 및 템플릿 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# 정적 파일 및 템플릿 설정 (안전하게)
try:
    if os.path.exists(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
        logger.info(f"✅ 정적 파일 설정 완료: {STATIC_DIR}")
    else:
        logger.warning(f"⚠️ 정적 파일 디렉토리 없음: {STATIC_DIR}")

    if os.path.exists(TEMPLATES_DIR):
        templates = Jinja2Templates(directory=TEMPLATES_DIR)
        logger.info(f"✅ 템플릿 설정 완료: {TEMPLATES_DIR}")
        TEMPLATES_AVAILABLE = True
    else:
        logger.warning(f"⚠️ 템플릿 디렉토리 없음: {TEMPLATES_DIR}")
        templates = None
        TEMPLATES_AVAILABLE = False
except Exception as e:
    logger.error(f"❌ 정적 파일/템플릿 설정 실패: {e}")
    templates = None
    TEMPLATES_AVAILABLE = False

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 핵심 API 라우터만 등록 (확실히 작동하는 것)
app.include_router(
    papers.router,
    prefix=f"{settings.API_V1_STR}/papers",
    tags=["papers"]
)

# 선택적 라우터들 (안전하게 로드)
try:
    from app.api.v1.endpoints import upload
    app.include_router(
        upload.router,
        prefix=f"{settings.API_V1_STR}/upload",
        tags=["upload"]
    )
    logger.info("✅ Upload 라우터 로드 완료")
except ImportError as e:
    logger.warning(f"⚠️ Upload 라우터 로드 실패: {e}")

try:
    from app.api.v1.endpoints import embedding_analysis
    app.include_router(
        embedding_analysis.router,
        prefix=f"{settings.API_V1_STR}/embedding",
        tags=["embedding"]
    )
    logger.info("✅ Embedding 라우터 로드 완료")
except ImportError as e:
    logger.warning(f"⚠️ Embedding 라우터 로드 실패: {e}")

try:
    from app.api.v1.endpoints.reviews import router as reviews_router
    app.include_router(
        reviews_router,
        prefix=f"{settings.API_V1_STR}",
        tags=["reviews"]
    )
    logger.info("✅ Reviews 라우터 로드 완료")
except ImportError as e:
    logger.warning(f"⚠️ Reviews 라우터 로드 실패: {e}")

try:
    from app.api.v1.endpoints import sql_semantic
    app.include_router(
        sql_semantic.router,
        prefix=f"{settings.API_V1_STR}/sql-semantic",
        tags=["sql-semantic"]
    )
    logger.info("✅ SQL Semantic 라우터 로드 완료")
except ImportError as e:
    logger.warning(f"⚠️ SQL Semantic 라우터 로드 실패: {e}")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """메인 페이지"""
    if TEMPLATES_AVAILABLE and templates:
        try:
            return templates.TemplateResponse("index.html", {"request": request})
        except Exception as e:
            logger.warning(f"템플릿 렌더링 실패: {e}")

    # 템플릿이 없거나 실패한 경우 기본 HTML 반환
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
        <head>
            <title>📚 Paper Reviewer System</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                .container { max-width: 800px; margin: 0 auto; }
                .status { background: #e8f5e8; padding: 20px; border-radius: 10px; margin: 20px 0; }
                button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 Paper Reviewer System</h1>
                <div class="status">
                    <h3>✅ 시스템 상태: 정상 작동</h3>
                    <p>서버가 성공적으로 실행되고 있습니다!</p>
                </div>
                <ul>
                    <li><a href="/docs">📚 API 문서</a></li>
                    <li><a href="/health">💊 헬스 체크</a></li>
                </ul>
                <div style="margin-top: 20px;">
                    <button onclick="generateSampleData()">샘플 데이터 생성</button>
                    <script>
                        async function generateSampleData() {
                            try {
                                const response = await fetch('/api/v1/papers/sample-data', {
                                    method: 'POST'
                                });
                                const result = await response.json();
                                alert('샘플 데이터 생성 완료: ' + result.message);
                            } catch (error) {
                                alert('오류 발생: ' + error.message);
                            }
                        }
                    </script>
                </div>
            </div>
        </body>
    </html>
    """)

@app.get("/health", response_model=HealthCheck)
async def health_check(db: Session = Depends(get_db)):
    """헬스 체크 엔드포인트"""
    try:
        # 데이터베이스 연결 확인
        db.execute(text("SELECT 1"))
        database_status = "connected"
        status = "healthy"
    except Exception as e:
        logger.error(f"데이터베이스 연결 실패: {e}")
        database_status = "disconnected"
        status = "unhealthy"

    return HealthCheck(
        status=status,
        database=database_status
    )

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행되는 이벤트"""
    logger.info("🚀 논문 리뷰어 시스템 시작")
    logger.info("📊 데이터베이스 설정 로드 완료")
    logger.info(f"🌐 서버 주소: http://localhost:8000")
    logger.info(f"📁 작업 디렉토리: {os.getcwd()}")
    logger.info(f"📁 베이스 디렉토리: {BASE_DIR}")

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 실행되는 이벤트"""
    logger.info("논문 리뷰어 시스템 종료")

if __name__ == "__main__":
    import uvicorn
    print("🚀 논문 리뷰어 시스템을 시작합니다...")
    print(f"📁 작업 디렉토리: {os.getcwd()}")
    print(f"📁 베이스 디렉토리: {BASE_DIR}")
    uvicorn.run(
        "app.main:app",  # 수정: main:app -> app.main:app
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
