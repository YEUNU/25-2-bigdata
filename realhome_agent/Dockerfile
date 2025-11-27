# =============================================================================
# 리얼홈 에이전트 Dockerfile
# =============================================================================
# Python 3.9 + CUDA 지원 (임베딩 모델용)

FROM python:3.9-slim

# 메타데이터
LABEL maintainer="RealHome Agent Team"
LABEL version="1.0.0"
LABEL description="라이프스타일 기반 리얼홈 에이전트 - 서울시 아파트 맞춤 추천 서비스"

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 의존성 설치 (캐시 활용)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 데이터 디렉토리 생성
RUN mkdir -p /app/data /app/logs

# Streamlit 포트 노출
EXPOSE 8501

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 실행 명령
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
