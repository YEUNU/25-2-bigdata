# =============================================================================
# 리얼홈 에이전트 Dockerfile (NVIDIA CUDA 가속 지원)
# =============================================================================
# Python 3.10 + CUDA 12.8 지원 (임베딩 모델 GPU 가속)
# 빌드 캐시 최적화 적용

# NVIDIA CUDA 베이스 이미지 사용
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# 메타데이터
LABEL maintainer="RealHome Agent Team"
LABEL version="1.0.0"
LABEL description="라이프스타일 기반 리얼홈 에이전트 - 서울시 아파트 맞춤 추천 서비스 (CUDA 가속)"

# 환경 변수 설정 (PIP_NO_CACHE_DIR 제거하여 캐시 활용)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Python 3.10 및 시스템 패키지 설치 (Ubuntu 22.04 기본 Python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    curl \
    git \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# ============================================================================
# 의존성 설치 (캐시 최적화)
# ============================================================================
# 1단계: PyTorch CUDA 버전 먼저 설치 (가장 무거움 - 캐시 우선순위 높음)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2단계: 나머지 의존성 설치 (requirements.txt에서 torch 제외)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# ============================================================================
# 애플리케이션 코드 복사 (가장 자주 변경됨 - 마지막에 배치)
# ============================================================================
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
