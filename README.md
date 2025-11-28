# 리얼홈 에이전트 (RealHome Agent)

라이프스타일 기반 서울시 아파트 추천 AI 챗봇

## 폴더 구조

```
realhome_agent/
├── agent_core.py      # LangGraph ReAct 에이전트 핵심 로직
├── app.py             # Streamlit UI 애플리케이션
├── custom_tools.py    # 검색, 정책, 대출계산 도구
├── indexer.py         # ElasticSearch 데이터 인덱싱
├── models.py          # Pydantic 데이터 모델
├── search_engine.py   # 하이브리드 검색 엔진 (BM25 + kNN)
├── requirements.txt   # Python 패키지 의존성
├── Dockerfile         # 애플리케이션 Docker 이미지
├── docker-compose.yml # Docker Compose 설정
├── .env.example       # 환경변수 템플릿
├── .env               # 실제 환경변수 (git 제외)
├── elasticsearch/
│   └── Dockerfile     # ES + Nori 플러그인 이미지
├── tests/             # 테스트 코드
│   ├── test_agent_core.py
│   ├── test_app.py
│   ├── test_custom_tools.py
│   ├── test_indexer.py
│   ├── test_integration.py
│   ├── test_models.py
│   └── test_search_engine.py
├── data/              # 데이터 저장 디렉토리
└── logs/              # 로그 저장 디렉토리
```

## 환경변수

모든 설정은 환경변수로 관리합니다 (`os.getenv()` 사용):

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 키 | (필수) |
| `OPENAI_MODEL` | 사용할 모델 | `gpt-5-mini-2025-08-07` |
| `OPENAI_TEMPERATURE` | 응답 창의성 | `0.3` |
| `ES_HOST` | ElasticSearch 호스트 | `elasticsearch` |
| `ES_PORT` | ElasticSearch 포트 | `9200` |
| `ES_INDEX` | 인덱스 이름 | `realhome_apartments` |
| `EMBEDDING_MODEL` | 임베딩 모델 | `BAAI/bge-m3` |
| `EMBEDDING_DEVICE` | 임베딩 디바이스 | `cuda` (GPU) / `cpu` |
| `GOOGLE_API_KEY` | Google Search API 키 | (선택) |
| `GOOGLE_SEARCH_ENGINE_ID` | Search Engine ID | (선택) |

## 실행 방법

### 1. 환경 설정
```bash
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 등 설정
```

### 2. Docker 실행

#### GPU 환경 (NVIDIA CUDA 가속)
```bash
# 사전 요구사항: NVIDIA Driver, NVIDIA Container Toolkit 설치 필요
# 설치 확인: nvidia-smi

# 전체 시스템 시작 (GPU 가속)
docker-compose up -d

# 데이터 인덱싱 (최초 1회, GPU 가속)
docker-compose --profile indexing up indexer

# 로그 확인
docker-compose logs -f realhome-agent
```

#### CPU 전용 환경 (GPU 없음)
```bash
# CPU 전용 docker-compose 사용
docker-compose -f docker-compose.cpu.yml up -d

# 데이터 인덱싱 (CPU 전용)
docker-compose -f docker-compose.cpu.yml --profile indexing up indexer
```

### 3. 접속
- Streamlit UI: http://localhost:8501
- ElasticSearch: http://localhost:9200

### NVIDIA Container Toolkit 설치 (Ubuntu/WSL2)
```bash
# 저장소 설정
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 설치
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Docker 재시작
sudo systemctl restart docker

# 테스트
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

## 테스트

```bash
# 로컬 테스트
cd realhome_agent
pip install -r requirements.txt
pytest tests/ -v
```

## 기술 스택

- **LLM**: OpenAI GPT (gpt-5-mini-2025-08-07)
- **Agent**: LangGraph ReAct 패턴
- **검색**: ElasticSearch 8.11 (Nori 한국어 분석기)
- **임베딩**: BAAI/bge-m3 (1024차원, 다국어)
- **UI**: Streamlit
