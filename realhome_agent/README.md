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

# 아파트 데이터 인덱싱 (최초 1회, GPU 가속)
docker-compose --profile indexing up indexer

# 정책 문서 PDF 인덱싱
docker-compose exec realhome-agent python policy_indexer.py

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

## 정책 문서 인덱싱

PDF 형식의 정책 문서를 OCR로 텍스트 추출 후 ElasticSearch에 인덱싱합니다.

### 로컬 실행 (Python 환경)
```bash
# 의존성 설치
pip install -r requirements.txt

# Tesseract OCR 설치 (Ubuntu/WSL)
sudo apt-get install tesseract-ocr tesseract-ocr-kor poppler-utils

# Tesseract OCR 설치 (Windows)
# https://github.com/UB-Mannheim/tesseract/wiki 에서 설치

# PDF 파일을 상위 디렉토리에 배치 (예: R25_1015.pdf)
# 실행
python policy_indexer.py

# OCR 사용 (이미지 기반 PDF)
USE_OCR=true python policy_indexer.py
```

### Docker 환경에서 실행
```bash
# 컨테이너 내부에서 실행
docker-compose exec realhome-agent python policy_indexer.py

# 또는 docker run
docker-compose run --rm realhome-agent python policy_indexer.py
```

### 정책 검색 테스트
```python
from policy_indexer import PolicyIndexer

indexer = PolicyIndexer()
indexer.connect()

# 검색
results = indexer.search("LTV 규제", size=5)
for result in results:
    print(f"제목: {result['document']['title']}")
    print(f"점수: {result['score']}")
    print(f"키워드: {result['document']['keywords']}")
```

## 주요 기능

1. **아파트 검색**: ElasticSearch 하이브리드 검색 (BM25 + kNN 임베딩)
2. **정책 검색**: PDF 문서 OCR + ElasticSearch 인덱싱
3. **대출 계산**: LTV/DSR 기반 대출 가능 금액 계산
4. **AI 에이전트**: LangGraph ReAct 패턴 기반 대화형 추천
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
