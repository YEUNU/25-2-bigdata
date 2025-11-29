# 리얼홈 에이전트 (RealHome Agent)

라이프스타일 기반 서울시 아파트 추천 AI 챗봇

## 📁 폴더 구조

```
realhome_agent/
├── agent_core.py       # LangGraph ReAct 에이전트 핵심 로직
├── app.py              # Streamlit UI 애플리케이션
├── custom_tools.py     # 검색, 정책, 대출계산 도구
├── indexer.py          # ElasticSearch 아파트 데이터 인덱싱
├── policy_indexer.py   # PDF 정책 문서 OCR 및 인덱싱
├── models.py           # Pydantic 데이터 모델
├── search_engine.py    # 하이브리드 검색 엔진 (BM25 + kNN)
├── requirements.txt    # Python 패키지 의존성
├── Dockerfile          # GPU 지원 Docker 이미지 (CUDA)
├── Dockerfile.cpu      # CPU 전용 Docker 이미지
├── docker-compose.yml  # GPU 환경 Docker Compose
├── docker-compose.cpu.yml # CPU 환경 Docker Compose
├── auto-deploy.ps1     # 자동 배포 스크립트 (Windows)
├── auto-deploy.sh      # 자동 배포 스크립트 (Linux/Mac)
├── .env.example        # 환경변수 템플릿
├── .env                # 실제 환경변수 (git 제외)
├── elasticsearch/
│   └── Dockerfile      # ES + Nori 플러그인 이미지
├── tests/              # 테스트 코드
├── data/               # 데이터 저장 디렉토리
└── logs/               # 로그 저장 디렉토리
```

## 🔧 환경변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 키 | (필수) |
| `OPENAI_MODEL` | 사용할 모델 | `gpt-4o-mini` |
| `OPENAI_TEMPERATURE` | 응답 창의성 | `0.3` |
| `ES_HOST` | ElasticSearch 호스트 | `elasticsearch` |
| `ES_PORT` | ElasticSearch 포트 | `9200` |
| `ES_INDEX` | 아파트 인덱스 이름 | `realhome_apartments` |
| `EMBEDDING_MODEL` | 임베딩 모델 | `BAAI/bge-m3` |
| `EMBEDDING_DEVICE` | 임베딩 디바이스 | `cuda` / `cpu` |

## 🚀 실행 방법

### 1. 환경 설정
```bash
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 설정
```

### 2. Docker 실행

#### GPU 환경 (NVIDIA CUDA 가속)
```bash
# 전체 시스템 시작
docker-compose up -d

# 아파트 데이터 인덱싱 (최초 1회)
docker-compose --profile indexing up indexer

# 정책 문서 PDF 인덱싱
docker-compose exec realhome-agent python policy_indexer.py

# 로그 확인
docker-compose logs -f realhome-agent
```

#### CPU 전용 환경
```bash
docker-compose -f docker-compose.cpu.yml up -d
docker-compose -f docker-compose.cpu.yml --profile indexing up indexer
```

### 3. 접속
- **Streamlit UI**: http://localhost:8501
- **ElasticSearch**: http://localhost:9200

## 📊 ElasticSearch 인덱스

| 인덱스 | 데이터 소스 | 용도 |
|--------|------------|------|
| `realhome_apartments` | CSV (아파트, 리뷰, 실거래가) | 아파트 매물 검색 |
| `realhome_policies` | PDF (R25_*.pdf) | 부동산 정책 검색 |

```bash
# 인덱스 확인
docker-compose exec elasticsearch curl -s "http://localhost:9200/_cat/indices?v"
```

## 🛠️ 주요 기능

| 기능 | 설명 |
|------|------|
| 🏢 **아파트 검색** | 가격, 면적, 지역, 라이프스타일 조건 검색 |
| 📋 **정책 검색** | PDF 정책 문서에서 LTV/DSR 규제 등 검색 |
| 💰 **대출 계산** | LTV/DSR 기반 대출 가능 금액 산출 |
| 🤖 **AI 에이전트** | LangGraph ReAct 패턴 대화형 추천 |

## 🏗️ 기술 스택

- **LLM**: OpenAI GPT-4o-mini
- **Agent**: LangGraph ReAct 패턴
- **검색**: ElasticSearch 8.x + Nori 한국어 분석기
- **임베딩**: BAAI/bge-m3
- **OCR**: Tesseract (한국어)
- **UI**: Streamlit
- **GPU**: NVIDIA CUDA 12.8
