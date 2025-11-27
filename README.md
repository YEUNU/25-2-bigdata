# 리얼홈 에이전트 (RealHome Agent)

서울시 아파트 매수 희망자를 위한 **라이프스타일 기반 맞춤형 부동산 추천 챗봇** 서비스입니다.

## 🏠 프로젝트 개요

- **대상 지역**: 서울시 송파구, 마포구, 노원구
- **핵심 기능**: 
  - 정형 조건(예산, 평수) + 비정형 라이프스타일(육아, 문화생활) 분석
  - 맞춤형 부동산 추천
  - 2025년 기준 정책 정보 및 대출 계산

## 🛠️ 기술 스택

| 구분 | 기술 |
|------|------|
| 언어 | Python 3.9+ |
| LLM 프레임워크 | LangChain (ReAct 패턴) |
| 검색 엔진 | ElasticSearch (BM25 + kNN 하이브리드) |
| 임베딩 모델 | google/embeddinggemma-300m |
| 외부 API | Google Search API |
| 메모리 | ConversationBufferMemory |
| UI | Streamlit |
| 배포 | Docker, Docker Compose |

## 📁 프로젝트 구조

```
realhome_agent/
├── models.py          # Pydantic 데이터 모델
├── search_engine.py   # ElasticSearch 인덱싱/검색
├── custom_tools.py    # LangChain 도구 정의
├── agent_core.py      # ReAct 에이전트 핵심 로직
├── app.py             # Streamlit UI
├── indexer.py         # 데이터 인덱싱 스크립트
├── config.py          # 설정 관리
├── requirements.txt   # Python 의존성
├── Dockerfile         # Docker 이미지 빌드
├── docker-compose.yml # Docker Compose 설정
└── .env.example       # 환경 변수 예시
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
cd realhome_agent

# 환경 변수 설정
cp .env.example .env
# .env 파일 편집하여 OPENAI_API_KEY 등 설정
```

### 2. Docker로 실행 (권장)

```bash
# 서비스 시작
docker compose up -d

# 데이터 인덱싱 (최초 1회)
docker compose --profile indexing up indexer

# 로그 확인
docker compose logs -f realhome-agent
```

### 3. 로컬 실행

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# ElasticSearch 실행 (별도 터미널)
docker run -d --name elasticsearch \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0

# 데이터 인덱싱
python indexer.py

# Streamlit 앱 실행
streamlit run app.py
```

### 4. 접속

- **Streamlit UI**: http://localhost:8501
- **ElasticSearch**: http://localhost:9200
- **Kibana** (선택): http://localhost:5601

## 📚 모듈별 설명

### 1. `models.py` - Pydantic 데이터 모델

```python
from models import ApartmentSchema, SearchQuery, LoanCalculationRequest

# 아파트 스키마
apartment = ApartmentSchema(
    kapt_code="A12345",
    kapt_name="잠실엘스",
    gu="송파구",
    price_manwon=340000,
    area_m2=84.8
)

# 검색 쿼리
query = SearchQuery(
    districts=["송파구"],
    max_price=70000,
    lifestyle_keywords=["육아", "교통"]
)
```

### 2. `search_engine.py` - ElasticSearch 검색

```python
from search_engine import SearchEngine, ESConfig

# 검색 엔진 초기화
config = ESConfig(host="localhost", port=9200)
engine = SearchEngine(config)
engine.connect()

# 하이브리드 검색 (BM25 + Vector)
results = engine.hybrid_search(query, bm25_weight=0.5, vector_weight=0.5)
```

### 3. `custom_tools.py` - LangChain 도구

```python
from custom_tools import search_apartment_tool, policy_search_tool, loan_calculator_tool

# 아파트 검색
result = search_apartment_tool.invoke({
    "districts": ["송파구"],
    "max_price": 70000,
    "lifestyle_keywords": ["육아"]
})

# 대출 계산
result = loan_calculator_tool.invoke({
    "property_price": 70000,
    "annual_income": 8000,
    "is_first_home": True
})
```

### 4. `agent_core.py` - ReAct 에이전트

```python
from agent_core import RealHomeAgent

# 에이전트 생성
agent = RealHomeAgent(model_name="gpt-4o-mini")

# 대화
response = agent.chat("7억대 송파구 아파트 추천해줘")
print(response)

# 멀티턴 대화
response = agent.chat("더 저렴한 곳은 없어?")
```

## 🔧 환경 변수

| 변수명 | 필수 | 기본값 | 설명 |
|--------|------|--------|------|
| `OPENAI_API_KEY` | ✅ | - | OpenAI API 키 |
| `OPENAI_MODEL` | ❌ | gpt-4o-mini | 사용할 모델 |
| `ES_HOST` | ❌ | elasticsearch | ES 호스트 |
| `ES_PORT` | ❌ | 9200 | ES 포트 |
| `ES_INDEX` | ❌ | realhome_apartments | 인덱스명 |
| `GOOGLE_API_KEY` | ❌ | - | Google Search API 키 |

## 💡 사용 예시

### 매물 검색
```
사용자: 7억대 송파구 30평대 아파트 추천해줘
에이전트: 송파구 7억대(70,000~79,999만원) 30평대 아파트를 검색했습니다...
```

### 라이프스타일 기반 검색
```
사용자: 아이 키우기 좋은 조용한 동네 찾아줘
에이전트: 육아와 조용한 환경을 고려한 아파트를 추천해드립니다...
```

### 대출 계산
```
사용자: 연봉 8천만원인데 7억 아파트 살 수 있어?
에이전트: 2025년 규제 기준으로 대출 가능 금액을 계산해드립니다...
- LTV 80% (생애최초): 56,000만원
- DSR 40%: 48,000만원
- 필요 자기자본: 약 2.2억원
```

## 🧪 테스트

### 테스트 파일 구조

```
tests/
├── __init__.py           # 패키지 초기화
├── test_models.py        # Pydantic 모델 테스트
├── test_search_engine.py # ElasticSearch 검색 엔진 테스트
├── test_custom_tools.py  # LangChain 도구 테스트
├── test_agent_core.py    # ReAct 에이전트 테스트
├── test_indexer.py       # 데이터 인덱서 테스트
├── test_app.py           # Streamlit UI 테스트
└── test_integration.py   # 통합 테스트
```

### 테스트 실행

```bash
# 테스트 의존성 설치
pip install pytest pytest-cov pytest-asyncio

# 전체 테스트 실행
pytest tests/ -v

# 개별 모듈 테스트
pytest tests/test_models.py -v          # 모델
pytest tests/test_search_engine.py -v   # 검색 엔진
pytest tests/test_custom_tools.py -v    # 도구
pytest tests/test_agent_core.py -v      # 에이전트
pytest tests/test_indexer.py -v         # 인덱서
pytest tests/test_app.py -v             # UI
pytest tests/test_integration.py -v     # 통합

# 커버리지 포함 실행
pytest tests/ -v --cov=. --cov-report=html

# HTML 리포트 확인 (htmlcov/index.html)
```

### 테스트 종류

| 테스트 파일 | 설명 | 주요 테스트 케이스 |
|-------------|------|-------------------|
| `test_models.py` | Pydantic 모델 검증 | 필수 필드, 기본값, 유효성 검사 |
| `test_search_engine.py` | ES 검색 엔진 | 연결, 인덱싱, 하이브리드 검색 |
| `test_custom_tools.py` | LangChain 도구 | 아파트 검색, 정책 검색, 대출 계산 |
| `test_agent_core.py` | ReAct 에이전트 | 쿼리 파싱, 대화 메모리, 응답 생성 |
| `test_indexer.py` | 데이터 인덱서 | CSV 로드, 데이터 병합, 인덱싱 |
| `test_app.py` | Streamlit UI | 세션 상태, 메시지 표시, 사이드바 |
| `test_integration.py` | E2E 통합 테스트 | 전체 흐름, 오류 처리 |

## 📝 라이선스

MIT License

## 👥 기여자

RealHome Agent Team
