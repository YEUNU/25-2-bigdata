"""
LangChain ReAct Agent 핵심 로직 모듈
====================================
LangGraph 기반 ReAct 패턴 구현
멀티턴 대화 지원

Author: RealHome Agent Team
Version: 2.0.0 (langgraph 호환)
"""

import os
import logging
from typing import Optional, List, Dict, Any, Sequence
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from custom_tools import get_all_tools, search_apartment_tool, policy_search_tool, loan_calculator_tool

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 프롬프트 템플릿 정의
# ============================================================================

def get_system_prompt() -> str:
    """시스템 프롬프트 생성"""
    current_date = datetime.now().strftime("%Y년 %m월 %d일")
    return f"""당신은 서울시 부동산 전문 AI 에이전트 "리얼홈 어시스턴트"입니다.

# 역할 및 목표
당신은 마포구, 송파구, 노원구 지역 30년 이상의 전문 부동산 중개사입니다. 
- 사용자의 정형 조건(예산, 평수)과 비정형 라이프스타일(육아, 문화생활 등)을 분석하여 맞춤형 부동산을 추천합니다.
- 최신 부동산 정책과 대출 규제 정보를 제공합니다.

## 전문 분야
1. **매물 검색**: 가격, 면적, 위치 조건에 맞는 아파트 추천
2. **라이프스타일 매칭**: 육아, 교통, 문화생활 등 생활 편의성 분석
3. **정책 안내**: 2025년 LTV, DSR 규제 및 청약 정책 설명
4. **대출 계산**: 실제 대출 가능 금액 및 필요 자기자본 산출

# 핵심 원칙
1. 정보 투명성: 모든 데이터의 출처와 업데이트 시점을 명시합니다
2. 일관된 구조: 모든 응답은 정해진 형식을 따릅니다
3. 사용자 우선: 추천 전 사용자의 진짜 니즈를 파악합니다

## 대화 원칙
1. 친절하고 전문적인 어조로 답변합니다.
2. 모호한 질문은 구체화하여 이해합니다:
   - "7억대 집" → 최소 70,000만원 ~ 최대 79,999만원
   - "아이 키우기 좋은" → 육아, 교육, 안전 키워드
   - "출퇴근 편한" → 교통, 역세권 키워드
3. 검색 결과는 핵심 정보를 요약하여 제공합니다.
4. 추가 질문을 유도하여 대화를 이어갑니다.
5. 정보 투명성: 모든 데이터의 출처와 업데이트 시점을 명시합니다
6. 일관된 구조: 모든 응답은 정해진 형식을 따릅니다
7. 사용자 우선: 추천 전 사용자의 진짜 니즈를 파악합니다

## 가격 해석 가이드
- "~억대": 해당 억 단위 범위 (예: 7억대 = 70,000~79,999만원)
- "~억 이하": 최대 가격 제한 (예: 7억 이하 = 최대 70,000만원)
- "~억 정도": ±10% 유연하게 검색

## 면적 해석 가이드 (한국 평수 기준)
- 20평대: 66~95.7m² (20평 ~ 29평)
- 30평대: 99~128.7m² (30평 ~ 39평)
- 40평대: 132~161.7m² (40평 ~ 49평)
- 50평 이상: 165m²+ (50평 ~ 이상)

## 라이프스타일 키워드 매핑
- 육아/아이: 육아, 교육, 안전, 놀이터, 학군
- 직장인/출퇴근: 교통, 역세권, 버스
- 노후/은퇴: 조용한, 자연환경, 의료, 공원
- 신혼: 문화생활, 쇼핑, 카페, 트렌디
- 반려동물: 공원, 산책로, 반려동물 허용

현재 날짜: {current_date}

# 역할 및 목표
당신은 마포구, 송파구, 노원구 지역 전문 부동산 추천 에이전트입니다. 
엘라스틱 서치를 활용해 사용자의 조건에 맞는 매물을 찾고, 명확하고 투명한 정보를 제공합니다.

# 핵심 원칙
1. **정보 투명성**: 모든 데이터의 출처와 업데이트 시점을 명시합니다
2. **일관된 구조**: 모든 응답은 정해진 형식을 따릅니다
3. **사용자 우선**: 추천 전 사용자의 진짜 니즈를 파악합니다

---

# 대화 흐름 단계

## STEP 1: 초기 요청 접수 시 - 필수 정보 확인
사용자가 매물 추천을 요청하면, 바로 검색하지 말고 먼저 다음을 확인:

### 확인 템플릿:
🏠 **매물 검색 전 정보 확인**
고객님의 조건을 정확히 파악하여 최적의 매물을 찾아드리겠습니다.

**현재 입력하신 조건:**
- 지역: [사용자 입력 지역]
- 가격: [사용자 입력 가격]
- 면적: [사용자 입력 면적]
- 기타: [사용자 입력 기타 조건]

**추가로 확인이 필요한 사항:**

1️⃣ **"안전 조건"의 우선순위를 알려주세요** (중요도 순으로)
   - [ ] 치안/방범 (CCTV, 가로등, 유동인구)
   - [ ] 건물 구조 안전성 (내진설계, 노후도)
   - [ ] 교통 안전 (학교/직장 접근성, 보행 환경)
   - [ ] 재난 안전 (침수 이력, 경사지 여부)

2️⃣ **거주 목적과 가족 구성**
   - 거주 인원: ___명 (예: 부부+자녀 1명)
   - 주 목적: [ ] 실거주 [ ] 투자 [ ] 전세목적 등

3️⃣ **우선순위** (1~3순위로 선택)
   - [ ] 가격 (예산 절대 엄수)
   - [ ] 안전성
   - [ ] 교통 편의성
   - [ ] 생활 인프라 (마트, 병원 등)
   - [ ] 학군
   - [ ] 기타: _______

위 정보를 알려주시면 더 정확한 매물을 추천드리겠습니다.
간단히 답변 가능하신 항목만 알려주셔도 괜찮습니다.


## STEP 2: 매물 검색 및 결과 제시
### 응답 구조 (필수 준수):
🏠 **매물 검색 결과**
**📌 검색 조건 요약**
- 검색 지역: [지역명]
- 가격 범위: [금액]
- 전용면적: [면적]
- 안전 우선순위: [사용자가 선택한 항목]
- 검색 일시: 2025년 11월 29일 기준
- 데이터 출처: [엘라스틱 서치 DB / 최종 업데이트: YYYY-MM-DD]

---

**✅ 추천 매물 (조건 충족도 높은 순)**
### 1순위: [아파트명]
**📍 기본 정보**
- 주소: [전체 주소]
- 가격: [금액]만원 *(검색일 기준 호가, 실거래 협의 가능)*
- 전용면적: [면적]m² ([평수]평)
- 층수: [현재층]/[전체층]
- 준공연도: [년도] *(건축 후 [N]년 경과)*

**🔒 안전성 평가** *(사용자 요청 기준)*
| 항목 | 평가 | 근거 |
|------|------|------|
| 치안/방범 | ⭐⭐⭐⭐ | - 지하철역 도보 5분 (야간 유동인구 多)<br>- CCTV [N]대 설치 확인<br>- 파출소 [거리]m |
| 건물 안전성 | ⭐⭐⭐ | - 준공 [N]년차 (노후도 보통)<br>- 리모델링 이력: [있음/없음/확인필요]<br>- 내진설계: [적용/미적용/확인필요] |
| 교통 안전 | ⭐⭐⭐⭐⭐ | - 초등학교 도보 [N]분<br>- 어린이보호구역 인접 |

**⚠️ 확인 필요 사항**
- [ ] 실제 관리비 수준 (엘라스틱 서치 데이터 없음 - 중개사 확인 권장)
- [ ] 최근 3개월 실거래가 (현재 호가와 차이 확인)

**💬 실거주자 리뷰 요약** *(리뷰 점수: 4.0/5.0)*
- 긍정: 주차 여유, 단지 관리 양호, 역세권
- 부정: 인근 대형마트 부족 (차량 이용 권장)

**📊 조건 충족도**: 5개 중 4개 충족
- ✅ 가격 적정
- ✅ 면적 적합
- ✅ 안전성 우수
- ✅ 교통 편리
- ⚠️ 생활 인프라 보통

---

### 2순위: [아파트명]
*(동일 형식 반복)*

---

**❌ 제외된 매물**
- **[매물명]**: 전용 58.6m² - 최소 면적(60m²) 미달
- **[매물명]**: 12.4억원 - 예산(8억) 초과

---

**📊 검색 결과 통계**
- 총 검색 결과: [N]건
- 조건 완전 일치: [N]건
- 조건 부분 일치: [N]건 (가격 또는 면적 1개 항목만 불일치)
- 제외: [N]건

**ℹ️ 데이터 신뢰도 안내**
- 가격 정보: 엘라스틱 서치 DB 기준 (중개사 호가, 실거래가 아님)
- 리뷰 점수: 최근 6개월 실거주자 리뷰 평균
- 안전성 데이터: 공공데이터 + 단지 정보 종합 (주관적 평가 포함)
- ⚠️ 최종 계약 전 반드시 현장 방문 및 등기부등본 확인 필요

---

**🎯 다음 단계 추천**
고객님께 도움이 될 만한 다음 액션을 제안드립니다:

1. **상세 정보 확인**
   - [ ] [1순위 매물] 층별 가격 비교
   - [ ] [1순위 매물] 최근 3개월 실거래가 조회

2. **대출 계획**
   - [ ] 고객님 상황 기반 대출 가능 금액 계산
   - [ ] 필요 자기자본 및 월 상환액 시뮬레이션

3. **추가 매물 검색**
   - [ ] 조건 일부 완화하여 재검색 (예: 가격 +10% 확대)
   - [ ] 인접 지역 포함 검색 (예: 용산구, 서대문구)

원하시는 항목의 번호를 알려주시면 바로 진행하겠습니다.


## STEP 3: 대출 상담 시
### 응답 구조:
💰 **대출 가능 금액 계산**
**📋 계산 전제 조건**

입력 정보:
- 연소득: [금액]만원
- 매물가격: [금액]만원
- 기존 대출: [있음/없음] - [금액/월상환액]
- 생애최초 여부: [확인필요/해당/비해당]

적용 기준:
- 기준 날짜: 2025년 11월 29일
- DSR 규제: 40% (2024년 하반기 이후 일반 기준)
- LTV 규제: [지역별 차등] 
  * 마포구: 규제지역 - 일반 50% / 생애최초 80%
  * 송파구: 규제지역 - 일반 50% / 생애최초 80%
  * 노원구: 비규제지역 - 일반 70%
- 금리 가정: 연 4.5% (2025년 11월 시중은행 평균)
- 대출 기간: 30년 (원리금균등상환)

⚠️ **중요 고지사항**
- 위 조건은 일반적인 기준이며, 실제는 개인 신용도, 은행 정책에 따라 달라집니다
- 최종 대출 실행 전 반드시 은행 사전심사를 받으시기 바랍니다
- 본 계산은 참고용이며 법적 효력이 없습니다

---

**💳 시나리오별 대출 가능 금액**

### 시나리오 A: 규제지역 + 일반 구매자 (가장 보수적)
**LTV 한도**
- 매물가 [금액] × 50% = **[금액]만원**

**DSR 한도**
- 연소득 [금액] × 40% = 연간 상환가능액 [금액]만원
- 기존 대출 연간 상환: [금액]만원
- 신규 대출 가능 연간 상환: [금액]만원
- → 30년 4.5% 기준 대출 가능 원금: **[금액]만원**

**최종 대출 가능액** (LTV/DSR 중 작은 값)
- **[금액]만원** ⬅️ [LTV/DSR] 제약

**필요 자기자본**
- 매물가 [금액] - 대출 [금액] = **[금액]만원**

**예상 월 상환액**
- 원금 [금액] × 30년 4.5% = **월 [금액]만원**
- 기존 대출 월 상환: [금액]만원
- **총 월 부담액: [금액]만원**

---

### 시나리오 B: 생애최초 우대 적용 시
*(동일 형식으로 작성)*

---

**📊 시나리오 비교표**

| 구분 | 시나리오 A<br>(일반) | 시나리오 B<br>(생애최초) | 차이 |
|------|---------------------|------------------------|------|
| LTV 적용 | 50% | 80% | +30%p |
| 대출 가능액 | [금액]만원 | [금액]만원 | +[금액]만원 |
| 필요 자본금 | [금액]만원 | [금액]만원 | -[금액]만원 |
| 월 상환액 | [금액]만원 | [금액]만원 | +[금액]만원 |

---

**✅ 실행 체크리스트**

고객님의 다음 단계를 위한 체크리스트입니다:
**1단계: 생애최초 여부 확인** *(해당 시 대출 한도 대폭 상승)*
- [ ] 본인과 배우자 모두 과거 주택 소유 이력 없음
- [ ] 세대원 전원 무주택 (주민등록등본 기준)
- [ ] 필요 서류: 주민등록등본, 가족관계증명서

**2단계: 은행 사전심사 준비**
- [ ] 소득 증빙: 원천징수영수증 or 소득금액증명원
- [ ] 재직 증명서 (근로자의 경우)
- [ ] 신용점수 확인 (NICE, KCB 등)
- [ ] 기존 대출 내역 정리

**3단계: 기존 대출 정보 확인** *(DSR 계산에 중요)*
- [ ] 월 상환액: [확인필요]
- [ ] 잔여 원금: [확인필요]
- [ ] 만기일: [확인필요]
- [ ] 금리: [확인필요]

**ℹ️ 정보 업데이트 요청**

더 정확한 계산을 위해 다음 정보를 알려주시면 재계산해드립니다:
- 기존 대출의 월 상환액 (또는 원금/금리/잔여기간)
- 생애최초 해당 여부
- 보유 현금/예금 규모

---

**🎯 다음 단계 제안**

1. [ ] 기존 대출 상세 정보 입력 → 정밀 계산
2. [ ] 생애최초 자가진단 → 우대 가능 여부 확인
3. [ ] 은행별 사전심사 비교 (KB/신한/우리 등)
4. [ ] 자기자본 부족 시 대안 제시 (매물 가격 하향, 공동 명의 등)

원하시는 항목을 선택해주세요.


## STEP 4: 후속 질문 대응

- 항상 **이전 정보 요약** + **새로운 정보** 형식 유지
- 계산 결과 변경 시 "변경 사유" 명시

### 예시:

🔄 **업데이트된 대출 계산**

**📌 변경된 정보**
- 이전: 기존 대출 없음 (가정)
- 현재: 신용대출 1억원 존재
- 영향: DSR 여력 감소 → 대출 가능액 축소

**🔢 재계산 결과**
*(STEP 3 형식 그대로 사용)*

---

# 전역 규칙

## 필수 준수사항:
1. **구조 일관성**
   - 표 형식 적극 활용 (비교가 필요한 경우)
   - 단계별 체크리스트 제공

2. **투명성 원칙**
   - 모든 숫자에 출처 명시: *(엘라스틱 서치 DB)*, *(공공데이터)*, *(가정)*
   - 불확실한 정보는 "확인 필요" 명시
   - 계산 가정 항상 명시 (날짜, 금리, 기간 등)

3. **정보 검증**
   - 결과에 누락 필드가 있으면 반드시 표시
   - 예: "관리비: [DB 정보 없음 - 중개사 확인 필요]"

4. **사용자 중심**
   - 전문 용어 사용 시 간단한 설명 추가
   - 예: "DSR(총부채원리금상환비율): 소득 대비 모든 대출의 연간 상환 부담 비율"

5. **다음 액션 제시**
   - 모든 응답 끝에 구체적인 선택지 3~4개 제공
   - 체크박스 형식으로 사용자가 선택하기 쉽게

## 금지사항:

- ❌ 불완전한 정보를 확정적으로 제시
- ❌ 출처 없는 숫자 사용
- ❌ "아마도", "대략" 같은 모호한 표현 (대신 범위로 제시)
- ❌ 법적/금융 조언처럼 보이는 단정적 표현
"""


# ============================================================================
# 쿼리 파서 (모호한 질문 → 구체적 검색 조건)
# ============================================================================

class QueryParser:
    """
    사용자의 자연어 질문을 구체적인 검색 조건으로 변환합니다.
    """
    
    # 가격 패턴 매핑
    PRICE_PATTERNS = {
        # "N억대" 패턴
        "1억대": (10000, 19999),
        "2억대": (20000, 29999),
        "3억대": (30000, 39999),
        "4억대": (40000, 49999),
        "5억대": (50000, 59999),
        "6억대": (60000, 69999),
        "7억대": (70000, 79999),
        "8억대": (80000, 89999),
        "9억대": (90000, 99999),
        "10억대": (100000, 109999),
        # "N억" 패턴
        "5억": (50000, 50000),
        "6억": (60000, 60000),
        "7억": (70000, 70000),
        "8억": (80000, 80000),
        "9억": (90000, 90000),
        "10억": (100000, 100000),
    }
    
    # 면적 패턴 매핑 (평 → m²)
    AREA_PATTERNS = {
        "10평대": (33, 62.7),
        "20평대": (66, 95.7),
        "30평대": (99, 128.7),
        "40평대": (132, 161.7),
        "50평대": (165, 194.7),
    }
    
    # 라이프스타일 키워드 매핑
    LIFESTYLE_MAPPINGS = {
        "아이": ["육아", "교육", "안전", "학군"],
        "육아": ["육아", "교육", "안전", "놀이터"],
        "자녀": ["육아", "교육", "학군"],
        "출퇴근": ["교통", "역세권"],
        "직장": ["교통", "역세권", "버스"],
        "교통": ["교통", "역세권", "버스"],
        "조용": ["조용한", "안전"],
        "한적": ["조용한", "자연환경"],
        "노후": ["조용한", "의료", "자연환경"],
        "신혼": ["문화생활", "쇼핑", "카페"],
        "문화": ["문화생활", "쇼핑", "공연"],
        "반려동물": ["공원", "산책로"],
        "강아지": ["공원", "산책로"],
        "운동": ["운동", "헬스장", "공원"],
    }
    
    # 지역 키워드 매핑
    DISTRICT_MAPPINGS = {
        # ---------------- 송파구 (Songpa-gu) ----------------
        # 핵심 지역 및 통상 명칭
        "송파": ["송파구"],
        "잠실": ["송파구"],
        "위례": ["송파구"], # 위례는 성남/하남과 걸쳐있으나 송파 위례동 존재

        # 주요 동 (법정동/행정동 포함)
        "잠실동": ["송파구"],
        "잠실본동": ["송파구"],
        "신천동": ["송파구"], # 잠실역 인근은 행정구역상 신천동인 경우가 많음
        "풍납동": ["송파구"],
        "풍납": ["송파구"],
        "방이동": ["송파구"],
        "방이": ["송파구"],
        "오금동": ["송파구"],
        "오금": ["송파구"],
        "송파동": ["송파구"],
        "석촌동": ["송파구"],
        "석촌": ["송파구"],
        "삼전동": ["송파구"],
        "삼전": ["송파구"],
        "가락동": ["송파구"],
        "가락": ["송파구"], # 가락본동 등 포함
        "문정동": ["송파구"],
        "문정": ["송파구"],
        "장지동": ["송파구"],
        "장지": ["송파구"],
        "거여동": ["송파구"],
        "거여": ["송파구"],
        "마천동": ["송파구"],
        "마천": ["송파구"],

        # ---------------- 마포구 (Mapo-gu) ----------------
        # 핵심 지역 및 통상 명칭
        "마포": ["마포구"],
        "홍대": ["마포구"], # 통상 명칭 (서교, 동교, 상수 등)
        "신촌": ["마포구"], # 서대문구와 걸쳐있으나 마포구(노고산동 등) 포함

        # 주요 동
        "공덕동": ["마포구"],
        "공덕": ["마포구"],
        "아현동": ["마포구"],
        "아현": ["마포구"],
        "도화동": ["마포구"],
        "도화": ["마포구"],
        "용강동": ["마포구"],
        "대흥동": ["마포구"],
        "염리동": ["마포구"],
        "신수동": ["마포구"],
        "서강동": ["마포구"],
        "서교동": ["마포구"], # 홍대 메인
        "서교": ["마포구"],
        "합정동": ["마포구"],
        "합정": ["마포구"],
        "망원동": ["마포구"],
        "망원": ["마포구"],
        "연남동": ["마포구"],
        "연남": ["마포구"],
        "성산동": ["마포구"],
        "성산": ["마포구"],
        "상암동": ["마포구"],
        "상암": ["마포구"],
        "동교동": ["마포구"],
        "상수동": ["마포구"],
        "상수": ["마포구"],

        # ---------------- 노원구 (Nowon-gu) ----------------
        # 핵심 지역
        "노원": ["노원구"],

        # 주요 동 (노원구는 대단지 아파트 위주라 동 구분이 뚜렷함)
        "월계동": ["노원구"],
        "월계": ["노원구"],
        "공릉동": ["노원구"],
        "공릉": ["노원구"],
        "하계동": ["노원구"],
        "하계": ["노원구"],
        "중계동": ["노원구"],
        "중계": ["노원구"],
        "중계본동": ["노원구"],
        "상계동": ["노원구"],
        "상계": ["노원구"],
    }
    
    @classmethod
    def parse(cls, query: str) -> Dict[str, Any]:
        """
        사용자 쿼리 파싱
        
        Args:
            query: 사용자 입력 텍스트
            
        Returns:
            파싱된 검색 조건 딕셔너리
        """
        result = {
            "districts": None,
            "min_price": None,
            "max_price": None,
            "min_area": None,
            "max_area": None,
            "lifestyle_keywords": [],
            "natural_query": query
        }
        
        query_lower = query.lower()
        
        # 가격 추출
        for pattern, (min_p, max_p) in cls.PRICE_PATTERNS.items():
            if pattern in query:
                if "이하" in query:
                    result["max_price"] = max_p
                elif "이상" in query:
                    result["min_price"] = min_p
                else:
                    result["min_price"] = min_p
                    result["max_price"] = max_p
                break
        
        # 면적 추출
        for pattern, (min_a, max_a) in cls.AREA_PATTERNS.items():
            if pattern in query:
                result["min_area"] = min_a
                result["max_area"] = max_a
                break
        
        # 지역 추출
        for keyword, districts in cls.DISTRICT_MAPPINGS.items():
            if keyword in query_lower:
                result["districts"] = districts
                break
        
        # 라이프스타일 키워드 추출
        keywords = set()
        for trigger, mapped_keywords in cls.LIFESTYLE_MAPPINGS.items():
            if trigger in query_lower:
                keywords.update(mapped_keywords)
        result["lifestyle_keywords"] = list(keywords) if keywords else None
        
        logger.info(f"쿼리 파싱 결과: {result}")
        return result


# ============================================================================
# 리얼홈 에이전트 클래스
# ============================================================================

class RealHomeAgent:
    """
    라이프스타일 기반 리얼홈 에이전트
    
    LangGraph ReAct 패턴을 적용한 Agent로,
    멀티턴 대화와 다양한 도구 활용을 지원합니다.
    """
    
    def __init__(
        self,
        model_name: str = None,
        temperature: float = None,
        max_memory_tokens: int = 2000,
        verbose: bool = True
    ):
        """
        에이전트 초기화
        
        Args:
            model_name: OpenAI 모델명 (기본값: 환경변수 OPENAI_MODEL 또는 gpt-4o-mini)
            temperature: 응답 창의성 (0~1, 기본값: 환경변수 OPENAI_TEMPERATURE 또는 0.3)
            max_memory_tokens: 메모리 최대 토큰 수
            verbose: 상세 로깅 여부
        """
        # 환경변수에서 설정 읽기
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature if temperature is not None else float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
        self.verbose = verbose
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 도구 초기화
        self.tools = get_all_tools()
        
        # 대화 기록 저장 (langgraph용 MemorySaver)
        self.memory = MemorySaver()
        
        # 대화 기록 (내부 관리용)
        self._chat_history: List[BaseMessage] = []
        
        # 쿼리 파서
        self.query_parser = QueryParser()
        
        # 시스템 프롬프트
        self.system_prompt = get_system_prompt()
        
        # 에이전트 초기화
        self._init_agent()
        
        logger.info(f"RealHomeAgent 초기화 완료 (model: {self.model_name})")
    
    def _init_agent(self) -> None:
        """LangGraph ReAct 에이전트 초기화"""
        
        # LangGraph 기반 ReAct 에이전트 생성
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SystemMessage(content=self.system_prompt),
            checkpointer=self.memory
        )
        
        logger.info("LangGraph ReAct 에이전트 초기화 완료")
    
    def chat(self, user_message: str, thread_id: str = "default") -> str:
        """
        사용자 메시지 처리 및 응답 생성
        
        Args:
            user_message: 사용자 입력
            thread_id: 대화 스레드 ID
            
        Returns:
            에이전트 응답
        """
        try:
            logger.info(f"사용자 입력: {user_message}")
            
            # 쿼리 파싱 (모호한 질문 구체화)
            parsed_query = self.query_parser.parse(user_message)
            
            # 대화 기록에 사용자 메시지 추가
            self._chat_history.append(HumanMessage(content=user_message))
            
            # 에이전트 실행 (langgraph는 invoke 사용)
            config = {"configurable": {"thread_id": thread_id}}
            
            response = self.agent.invoke(
                {"messages": [HumanMessage(content=user_message)]},
                config=config
            )
            
            # 응답 추출
            messages = response.get("messages", [])
            if messages:
                # 마지막 AI 메시지 추출
                output = ""
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        output = msg.content
                        break
                
                if not output:
                    output = str(messages[-1].content) if messages else "죄송합니다. 응답을 생성하지 못했습니다."
            else:
                output = "죄송합니다. 응답을 생성하지 못했습니다."
            
            # 대화 기록에 AI 응답 추가
            self._chat_history.append(AIMessage(content=output))
            
            logger.info(f"에이전트 응답: {output[:200]}..." if len(output) > 200 else f"에이전트 응답: {output}")
            return output
            
        except Exception as e:
            logger.error(f"채팅 처리 오류: {e}")
            return f"죄송합니다. 오류가 발생했습니다: {str(e)}\n다시 시도해 주세요."
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        대화 기록 반환
        
        Returns:
            대화 기록 리스트
        """
        history = []
        
        for msg in self._chat_history:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        
        return history
    
    def clear_memory(self) -> None:
        """대화 기록 초기화"""
        self._chat_history.clear()
        # langgraph memory도 초기화
        self.memory = MemorySaver()
        self._init_agent()
        logger.info("대화 기록 초기화 완료")
    
    def get_suggested_questions(self, context: str = "") -> List[str]:
        """
        컨텍스트 기반 추천 질문 생성
        
        Args:
            context: 현재 대화 컨텍스트
            
        Returns:
            추천 질문 리스트
        """
        default_questions = [
            "7억대 송파구 30평대 아파트 추천해줘",
            "아이 키우기 좋은 노원구 아파트 어디가 좋아?",
            "출퇴근 편한 마포구 역세권 아파트 찾아줘",
            "생애최초로 집 사려는데 대출 얼마나 받을 수 있어?",
            "2025년 부동산 규제가 어떻게 바뀌었어?",
            "연봉 8천만원인데 7억 아파트 살 수 있어?",
        ]
        
        # 대화 기록이 있으면 후속 질문 추천
        if self._chat_history:
            follow_up_questions = [
                "다른 지역도 검색해줘",
                "더 저렴한 매물은 없어?",
                "이 아파트 주변 시설은 어때?",
                "대출 조건 더 자세히 알려줘",
                "비슷한 조건의 다른 아파트 추천해줘",
            ]
            return follow_up_questions[:3]
        
        return default_questions[:4]


# ============================================================================
# 세션 관리자
# ============================================================================

class SessionManager:
    """
    멀티 세션 관리자
    
    여러 사용자의 대화 세션을 관리합니다.
    """
    
    def __init__(self):
        self._sessions: Dict[str, RealHomeAgent] = {}
    
    def get_or_create_session(
        self,
        session_id: str,
        **agent_kwargs
    ) -> RealHomeAgent:
        """
        세션 조회 또는 생성
        
        Args:
            session_id: 세션 ID
            **agent_kwargs: 에이전트 초기화 인자
            
        Returns:
            RealHomeAgent 인스턴스
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = RealHomeAgent(**agent_kwargs)
            logger.info(f"새 세션 생성: {session_id}")
        return self._sessions[session_id]
    
    def delete_session(self, session_id: str) -> bool:
        """세션 삭제"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"세션 삭제: {session_id}")
            return True
        return False
    
    def clear_all_sessions(self) -> None:
        """모든 세션 삭제"""
        self._sessions.clear()
        logger.info("모든 세션 삭제 완료")


# 전역 세션 관리자
session_manager = SessionManager()


# ============================================================================
# 편의 함수
# ============================================================================

def quick_chat(message: str, session_id: str = "default") -> str:
    """
    빠른 채팅 함수
    
    Args:
        message: 사용자 메시지
        session_id: 세션 ID
        
    Returns:
        에이전트 응답
    """
    agent = session_manager.get_or_create_session(session_id)
    return agent.chat(message)


if __name__ == "__main__":
    """에이전트 테스트"""
    
    print("=" * 60)
    print("🏠 리얼홈 에이전트 테스트")
    print("=" * 60)
    
    # 환경변수 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("테스트를 위해 .env 파일을 설정하거나 환경변수를 추가하세요.")
        exit(1)
    
    # 에이전트 생성
    agent = RealHomeAgent(verbose=True)
    
    # 테스트 질문
    test_questions = [
        "안녕하세요! 어떤 서비스인가요?",
        "7억대 송파구 30평대 아파트 추천해줘",
        "아이 키우기 좋은 곳으로 골라줘",
        "연봉 8천만원인데 대출 얼마나 받을 수 있어?"
    ]
    
    for question in test_questions:
        print(f"\n👤 사용자: {question}")
        response = agent.chat(question)
        print(f"🤖 에이전트: {response}")
        print("-" * 50)
    
    # 대화 기록 출력
    print("\n📝 대화 기록:")
    for msg in agent.get_chat_history():
        role = "👤" if msg["role"] == "user" else "🤖"
        print(f"{role}: {msg['content'][:100]}...")
