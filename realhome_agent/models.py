"""
Pydantic 데이터 모델 정의
=========================
아파트 매물 및 검색 쿼리에 대한 데이터 검증 모델

Author: RealHome Agent Team
Version: 1.0.0
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal
from datetime import date
from enum import Enum


class District(str, Enum):
    """서울시 대상 구역 Enum"""
    SONGPA = "송파구"
    MAPO = "마포구"
    NOWON = "노원구"


class LifestyleKeyword(str, Enum):
    """라이프스타일 키워드 Enum"""
    CHILDCARE = "육아"
    EDUCATION = "교육"
    CULTURE = "문화생활"
    TRANSPORTATION = "교통"
    NATURE = "자연환경"
    SHOPPING = "쇼핑"
    QUIET = "조용한"
    SAFETY = "안전"
    MEDICAL = "의료"
    EXERCISE = "운동"


class ApartmentSchema(BaseModel):
    """
    아파트 매물 정보 스키마
    
    정형 데이터와 비정형 데이터를 모두 포함하여
    ElasticSearch 인덱싱 및 검색에 활용됩니다.
    """
    # 기본 식별 정보
    kapt_code: str = Field(..., description="아파트 고유 코드")
    kapt_name: str = Field(..., description="아파트명")
    
    # 위치 정보
    doro_juso: Optional[str] = Field(None, description="도로명 주소")
    gu: str = Field(..., description="구 (송파구/마포구/노원구)")
    dong: Optional[str] = Field(None, description="동")
    
    # 정형 데이터 (가격, 면적 등)
    price_manwon: Optional[float] = Field(None, ge=0, description="거래가격 (만원)")
    price_krw: Optional[float] = Field(None, ge=0, description="거래가격 (원)")
    area_m2: Optional[float] = Field(None, ge=0, description="전용면적 (m²)")
    floor: Optional[int] = Field(None, description="층수")
    year_built: Optional[int] = Field(None, ge=1970, le=2030, description="준공연도")
    
    # 비정형 데이터 (리뷰, 텍스트)
    review_score: Optional[float] = Field(None, ge=0, le=5, description="평균 리뷰 점수")
    pros: Optional[str] = Field(None, description="장점 리뷰 텍스트")
    cons: Optional[str] = Field(None, description="단점 리뷰 텍스트")
    combined_review: Optional[str] = Field(None, description="통합 리뷰 텍스트 (검색용)")
    
    # 임베딩 벡터 (검색 시 활용)
    embedding: Optional[List[float]] = Field(None, description="텍스트 임베딩 벡터")
    
    @field_validator('gu')
    @classmethod
    def validate_district(cls, v: str) -> str:
        """대상 구역 검증"""
        valid_districts = ["송파구", "마포구", "노원구"]
        if v not in valid_districts:
            raise ValueError(f"지원하지 않는 구역입니다. 지원 구역: {valid_districts}")
        return v
    
    @field_validator('area_m2')
    @classmethod
    def validate_area(cls, v: Optional[float]) -> Optional[float]:
        """면적 범위 검증 (5~500 m²)"""
        if v is not None and (v < 5 or v > 500):
            raise ValueError("면적은 5~500 m² 범위여야 합니다.")
        return v


class SearchQuery(BaseModel):
    """
    사용자 검색 쿼리 스키마
    
    사용자의 정형 조건과 비정형 라이프스타일 선호도를
    구조화하여 검색에 활용합니다.
    """
    # 위치 조건
    districts: Optional[List[str]] = Field(
        default=None,
        description="검색 대상 구역 리스트 (예: ['송파구', '노원구'])"
    )
    dong: Optional[str] = Field(None, description="특정 동 검색")
    
    # 가격 조건
    min_price: Optional[float] = Field(None, ge=0, description="최소 가격 (만원)")
    max_price: Optional[float] = Field(None, ge=0, description="최대 가격 (만원)")
    
    # 면적 조건
    min_area: Optional[float] = Field(None, ge=0, description="최소 면적 (m²)")
    max_area: Optional[float] = Field(None, ge=0, description="최대 면적 (m²)")
    
    # 기타 정형 조건
    min_floor: Optional[int] = Field(None, description="최소 층수")
    max_floor: Optional[int] = Field(None, description="최대 층수")
    min_year_built: Optional[int] = Field(None, description="최소 준공연도")
    
    # 비정형 조건 (라이프스타일)
    lifestyle_keywords: Optional[List[str]] = Field(
        default=None,
        description="라이프스타일 키워드 (예: ['육아', '교통', '조용한'])"
    )
    natural_query: Optional[str] = Field(
        None,
        description="자연어 검색 쿼리 (예: '아이 키우기 좋은 조용한 동네')"
    )
    
    # 검색 설정
    top_k: int = Field(default=10, ge=1, le=50, description="반환할 결과 수")
    hybrid_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="하이브리드 검색 가중치 (0: BM25만, 1: Vector만)"
    )
    
    @field_validator('max_price')
    @classmethod
    def validate_price_range(cls, v: Optional[float], info) -> Optional[float]:
        """가격 범위 유효성 검증"""
        min_price = info.data.get('min_price')
        if v is not None and min_price is not None and v < min_price:
            raise ValueError("최대 가격은 최소 가격보다 커야 합니다.")
        return v
    
    @field_validator('max_area')
    @classmethod
    def validate_area_range(cls, v: Optional[float], info) -> Optional[float]:
        """면적 범위 유효성 검증"""
        min_area = info.data.get('min_area')
        if v is not None and min_area is not None and v < min_area:
            raise ValueError("최대 면적은 최소 면적보다 커야 합니다.")
        return v


class DealRecord(BaseModel):
    """
    실거래가 기록 스키마
    
    과거 거래 이력 데이터를 저장합니다.
    """
    gu: str = Field(..., description="구")
    dong: str = Field(..., description="동")
    apt_name: str = Field(..., description="아파트명")
    deal_date: date = Field(..., description="거래일")
    area_m2: float = Field(..., ge=0, description="전용면적 (m²)")
    floor: Optional[int] = Field(None, description="층수")
    year_built: Optional[int] = Field(None, description="준공연도")
    price_manwon: float = Field(..., ge=0, description="거래가격 (만원)")
    price_krw: float = Field(..., ge=0, description="거래가격 (원)")


class ReviewData(BaseModel):
    """
    리뷰 데이터 스키마
    
    아파트 리뷰 정보를 구조화합니다.
    """
    kapt_name: str = Field(..., description="아파트명")
    doro_juso: Optional[str] = Field(None, description="도로명 주소")
    score: float = Field(..., ge=0, le=5, description="리뷰 점수")
    pros: str = Field(..., description="장점")
    cons: str = Field(..., description="단점")
    source_index: Optional[int] = Field(None, description="원본 인덱스")


class LoanCalculationRequest(BaseModel):
    """
    대출 계산 요청 스키마
    
    LTV, DSR 기반 대출 가능 금액 계산에 필요한 정보입니다.
    """
    # 구매 정보
    property_price: float = Field(..., gt=0, description="매물 가격 (만원)")
    
    # 소득 정보
    annual_income: float = Field(..., gt=0, description="연 소득 (만원)")
    
    # 기존 부채
    existing_debt_payment: float = Field(
        default=0,
        ge=0,
        description="기존 연간 부채 상환액 (만원)"
    )
    
    # 대출 조건
    loan_term_years: int = Field(default=30, ge=1, le=40, description="대출 기간 (년)")
    interest_rate: float = Field(default=4.5, ge=0, le=20, description="금리 (%)")
    
    # 규제 지역 여부
    is_regulated_area: bool = Field(default=True, description="규제 지역 여부")
    is_first_home: bool = Field(default=True, description="생애 최초 주택 여부")
    house_count: int = Field(default=0, ge=0, description="보유 주택 수")


class LoanCalculationResult(BaseModel):
    """
    대출 계산 결과 스키마
    """
    # LTV 관련
    ltv_limit: float = Field(..., description="적용 LTV 한도 (%)")
    ltv_max_loan: float = Field(..., description="LTV 기준 최대 대출액 (만원)")
    
    # DSR 관련
    dsr_limit: float = Field(..., description="적용 DSR 한도 (%)")
    dsr_max_loan: float = Field(..., description="DSR 기준 최대 대출액 (만원)")
    
    # 최종 결과
    final_max_loan: float = Field(..., description="최종 최대 대출 가능액 (만원)")
    required_down_payment: float = Field(..., description="필요 자기자본 (만원)")
    monthly_payment: float = Field(..., description="예상 월 상환액 (만원)")
    
    # 안내 메시지
    regulation_notes: List[str] = Field(default=[], description="적용된 규제 안내")


class PolicySearchResult(BaseModel):
    """
    정책 검색 결과 스키마
    """
    title: str = Field(..., description="정책/뉴스 제목")
    snippet: str = Field(..., description="내용 요약")
    link: str = Field(..., description="원문 링크")
    source: str = Field(..., description="출처")
    published_date: Optional[str] = Field(None, description="게시일")


class AgentResponse(BaseModel):
    """
    에이전트 응답 스키마
    """
    answer: str = Field(..., description="에이전트 답변")
    apartments: Optional[List[ApartmentSchema]] = Field(
        None,
        description="추천 아파트 목록"
    )
    loan_info: Optional[LoanCalculationResult] = Field(
        None,
        description="대출 계산 결과"
    )
    policy_info: Optional[List[PolicySearchResult]] = Field(
        None,
        description="관련 정책 정보"
    )
    follow_up_questions: Optional[List[str]] = Field(
        None,
        description="추천 후속 질문"
    )


class ConversationMessage(BaseModel):
    """
    대화 메시지 스키마
    """
    role: Literal["user", "assistant", "system"] = Field(..., description="메시지 역할")
    content: str = Field(..., description="메시지 내용")
    timestamp: Optional[str] = Field(None, description="타임스탬프")


if __name__ == "__main__":
    # 모델 테스트
    import json
    
    # SearchQuery 예시
    query = SearchQuery(
        districts=["송파구", "노원구"],
        max_price=70000,
        min_area=60,
        lifestyle_keywords=["육아", "교통"],
        natural_query="아이 키우기 좋은 조용한 동네"
    )
    print("SearchQuery 예시:")
    print(json.dumps(query.model_dump(), ensure_ascii=False, indent=2))
    
    # LoanCalculationRequest 예시
    loan_request = LoanCalculationRequest(
        property_price=70000,
        annual_income=8000,
        loan_term_years=30,
        interest_rate=4.5,
        is_regulated_area=True,
        is_first_home=True
    )
    print("\nLoanCalculationRequest 예시:")
    print(json.dumps(loan_request.model_dump(), ensure_ascii=False, indent=2))
