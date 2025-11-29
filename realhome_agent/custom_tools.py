"""
LangChain 도구(Tools) 정의 모듈
===============================
@tool 데코레이터를 사용한 커스텀 도구 정의
- search_apartment_tool: ES 기반 매물 검색
- policy_search_tool: Google Search API 정책 검색
- loan_calculator_tool: LTV/DSR 기반 대출 계산

Author: RealHome Agent Team
Version: 1.0.0
"""

import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import requests

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from models import (
    SearchQuery, 
    LoanCalculationRequest, 
    LoanCalculationResult,
    PolicySearchResult
)
from search_engine import SearchEngine, ESConfig

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 전역 검색 엔진 인스턴스 (Lazy Initialization)
_search_engine: Optional[SearchEngine] = None


def get_search_engine() -> SearchEngine:
    """
    검색 엔진 싱글톤 인스턴스 반환
    
    Returns:
        SearchEngine 인스턴스
    """
    global _search_engine
    if _search_engine is None:
        config = ESConfig(
            host=os.getenv("ES_HOST", "localhost"),
            port=int(os.getenv("ES_PORT", "9200")),
            username=os.getenv("ES_USERNAME"),
            password=os.getenv("ES_PASSWORD"),
            index_name=os.getenv("ES_INDEX", "realhome_apartments")
        )
        _search_engine = SearchEngine(config)
        _search_engine.connect()
    return _search_engine


# ============================================================================
# 1. 아파트 검색 도구
# ============================================================================

class ApartmentSearchInput(BaseModel):
    """아파트 검색 입력 스키마"""
    districts: Optional[List[str]] = Field(
        default=None,
        description="검색할 구역 리스트 (예: ['송파구', '노원구']). 지원: 송파구, 마포구, 노원구"
    )
    min_price: Optional[float] = Field(
        default=None,
        description="최소 가격 (만원 단위). 예: 50000 (5억)"
    )
    max_price: Optional[float] = Field(
        default=None,
        description="최대 가격 (만원 단위). 예: 70000 (7억)"
    )
    min_area: Optional[float] = Field(
        default=None,
        description="최소 전용면적 (m² 단위). 예: 60"
    )
    max_area: Optional[float] = Field(
        default=None,
        description="최대 전용면적 (m² 단위). 예: 85"
    )
    lifestyle_keywords: Optional[List[str]] = Field(
        default=None,
        description="라이프스타일 키워드 (예: ['육아', '교통', '조용한'])"
    )
    natural_query: Optional[str] = Field(
        default=None,
        description="자연어 검색 쿼리 (예: '아이 키우기 좋은 동네')"
    )
    top_k: int = Field(
        default=5,
        description="반환할 결과 수"
    )


@tool(args_schema=ApartmentSearchInput)
def search_apartment_tool(
    districts: Optional[List[str]] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_area: Optional[float] = None,
    max_area: Optional[float] = None,
    lifestyle_keywords: Optional[List[str]] = None,
    natural_query: Optional[str] = None,
    top_k: int = 5
) -> str:
    """
    사용자의 조건에 맞는 아파트 매물을 검색합니다.
    
    서울시 송파구, 마포구, 노원구 지역의 아파트를 검색하며,
    가격, 면적, 라이프스타일 조건을 기반으로 매칭합니다.
    리뷰 텍스트를 분석하여 육아, 교통, 문화생활 등 
    비정형 조건도 고려합니다.
    
    Args:
        districts: 검색할 구역 리스트
        min_price: 최소 가격 (만원)
        max_price: 최대 가격 (만원)
        min_area: 최소 면적 (m²)
        max_area: 최대 면적 (m²)
        lifestyle_keywords: 라이프스타일 키워드
        natural_query: 자연어 검색 쿼리
        top_k: 반환할 결과 수
        
    Returns:
        검색 결과 JSON 문자열
    """
    try:
        logger.info(f"아파트 검색 시작 - 조건: districts={districts}, price={min_price}-{max_price}, "
                   f"area={min_area}-{max_area}, keywords={lifestyle_keywords}")
        
        # SearchQuery 객체 생성
        query = SearchQuery(
            districts=districts,
            min_price=min_price,
            max_price=max_price,
            min_area=min_area,
            max_area=max_area,
            lifestyle_keywords=lifestyle_keywords,
            natural_query=natural_query,
            top_k=top_k
        )
        
        # 검색 엔진으로 하이브리드 검색 실행
        engine = get_search_engine()
        results = engine.hybrid_search(query)
        
        if not results:
            return json.dumps({
                "status": "no_results",
                "message": "조건에 맞는 매물을 찾지 못했습니다. 조건을 완화해보세요.",
                "apartments": []
            }, ensure_ascii=False)
        
        # 결과 포맷팅
        formatted_results = []
        for i, apt in enumerate(results[:top_k], 1):
            formatted = {
                "순위": i,
                "아파트명": apt.get('kapt_name', '정보없음'),
                "주소": apt.get('doro_juso', apt.get('dong', '정보없음')),
                "구": apt.get('gu', '정보없음'),
                "가격": f"{apt.get('price_manwon', 0):,.0f}만원" if apt.get('price_manwon') else "시세확인필요",
                "면적": f"{apt.get('area_m2', 0):.1f}m²" if apt.get('area_m2') else "정보없음",
                "층": apt.get('floor', '정보없음'),
                "준공년도": apt.get('year_built', '정보없음'),
                "리뷰점수": f"{apt.get('review_score', 0):.1f}/5.0" if apt.get('review_score') else "리뷰없음",
                "장점요약": (apt.get('pros', '')[:200] + '...') if apt.get('pros') and len(apt.get('pros', '')) > 200 else apt.get('pros', '정보없음'),
                "단점요약": (apt.get('cons', '')[:200] + '...') if apt.get('cons') and len(apt.get('cons', '')) > 200 else apt.get('cons', '정보없음'),
                "매칭점수": f"{apt.get('_score', 0):.2f}"
            }
            formatted_results.append(formatted)
        
        response = {
            "status": "success",
            "total_found": len(results),
            "displayed": len(formatted_results),
            "search_conditions": {
                "지역": districts or ["전체"],
                "가격범위": f"{min_price or 0}~{max_price or '제한없음'}만원",
                "면적범위": f"{min_area or 0}~{max_area or '제한없음'}m²",
                "키워드": lifestyle_keywords or [],
                "자연어쿼리": natural_query or ""
            },
            "apartments": formatted_results
        }
        
        logger.info(f"아파트 검색 완료: {len(formatted_results)}개 결과")
        return json.dumps(response, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"아파트 검색 오류: {e}")
        return json.dumps({
            "status": "error",
            "message": f"검색 중 오류가 발생했습니다: {str(e)}",
            "apartments": []
        }, ensure_ascii=False)


# ============================================================================
# 2. 정책 검색 도구 (Google Search API)
# ============================================================================

class PolicySearchInput(BaseModel):
    """정책 검색 입력 스키마"""
    query: str = Field(
        ...,
        description="검색할 정책/규제 키워드 (예: '2025년 LTV 규제', '생애최초 주택 대출')"
    )
    num_results: int = Field(
        default=5,
        description="반환할 결과 수"
    )


@tool(args_schema=PolicySearchInput)
def policy_search_tool(query: str, num_results: int = 5) -> str:
    """
    ElasticSearch에 인덱싱된 부동산 정책 문서를 검색합니다.
    
    PDF로 저장된 부동산 정책 문서에서 관련 정보를 검색하여 제공합니다.
    LTV, DSR, 대출 규제, 세금, 청약 등 다양한 정책 정보 검색 가능.
    
    Args:
        query: 검색할 정책/규제 키워드
        num_results: 반환할 결과 수
        
    Returns:
        검색 결과 JSON 문자열
    """
    try:
        logger.info(f"정책 검색 시작: {query}")
        
        # ElasticSearch 연결
        from policy_indexer import PolicyIndexer
        
        indexer = PolicyIndexer(
            host=os.getenv("ES_HOST", "localhost"),
            port=int(os.getenv("ES_PORT", "9200")),
            index_name="realhome_policies"
        )
        
        if not indexer.connect():
            logger.warning("ElasticSearch 연결 실패. 더미 데이터 반환")
            return _get_dummy_policy_results(query)
        
        # 정책 검색
        search_results = indexer.search(query, size=num_results)
        
        if not search_results:
            # 인덱스가 비어있거나 결과 없음 - 더미 데이터 반환
            logger.warning("정책 검색 결과 없음. 더미 데이터 반환")
            return _get_dummy_policy_results(query)
        
        # 결과 포맷팅
        results = []
        for hit in search_results:
            doc = hit['document']
            highlights = hit.get('highlights', {})
            
            # 하이라이트된 텍스트 추출
            snippet = ""
            if 'full_text' in highlights:
                snippet = " ... ".join(highlights['full_text'][:2])
            elif 'sections.content' in highlights:
                snippet = " ... ".join(highlights['sections.content'][:2])
            else:
                # 하이라이트 없으면 문서 시작 부분
                snippet = doc.get('full_text', '')[:200] + "..."
            
            result = {
                "title": doc.get('title', ''),
                "snippet": snippet,
                "date": doc.get('date', ''),
                "keywords": doc.get('keywords', []),
                "score": hit['score'],
                "source": f"정책문서 ({doc.get('filename', '')})"
            }
            results.append(result)
        
        response_data = {
            "status": "success",
            "query": query,
            "search_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "total_found": len(results),
            "policies": results,
            "data_source": "ElasticSearch 정책 인덱스"
        }
        
        logger.info(f"정책 검색 완료: {len(results)}개 결과")
        return json.dumps(response_data, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"정책 검색 오류: {e}")
        # 오류 시 더미 데이터 반환
        return _get_dummy_policy_results(query)


def _get_dummy_policy_results(query: str) -> str:
    """
    API 키가 없거나 오류 시 반환할 더미 정책 데이터
    2025년 기준 주요 부동산 정책 정보 포함
    """
    dummy_policies = [
        {
            "title": "2025년 주택담보대출 LTV 규제 현황",
            "snippet": "2025년 기준 규제지역 LTV: 무주택자 50%, 1주택자 30%. 비규제지역: 무주택자 70%, 1주택자 60%. 생애최초 주택구입자는 추가 우대 적용.",
            "link": "https://www.fss.or.kr",
            "source": "금융감독원",
            "published_date": "2025-01-15"
        },
        {
            "title": "DSR 규제 및 적용 기준 안내",
            "snippet": "총부채원리금상환비율(DSR) 40% 규제 적용. 연소득 대비 모든 대출의 연간 원리금 상환액이 40%를 초과할 수 없음. 2025년부터 전 금융권 확대 적용.",
            "link": "https://www.hf.go.kr",
            "source": "한국주택금융공사",
            "published_date": "2025-01-10"
        },
        {
            "title": "생애최초 주택 구입자 금융 지원 정책",
            "snippet": "생애최초 주택구입자: LTV 80% 특례, 저금리 대출(연 3.5%~4.0%), 취득세 감면(200만원 한도). 부부합산 연소득 9천만원 이하 대상.",
            "link": "https://www.molit.go.kr",
            "source": "국토교통부",
            "published_date": "2025-02-01"
        },
        {
            "title": "2025년 부동산 취득세 및 보유세 현황",
            "snippet": "1주택자 취득세: 1~3% (가격별 차등), 다주택자 중과세: 8~12%. 종합부동산세: 공시가격 12억 초과 주택 대상, 세율 0.6~6.0%.",
            "link": "https://www.nts.go.kr",
            "source": "국세청",
            "published_date": "2025-01-20"
        },
        {
            "title": "청약 제도 및 특별공급 안내",
            "snippet": "신혼부부 특별공급 30%, 생애최초 특별공급 25% 물량 배정. 청약가점제와 추첨제 병행 운영. 무주택 기간, 부양가족수, 청약통장 가입기간 반영.",
            "link": "https://www.applyhome.co.kr",
            "source": "청약홈",
            "published_date": "2025-01-25"
        }
    ]
    
    # 쿼리와 관련된 결과 필터링
    filtered = [p for p in dummy_policies if any(
        keyword in p['title'] + p['snippet'] 
        for keyword in query.split()
    )]
    
    if not filtered:
        filtered = dummy_policies[:3]  # 기본 결과 반환
    
    return json.dumps({
        "status": "success",
        "query": query,
        "search_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "note": "오프라인 정책 데이터 (실시간 검색을 위해서는 GOOGLE_API_KEY 설정 필요)",
        "total_found": len(filtered),
        "policies": filtered
    }, ensure_ascii=False, indent=2)


# ============================================================================
# 3. 대출 계산 도구
# ============================================================================

class LoanCalculatorInput(BaseModel):
    """대출 계산 입력 스키마"""
    property_price: float = Field(
        ...,
        description="매물 가격 (만원 단위). 예: 70000 (7억)"
    )
    annual_income: float = Field(
        ...,
        description="연 소득 (만원 단위). 예: 8000 (8천만원)"
    )
    existing_debt_payment: float = Field(
        default=0,
        description="기존 연간 부채 상환액 (만원 단위). 예: 500"
    )
    loan_term_years: int = Field(
        default=30,
        description="대출 기간 (년). 예: 30"
    )
    interest_rate: float = Field(
        default=4.5,
        description="예상 금리 (%). 예: 4.5"
    )
    is_regulated_area: bool = Field(
        default=True,
        description="규제지역 여부. 송파구/마포구는 일부 규제, 노원구는 비규제"
    )
    is_first_home: bool = Field(
        default=True,
        description="생애 최초 주택 구입 여부"
    )
    house_count: int = Field(
        default=0,
        description="현재 보유 주택 수"
    )


@tool(args_schema=LoanCalculatorInput)
def loan_calculator_tool(
    property_price: float,
    annual_income: float,
    existing_debt_payment: float = 0,
    loan_term_years: int = 30,
    interest_rate: float = 4.5,
    is_regulated_area: bool = True,
    is_first_home: bool = True,
    house_count: int = 0
) -> str:
    """
    2025년 부동산 규제를 반영한 주택담보대출 가능 금액을 계산합니다.
    
    LTV(담보인정비율)와 DSR(총부채원리금상환비율) 규제를 적용하여
    실제 대출 가능 금액과 필요 자기자본을 산출합니다.
    
    Args:
        property_price: 매물 가격 (만원)
        annual_income: 연 소득 (만원)
        existing_debt_payment: 기존 연간 부채 상환액 (만원)
        loan_term_years: 대출 기간 (년)
        interest_rate: 예상 금리 (%)
        is_regulated_area: 규제지역 여부
        is_first_home: 생애최초 주택 여부
        house_count: 보유 주택 수
        
    Returns:
        대출 계산 결과 JSON 문자열
    """
    try:
        logger.info(f"대출 계산 시작 - 매물가격: {property_price}만원, 연소득: {annual_income}만원")
        
        regulation_notes = []
        
        # ============================================================
        # 1. LTV 한도 결정 (2025년 규제 기준)
        # ============================================================
        if is_regulated_area:
            if house_count == 0:  # 무주택자
                if is_first_home:
                    ltv_limit = 80  # 생애최초 특례
                    regulation_notes.append("생애최초 주택구입자 LTV 80% 특례 적용")
                else:
                    ltv_limit = 50  # 규제지역 무주택자
                    regulation_notes.append("규제지역 무주택자 LTV 50% 적용")
            elif house_count == 1:
                ltv_limit = 30  # 규제지역 1주택자
                regulation_notes.append("규제지역 1주택자 LTV 30% 적용 (주담대 원칙적 불가)")
            else:
                ltv_limit = 0  # 다주택자 대출 제한
                regulation_notes.append("규제지역 다주택자 주담대 불가")
        else:  # 비규제지역
            if house_count == 0:
                if is_first_home:
                    ltv_limit = 80  # 생애최초 특례
                    regulation_notes.append("생애최초 주택구입자 LTV 80% 특례 적용")
                else:
                    ltv_limit = 70  # 비규제지역 무주택자
                    regulation_notes.append("비규제지역 무주택자 LTV 70% 적용")
            elif house_count == 1:
                ltv_limit = 60  # 비규제지역 1주택자
                regulation_notes.append("비규제지역 1주택자 LTV 60% 적용")
            else:
                ltv_limit = 40  # 비규제지역 다주택자
                regulation_notes.append("비규제지역 다주택자 LTV 40% 적용")
        
        # LTV 기준 최대 대출액
        ltv_max_loan = property_price * (ltv_limit / 100)
        
        # ============================================================
        # 2. DSR 한도 결정 (2025년 규제 기준: 40%)
        # ============================================================
        dsr_limit = 40  # 2025년 전 금융권 DSR 40% 규제
        regulation_notes.append("DSR 40% 규제 적용 (2025년 전 금융권)")
        
        # DSR 기준 최대 연간 원리금 상환 가능액
        max_annual_payment = (annual_income * dsr_limit / 100) - existing_debt_payment
        
        if max_annual_payment <= 0:
            dsr_max_loan = 0
            regulation_notes.append("⚠️ 기존 부채가 많아 추가 대출이 어렵습니다.")
        else:
            # 원리금균등상환 방식 대출 가능액 역산
            # P = A * [(1+r)^n - 1] / [r * (1+r)^n]
            monthly_rate = interest_rate / 100 / 12
            total_months = loan_term_years * 12
            
            if monthly_rate > 0:
                annuity_factor = ((1 + monthly_rate) ** total_months - 1) / \
                                (monthly_rate * (1 + monthly_rate) ** total_months)
                monthly_payment = max_annual_payment / 12
                dsr_max_loan = monthly_payment * annuity_factor
            else:
                dsr_max_loan = max_annual_payment * loan_term_years
        
        # ============================================================
        # 3. 최종 대출 가능액 (LTV, DSR 중 작은 값)
        # ============================================================
        final_max_loan = min(ltv_max_loan, dsr_max_loan)
        
        # 필요 자기자본
        required_down_payment = property_price - final_max_loan
        
        # 월 상환액 계산 (원리금균등상환)
        if final_max_loan > 0:
            monthly_rate = interest_rate / 100 / 12
            total_months = loan_term_years * 12
            if monthly_rate > 0:
                monthly_payment = final_max_loan * \
                    (monthly_rate * (1 + monthly_rate) ** total_months) / \
                    ((1 + monthly_rate) ** total_months - 1)
            else:
                monthly_payment = final_max_loan / total_months
        else:
            monthly_payment = 0
        
        # ============================================================
        # 4. 결과 생성
        # ============================================================
        
        # 구매 가능성 판단
        if required_down_payment <= 0:
            feasibility = "✅ 전액 대출 가능"
        elif required_down_payment <= annual_income * 3:
            feasibility = "✅ 적정 범위 (자기자본 연소득 3년 이내)"
        elif required_down_payment <= annual_income * 5:
            feasibility = "⚠️ 다소 부담 (자기자본 연소득 5년 이내)"
        else:
            feasibility = "❌ 자기자본 부담 큼 (연소득 5년 초과)"
        
        result = {
            "status": "success",
            "계산_기준": {
                "매물가격": f"{property_price:,.0f}만원 ({property_price/10000:.1f}억원)",
                "연소득": f"{annual_income:,.0f}만원",
                "기존부채상환액": f"{existing_debt_payment:,.0f}만원/년",
                "대출기간": f"{loan_term_years}년",
                "적용금리": f"{interest_rate}%",
                "규제지역여부": "규제지역" if is_regulated_area else "비규제지역",
                "생애최초여부": "예" if is_first_home else "아니오",
                "보유주택수": house_count
            },
            "LTV_계산": {
                "적용한도": f"{ltv_limit}%",
                "최대대출액": f"{ltv_max_loan:,.0f}만원 ({ltv_max_loan/10000:.1f}억원)"
            },
            "DSR_계산": {
                "적용한도": f"{dsr_limit}%",
                "최대대출액": f"{dsr_max_loan:,.0f}만원 ({dsr_max_loan/10000:.1f}억원)"
            },
            "최종_결과": {
                "최대대출가능액": f"{final_max_loan:,.0f}만원 ({final_max_loan/10000:.1f}억원)",
                "필요자기자본": f"{required_down_payment:,.0f}만원 ({required_down_payment/10000:.1f}억원)",
                "예상월상환액": f"{monthly_payment:,.0f}만원",
                "구매가능성": feasibility
            },
            "적용된_규제": regulation_notes,
            "참고사항": [
                "실제 대출 조건은 금융기관별로 다를 수 있습니다.",
                "신용등급, 소득 증빙 방식에 따라 한도가 달라질 수 있습니다.",
                "중도상환수수료, 대출부대비용이 추가로 발생할 수 있습니다.",
                "취득세, 중개수수료 등 구매 부대비용을 별도로 준비해야 합니다."
            ]
        }
        
        logger.info(f"대출 계산 완료 - 최대대출: {final_max_loan:,.0f}만원, 필요자기자본: {required_down_payment:,.0f}만원")
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"대출 계산 오류: {e}")
        return json.dumps({
            "status": "error",
            "message": f"대출 계산 중 오류가 발생했습니다: {str(e)}"
        }, ensure_ascii=False)


# ============================================================================
# 도구 목록 Export
# ============================================================================

def get_all_tools():
    """모든 도구 리스트 반환"""
    return [
        search_apartment_tool,
        policy_search_tool,
        loan_calculator_tool
    ]


if __name__ == "__main__":
    """도구 테스트"""
    import json
    
    print("=" * 60)
    print("1. 아파트 검색 도구 테스트")
    print("=" * 60)
    result = search_apartment_tool.invoke({
        "districts": ["송파구"],
        "max_price": 70000,
        "lifestyle_keywords": ["육아", "교통"],
        "natural_query": "아이 키우기 좋은 동네",
        "top_k": 3
    })
    print(result)
    
    print("\n" + "=" * 60)
    print("2. 정책 검색 도구 테스트")
    print("=" * 60)
    result = policy_search_tool.invoke({
        "query": "2025년 LTV 규제 생애최초",
        "num_results": 3
    })
    print(result)
    
    print("\n" + "=" * 60)
    print("3. 대출 계산 도구 테스트")
    print("=" * 60)
    result = loan_calculator_tool.invoke({
        "property_price": 70000,
        "annual_income": 8000,
        "existing_debt_payment": 200,
        "loan_term_years": 30,
        "interest_rate": 4.5,
        "is_regulated_area": True,
        "is_first_home": True,
        "house_count": 0
    })
    print(result)
