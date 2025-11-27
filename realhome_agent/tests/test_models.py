"""
Pydantic 모델 테스트
====================
models.py의 데이터 검증 로직 테스트

실행: pytest tests/test_models.py -v
"""

import pytest
from datetime import date
import sys
import os

# 상위 디렉토리 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    ApartmentSchema,
    SearchQuery,
    DealRecord,
    ReviewData,
    LoanCalculationRequest,
    LoanCalculationResult,
    PolicySearchResult,
    ConversationMessage,
    District,
    LifestyleKeyword
)


class TestApartmentSchema:
    """ApartmentSchema 테스트"""
    
    def test_valid_apartment(self):
        """유효한 아파트 데이터 생성"""
        apt = ApartmentSchema(
            kapt_code="A12345",
            kapt_name="잠실엘스",
            gu="송파구",
            dong="잠실동",
            doro_juso="서울특별시 송파구 올림픽로 212",
            price_manwon=340000,
            area_m2=84.8,
            floor=15,
            year_built=2008,
            review_score=4.5,
            pros="교통 편리, 학군 좋음",
            cons="관리비 비쌈"
        )
        
        assert apt.kapt_code == "A12345"
        assert apt.kapt_name == "잠실엘스"
        assert apt.gu == "송파구"
        assert apt.price_manwon == 340000
    
    def test_invalid_district(self):
        """지원하지 않는 구역 검증"""
        with pytest.raises(ValueError, match="지원하지 않는 구역"):
            ApartmentSchema(
                kapt_code="A12345",
                kapt_name="테스트아파트",
                gu="강남구"  # 지원하지 않는 구역
            )
    
    def test_valid_districts(self):
        """지원하는 모든 구역 테스트"""
        for district in ["송파구", "마포구", "노원구"]:
            apt = ApartmentSchema(
                kapt_code="A12345",
                kapt_name="테스트아파트",
                gu=district
            )
            assert apt.gu == district
    
    def test_area_validation(self):
        """면적 범위 검증"""
        # 너무 작은 면적
        with pytest.raises(ValueError, match="면적은 5~500"):
            ApartmentSchema(
                kapt_code="A12345",
                kapt_name="테스트",
                gu="송파구",
                area_m2=3
            )
        
        # 너무 큰 면적
        with pytest.raises(ValueError, match="면적은 5~500"):
            ApartmentSchema(
                kapt_code="A12345",
                kapt_name="테스트",
                gu="송파구",
                area_m2=600
            )
    
    def test_optional_fields(self):
        """선택 필드 None 허용"""
        apt = ApartmentSchema(
            kapt_code="A12345",
            kapt_name="테스트아파트",
            gu="송파구"
        )
        
        assert apt.doro_juso is None
        assert apt.price_manwon is None
        assert apt.area_m2 is None
        assert apt.embedding is None


class TestSearchQuery:
    """SearchQuery 테스트"""
    
    def test_basic_query(self):
        """기본 검색 쿼리 생성"""
        query = SearchQuery(
            districts=["송파구"],
            max_price=70000,
            lifestyle_keywords=["육아", "교통"]
        )
        
        assert query.districts == ["송파구"]
        assert query.max_price == 70000
        assert "육아" in query.lifestyle_keywords
    
    def test_price_range_validation(self):
        """가격 범위 유효성 검증"""
        # 최대가격 < 최소가격
        with pytest.raises(ValueError, match="최대 가격은 최소 가격보다"):
            SearchQuery(
                min_price=80000,
                max_price=50000
            )
    
    def test_area_range_validation(self):
        """면적 범위 유효성 검증"""
        with pytest.raises(ValueError, match="최대 면적은 최소 면적보다"):
            SearchQuery(
                min_area=100,
                max_area=50
            )
    
    def test_default_values(self):
        """기본값 테스트"""
        query = SearchQuery()
        
        assert query.top_k == 10
        assert query.hybrid_weight == 0.5
        assert query.districts is None
        assert query.lifestyle_keywords is None
    
    def test_top_k_bounds(self):
        """top_k 범위 테스트"""
        # 유효 범위
        query = SearchQuery(top_k=50)
        assert query.top_k == 50
        
        # 범위 초과
        with pytest.raises(ValueError):
            SearchQuery(top_k=100)
    
    def test_hybrid_weight_bounds(self):
        """hybrid_weight 범위 테스트"""
        # 유효 범위
        query1 = SearchQuery(hybrid_weight=0.0)
        query2 = SearchQuery(hybrid_weight=1.0)
        assert query1.hybrid_weight == 0.0
        assert query2.hybrid_weight == 1.0
        
        # 범위 초과
        with pytest.raises(ValueError):
            SearchQuery(hybrid_weight=1.5)
    
    def test_natural_query(self):
        """자연어 쿼리 테스트"""
        query = SearchQuery(
            natural_query="아이 키우기 좋은 7억대 아파트"
        )
        assert query.natural_query == "아이 키우기 좋은 7억대 아파트"


class TestLoanCalculationRequest:
    """LoanCalculationRequest 테스트"""
    
    def test_valid_request(self):
        """유효한 대출 계산 요청"""
        request = LoanCalculationRequest(
            property_price=70000,
            annual_income=8000,
            existing_debt_payment=200,
            loan_term_years=30,
            interest_rate=4.5,
            is_regulated_area=True,
            is_first_home=True,
            house_count=0
        )
        
        assert request.property_price == 70000
        assert request.annual_income == 8000
        assert request.is_first_home is True
    
    def test_default_values(self):
        """기본값 테스트"""
        request = LoanCalculationRequest(
            property_price=70000,
            annual_income=8000
        )
        
        assert request.existing_debt_payment == 0
        assert request.loan_term_years == 30
        assert request.interest_rate == 4.5
        assert request.is_regulated_area is True
        assert request.is_first_home is True
    
    def test_invalid_property_price(self):
        """잘못된 매물 가격"""
        with pytest.raises(ValueError):
            LoanCalculationRequest(
                property_price=0,  # 0 이하 불가
                annual_income=8000
            )
    
    def test_loan_term_bounds(self):
        """대출 기간 범위 테스트"""
        # 유효 범위
        request = LoanCalculationRequest(
            property_price=70000,
            annual_income=8000,
            loan_term_years=40
        )
        assert request.loan_term_years == 40
        
        # 범위 초과
        with pytest.raises(ValueError):
            LoanCalculationRequest(
                property_price=70000,
                annual_income=8000,
                loan_term_years=50
            )


class TestDealRecord:
    """DealRecord 테스트"""
    
    def test_valid_deal_record(self):
        """유효한 거래 기록"""
        deal = DealRecord(
            gu="송파구",
            dong="잠실동",
            apt_name="잠실엘스",
            deal_date=date(2025, 10, 15),
            area_m2=84.8,
            floor=15,
            year_built=2008,
            price_manwon=340000,
            price_krw=3400000000
        )
        
        assert deal.gu == "송파구"
        assert deal.apt_name == "잠실엘스"
        assert deal.price_manwon == 340000


class TestReviewData:
    """ReviewData 테스트"""
    
    def test_valid_review(self):
        """유효한 리뷰 데이터"""
        review = ReviewData(
            kapt_name="잠실엘스",
            doro_juso="서울특별시 송파구 올림픽로 212",
            score=4.5,
            pros="교통 편리, 학군 좋음",
            cons="관리비 비쌈"
        )
        
        assert review.kapt_name == "잠실엘스"
        assert review.score == 4.5
    
    def test_score_bounds(self):
        """점수 범위 테스트"""
        # 유효 범위
        review = ReviewData(
            kapt_name="테스트",
            score=5.0,
            pros="좋음",
            cons="없음"
        )
        assert review.score == 5.0
        
        # 범위 초과
        with pytest.raises(ValueError):
            ReviewData(
                kapt_name="테스트",
                score=6.0,
                pros="좋음",
                cons="없음"
            )


class TestConversationMessage:
    """ConversationMessage 테스트"""
    
    def test_user_message(self):
        """사용자 메시지"""
        msg = ConversationMessage(
            role="user",
            content="7억대 아파트 추천해줘"
        )
        assert msg.role == "user"
    
    def test_assistant_message(self):
        """어시스턴트 메시지"""
        msg = ConversationMessage(
            role="assistant",
            content="추천 결과입니다..."
        )
        assert msg.role == "assistant"
    
    def test_invalid_role(self):
        """잘못된 역할"""
        with pytest.raises(ValueError):
            ConversationMessage(
                role="invalid",
                content="테스트"
            )


class TestEnums:
    """Enum 테스트"""
    
    def test_district_enum(self):
        """District Enum 테스트"""
        assert District.SONGPA.value == "송파구"
        assert District.MAPO.value == "마포구"
        assert District.NOWON.value == "노원구"
    
    def test_lifestyle_keyword_enum(self):
        """LifestyleKeyword Enum 테스트"""
        assert LifestyleKeyword.CHILDCARE.value == "육아"
        assert LifestyleKeyword.TRANSPORTATION.value == "교통"
        assert LifestyleKeyword.CULTURE.value == "문화생활"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
