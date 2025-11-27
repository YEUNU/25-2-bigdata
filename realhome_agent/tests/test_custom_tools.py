"""
LangChain 도구(Tools) 테스트
============================
custom_tools.py의 도구 기능 테스트

실행: pytest tests/test_custom_tools.py -v
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import json
import sys
import os

# 상위 디렉토리 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_tools import (
    search_apartment_tool,
    policy_search_tool,
    loan_calculator_tool,
    get_all_tools,
    _get_dummy_policy_results,
    ApartmentSearchInput,
    PolicySearchInput,
    LoanCalculatorInput
)


class TestApartmentSearchTool:
    """아파트 검색 도구 테스트"""
    
    def test_tool_metadata(self):
        """도구 메타데이터 테스트"""
        assert search_apartment_tool.name == "search_apartment_tool"
        assert "아파트" in search_apartment_tool.description
    
    def test_input_schema(self):
        """입력 스키마 테스트"""
        schema = ApartmentSearchInput(
            districts=["송파구"],
            max_price=70000,
            lifestyle_keywords=["육아"]
        )
        
        assert schema.districts == ["송파구"]
        assert schema.max_price == 70000
        assert schema.top_k == 5  # 기본값
    
    @patch('custom_tools.get_search_engine')
    def test_search_with_results(self, mock_get_engine):
        """검색 결과 반환 테스트"""
        # 모킹 설정
        mock_engine = MagicMock()
        mock_engine.hybrid_search.return_value = [
            {
                'kapt_name': '테스트아파트',
                'doro_juso': '서울시 송파구',
                'gu': '송파구',
                'price_manwon': 70000,
                'area_m2': 84.5,
                'floor': 10,
                'year_built': 2015,
                'review_score': 4.5,
                'pros': '교통 편리',
                'cons': '관리비 비쌈',
                '_score': 0.95
            }
        ]
        mock_get_engine.return_value = mock_engine
        
        result = search_apartment_tool.invoke({
            "districts": ["송파구"],
            "max_price": 80000,
            "top_k": 3
        })
        
        result_dict = json.loads(result)
        assert result_dict["status"] == "success"
        assert len(result_dict["apartments"]) == 1
        assert result_dict["apartments"][0]["아파트명"] == "테스트아파트"
    
    @patch('custom_tools.get_search_engine')
    def test_search_no_results(self, mock_get_engine):
        """검색 결과 없음 테스트"""
        mock_engine = MagicMock()
        mock_engine.hybrid_search.return_value = []
        mock_get_engine.return_value = mock_engine
        
        result = search_apartment_tool.invoke({
            "districts": ["송파구"],
            "max_price": 10000
        })
        
        result_dict = json.loads(result)
        assert result_dict["status"] == "no_results"
        assert "조건에 맞는 매물을 찾지 못했습니다" in result_dict["message"]
    
    @patch('custom_tools.get_search_engine')
    def test_search_with_lifestyle_keywords(self, mock_get_engine):
        """라이프스타일 키워드 검색 테스트"""
        mock_engine = MagicMock()
        mock_engine.hybrid_search.return_value = []
        mock_get_engine.return_value = mock_engine
        
        result = search_apartment_tool.invoke({
            "lifestyle_keywords": ["육아", "교통", "학군"],
            "natural_query": "아이 키우기 좋은 동네"
        })
        
        # hybrid_search 호출 확인
        mock_engine.hybrid_search.assert_called_once()
        call_args = mock_engine.hybrid_search.call_args[0][0]
        assert "육아" in call_args.lifestyle_keywords


class TestPolicySearchTool:
    """정책 검색 도구 테스트"""
    
    def test_tool_metadata(self):
        """도구 메타데이터 테스트"""
        assert policy_search_tool.name == "policy_search_tool"
        assert "정책" in policy_search_tool.description or "Google" in policy_search_tool.description
    
    def test_input_schema(self):
        """입력 스키마 테스트"""
        schema = PolicySearchInput(
            query="2025년 LTV 규제",
            num_results=5
        )
        
        assert schema.query == "2025년 LTV 규제"
        assert schema.num_results == 5
    
    def test_dummy_policy_results_ltv(self):
        """더미 정책 결과 (LTV 관련) 테스트"""
        result = _get_dummy_policy_results("LTV 규제")
        result_dict = json.loads(result)
        
        assert result_dict["status"] == "success"
        assert len(result_dict["policies"]) > 0
        
        # LTV 관련 정책 포함 확인
        policy_texts = " ".join([p["title"] + p["snippet"] for p in result_dict["policies"]])
        assert "LTV" in policy_texts
    
    def test_dummy_policy_results_dsr(self):
        """더미 정책 결과 (DSR 관련) 테스트"""
        result = _get_dummy_policy_results("DSR 한도")
        result_dict = json.loads(result)
        
        assert result_dict["status"] == "success"
        policy_texts = " ".join([p["title"] + p["snippet"] for p in result_dict["policies"]])
        assert "DSR" in policy_texts
    
    def test_dummy_policy_results_first_home(self):
        """더미 정책 결과 (생애최초) 테스트"""
        result = _get_dummy_policy_results("생애최초 주택")
        result_dict = json.loads(result)
        
        assert result_dict["status"] == "success"
        policy_texts = " ".join([p["title"] + p["snippet"] for p in result_dict["policies"]])
        assert "생애최초" in policy_texts
    
    @patch.dict(os.environ, {}, clear=True)
    def test_policy_search_without_api_key(self):
        """API 키 없이 정책 검색 테스트"""
        result = policy_search_tool.invoke({
            "query": "2025년 LTV",
            "num_results": 3
        })
        
        result_dict = json.loads(result)
        assert result_dict["status"] == "success"
        assert "오프라인 정책 데이터" in result_dict.get("note", "")


class TestLoanCalculatorTool:
    """대출 계산 도구 테스트"""
    
    def test_tool_metadata(self):
        """도구 메타데이터 테스트"""
        assert loan_calculator_tool.name == "loan_calculator_tool"
        assert "대출" in loan_calculator_tool.description
    
    def test_input_schema(self):
        """입력 스키마 테스트"""
        schema = LoanCalculatorInput(
            property_price=70000,
            annual_income=8000,
            is_first_home=True
        )
        
        assert schema.property_price == 70000
        assert schema.annual_income == 8000
        assert schema.is_first_home is True
    
    def test_first_home_regulated_area(self):
        """생애최초 + 규제지역 대출 계산"""
        result = loan_calculator_tool.invoke({
            "property_price": 70000,
            "annual_income": 8000,
            "is_regulated_area": True,
            "is_first_home": True,
            "house_count": 0
        })
        
        result_dict = json.loads(result)
        
        assert result_dict["status"] == "success"
        assert result_dict["LTV_계산"]["적용한도"] == "80%"  # 생애최초 특례
        assert result_dict["DSR_계산"]["적용한도"] == "40%"
        assert "생애최초" in " ".join(result_dict["적용된_규제"])
    
    def test_non_first_home_regulated_area(self):
        """비생애최초 + 규제지역 무주택자"""
        result = loan_calculator_tool.invoke({
            "property_price": 70000,
            "annual_income": 8000,
            "is_regulated_area": True,
            "is_first_home": False,
            "house_count": 0
        })
        
        result_dict = json.loads(result)
        
        assert result_dict["status"] == "success"
        assert result_dict["LTV_계산"]["적용한도"] == "50%"  # 규제지역 무주택자
    
    def test_one_house_owner_regulated(self):
        """1주택자 + 규제지역"""
        result = loan_calculator_tool.invoke({
            "property_price": 70000,
            "annual_income": 8000,
            "is_regulated_area": True,
            "is_first_home": False,
            "house_count": 1
        })
        
        result_dict = json.loads(result)
        
        assert result_dict["status"] == "success"
        assert result_dict["LTV_계산"]["적용한도"] == "30%"  # 1주택자
    
    def test_multi_house_owner_regulated(self):
        """다주택자 + 규제지역"""
        result = loan_calculator_tool.invoke({
            "property_price": 70000,
            "annual_income": 8000,
            "is_regulated_area": True,
            "is_first_home": False,
            "house_count": 2
        })
        
        result_dict = json.loads(result)
        
        assert result_dict["status"] == "success"
        assert result_dict["LTV_계산"]["적용한도"] == "0%"  # 다주택자 대출 불가
    
    def test_non_regulated_area(self):
        """비규제지역 테스트"""
        result = loan_calculator_tool.invoke({
            "property_price": 50000,
            "annual_income": 6000,
            "is_regulated_area": False,
            "is_first_home": False,
            "house_count": 0
        })
        
        result_dict = json.loads(result)
        
        assert result_dict["status"] == "success"
        assert result_dict["LTV_계산"]["적용한도"] == "70%"  # 비규제지역 무주택자
    
    def test_high_existing_debt(self):
        """기존 부채 높은 경우"""
        result = loan_calculator_tool.invoke({
            "property_price": 70000,
            "annual_income": 5000,
            "existing_debt_payment": 2500,  # 연소득 50% 이미 부채 상환
            "is_regulated_area": True,
            "is_first_home": True,
            "house_count": 0
        })
        
        result_dict = json.loads(result)
        
        assert result_dict["status"] == "success"
        # DSR 기준 대출 제한적
        dsr_max = float(result_dict["DSR_계산"]["최대대출액"].replace(",", "").replace("만원", "").split("(")[0])
        assert dsr_max < 50000  # 제한적인 금액
    
    def test_monthly_payment_calculation(self):
        """월 상환액 계산 검증"""
        result = loan_calculator_tool.invoke({
            "property_price": 70000,
            "annual_income": 10000,
            "loan_term_years": 30,
            "interest_rate": 4.5,
            "is_regulated_area": True,
            "is_first_home": True,
            "house_count": 0
        })
        
        result_dict = json.loads(result)
        
        # 월 상환액이 계산되었는지 확인
        monthly_payment_str = result_dict["최종_결과"]["예상월상환액"]
        assert "만원" in monthly_payment_str
    
    def test_feasibility_assessment(self):
        """구매 가능성 평가 테스트"""
        # 적정 범위 케이스
        result1 = loan_calculator_tool.invoke({
            "property_price": 30000,
            "annual_income": 8000,
            "is_regulated_area": True,
            "is_first_home": True,
            "house_count": 0
        })
        result1_dict = json.loads(result1)
        assert "✅" in result1_dict["최종_결과"]["구매가능성"]
        
        # 부담 큰 케이스
        result2 = loan_calculator_tool.invoke({
            "property_price": 100000,
            "annual_income": 5000,
            "is_regulated_area": True,
            "is_first_home": True,
            "house_count": 0
        })
        result2_dict = json.loads(result2)
        # 자기자본 부담이 크면 경고
        feasibility = result2_dict["최종_결과"]["구매가능성"]
        assert "⚠️" in feasibility or "❌" in feasibility


class TestToolsList:
    """도구 목록 테스트"""
    
    def test_get_all_tools(self):
        """모든 도구 리스트 반환 테스트"""
        tools = get_all_tools()
        
        assert len(tools) == 3
        tool_names = [t.name for t in tools]
        
        assert "search_apartment_tool" in tool_names
        assert "policy_search_tool" in tool_names
        assert "loan_calculator_tool" in tool_names


class TestEdgeCases:
    """엣지 케이스 테스트"""
    
    def test_loan_zero_interest_rate(self):
        """금리 0% 케이스"""
        result = loan_calculator_tool.invoke({
            "property_price": 50000,
            "annual_income": 8000,
            "interest_rate": 0.0,
            "is_regulated_area": False,
            "is_first_home": True,
            "house_count": 0
        })
        
        result_dict = json.loads(result)
        assert result_dict["status"] == "success"
    
    @patch('custom_tools.get_search_engine')
    def test_search_exception_handling(self, mock_get_engine):
        """검색 예외 처리 테스트"""
        mock_engine = MagicMock()
        mock_engine.hybrid_search.side_effect = Exception("Search error")
        mock_get_engine.return_value = mock_engine
        
        result = search_apartment_tool.invoke({
            "districts": ["송파구"]
        })
        
        result_dict = json.loads(result)
        assert result_dict["status"] == "error"
        assert "오류" in result_dict["message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
