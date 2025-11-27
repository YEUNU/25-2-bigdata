"""
통합 테스트
===========
전체 시스템 흐름 테스트

실행: pytest tests/test_integration.py -v
참고: ES 서버 및 OPENAI_API_KEY 필요
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import json
import sys
import os

# 상위 디렉토리 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEndToEndFlow:
    """End-to-End 흐름 테스트"""
    
    @patch('custom_tools.get_search_engine')
    @patch('agent_core.ChatOpenAI')
    @patch('agent_core.create_react_agent')
    @patch('agent_core.AgentExecutor')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_apartment_search_flow(self, mock_executor, mock_create_agent, 
                                    mock_llm, mock_get_engine):
        """아파트 검색 전체 흐름"""
        # 1. 검색 엔진 모킹
        mock_engine = MagicMock()
        mock_engine.hybrid_search.return_value = [
            {
                'kapt_name': '잠실엘스',
                'doro_juso': '서울시 송파구 잠실동',
                'gu': '송파구',
                'price_manwon': 340000,
                'area_m2': 84.8,
                'floor': 15,
                'year_built': 2008,
                'review_score': 4.5,
                'pros': '교통 편리, 학군 좋음',
                'cons': '관리비 비쌈',
                '_score': 0.95
            }
        ]
        mock_get_engine.return_value = mock_engine
        
        # 2. 에이전트 모킹
        mock_llm.return_value = MagicMock()
        mock_create_agent.return_value = MagicMock()
        
        mock_executor_instance = MagicMock()
        mock_executor_instance.invoke.return_value = {
            "output": "송파구 7억대 아파트를 검색했습니다. 잠실엘스를 추천드립니다."
        }
        mock_executor.return_value = mock_executor_instance
        
        # 3. 에이전트 생성 및 채팅
        from agent_core import RealHomeAgent
        agent = RealHomeAgent(verbose=False)
        
        response = agent.chat("7억대 송파구 아파트 추천해줘")
        
        assert "아파트" in response or "추천" in response
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_loan_calculation_flow(self):
        """대출 계산 전체 흐름"""
        from custom_tools import loan_calculator_tool
        
        # 1. 대출 계산 도구 호출
        result = loan_calculator_tool.invoke({
            "property_price": 70000,
            "annual_income": 8000,
            "is_regulated_area": True,
            "is_first_home": True,
            "house_count": 0
        })
        
        result_dict = json.loads(result)
        
        # 2. 결과 검증
        assert result_dict["status"] == "success"
        assert "LTV_계산" in result_dict
        assert "DSR_계산" in result_dict
        assert "최종_결과" in result_dict
        
        # 3. 생애최초 LTV 80% 적용 확인
        assert result_dict["LTV_계산"]["적용한도"] == "80%"
    
    @patch.dict(os.environ, {}, clear=True)
    def test_policy_search_flow_without_api(self):
        """API 키 없이 정책 검색 흐름"""
        from custom_tools import policy_search_tool
        
        result = policy_search_tool.invoke({
            "query": "2025년 LTV 규제",
            "num_results": 3
        })
        
        result_dict = json.loads(result)
        
        # 오프라인 더미 데이터 반환
        assert result_dict["status"] == "success"
        assert len(result_dict["policies"]) > 0


class TestQueryToSearchFlow:
    """쿼리 → 검색 흐름 테스트"""
    
    def test_natural_query_parsing(self):
        """자연어 쿼리 파싱"""
        from agent_core import QueryParser
        from models import SearchQuery
        
        # 1. 사용자 쿼리 파싱
        parsed = QueryParser.parse("7억대 송파구 30평대 아이 키우기 좋은 아파트")
        
        # 2. SearchQuery 생성
        query = SearchQuery(
            districts=parsed["districts"],
            min_price=parsed["min_price"],
            max_price=parsed["max_price"],
            min_area=parsed["min_area"],
            max_area=parsed["max_area"],
            lifestyle_keywords=parsed["lifestyle_keywords"],
            natural_query=parsed["natural_query"]
        )
        
        # 3. 검증
        assert query.districts == ["송파구"]
        assert query.min_price == 70000
        assert query.max_price == 79999
        assert "육아" in query.lifestyle_keywords


class TestMultiTurnConversation:
    """멀티턴 대화 테스트"""
    
    @patch('agent_core.ChatOpenAI')
    @patch('agent_core.create_react_agent')
    @patch('agent_core.AgentExecutor')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_conversation_memory(self, mock_executor, mock_create_agent, mock_llm):
        """대화 메모리 유지 테스트"""
        mock_llm.return_value = MagicMock()
        mock_create_agent.return_value = MagicMock()
        
        responses = [
            {"output": "송파구 아파트를 추천해드립니다."},
            {"output": "네, 더 저렴한 매물을 찾아봤습니다."},
        ]
        mock_executor_instance = MagicMock()
        mock_executor_instance.invoke.side_effect = responses
        mock_executor.return_value = mock_executor_instance
        
        from agent_core import RealHomeAgent
        agent = RealHomeAgent(verbose=False)
        
        # 첫 번째 질문
        response1 = agent.chat("7억대 송파구 아파트 추천해줘")
        
        # 두 번째 질문 (컨텍스트 유지)
        response2 = agent.chat("더 저렴한 곳은 없어?")
        
        # 대화 기록 확인
        history = agent.get_chat_history()
        assert len(history) >= 2


class TestToolIntegration:
    """도구 통합 테스트"""
    
    def test_all_tools_available(self):
        """모든 도구 사용 가능 확인"""
        from custom_tools import get_all_tools
        
        tools = get_all_tools()
        tool_names = [t.name for t in tools]
        
        assert "search_apartment_tool" in tool_names
        assert "policy_search_tool" in tool_names
        assert "loan_calculator_tool" in tool_names
    
    def test_tool_descriptions(self):
        """도구 설명 확인"""
        from custom_tools import get_all_tools
        
        tools = get_all_tools()
        
        for tool in tools:
            assert tool.description is not None
            assert len(tool.description) > 10


class TestDataModelIntegration:
    """데이터 모델 통합 테스트"""
    
    def test_apartment_to_search_result(self):
        """아파트 모델 → 검색 결과 변환"""
        from models import ApartmentSchema
        
        apt = ApartmentSchema(
            kapt_code="A12345",
            kapt_name="잠실엘스",
            gu="송파구",
            dong="잠실동",
            doro_juso="서울시 송파구 잠실동 올림픽로 212",
            price_manwon=340000,
            area_m2=84.8,
            review_score=4.5,
            pros="교통 편리",
            cons="관리비 비쌈"
        )
        
        # 검색 결과 포맷
        result = {
            "아파트명": apt.kapt_name,
            "주소": apt.doro_juso,
            "가격": f"{apt.price_manwon:,.0f}만원",
            "면적": f"{apt.area_m2}m²"
        }
        
        assert result["아파트명"] == "잠실엘스"
        assert "340,000만원" in result["가격"]
    
    def test_loan_request_to_result(self):
        """대출 요청 → 결과 변환"""
        from models import LoanCalculationRequest
        from custom_tools import loan_calculator_tool
        import json
        
        request = LoanCalculationRequest(
            property_price=70000,
            annual_income=8000,
            is_first_home=True
        )
        
        result = loan_calculator_tool.invoke({
            "property_price": request.property_price,
            "annual_income": request.annual_income,
            "is_first_home": request.is_first_home
        })
        
        result_dict = json.loads(result)
        assert result_dict["status"] == "success"


class TestErrorHandling:
    """오류 처리 통합 테스트"""
    
    @patch('custom_tools.get_search_engine')
    def test_search_engine_error(self, mock_get_engine):
        """검색 엔진 오류 처리"""
        mock_engine = MagicMock()
        mock_engine.hybrid_search.side_effect = Exception("Connection failed")
        mock_get_engine.return_value = mock_engine
        
        from custom_tools import search_apartment_tool
        
        result = search_apartment_tool.invoke({
            "districts": ["송파구"]
        })
        
        result_dict = json.loads(result)
        assert result_dict["status"] == "error"
    
    @patch('agent_core.ChatOpenAI')
    @patch('agent_core.create_react_agent')
    @patch('agent_core.AgentExecutor')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_agent_error_recovery(self, mock_executor, mock_create_agent, mock_llm):
        """에이전트 오류 복구"""
        mock_llm.return_value = MagicMock()
        mock_create_agent.return_value = MagicMock()
        
        mock_executor_instance = MagicMock()
        mock_executor_instance.invoke.side_effect = Exception("OpenAI API Error")
        mock_executor.return_value = mock_executor_instance
        
        from agent_core import RealHomeAgent
        agent = RealHomeAgent(verbose=False)
        
        response = agent.chat("테스트 질문")
        
        # 오류 메시지 반환
        assert "오류" in response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
