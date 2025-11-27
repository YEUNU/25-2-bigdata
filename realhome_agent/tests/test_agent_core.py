"""
Agent Core 테스트
=================
agent_core.py의 에이전트 및 쿼리 파서 테스트

실행: pytest tests/test_agent_core.py -v
참고: 일부 테스트는 OPENAI_API_KEY 필요
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# 상위 디렉토리 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_core import (
    QueryParser,
    RealHomeAgent,
    SessionManager,
    quick_chat,
    session_manager,
    SYSTEM_PROMPT,
    REACT_PROMPT_TEMPLATE
)


class TestQueryParser:
    """QueryParser 테스트"""
    
    def test_parse_price_7억대(self):
        """7억대 가격 파싱"""
        result = QueryParser.parse("7억대 아파트 찾아줘")
        
        assert result["min_price"] == 70000
        assert result["max_price"] == 79999
    
    def test_parse_price_이하(self):
        """가격 이하 파싱"""
        result = QueryParser.parse("7억 이하 아파트")
        
        assert result["max_price"] == 70000
        assert result["min_price"] is None
    
    def test_parse_price_이상(self):
        """가격 이상 파싱"""
        result = QueryParser.parse("5억 이상 아파트")
        
        assert result["min_price"] == 50000
    
    def test_parse_area_30평대(self):
        """30평대 면적 파싱"""
        result = QueryParser.parse("30평대 아파트")
        
        assert result["min_area"] == 85
        assert result["max_area"] == 100
    
    def test_parse_area_20평대(self):
        """20평대 면적 파싱"""
        result = QueryParser.parse("20평대 아파트")
        
        assert result["min_area"] == 59
        assert result["max_area"] == 75
    
    def test_parse_district_송파(self):
        """송파구 지역 파싱"""
        result = QueryParser.parse("송파 아파트 추천해줘")
        
        assert result["districts"] == ["송파구"]
    
    def test_parse_district_잠실(self):
        """잠실(송파구) 지역 파싱"""
        result = QueryParser.parse("잠실 아파트")
        
        assert result["districts"] == ["송파구"]
    
    def test_parse_district_마포(self):
        """마포구 지역 파싱"""
        result = QueryParser.parse("마포구 아파트")
        
        assert result["districts"] == ["마포구"]
    
    def test_parse_district_홍대(self):
        """홍대(마포구) 지역 파싱"""
        result = QueryParser.parse("홍대 근처 집")
        
        assert result["districts"] == ["마포구"]
    
    def test_parse_district_노원(self):
        """노원구 지역 파싱"""
        result = QueryParser.parse("노원 아파트")
        
        assert result["districts"] == ["노원구"]
    
    def test_parse_lifestyle_육아(self):
        """육아 키워드 파싱"""
        result = QueryParser.parse("아이 키우기 좋은 아파트")
        
        assert "육아" in result["lifestyle_keywords"]
        assert "교육" in result["lifestyle_keywords"]
    
    def test_parse_lifestyle_교통(self):
        """교통 키워드 파싱"""
        result = QueryParser.parse("출퇴근 편한 아파트")
        
        assert "교통" in result["lifestyle_keywords"]
        assert "역세권" in result["lifestyle_keywords"]
    
    def test_parse_lifestyle_조용(self):
        """조용한 키워드 파싱"""
        result = QueryParser.parse("조용한 동네 아파트")
        
        assert "조용한" in result["lifestyle_keywords"]
    
    def test_parse_lifestyle_신혼(self):
        """신혼부부 키워드 파싱"""
        result = QueryParser.parse("신혼부부 추천 아파트")
        
        assert "문화생활" in result["lifestyle_keywords"]
        assert "쇼핑" in result["lifestyle_keywords"]
    
    def test_parse_complex_query(self):
        """복합 쿼리 파싱"""
        result = QueryParser.parse("7억대 송파 30평대 아이 키우기 좋은 조용한 아파트")
        
        assert result["min_price"] == 70000
        assert result["max_price"] == 79999
        assert result["districts"] == ["송파구"]
        assert result["min_area"] == 85
        assert result["max_area"] == 100
        assert "육아" in result["lifestyle_keywords"]
        assert "조용한" in result["lifestyle_keywords"]
    
    def test_parse_natural_query_preserved(self):
        """자연어 쿼리 원본 보존"""
        original_query = "아이 키우기 좋은 7억대 아파트"
        result = QueryParser.parse(original_query)
        
        assert result["natural_query"] == original_query
    
    def test_parse_empty_query(self):
        """빈 쿼리 파싱"""
        result = QueryParser.parse("")
        
        assert result["districts"] is None
        assert result["min_price"] is None
        assert result["lifestyle_keywords"] == []


class TestSessionManager:
    """SessionManager 테스트"""
    
    def test_get_or_create_new_session(self):
        """새 세션 생성"""
        manager = SessionManager()
        
        with patch('agent_core.RealHomeAgent') as MockAgent:
            MockAgent.return_value = MagicMock()
            agent = manager.get_or_create_session("test_session_1")
            
            assert agent is not None
            MockAgent.assert_called_once()
    
    def test_get_existing_session(self):
        """기존 세션 반환"""
        manager = SessionManager()
        
        with patch('agent_core.RealHomeAgent') as MockAgent:
            mock_agent = MagicMock()
            MockAgent.return_value = mock_agent
            
            agent1 = manager.get_or_create_session("test_session_2")
            agent2 = manager.get_or_create_session("test_session_2")
            
            assert agent1 is agent2
            assert MockAgent.call_count == 1  # 한 번만 생성
    
    def test_delete_session(self):
        """세션 삭제"""
        manager = SessionManager()
        
        with patch('agent_core.RealHomeAgent') as MockAgent:
            MockAgent.return_value = MagicMock()
            manager.get_or_create_session("test_session_3")
            
            result = manager.delete_session("test_session_3")
            assert result is True
            
            result2 = manager.delete_session("nonexistent")
            assert result2 is False
    
    def test_clear_all_sessions(self):
        """모든 세션 삭제"""
        manager = SessionManager()
        
        with patch('agent_core.RealHomeAgent') as MockAgent:
            MockAgent.return_value = MagicMock()
            manager.get_or_create_session("session_a")
            manager.get_or_create_session("session_b")
            
            manager.clear_all_sessions()
            
            # 새로 생성해야 함
            agent = manager.get_or_create_session("session_a")
            assert MockAgent.call_count == 3


class TestRealHomeAgent:
    """RealHomeAgent 테스트 (모킹 사용)"""
    
    @patch('agent_core.ChatOpenAI')
    @patch('agent_core.create_react_agent')
    @patch('agent_core.AgentExecutor')
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_agent_initialization(self, mock_executor, mock_create_agent, mock_llm):
        """에이전트 초기화 테스트"""
        mock_llm.return_value = MagicMock()
        mock_create_agent.return_value = MagicMock()
        mock_executor.return_value = MagicMock()
        
        agent = RealHomeAgent(model_name="gpt-4o-mini", verbose=False)
        
        assert agent.model_name == "gpt-4o-mini"
        mock_llm.assert_called_once()
    
    @patch('agent_core.ChatOpenAI')
    @patch('agent_core.create_react_agent')
    @patch('agent_core.AgentExecutor')
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_chat_success(self, mock_executor, mock_create_agent, mock_llm):
        """채팅 성공 테스트"""
        mock_llm.return_value = MagicMock()
        mock_create_agent.return_value = MagicMock()
        
        mock_executor_instance = MagicMock()
        mock_executor_instance.invoke.return_value = {
            "output": "7억대 송파구 아파트를 추천해드립니다."
        }
        mock_executor.return_value = mock_executor_instance
        
        agent = RealHomeAgent(verbose=False)
        response = agent.chat("7억대 송파구 아파트 추천해줘")
        
        assert "아파트" in response
        mock_executor_instance.invoke.assert_called_once()
    
    @patch('agent_core.ChatOpenAI')
    @patch('agent_core.create_react_agent')
    @patch('agent_core.AgentExecutor')
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_chat_error_handling(self, mock_executor, mock_create_agent, mock_llm):
        """채팅 오류 처리 테스트"""
        mock_llm.return_value = MagicMock()
        mock_create_agent.return_value = MagicMock()
        
        mock_executor_instance = MagicMock()
        mock_executor_instance.invoke.side_effect = Exception("API Error")
        mock_executor.return_value = mock_executor_instance
        
        agent = RealHomeAgent(verbose=False)
        response = agent.chat("테스트 질문")
        
        assert "오류" in response
    
    @patch('agent_core.ChatOpenAI')
    @patch('agent_core.create_react_agent')
    @patch('agent_core.AgentExecutor')
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_clear_memory(self, mock_executor, mock_create_agent, mock_llm):
        """메모리 초기화 테스트"""
        mock_llm.return_value = MagicMock()
        mock_create_agent.return_value = MagicMock()
        mock_executor.return_value = MagicMock()
        
        agent = RealHomeAgent(verbose=False)
        agent.clear_memory()
        
        history = agent.get_chat_history()
        assert len(history) == 0
    
    @patch('agent_core.ChatOpenAI')
    @patch('agent_core.create_react_agent')
    @patch('agent_core.AgentExecutor')
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_get_suggested_questions_initial(self, mock_executor, mock_create_agent, mock_llm):
        """초기 추천 질문 테스트"""
        mock_llm.return_value = MagicMock()
        mock_create_agent.return_value = MagicMock()
        mock_executor.return_value = MagicMock()
        
        agent = RealHomeAgent(verbose=False)
        questions = agent.get_suggested_questions()
        
        assert len(questions) > 0
        assert any("아파트" in q for q in questions)


class TestPrompts:
    """프롬프트 템플릿 테스트"""
    
    def test_system_prompt_contains_regions(self):
        """시스템 프롬프트에 대상 지역 포함"""
        assert "송파구" in SYSTEM_PROMPT
        assert "마포구" in SYSTEM_PROMPT
        assert "노원구" in SYSTEM_PROMPT
    
    def test_system_prompt_contains_features(self):
        """시스템 프롬프트에 기능 설명 포함"""
        assert "매물 검색" in SYSTEM_PROMPT
        assert "라이프스타일" in SYSTEM_PROMPT
        assert "정책" in SYSTEM_PROMPT
        assert "대출" in SYSTEM_PROMPT
    
    def test_system_prompt_contains_price_guide(self):
        """시스템 프롬프트에 가격 가이드 포함"""
        assert "억대" in SYSTEM_PROMPT
        assert "억 이하" in SYSTEM_PROMPT
    
    def test_react_prompt_contains_format(self):
        """ReAct 프롬프트에 형식 포함"""
        assert "Thought" in REACT_PROMPT_TEMPLATE
        assert "Action" in REACT_PROMPT_TEMPLATE
        assert "Observation" in REACT_PROMPT_TEMPLATE
        assert "Final Answer" in REACT_PROMPT_TEMPLATE


class TestQueryParserEdgeCases:
    """QueryParser 엣지 케이스 테스트"""
    
    def test_parse_multiple_prices(self):
        """여러 가격 언급 (첫 번째 것 사용)"""
        result = QueryParser.parse("5억에서 7억대 아파트")
        
        # 첫 번째 매칭된 가격 패턴 사용
        assert result["min_price"] is not None or result["max_price"] is not None
    
    def test_parse_korean_digits(self):
        """한글 숫자 처리"""
        result = QueryParser.parse("칠억대 아파트")
        
        # 현재 구현은 아라비아 숫자만 지원
        # 향후 한글 숫자 지원 시 테스트 업데이트
        assert result["natural_query"] == "칠억대 아파트"
    
    def test_parse_case_insensitive_district(self):
        """대소문자 구분 없이 지역 파싱"""
        result = QueryParser.parse("SONGPA 아파트")
        
        # 한글만 지원하므로 None
        assert result["districts"] is None
    
    def test_parse_with_special_characters(self):
        """특수문자 포함 쿼리"""
        result = QueryParser.parse("7억대 송파구 아파트!! 추천해줘~")
        
        assert result["min_price"] == 70000
        assert result["districts"] == ["송파구"]
    
    def test_parse_반려동물(self):
        """반려동물 키워드 파싱"""
        result = QueryParser.parse("강아지 키우기 좋은 아파트")
        
        assert "공원" in result["lifestyle_keywords"]
        assert "산책로" in result["lifestyle_keywords"]
    
    def test_parse_운동(self):
        """운동 키워드 파싱"""
        result = QueryParser.parse("운동하기 좋은 아파트")
        
        assert "운동" in result["lifestyle_keywords"]


class TestQuickChat:
    """quick_chat 함수 테스트"""
    
    @patch('agent_core.session_manager')
    def test_quick_chat_default_session(self, mock_manager):
        """기본 세션 사용 테스트"""
        mock_agent = MagicMock()
        mock_agent.chat.return_value = "응답입니다"
        mock_manager.get_or_create_session.return_value = mock_agent
        
        response = quick_chat("테스트 메시지")
        
        mock_manager.get_or_create_session.assert_called_with("default")
        assert response == "응답입니다"
    
    @patch('agent_core.session_manager')
    def test_quick_chat_custom_session(self, mock_manager):
        """커스텀 세션 사용 테스트"""
        mock_agent = MagicMock()
        mock_agent.chat.return_value = "응답"
        mock_manager.get_or_create_session.return_value = mock_agent
        
        response = quick_chat("메시지", session_id="custom_session")
        
        mock_manager.get_or_create_session.assert_called_with("custom_session")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
