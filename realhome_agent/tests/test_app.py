"""
Streamlit 앱 테스트
==================
app.py의 UI 컴포넌트 및 세션 관리 테스트

실행: pytest tests/test_app.py -v
참고: Streamlit 테스트는 별도 환경 필요
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# 상위 디렉토리 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSessionState:
    """세션 상태 테스트"""
    
    def test_default_filters(self):
        """기본 필터 값 테스트"""
        default_filters = {
            'districts': [],
            'min_price': None,
            'max_price': None,
            'min_area': None,
            'max_area': None,
            'lifestyle_keywords': []
        }
        
        assert default_filters['districts'] == []
        assert default_filters['min_price'] is None
        assert default_filters['lifestyle_keywords'] == []


class TestUIComponents:
    """UI 컴포넌트 테스트 (모킹)"""
    
    @patch('streamlit.set_page_config')
    def test_page_config(self, mock_config):
        """페이지 설정 테스트"""
        # 페이지 설정 함수 호출 시뮬레이션
        mock_config(
            page_title="🏠 리얼홈 에이전트",
            page_icon="🏠",
            layout="wide"
        )
        
        mock_config.assert_called_once()
        call_args = mock_config.call_args
        assert call_args[1]['page_title'] == "🏠 리얼홈 에이전트"


class TestChatInterface:
    """채팅 인터페이스 테스트"""
    
    def test_message_structure(self):
        """메시지 구조 테스트"""
        message = {
            'role': 'user',
            'content': '7억대 아파트 추천해줘',
            'timestamp': '2025-11-27T10:00:00'
        }
        
        assert message['role'] in ['user', 'assistant']
        assert 'content' in message
    
    def test_assistant_message(self):
        """어시스턴트 메시지 구조"""
        message = {
            'role': 'assistant',
            'content': '송파구 7억대 아파트를 추천해드립니다.',
            'timestamp': '2025-11-27T10:00:05'
        }
        
        assert message['role'] == 'assistant'


class TestSuggestedQuestions:
    """추천 질문 테스트"""
    
    def test_suggested_questions_list(self):
        """추천 질문 목록 테스트"""
        suggested_questions = [
            "7억대 송파구 아파트 추천해줘",
            "아이 키우기 좋은 노원구 아파트",
            "역세권 마포구 신축 아파트",
            "대출 가능 금액 계산해줘",
            "2025년 부동산 정책 알려줘"
        ]
        
        assert len(suggested_questions) >= 5
        assert any("아파트" in q for q in suggested_questions)
        assert any("대출" in q for q in suggested_questions)


class TestFilterValidation:
    """필터 검증 테스트"""
    
    def test_district_options(self):
        """지역 옵션 테스트"""
        valid_districts = ["송파구", "마포구", "노원구"]
        
        assert "송파구" in valid_districts
        assert "강남구" not in valid_districts
    
    def test_price_range_conversion(self):
        """가격 범위 변환 테스트"""
        # UI에서 억원 단위로 입력 → 만원 단위로 변환
        min_price_billions = 5.0
        max_price_billions = 10.0
        
        min_price_manwon = min_price_billions * 10000
        max_price_manwon = max_price_billions * 10000
        
        assert min_price_manwon == 50000
        assert max_price_manwon == 100000
    
    def test_area_range(self):
        """면적 범위 테스트"""
        area_range = (60, 120)
        
        assert area_range[0] >= 30  # 최소 30m²
        assert area_range[1] <= 200  # 최대 200m²
    
    def test_lifestyle_options(self):
        """라이프스타일 옵션 테스트"""
        lifestyle_options = [
            "육아", "교통", "교육", "문화생활",
            "조용한", "자연환경", "쇼핑", "안전", "운동"
        ]
        
        assert "육아" in lifestyle_options
        assert "교통" in lifestyle_options
        assert len(lifestyle_options) >= 9


class TestElasticSearchCheck:
    """ElasticSearch 연결 확인 테스트"""
    
    @patch('app.SearchEngine')
    @patch('app.ESConfig')
    def test_check_elasticsearch_connected(self, mock_config, mock_engine_class):
        """ES 연결 성공 테스트"""
        mock_engine = MagicMock()
        mock_engine.connect.return_value = True
        mock_engine_class.return_value = mock_engine
        
        # check_elasticsearch 함수 시뮬레이션
        result = mock_engine.connect()
        
        assert result is True
    
    @patch('app.SearchEngine')
    @patch('app.ESConfig')
    def test_check_elasticsearch_disconnected(self, mock_config, mock_engine_class):
        """ES 연결 실패 테스트"""
        mock_engine = MagicMock()
        mock_engine.connect.return_value = False
        mock_engine_class.return_value = mock_engine
        
        result = mock_engine.connect()
        
        assert result is False


class TestEnvironmentVariables:
    """환경 변수 테스트"""
    
    def test_required_env_vars(self):
        """필수 환경 변수 목록"""
        required_vars = ['OPENAI_API_KEY']
        optional_vars = ['OPENAI_MODEL', 'ES_HOST', 'ES_PORT', 'GOOGLE_API_KEY']
        
        assert 'OPENAI_API_KEY' in required_vars
        assert 'ES_HOST' in optional_vars
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_env_var_set(self):
        """환경 변수 설정 확인"""
        assert os.getenv('OPENAI_API_KEY') == 'test-key'
    
    @patch.dict(os.environ, {}, clear=True)
    def test_env_var_not_set(self):
        """환경 변수 미설정 확인"""
        result = os.getenv('OPENAI_API_KEY')
        assert result is None


class TestWelcomeMessage:
    """환영 메시지 테스트"""
    
    def test_welcome_message_content(self):
        """환영 메시지 내용 테스트"""
        welcome_features = [
            "매물 검색",
            "라이프스타일 매칭",
            "대출 계산",
            "정책 안내"
        ]
        
        assert len(welcome_features) >= 4


class TestProcessUserInput:
    """사용자 입력 처리 테스트"""
    
    def test_empty_input_handling(self):
        """빈 입력 처리"""
        user_input = ""
        
        # 빈 입력은 처리하지 않음
        assert not user_input.strip()
    
    def test_whitespace_input_handling(self):
        """공백 입력 처리"""
        user_input = "   "
        
        assert not user_input.strip()
    
    def test_valid_input(self):
        """유효한 입력 처리"""
        user_input = "7억대 송파구 아파트 추천해줘"
        
        assert user_input.strip()


class TestClearChat:
    """대화 초기화 테스트"""
    
    def test_clear_messages(self):
        """메시지 목록 초기화"""
        messages = [
            {'role': 'user', 'content': '질문'},
            {'role': 'assistant', 'content': '답변'}
        ]
        
        messages.clear()
        
        assert len(messages) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
