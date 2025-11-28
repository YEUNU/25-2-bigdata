"""
Streamlit 기반 리얼홈 에이전트 UI
=================================
사용자 친화적인 챗봇 인터페이스 제공

Author: RealHome Agent Team
Version: 1.0.0
"""

import os
import sys
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st
from streamlit_chat import message

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_core import RealHomeAgent, session_manager
from models import SearchQuery

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 페이지 설정
# ============================================================================

st.set_page_config(
    page_title="🏠 리얼홈 에이전트",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    /* 전체 스타일 */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* 채팅 메시지 스타일 */
    .user-message {
        background-color: #E3F2FD;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 5px 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    
    .assistant-message {
        background-color: #F5F5F5;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 5px 0;
        max-width: 80%;
        float: left;
        clear: both;
    }
    
    /* 사이드바 스타일 */
    .sidebar-section {
        background-color: #FAFAFA;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        padding: 0.5rem 1rem;
    }
    
    /* 추천 질문 스타일 */
    .suggestion-btn {
        background-color: #E8F5E9;
        border: 1px solid #81C784;
        border-radius: 20px;
        padding: 8px 16px;
        margin: 5px;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .suggestion-btn:hover {
        background-color: #C8E6C9;
    }
    
    /* 결과 카드 스타일 */
    .result-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* 로딩 스피너 */
    .loading {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 세션 상태 초기화
# ============================================================================

def init_session_state():
    """세션 상태 초기화"""
    
    # 세션 ID 생성
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # 대화 기록
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # 에이전트 인스턴스
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    
    # 검색 필터 상태
    if 'filters' not in st.session_state:
        st.session_state.filters = {
            'districts': [],
            'min_price': None,
            'max_price': None,
            'min_area': None,
            'max_area': None,
            'lifestyle_keywords': []
        }
    
    # 로딩 상태
    if 'is_loading' not in st.session_state:
        st.session_state.is_loading = False


def get_agent() -> RealHomeAgent:
    """에이전트 인스턴스 반환"""
    if st.session_state.agent is None:
        try:
            st.session_state.agent = RealHomeAgent(
                model_name=os.getenv("OPENAI_MODEL", "gpt-5-mini-2025-08-07"),
                temperature=0.3,
                verbose=False
            )
            logger.info("에이전트 초기화 완료")
        except Exception as e:
            logger.error(f"에이전트 초기화 실패: {e}")
            st.error(f"에이전트 초기화 실패: {e}")
            return None
    return st.session_state.agent


# ============================================================================
# 사이드바 컴포넌트
# ============================================================================

def render_sidebar():
    """사이드바 렌더링"""
    
    with st.sidebar:
        st.markdown("### 🏠 리얼홈 에이전트")
        st.markdown("서울시 아파트 맞춤 추천 서비스")
        
        st.markdown("---")
        
        # 검색 필터 섹션
        st.markdown("#### 🔍 검색 필터")
        
        # 지역 선택
        districts = st.multiselect(
            "지역 선택",
            options=["송파구", "마포구", "노원구"],
            default=st.session_state.filters.get('districts', []),
            help="원하는 지역을 선택하세요"
        )
        st.session_state.filters['districts'] = districts
        
        # 가격 범위
        st.markdown("##### 💰 가격 범위 (억원)")
        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input(
                "최소",
                min_value=0.0,
                max_value=50.0,
                value=0.0,
                step=0.5,
                key="min_price_input"
            )
        with col2:
            max_price = st.number_input(
                "최대",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=0.5,
                key="max_price_input"
            )
        st.session_state.filters['min_price'] = min_price * 10000 if min_price > 0 else None
        st.session_state.filters['max_price'] = max_price * 10000 if max_price > 0 else None
        
        # 면적 범위
        st.markdown("##### 📐 면적 범위 (m²)")
        area_range = st.slider(
            "전용면적",
            min_value=30,
            max_value=200,
            value=(60, 120),
            step=5,
            key="area_range_slider"
        )
        st.session_state.filters['min_area'] = area_range[0]
        st.session_state.filters['max_area'] = area_range[1]
        
        # 라이프스타일 키워드
        st.markdown("##### 🎯 라이프스타일")
        lifestyle_options = ["육아", "교통", "교육", "문화생활", "조용한", "자연환경", "쇼핑", "안전", "운동"]
        lifestyle = st.multiselect(
            "관심 키워드",
            options=lifestyle_options,
            default=st.session_state.filters.get('lifestyle_keywords', []),
            help="원하는 라이프스타일을 선택하세요"
        )
        st.session_state.filters['lifestyle_keywords'] = lifestyle
        
        # 필터 적용 버튼
        if st.button("🔍 필터로 검색", use_container_width=True):
            apply_filter_search()
        
        st.markdown("---")
        
        # 추천 질문
        st.markdown("#### 💡 추천 질문")
        
        suggested_questions = [
            "7억대 송파구 아파트 추천해줘",
            "아이 키우기 좋은 노원구 아파트",
            "역세권 마포구 신축 아파트",
            "대출 가능 금액 계산해줘",
            "2025년 부동산 정책 알려줘"
        ]
        
        for question in suggested_questions:
            if st.button(f"💬 {question}", key=f"q_{question}", use_container_width=True):
                process_user_input(question)
        
        st.markdown("---")
        
        # 세션 관리
        st.markdown("#### ⚙️ 설정")
        
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            clear_chat()
        
        # API 키 상태
        api_key_status = "✅ 설정됨" if os.getenv("OPENAI_API_KEY") else "❌ 미설정"
        st.markdown(f"**OpenAI API**: {api_key_status}")
        
        es_status = "✅ 연결됨" if check_elasticsearch() else "⚠️ 미연결"
        st.markdown(f"**ElasticSearch**: {es_status}")


def apply_filter_search():
    """필터 기반 검색 실행"""
    filters = st.session_state.filters
    
    # 검색 쿼리 생성
    query_parts = []
    
    if filters['districts']:
        query_parts.append(f"{', '.join(filters['districts'])} 지역")
    
    if filters['max_price']:
        price_billions = filters['max_price'] / 10000
        query_parts.append(f"{price_billions:.1f}억 이하")
    
    if filters['min_area'] and filters['max_area']:
        query_parts.append(f"{filters['min_area']}~{filters['max_area']}m²")
    
    if filters['lifestyle_keywords']:
        query_parts.append(f"{', '.join(filters['lifestyle_keywords'])} 조건")
    
    if query_parts:
        search_query = f"{'의 '.join(query_parts)} 아파트 추천해줘"
        process_user_input(search_query)
    else:
        st.warning("최소 하나의 필터 조건을 선택해주세요.")


def check_elasticsearch() -> bool:
    """ElasticSearch 연결 상태 확인"""
    try:
        from search_engine import SearchEngine, ESConfig
        config = ESConfig(
            host=os.getenv("ES_HOST", "localhost"),
            port=int(os.getenv("ES_PORT", "9200"))
        )
        engine = SearchEngine(config)
        return engine.connect()
    except:
        return False


# ============================================================================
# 채팅 인터페이스
# ============================================================================

def render_chat_interface():
    """채팅 인터페이스 렌더링"""
    
    # 헤더
    st.markdown('<h1 class="main-header">🏠 리얼홈 에이전트</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">서울시 송파구, 마포구, 노원구의 맞춤형 아파트를 추천해드립니다</p>', unsafe_allow_html=True)
    
    # 채팅 컨테이너
    chat_container = st.container()
    
    with chat_container:
        # 대화 기록 표시
        for i, msg in enumerate(st.session_state.messages):
            if msg['role'] == 'user':
                with st.chat_message("user", avatar="👤"):
                    st.markdown(msg['content'])
            else:
                with st.chat_message("assistant", avatar="🏠"):
                    st.markdown(msg['content'])
    
    # 입력 영역
    st.markdown("---")
    
    # 환영 메시지 (첫 방문시)
    if not st.session_state.messages:
        render_welcome_message()
    
    # 사용자 입력
    user_input = st.chat_input(
        "무엇이든 물어보세요! (예: 7억대 송파구 아파트 추천해줘)",
        key="user_input"
    )
    
    if user_input:
        process_user_input(user_input)


def render_welcome_message():
    """환영 메시지 렌더링"""
    
    st.markdown("""
    ### 👋 안녕하세요! 리얼홈 에이전트입니다.
    
    서울시 **송파구, 마포구, 노원구** 지역의 아파트 매물을 추천해드립니다.
    
    #### 🎯 이런 것들을 도와드릴 수 있어요:
    
    | 기능 | 예시 질문 |
    |------|----------|
    | 🏢 **매물 검색** | "7억대 송파구 30평대 아파트 추천해줘" |
    | 👶 **라이프스타일 매칭** | "아이 키우기 좋은 조용한 동네 찾아줘" |
    | 💰 **대출 계산** | "연봉 8천만원인데 대출 얼마나 받을 수 있어?" |
    | 📋 **정책 안내** | "2025년 LTV 규제가 어떻게 되나요?" |
    
    ---
    
    **💡 Tip**: 왼쪽 사이드바에서 필터를 설정하면 더 정확한 검색이 가능해요!
    """)


def process_user_input(user_input: str):
    """사용자 입력 처리"""
    
    if not user_input.strip():
        return
    
    # 사용자 메시지 추가
    st.session_state.messages.append({
        'role': 'user',
        'content': user_input,
        'timestamp': datetime.now().isoformat()
    })
    
    # 에이전트 응답 생성
    with st.spinner("🔍 검색 중..."):
        try:
            agent = get_agent()
            if agent:
                response = agent.chat(user_input)
            else:
                response = "죄송합니다. 에이전트 초기화에 실패했습니다. OPENAI_API_KEY가 설정되어 있는지 확인해주세요."
        except Exception as e:
            logger.error(f"응답 생성 오류: {e}")
            response = f"죄송합니다. 오류가 발생했습니다: {str(e)}"
    
    # 어시스턴트 메시지 추가
    st.session_state.messages.append({
        'role': 'assistant',
        'content': response,
        'timestamp': datetime.now().isoformat()
    })
    
    # 페이지 새로고침
    st.rerun()


def clear_chat():
    """대화 초기화"""
    st.session_state.messages = []
    if st.session_state.agent:
        st.session_state.agent.clear_memory()
    st.rerun()


# ============================================================================
# 결과 표시 컴포넌트
# ============================================================================

def render_apartment_cards(apartments: List[Dict]):
    """아파트 결과 카드 렌더링"""
    
    if not apartments:
        st.info("조건에 맞는 매물이 없습니다.")
        return
    
    for apt in apartments:
        with st.expander(f"🏢 {apt.get('아파트명', '정보없음')} - {apt.get('가격', '시세확인필요')}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**📍 주소**: {apt.get('주소', '정보없음')}")
                st.markdown(f"**💰 가격**: {apt.get('가격', '시세확인필요')}")
                st.markdown(f"**📐 면적**: {apt.get('면적', '정보없음')}")
            
            with col2:
                st.markdown(f"**🏗️ 층수**: {apt.get('층', '정보없음')}")
                st.markdown(f"**📅 준공년도**: {apt.get('준공년도', '정보없음')}")
                st.markdown(f"**⭐ 리뷰점수**: {apt.get('리뷰점수', '리뷰없음')}")
            
            st.markdown("---")
            st.markdown(f"**👍 장점**: {apt.get('장점요약', '정보없음')}")
            st.markdown(f"**👎 단점**: {apt.get('단점요약', '정보없음')}")


def render_loan_result(loan_info: Dict):
    """대출 계산 결과 렌더링"""
    
    st.markdown("### 💰 대출 계산 결과")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "최대 대출 가능액",
            loan_info.get("최대대출가능액", "0만원"),
            help="LTV, DSR 규제를 모두 반영한 금액"
        )
    
    with col2:
        st.metric(
            "필요 자기자본",
            loan_info.get("필요자기자본", "0만원"),
            help="매물 가격 - 최대 대출액"
        )
    
    with col3:
        st.metric(
            "예상 월 상환액",
            loan_info.get("예상월상환액", "0만원"),
            help="원리금균등상환 기준"
        )
    
    st.markdown(f"**📊 구매 가능성**: {loan_info.get('구매가능성', '분석 필요')}")


# ============================================================================
# 메인 함수
# ============================================================================

def main():
    """메인 실행 함수"""
    
    # 세션 상태 초기화
    init_session_state()
    
    # 환경변수 확인
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("""
        ⚠️ **OPENAI_API_KEY가 설정되지 않았습니다.**
        
        에이전트 기능을 사용하려면 환경변수를 설정해주세요:
        ```bash
        export OPENAI_API_KEY="your-api-key"
        ```
        
        또는 `.env` 파일에 추가해주세요.
        """)
    
    # 사이드바 렌더링
    render_sidebar()
    
    # 채팅 인터페이스 렌더링
    render_chat_interface()
    
    # 푸터
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888;'>"
        "🏠 리얼홈 에이전트 | 서울시 아파트 맞춤 추천 서비스 | "
        "© 2025 RealHome Agent Team"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
