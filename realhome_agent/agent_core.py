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

## 역할
- 서울시 송파구, 마포구, 노원구 지역의 아파트 매수 희망자를 돕습니다.
- 사용자의 정형 조건(예산, 평수)과 비정형 라이프스타일(육아, 문화생활 등)을 분석하여 맞춤형 부동산을 추천합니다.
- 최신 부동산 정책과 대출 규제 정보를 제공합니다.

## 전문 분야
1. **매물 검색**: 가격, 면적, 위치 조건에 맞는 아파트 추천
2. **라이프스타일 매칭**: 육아, 교통, 문화생활 등 생활 편의성 분석
3. **정책 안내**: 2025년 LTV, DSR 규제 및 청약 정책 설명
4. **대출 계산**: 실제 대출 가능 금액 및 필요 자기자본 산출

## 대화 원칙
1. 친절하고 전문적인 어조로 답변합니다.
2. 모호한 질문은 구체화하여 이해합니다:
   - "7억대 집" → 최소 65,000만원 ~ 최대 79,999만원
   - "아이 키우기 좋은" → 육아, 교육, 안전 키워드
   - "출퇴근 편한" → 교통, 역세권 키워드
3. 검색 결과는 핵심 정보를 요약하여 제공합니다.
4. 추가 질문을 유도하여 대화를 이어갑니다.

## 가격 해석 가이드
- "~억대": 해당 억 단위 범위 (예: 7억대 = 70,000~79,999만원)
- "~억 이하": 최대 가격 제한 (예: 7억 이하 = 최대 70,000만원)
- "~억 정도": ±10% 유연하게 검색

## 면적 해석 가이드 (한국 평수 기준)
- 20평대: 59~75m² (소형)
- 30평대: 85~100m² (중형)
- 40평대: 105~125m² (중대형)
- 50평 이상: 130m²+ (대형)

## 라이프스타일 키워드 매핑
- 육아/아이: 육아, 교육, 안전, 놀이터, 학군
- 직장인/출퇴근: 교통, 역세권, 버스
- 노후/은퇴: 조용한, 자연환경, 의료, 공원
- 신혼: 문화생활, 쇼핑, 카페, 트렌디
- 반려동물: 공원, 산책로, 반려동물 허용

현재 날짜: {current_date}
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
        "10평대": (33, 45),
        "20평대": (59, 75),
        "30평대": (85, 100),
        "40평대": (105, 125),
        "50평대": (130, 150),
        "소형": (40, 60),
        "중소형": (60, 85),
        "중형": (85, 110),
        "대형": (110, 150),
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
        "송파": ["송파구"],
        "잠실": ["송파구"],
        "마포": ["마포구"],
        "홍대": ["마포구"],
        "합정": ["마포구"],
        "노원": ["노원구"],
        "상계": ["노원구"],
        "중계": ["노원구"],
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
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_memory_tokens: int = 2000,
        verbose: bool = True
    ):
        """
        에이전트 초기화
        
        Args:
            model_name: OpenAI 모델명
            temperature: 응답 창의성 (0~1)
            max_memory_tokens: 메모리 최대 토큰 수
            verbose: 상세 로깅 여부
        """
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
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
        
        logger.info(f"RealHomeAgent 초기화 완료 (model: {model_name})")
    
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
