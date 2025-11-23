# agentic_rag.py
# 부동산 의사결정 지원 시스템용 Agentic RAG 파이프라인 예시
#
# 1단계: 사용자의 자연어 질의를 구조화하는 "의도 분석 체인"
# 2단계: 분석 결과를 바탕으로 ES 하이브리드 검색 + 구글 검색을 수행하는 Agent
# 3단계: 수집한 정보를 기반으로 최종 의사결정 리포트를 생성하는 체인

from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain.agents import (
    AgentExecutor,
    create_tool_calling_agent,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# =========================
# 0. 공통 LLM 설정
# =========================
def get_llm() -> ChatOpenAI:
    """
    공통으로 사용할 LLM 설정.
    필요에 따라 모델명, temperature, API Key(.env) 등을 수정.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # 또는 gpt-4o 등
        temperature=0.2,
    )
    return llm


llm = get_llm()


# =========================
# 1. 의도 분석 체인
# =========================


INTENT_TEMPLATE = """
당신은 부동산 의사결정 지원 시스템의 '질문 분석기'입니다.
사용자의 발화를 읽고, 집 추천 / 대출 정책 / 동네 비교 등에 필요한 정보를
구조화된 JSON 형식으로 추출하세요.

[사용자 질문]
{question}

[출력 형식(JSON)]
{{
  "question_type": "집추천 | 대출정책질문 | 동네비교 | 기타",
  "budget": "숫자만 또는 '모름' (예: '700000000', '모름')",
  "deal_type": "매매 | 전세 | 월세 | 모름",
  "preferred_areas": "콤마로 구분된 구/동 이름들 (예: '강동구, 마포구')",
  "commute_place": "주요 출근지 (예: '강남역', '여의도역') 또는 '모름'",
  "household": "1인/2인/3인 이상 등 가구 형태 또는 '모름'",
  "lifestyle": "학군, 육아, 조용한 동네, 문화생활, 자연환경 등 핵심 키워드 요약",
  "risk_preference": "안정형 | 중립 | 공격형 | 모름",
  "extra_constraints": "입주 시점, 대출 여부, 투자/실거주 등 추가 제약사항 요약"
}}

주의:
- 반드시 위 JSON 형식을 그대로 사용하고, 한국어 설명 문장은 쓰지 마세요.
- 값이 불명확하면 '모름'으로 채우세요.
"""

intent_prompt = PromptTemplate(
    template=INTENT_TEMPLATE,
    input_variables=["question"],
)

intent_chain = LLMChain(
    llm=llm,
    prompt=intent_prompt,
    output_key="intent_json",
)


# =========================
# 2. 도구 정의 (ES + Google Search)
# =========================


@tool
def es_hybrid_search(query: str) -> str:
    """
    ElasticSearch 하이브리드 검색(실거래가, 호가, 리뷰, 정책 문서 등)에 사용되는 도구.

    Args:
        query: '지역, 예산, 평형, 라이프스타일' 등이 반영된 검색 질의 문자열

    Returns:
        검색 결과를 요약한 문자열 (실제 구현에서는 상위 N개 매물/단지/문서 요약)
    """
    # TODO:
    #  - 실제 환경에서는 requests 또는 Elasticsearch Python client를 사용하여
    #    ES 인덱스(실거래가, 호가, 정책문서, 리뷰 등)에 하이브리드 검색 수행
    #  - BM25 + dense vector (knn) 결과를 RRF 등으로 합산해 상위 문서 요약 반환
    #  - 아래는 데모용 mock 응답
    return (
        f"[ES-HYBRID-RESULT]\n"
        f"쿼리: {query}\n"
        f"- 실거래가/호가 상위 매물 5건 요약\n"
        f"- 단지 특성(연식, 학군, 역세권, 생활편의) 요약\n"
        f"- 관련 정책/리뷰 문서 요약\n"
    )


@tool
def google_policy_search(query: str) -> str:
    """
    구글(또는 SerpAPI, DuckDuckGo)을 이용해 최신 대출/주택 정책, 뉴스 등을 검색하는 도구.

    Args:
        query: 정책/규제/금리/뉴스 위주의 검색 질의

    Returns:
        정책/뉴스 요약 문자열
    """
    # TODO:
    #  - 실제 환경에서는 serpapi/duckduckgo-search/wikipedia 등 LangChain 래퍼 활용
    #  - 예: DuckDuckGoSearchAPIWrapper().run(query)
    #  - 아래는 데모용 mock 응답
    return (
        f"[GOOGLE-POLICY-RESULT]\n"
        f"쿼리: {query}\n"
        f"- 최근 6개월 주요 주택 대출 규제/완화 이슈 요약\n"
        f"- 해당 지역(또는 수도권) 공급/정책 관련 기사 핵심 포인트\n"
    )


tools = [es_hybrid_search, google_policy_search]


# =========================
# 3. Agentic RAG 에이전트 (도구 사용 단계)
# =========================

RAG_SYSTEM_PROMPT = """
당신은 부동산 의사결정 Agent입니다.
다음 도구를 활용하여 사용자의 질문에 답하기 위한 '분석 메모'를 작성하는 것이 목표입니다.

[사용 가능한 도구]
- es_hybrid_search: 실거래가/호가/리뷰/정책 문서가 들어 있는 ElasticSearch 하이브리드 검색
- google_policy_search: 최신 대출/주택 정책, 뉴스, 규제 동향 검색

[입력 정보]
- 원본 사용자 질문
- 질문에 대한 구조화된 분석 JSON(intent_json)

[역할]
1. intent_json을 해석하여 어떤 정보가 필요한지 판단합니다.
2. 필요에 따라 es_hybrid_search, google_policy_search를 여러 번 호출해 충분한 정보를 수집합니다.
3. 수집된 정보를 바탕으로 '분석 메모'를 bullet 형태로 정리합니다.
   (아직 최종 리포트가 아니라, 리포트 작성을 위한 재료를 정리하는 단계입니다.)

[출력 형식]
- 제목: "분석 메모"
- 하위에 bullet 리스트로 정리
- 정량 정보(예산, 대출 가능 범위, 통근 시간, 가격 수준)와
  정성 정보(학군, 소음, 생활편의, 육아 환경, 정책 리스크)를 함께 포함
"""

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        (
            "human",
            "원본 질문:\n{question}\n\n"
            "질문 분석(JSON):\n{intent_json}\n"
            "위 정보를 바탕으로 필요한 도구를 호출해 분석 메모를 만들어 주세요.",
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

rag_llm = get_llm()
rag_agent_core = create_tool_calling_agent(rag_llm, tools, rag_prompt)

rag_executor = AgentExecutor(
    agent=rag_agent_core,
    tools=tools,
    verbose=True,
    max_iterations=8,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
)


def run_rag_agent(question: str, intent_json: str) -> Dict[str, Any]:
    """
    AgentExecutor를 한 번 감싼 helper 함수.
    - 반환값: {"output": "...", "intermediate_steps": [...]} 형태의 딕셔너리
    """
    result = rag_executor.invoke({"question": question, "intent_json": intent_json})
    return result


# =========================
# 4. 최종 리포트 생성 체인
# =========================


REPORT_TEMPLATE = """
당신은 '부동산 의사결정 리포트 생성기'입니다.
다음 정보를 바탕으로 사용자의 의사결정에 바로 활용 가능한 리포트를 작성하세요.

[1] 원본 질문
{question}

[2] 질문 분석(JSON)
{intent_json}

[3] 검색/분석 메모
{research_notes}

[리포트 작성 지침]
- 전체를 한국어로 작성합니다.
- 마크다운 없이, 워드에 바로 붙여넣을 수 있는 순수 텍스트 형식으로 작성합니다.
- 구조:
  1. 요약 (3~5줄)
     - 사용자의 상황, 주요 제약조건, 추천 방향을 한눈에 볼 수 있도록 정리
  2. 정량 분석
     - 예산, 예상 대출 가능 범위, 통근 시간, 해당 지역 가격 수준 등을 bullet로 정리
  3. 정성 분석
     - 학군, 육아 환경, 소음/쾌적성, 생활편의시설, 치안 등을 항목별로 비교/평가
  4. 정책 및 리스크
     - 현재 대출/세제/공급 정책 요약
     - 향후 정책/금리 변화에 따른 리스크 및 시나리오
  5. 추천 결론
     - 후보 지역/단지 A, B, C가 있다면 각각의 장단점을 요약
     - '이 사용자의 상황이라면 ○○을 우선 추천'처럼 명시적 결론 제시

리포트의 각 섹션에는 번호와 소제목을 붙여 주세요.
"""

report_prompt = PromptTemplate(
    template=REPORT_TEMPLATE,
    input_variables=["question", "intent_json", "research_notes"],
)

report_chain = LLMChain(
    llm=llm,
    prompt=report_prompt,
    output_key="report",
)


# =========================
# 5. 전체 Agentic RAG 파이프라인 함수
# =========================
def run_agentic_rag_pipeline(question: str) -> str:
    """
    프로젝트 기획서의 Agentic RAG 3단계
      1) 의도 분석 → 2) 도구 사용(ES + Google) → 3) 리포트 생성
    을 한 번에 실행하는 편의 함수.
    """
    # 1) 의도 분석
    intent_json: str = intent_chain.run(question=question)

    # 2) Agentic RAG (도구 호출 단계)
    rag_result = run_rag_agent(question, intent_json)
    research_notes: str = rag_result["output"]

    # 3) 최종 리포트 생성
    report: str = report_chain.run(
        question=question,
        intent_json=intent_json,
        research_notes=research_notes,
    )

    return report


# =========================
# 6. 간단한 실행 예시
# =========================
if __name__ == "__main__":
    sample_question = (
        "강남역으로 출근하는데 예산 7억 정도고, "
        "아이 초등학교 입학을 앞두고 있어서 학군이랑 육아 환경이 중요해요. "
        "강동구랑 마포구 중에 어디가 더 나을까요? 대출 규제도 같이 봐줘."
    )

    final_report = run_agentic_rag_pipeline(sample_question)
    print(final_report)
