"""
설정 관리 모듈
=============
환경 변수 및 애플리케이션 설정

Author: RealHome Agent Team
Version: 1.0.0
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # 애플리케이션 정보
    APP_NAME: str = "리얼홈 에이전트"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # OpenAI 설정
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.3
    
    # ElasticSearch 설정
    ES_HOST: str = "elasticsearch"
    ES_PORT: int = 9200
    ES_USERNAME: Optional[str] = None
    ES_PASSWORD: Optional[str] = None
    ES_INDEX: str = "realhome_apartments"
    
    # Google Search API 설정
    GOOGLE_API_KEY: Optional[str] = None
    GOOGLE_SEARCH_ENGINE_ID: Optional[str] = None
    
    # 임베딩 모델 설정
    EMBEDDING_MODEL: str = "google/embeddinggemma-300m"
    EMBEDDING_DEVICE: str = "cuda"  # cuda 또는 cpu
    
    # Streamlit 설정
    STREAMLIT_PORT: int = 8501
    
    # 로깅 설정
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """설정 싱글톤 인스턴스 반환"""
    return Settings()


# 편의 함수
def get_openai_api_key() -> str:
    """OpenAI API 키 반환"""
    return get_settings().OPENAI_API_KEY


def get_es_config() -> dict:
    """ElasticSearch 설정 반환"""
    settings = get_settings()
    return {
        "host": settings.ES_HOST,
        "port": settings.ES_PORT,
        "username": settings.ES_USERNAME,
        "password": settings.ES_PASSWORD,
        "index_name": settings.ES_INDEX
    }


def is_debug_mode() -> bool:
    """디버그 모드 여부"""
    return get_settings().DEBUG


if __name__ == "__main__":
    # 설정 테스트
    settings = get_settings()
    print(f"App: {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"Debug: {settings.DEBUG}")
    print(f"ES Host: {settings.ES_HOST}:{settings.ES_PORT}")
    print(f"OpenAI Model: {settings.OPENAI_MODEL}")
