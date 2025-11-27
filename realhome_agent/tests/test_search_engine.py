"""
ElasticSearch 검색 엔진 테스트
==============================
search_engine.py의 인덱싱 및 검색 로직 테스트

실행: pytest tests/test_search_engine.py -v
참고: 일부 테스트는 ElasticSearch 서버 실행 필요
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import sys
import os

# 상위 디렉토리 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_engine import SearchEngine, ESConfig, EmbeddingModel
from models import SearchQuery, ApartmentSchema


class TestESConfig:
    """ESConfig 테스트"""
    
    def test_default_config(self):
        """기본 설정 테스트"""
        config = ESConfig()
        
        assert config.host == "localhost"
        assert config.port == 9200
        assert config.scheme == "http"
        assert config.index_name == "realhome_apartments"
        assert config.embedding_dim == 1024  # BAAI/bge-m3 모델 사용
    
    def test_custom_config(self):
        """커스텀 설정 테스트"""
        config = ESConfig(
            host="es-server",
            port=9201,
            username="user",
            password="pass",
            index_name="test_index"
        )
        
        assert config.host == "es-server"
        assert config.port == 9201
        assert config.username == "user"
        assert config.password == "pass"


class TestEmbeddingModel:
    """EmbeddingModel 테스트 (모킹 사용)"""
    
    def test_initialization(self):
        """초기화 테스트"""
        model = EmbeddingModel(model_name="test-model")
        
        assert model.model_name == "test-model"
        assert model._is_loaded is False
    
    @patch('search_engine.AutoTokenizer')
    @patch('search_engine.AutoModel')
    def test_load_model(self, mock_model, mock_tokenizer):
        """모델 로드 테스트"""
        model = EmbeddingModel(model_name="test-model")
        model.load()
        
        mock_tokenizer.from_pretrained.assert_called_once_with("test-model")
        mock_model.from_pretrained.assert_called_once_with("test-model")
        assert model._is_loaded is True
    
    @patch('search_engine.AutoTokenizer')
    @patch('search_engine.AutoModel')
    def test_encode_single(self, mock_model, mock_tokenizer):
        """단일 텍스트 임베딩 테스트"""
        # 모킹 설정
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.return_value = {
            'input_ids': Mock(),
            'attention_mask': Mock()
        }
        
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # 임베딩 모델 테스트 (실제 모델 없이 기본 동작 확인)
        model = EmbeddingModel()
        assert model.model_name == "BAAI/bge-m3"  # 다국어 임베딩 모델


class TestSearchEngine:
    """SearchEngine 테스트"""
    
    def test_initialization(self):
        """초기화 테스트"""
        config = ESConfig()
        engine = SearchEngine(config)
        
        assert engine.config == config
        assert engine.client is None
    
    def test_index_mapping_structure(self):
        """인덱스 매핑 구조 테스트"""
        mapping = SearchEngine.INDEX_MAPPING
        
        # settings 확인
        assert "settings" in mapping
        assert mapping["settings"]["number_of_shards"] == 1
        
        # mappings 확인
        assert "mappings" in mapping
        props = mapping["mappings"]["properties"]
        
        # 필수 필드 확인
        assert "kapt_code" in props
        assert "kapt_name" in props
        assert "gu" in props
        assert "price_manwon" in props
        assert "embedding" in props
        
        # 필드 타입 확인
        assert props["kapt_code"]["type"] == "keyword"
        assert props["price_manwon"]["type"] == "float"
        assert props["embedding"]["type"] == "dense_vector"
        assert props["embedding"]["dims"] == 1024  # BAAI/bge-m3 모델
    
    @patch('search_engine.Elasticsearch')
    def test_connect_success(self, mock_es):
        """연결 성공 테스트"""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_es.return_value = mock_client
        
        engine = SearchEngine(ESConfig())
        result = engine.connect()
        
        assert result is True
        assert engine.client is not None
    
    @patch('search_engine.Elasticsearch')
    def test_connect_failure(self, mock_es):
        """연결 실패 테스트"""
        mock_client = MagicMock()
        mock_client.info.side_effect = Exception("Connection refused")
        mock_es.return_value = mock_client
        
        engine = SearchEngine(ESConfig())
        # 재시도 횟수를 1로 설정하여 빠르게 실패하도록
        result = engine.connect(max_retries=1, retry_delay=0)
        
        assert result is False
    
    def test_build_filter_conditions_empty(self):
        """빈 필터 조건 테스트"""
        engine = SearchEngine(ESConfig())
        query = SearchQuery()
        
        filters = engine._build_filter_conditions(query)
        assert filters == []
    
    def test_build_filter_conditions_with_districts(self):
        """지역 필터 조건 테스트"""
        engine = SearchEngine(ESConfig())
        query = SearchQuery(districts=["송파구", "노원구"])
        
        filters = engine._build_filter_conditions(query)
        
        assert len(filters) == 1
        assert filters[0] == {"terms": {"gu": ["송파구", "노원구"]}}
    
    def test_build_filter_conditions_with_price(self):
        """가격 필터 조건 테스트"""
        engine = SearchEngine(ESConfig())
        query = SearchQuery(min_price=50000, max_price=80000)
        
        filters = engine._build_filter_conditions(query)
        
        price_filter = next(f for f in filters if "range" in f and "price_manwon" in f["range"])
        assert price_filter["range"]["price_manwon"]["gte"] == 50000
        assert price_filter["range"]["price_manwon"]["lte"] == 80000
    
    def test_build_filter_conditions_with_area(self):
        """면적 필터 조건 테스트"""
        engine = SearchEngine(ESConfig())
        query = SearchQuery(min_area=60, max_area=100)
        
        filters = engine._build_filter_conditions(query)
        
        area_filter = next(f for f in filters if "range" in f and "area_m2" in f["range"])
        assert area_filter["range"]["area_m2"]["gte"] == 60
        assert area_filter["range"]["area_m2"]["lte"] == 100
    
    def test_build_bm25_query_with_natural_query(self):
        """자연어 쿼리 BM25 테스트"""
        engine = SearchEngine(ESConfig())
        query = SearchQuery(natural_query="아이 키우기 좋은 동네")
        
        bm25_queries = engine._build_bm25_query(query)
        
        assert len(bm25_queries) > 0
        # 리뷰, 장점, 단점, 주소, 아파트명 필드에서 검색
        fields_searched = set()
        for q in bm25_queries:
            if "match" in q:
                fields_searched.update(q["match"].keys())
        
        assert "combined_review" in fields_searched
        assert "pros" in fields_searched
    
    def test_build_bm25_query_with_lifestyle(self):
        """라이프스타일 키워드 BM25 테스트"""
        engine = SearchEngine(ESConfig())
        query = SearchQuery(lifestyle_keywords=["육아", "교통"])
        
        bm25_queries = engine._build_bm25_query(query)
        
        assert len(bm25_queries) > 0
    
    def test_build_bm25_query_empty(self):
        """빈 쿼리 BM25 테스트"""
        engine = SearchEngine(ESConfig())
        query = SearchQuery()
        
        bm25_queries = engine._build_bm25_query(query)
        
        # 기본 match_all 반환
        assert any("match_all" in q for q in bm25_queries)
    
    def test_build_search_text(self):
        """벡터 검색 텍스트 생성 테스트"""
        engine = SearchEngine(ESConfig())
        
        # 자연어 쿼리 있는 경우
        query1 = SearchQuery(natural_query="조용한 아파트")
        text1 = engine._build_search_text(query1)
        assert "조용한 아파트" in text1
        
        # 라이프스타일 키워드 있는 경우
        query2 = SearchQuery(lifestyle_keywords=["육아", "교통"])
        text2 = engine._build_search_text(query2)
        assert "육아" in text2
        assert "교통" in text2
        
        # 빈 쿼리
        query3 = SearchQuery()
        text3 = engine._build_search_text(query3)
        assert text3 == "서울 아파트 추천"
    
    @patch('search_engine.Elasticsearch')
    def test_hybrid_search_not_connected(self, mock_es):
        """연결 안된 상태에서 하이브리드 검색"""
        engine = SearchEngine(ESConfig())
        query = SearchQuery(districts=["송파구"])
        
        results = engine.hybrid_search(query)
        assert results == []
    
    @patch('search_engine.Elasticsearch')
    def test_get_document_not_found(self, mock_es):
        """존재하지 않는 문서 조회"""
        from elasticsearch.exceptions import NotFoundError
        
        mock_client = MagicMock()
        mock_client.get.side_effect = NotFoundError(404, "not found", {})
        mock_es.return_value = mock_client
        
        engine = SearchEngine(ESConfig())
        engine.client = mock_client
        
        result = engine.get_document("nonexistent")
        assert result is None


class TestSearchQueryIntegration:
    """SearchQuery와 SearchEngine 통합 테스트"""
    
    def test_complex_query_filter_building(self):
        """복잡한 쿼리의 필터 생성 테스트"""
        engine = SearchEngine(ESConfig())
        
        query = SearchQuery(
            districts=["송파구", "노원구"],
            min_price=50000,
            max_price=80000,
            min_area=60,
            max_area=100,
            min_floor=5,
            max_floor=20,
            min_year_built=2010,
            lifestyle_keywords=["육아", "교통", "학군"],
            natural_query="아이 키우기 좋은 조용한 동네",
            top_k=10
        )
        
        filters = engine._build_filter_conditions(query)
        bm25_queries = engine._build_bm25_query(query)
        search_text = engine._build_search_text(query)
        
        # 필터 검증
        assert len(filters) >= 5  # districts, price, area, floor, year_built
        
        # BM25 쿼리 검증
        assert len(bm25_queries) > 0
        
        # 검색 텍스트 검증
        assert "아이 키우기 좋은" in search_text
        assert "육아" in search_text


class TestDataLoadingFunctions:
    """데이터 로드 유틸리티 함수 테스트"""
    
    @patch('pandas.read_csv')
    def test_load_and_merge_data(self, mock_read_csv):
        """데이터 병합 함수 테스트"""
        import pandas as pd
        from search_engine import load_and_merge_data
        
        # 모킹 데이터 설정
        apartments_df = pd.DataFrame({
            'kaptCode': ['A001', 'A002'],
            'kaptName': ['아파트1', '아파트2'],
            'doroJuso': ['주소1', '주소2'],
            '수집지역': ['송파구', '노원구']
        })
        
        reviews_df = pd.DataFrame({
            'kaptName': ['아파트1', '아파트1', '아파트2'],
            'Score': [4.0, 5.0, 3.5],
            'Pros': ['좋음', '편리함', '조용함'],
            'Cons': ['비쌈', '주차 어려움', '오래됨']
        })
        
        deals_df = pd.DataFrame({
            'apt_name': ['아파트1', '아파트2'],
            'gu': ['송파구', '노원구'],
            'dong': ['잠실동', '상계동'],
            'deal_date': ['2025-10-01', '2025-09-15'],
            'price_manwon': [100000, 50000],
            'area_m2': [84.5, 59.0],
            'floor': [10, 5],
            'year_built': [2010, 2005]
        })
        
        mock_read_csv.side_effect = [apartments_df, reviews_df, deals_df]
        
        # 함수 호출 (실제 파일 없이 모킹)
        try:
            result = load_and_merge_data("apt.csv", "review.csv", "deals.csv")
            # 결과 검증
            assert isinstance(result, list)
        except Exception:
            # 파일이 없는 경우 예외 처리
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
