"""
데이터 인덱서 테스트
====================
indexer.py의 데이터 로드 및 병합 로직 테스트

실행: pytest tests/test_indexer.py -v
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import sys
import os

# 상위 디렉토리 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexer import (
    load_apartments_data,
    load_reviews_data,
    load_deals_data,
    merge_data,
    index_data
)
from search_engine import ESConfig


class TestLoadApartmentsData:
    """아파트 데이터 로드 테스트"""
    
    @patch('pandas.read_csv')
    def test_load_and_rename_columns(self, mock_read_csv):
        """컬럼명 변환 테스트"""
        mock_df = pd.DataFrame({
            'kaptCode': ['A001', 'A002'],
            'kaptName': ['아파트1', '아파트2'],
            'doroJuso': ['주소1', '주소2'],
            '수집지역': ['송파구', '노원구']
        })
        mock_read_csv.return_value = mock_df
        
        result = load_apartments_data("test.csv")
        
        assert len(result) == 2
        assert result[0]['kapt_code'] == 'A001'
        assert result[0]['kapt_name'] == '아파트1'
        assert result[0]['gu'] == '송파구'
    
    @patch('pandas.read_csv')
    def test_empty_dataframe(self, mock_read_csv):
        """빈 데이터프레임 처리"""
        mock_df = pd.DataFrame({
            'kaptCode': [],
            'kaptName': [],
            'doroJuso': [],
            '수집지역': []
        })
        mock_read_csv.return_value = mock_df
        
        result = load_apartments_data("empty.csv")
        
        assert result == []


class TestLoadReviewsData:
    """리뷰 데이터 로드 테스트"""
    
    @patch('pandas.read_csv')
    def test_aggregate_reviews(self, mock_read_csv):
        """리뷰 집계 테스트"""
        mock_df = pd.DataFrame({
            'kaptName': ['아파트1', '아파트1', '아파트1', '아파트2'],
            'Score': [4.0, 5.0, 3.0, 4.5],
            'Pros': ['좋음', '편리함', '깨끗함', '조용함'],
            'Cons': ['비쌈', '주차 어려움', '오래됨', '교통 불편']
        })
        mock_read_csv.return_value = mock_df
        
        result = load_reviews_data("reviews.csv")
        
        assert len(result) == 2  # 아파트 2개
        
        apt1_row = result[result['kapt_name'] == '아파트1'].iloc[0]
        assert apt1_row['review_score'] == 4.0  # (4+5+3)/3
        assert '좋음' in apt1_row['pros']
    
    @patch('pandas.read_csv')
    def test_limit_reviews(self, mock_read_csv):
        """리뷰 개수 제한 (5개) 테스트"""
        mock_df = pd.DataFrame({
            'kaptName': ['아파트1'] * 10,
            'Score': [4.0] * 10,
            'Pros': [f'장점{i}' for i in range(10)],
            'Cons': [f'단점{i}' for i in range(10)]
        })
        mock_read_csv.return_value = mock_df
        
        result = load_reviews_data("reviews.csv")
        
        # 상위 5개만 포함
        apt1_row = result[result['kapt_name'] == '아파트1'].iloc[0]
        pros_count = apt1_row['pros'].count('|') + 1
        assert pros_count <= 5


class TestLoadDealsData:
    """실거래가 데이터 로드 테스트"""
    
    @patch('pandas.read_csv')
    def test_load_and_get_latest(self, mock_read_csv):
        """최신 거래 데이터 추출 테스트"""
        mock_df = pd.DataFrame({
            'apt_name': ['아파트1', '아파트1', '아파트2'],
            'gu': ['송파구', '송파구', '노원구'],
            'dong': ['잠실동', '잠실동', '상계동'],
            'deal_date': ['2025-10-01', '2025-09-01', '2025-10-15'],
            'price_manwon': [100000, 95000, 50000],
            'area_m2': [84.5, 84.5, 59.0],
            'floor': [10, 5, 7],
            'year_built': [2010, 2010, 2005],
            'price_krw': [1000000000, 950000000, 500000000]
        })
        mock_read_csv.return_value = mock_df
        
        result = load_deals_data("deals.csv")
        
        assert len(result) == 2  # 아파트별 1건씩
        
        # 아파트1은 최신 거래 (2025-10-01)
        apt1_row = result[result['kapt_name'] == '아파트1'].iloc[0]
        assert apt1_row['price_manwon'] == 100000
    
    @patch('pandas.read_csv')
    def test_date_parsing(self, mock_read_csv):
        """날짜 파싱 테스트"""
        mock_df = pd.DataFrame({
            'apt_name': ['아파트1'],
            'gu': ['송파구'],
            'dong': ['잠실동'],
            'deal_date': ['2025-10-31'],
            'price_manwon': [100000],
            'area_m2': [84.5],
            'floor': [10],
            'year_built': [2010],
            'price_krw': [1000000000]
        })
        mock_read_csv.return_value = mock_df
        
        result = load_deals_data("deals.csv")
        
        assert len(result) == 1


class TestMergeData:
    """데이터 병합 테스트"""
    
    def test_merge_all_data(self):
        """전체 데이터 병합 테스트"""
        apartments = [
            {'kapt_code': 'A001', 'kapt_name': '아파트1', 'doro_juso': '주소1', 'gu': '송파구'},
            {'kapt_code': 'A002', 'kapt_name': '아파트2', 'doro_juso': '주소2', 'gu': '노원구'}
        ]
        
        reviews = pd.DataFrame({
            'kapt_name': ['아파트1', '아파트2'],
            'review_score': [4.5, 3.8],
            'pros': ['좋음', '조용함'],
            'cons': ['비쌈', '오래됨']
        })
        
        deals = pd.DataFrame({
            'kapt_name': ['아파트1'],
            'gu': ['송파구'],
            'dong': ['잠실동'],
            'price_manwon': [100000],
            'area_m2': [84.5],
            'floor': [10],
            'year_built': [2010]
        })
        
        result = merge_data(apartments, reviews, deals)
        
        assert len(result) == 2
        
        # 아파트1은 모든 데이터 병합됨
        apt1 = next(r for r in result if r['kapt_name'] == '아파트1')
        assert apt1['review_score'] == 4.5
        assert apt1['price_manwon'] == 100000
        
        # 아파트2는 리뷰만 있음
        apt2 = next(r for r in result if r['kapt_name'] == '아파트2')
        assert apt2['review_score'] == 3.8
        assert apt2['price_manwon'] is None
    
    def test_merge_empty_reviews(self):
        """빈 리뷰 데이터 병합"""
        apartments = [
            {'kapt_code': 'A001', 'kapt_name': '아파트1', 'gu': '송파구'}
        ]
        
        reviews = pd.DataFrame(columns=['kapt_name', 'review_score', 'pros', 'cons'])
        deals = pd.DataFrame(columns=['kapt_name', 'gu', 'dong', 'price_manwon', 'area_m2', 'floor', 'year_built'])
        
        result = merge_data(apartments, reviews, deals)
        
        assert len(result) == 1
        assert result[0]['kapt_name'] == '아파트1'


class TestIndexData:
    """데이터 인덱싱 테스트"""
    
    @patch('indexer.SearchEngine')
    def test_index_data_success(self, MockSearchEngine):
        """인덱싱 성공 테스트"""
        mock_engine = MagicMock()
        mock_engine.connect.return_value = True
        mock_engine.create_index.return_value = True
        mock_engine.bulk_index_documents.return_value = (10, 0)
        mock_engine.get_index_stats.return_value = {'doc_count': 10}
        MockSearchEngine.return_value = mock_engine
        
        documents = [
            {'kapt_code': 'A001', 'kapt_name': '아파트1'},
            {'kapt_code': 'A002', 'kapt_name': '아파트2'}
        ]
        
        config = ESConfig()
        index_data(documents, config)
        
        mock_engine.connect.assert_called_once()
        mock_engine.create_index.assert_called_once_with(delete_existing=True)
        mock_engine.bulk_index_documents.assert_called_once()
    
    @patch('indexer.SearchEngine')
    def test_index_data_connection_failure(self, MockSearchEngine):
        """연결 실패 테스트"""
        mock_engine = MagicMock()
        mock_engine.connect.return_value = False
        MockSearchEngine.return_value = mock_engine
        
        documents = [{'kapt_code': 'A001', 'kapt_name': '아파트1'}]
        config = ESConfig()
        
        # 연결 실패시 조기 종료
        index_data(documents, config)
        
        mock_engine.create_index.assert_not_called()
    
    @patch('indexer.SearchEngine')
    def test_index_data_create_index_failure(self, MockSearchEngine):
        """인덱스 생성 실패 테스트"""
        mock_engine = MagicMock()
        mock_engine.connect.return_value = True
        mock_engine.create_index.return_value = False
        MockSearchEngine.return_value = mock_engine
        
        documents = [{'kapt_code': 'A001', 'kapt_name': '아파트1'}]
        config = ESConfig()
        
        index_data(documents, config)
        
        mock_engine.bulk_index_documents.assert_not_called()


class TestMainFunction:
    """main 함수 테스트"""
    
    @patch('indexer.index_data')
    @patch('indexer.merge_data')
    @patch('indexer.load_deals_data')
    @patch('indexer.load_reviews_data')
    @patch('indexer.load_apartments_data')
    @patch.dict(os.environ, {'DATA_DIR': '/test/data'})
    def test_main_flow(self, mock_load_apt, mock_load_review, mock_load_deals, 
                       mock_merge, mock_index):
        """메인 함수 흐름 테스트"""
        mock_load_apt.return_value = [{'kapt_code': 'A001', 'kapt_name': '테스트'}]
        mock_load_review.return_value = pd.DataFrame()
        mock_load_deals.return_value = pd.DataFrame()
        mock_merge.return_value = [{'kapt_code': 'A001', 'kapt_name': '테스트'}]
        
        from indexer import main
        main()
        
        mock_load_apt.assert_called_once()
        mock_merge.assert_called_once()
        mock_index.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
