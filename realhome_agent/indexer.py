"""
데이터 인덱싱 스크립트
=====================
CSV 파일을 읽어 ElasticSearch에 인덱싱

Author: RealHome Agent Team
Version: 1.0.0
"""

import os
import sys
import logging
import pandas as pd
from typing import List, Dict, Any

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from search_engine import SearchEngine, ESConfig, load_and_merge_data

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_apartments_data(csv_path: str) -> List[Dict[str, Any]]:
    """
    아파트 기본 정보 CSV 로드
    """
    logger.info(f"아파트 데이터 로드: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        'kaptCode': 'kapt_code',
        'kaptName': 'kapt_name',
        'doroJuso': 'doro_juso',
        '수집지역': 'gu'
    })
    
    logger.info(f"아파트 데이터 로드 완료: {len(df)} 건")
    return df.to_dict('records')


def load_reviews_data(csv_path: str) -> pd.DataFrame:
    """
    리뷰 데이터 CSV 로드 및 집계
    """
    logger.info(f"리뷰 데이터 로드: {csv_path}")
    
    # 오류 행 skip 처리
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    
    # 아파트별 리뷰 집계
    agg_df = df.groupby('kaptName').agg({
        'Score': 'mean',
        'Pros': lambda x: ' | '.join(x.dropna().astype(str).head(5)),
        'Cons': lambda x: ' | '.join(x.dropna().astype(str).head(5))
    }).reset_index()
    
    agg_df.columns = ['kapt_name', 'review_score', 'pros', 'cons']
    
    logger.info(f"리뷰 데이터 집계 완료: {len(agg_df)} 개 아파트")
    return agg_df


def load_deals_data(csv_path: str) -> pd.DataFrame:
    """
    실거래가 데이터 CSV 로드 (최신 거래 기준)
    """
    logger.info(f"실거래가 데이터 로드: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df['deal_date'] = pd.to_datetime(df['deal_date'])
    
    # 아파트별 최신 거래 추출
    latest_df = df.sort_values('deal_date', ascending=False).drop_duplicates('apt_name')
    latest_df = latest_df[['apt_name', 'gu', 'dong', 'price_manwon', 'area_m2', 'floor', 'year_built', 'price_krw']]
    latest_df = latest_df.rename(columns={'apt_name': 'kapt_name'})
    
    logger.info(f"실거래가 데이터 로드 완료: {len(latest_df)} 건")
    return latest_df


def merge_data(
    apartments: List[Dict],
    reviews: pd.DataFrame,
    deals: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    모든 데이터 병합
    """
    logger.info("데이터 병합 시작...")
    
    # DataFrame 변환
    apt_df = pd.DataFrame(apartments)
    merged = apt_df
    
    # 리뷰 데이터 병합 (비어있지 않은 경우만)
    if not reviews.empty and 'kapt_name' in reviews.columns:
        merged = merged.merge(reviews, on='kapt_name', how='left')
    else:
        logger.warning("리뷰 데이터가 비어있거나 kapt_name 컬럼이 없습니다.")
    
    # 실거래가 데이터 병합 (비어있지 않은 경우만)
    if not deals.empty and 'kapt_name' in deals.columns:
        merged = merged.merge(deals, on='kapt_name', how='left', suffixes=('', '_deal'))
    else:
        logger.warning("실거래가 데이터가 비어있거나 kapt_name 컬럼이 없습니다.")
    
    # gu 필드 정리 (수집지역 우선, 없으면 실거래가 데이터)
    if 'gu_deal' in merged.columns:
        merged['gu'] = merged['gu'].fillna(merged['gu_deal'])
        merged = merged.drop(columns=['gu_deal'])
    
    # NaN 처리
    merged = merged.where(pd.notnull(merged), None)
    
    logger.info(f"데이터 병합 완료: {len(merged)} 건")
    return merged.to_dict('records')


def index_data(documents: List[Dict[str, Any]], config: ESConfig) -> None:
    """
    ElasticSearch에 데이터 인덱싱
    """
    logger.info("ElasticSearch 인덱싱 시작...")
    
    # 검색 엔진 초기화
    engine = SearchEngine(config)
    
    # 연결
    if not engine.connect():
        logger.error("ElasticSearch 연결 실패")
        return
    
    # 인덱스 생성 (기존 삭제)
    if not engine.create_index(delete_existing=True):
        logger.error("인덱스 생성 실패")
        return
    
    # 벌크 인덱싱
    success, fail = engine.bulk_index_documents(
        documents,
        generate_embeddings=True,
        batch_size=50
    )
    
    logger.info(f"인덱싱 완료: 성공 {success}, 실패 {fail}")
    
    # 통계 출력
    stats = engine.get_index_stats()
    logger.info(f"인덱스 통계: {stats}")


def main():
    """메인 실행 함수"""
    
    # 데이터 파일 경로
    DATA_DIR = os.getenv("DATA_DIR", "/data")
    
    apartments_csv = os.path.join(DATA_DIR, "아파트_수집_최종.csv")
    reviews_csv = os.path.join(DATA_DIR, "리뷰_구조화_결과.csv")
    deals_csv = os.path.join(DATA_DIR, "deals_2023_2025_min.csv")
    
    # 파일 존재 확인
    for path in [apartments_csv, reviews_csv, deals_csv]:
        if not os.path.exists(path):
            logger.warning(f"파일 없음: {path}")
    
    # 데이터 로드
    try:
        apartments = load_apartments_data(apartments_csv)
    except Exception as e:
        logger.error(f"아파트 데이터 로드 실패: {e}")
        apartments = []
    
    try:
        reviews = load_reviews_data(reviews_csv)
    except Exception as e:
        logger.error(f"리뷰 데이터 로드 실패: {e}")
        reviews = pd.DataFrame()
    
    try:
        deals = load_deals_data(deals_csv)
    except Exception as e:
        logger.error(f"실거래가 데이터 로드 실패: {e}")
        deals = pd.DataFrame()
    
    if not apartments:
        logger.error("인덱싱할 데이터가 없습니다.")
        return
    
    # 데이터 병합
    documents = merge_data(apartments, reviews, deals)
    
    # ElasticSearch 설정
    config = ESConfig(
        host=os.getenv("ES_HOST", "localhost"),
        port=int(os.getenv("ES_PORT", "9200")),
        username=os.getenv("ES_USERNAME"),
        password=os.getenv("ES_PASSWORD"),
        index_name=os.getenv("ES_INDEX", "realhome_apartments")
    )
    
    # 인덱싱 실행
    index_data(documents, config)
    
    logger.info("인덱싱 작업 완료!")


if __name__ == "__main__":
    main()
