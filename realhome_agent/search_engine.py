"""
ElasticSearch 데이터 인덱싱 및 하이브리드 검색 모듈
==================================================
BM25 + Dense Vector 하이브리드 검색 지원
google/embeddinggemma-300m 임베딩 모델 사용

Author: RealHome Agent Team
Version: 1.0.0
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import ConnectionError, NotFoundError, RequestError

import torch
from transformers import AutoTokenizer, AutoModel

from models import ApartmentSchema, SearchQuery

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ESConfig:
    """ElasticSearch 설정 클래스"""
    host: str = "localhost"
    port: int = 9200
    scheme: str = "http"
    username: Optional[str] = None
    password: Optional[str] = None
    index_name: str = "realhome_apartments"
    embedding_dim: int = 1024  # BAAI/bge-m3 출력 차원


class EmbeddingModel:
    """
    BAAI/bge-m3 임베딩 모델 래퍼
    
    다국어 지원 임베딩 생성을 담당합니다.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """
        임베딩 모델 초기화
        
        Args:
            model_name: HuggingFace 모델 이름
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._is_loaded = False
        
    def load(self) -> None:
        """모델 및 토크나이저 로드"""
        if self._is_loaded:
            return
            
        try:
            logger.info(f"임베딩 모델 로드 중: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            logger.info(f"임베딩 모델 로드 완료 (device: {self.device})")
        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {e}")
            raise
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        텍스트를 임베딩 벡터로 변환
        
        Args:
            texts: 변환할 텍스트 리스트
            batch_size: 배치 크기
            
        Returns:
            임베딩 벡터 배열 (shape: [len(texts), embedding_dim])
        """
        if not self._is_loaded:
            self.load()
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # 토큰화
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                # 임베딩 생성
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Mean pooling
                    attention_mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                all_embeddings.append(embeddings.cpu().numpy())
                
            except Exception as e:
                logger.error(f"임베딩 생성 오류 (batch {i}): {e}")
                # 오류 시 영벡터 반환
                zero_embeddings = np.zeros((len(batch_texts), 768))
                all_embeddings.append(zero_embeddings)
        
        return np.vstack(all_embeddings)
    
    def encode_single(self, text: str) -> List[float]:
        """
        단일 텍스트 임베딩 변환
        
        Args:
            text: 변환할 텍스트
            
        Returns:
            임베딩 벡터 리스트
        """
        embeddings = self.encode([text])
        return embeddings[0].tolist()


class SearchEngine:
    """
    ElasticSearch 기반 하이브리드 검색 엔진
    
    BM25(키워드)와 Dense Vector(임베딩)를 결합한
    하이브리드 검색을 제공합니다.
    """
    
    # 인덱스 매핑 정의 (정형 + 비정형 데이터 지원)
    INDEX_MAPPING = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "korean_analyzer": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer",
                        "filter": ["lowercase", "nori_readingform"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                # 기본 식별 정보
                "kapt_code": {"type": "keyword"},
                "kapt_name": {
                    "type": "text",
                    "analyzer": "korean_analyzer",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    }
                },
                
                # 위치 정보
                "doro_juso": {
                    "type": "text",
                    "analyzer": "korean_analyzer"
                },
                "gu": {"type": "keyword"},
                "dong": {"type": "keyword"},
                
                # 정형 데이터 (숫자형 - range query 지원)
                "price_manwon": {"type": "float"},
                "price_krw": {"type": "float"},
                "area_m2": {"type": "float"},
                "floor": {"type": "integer"},
                "year_built": {"type": "integer"},
                
                # 비정형 데이터 (텍스트 - 전문 검색)
                "review_score": {"type": "float"},
                "pros": {
                    "type": "text",
                    "analyzer": "korean_analyzer"
                },
                "cons": {
                    "type": "text",
                    "analyzer": "korean_analyzer"
                },
                "combined_review": {
                    "type": "text",
                    "analyzer": "korean_analyzer"
                },
                
                # Dense Vector (kNN 검색용)
                "embedding": {
                    "type": "dense_vector",
                    "dims": 1024,  # BAAI/bge-m3 차원
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    
    def __init__(self, config: Optional[ESConfig] = None):
        """
        검색 엔진 초기화
        
        Args:
            config: ElasticSearch 설정
        """
        self.config = config or ESConfig()
        self.client: Optional[Elasticsearch] = None
        self.embedding_model = EmbeddingModel()
        
    def connect(self, max_retries: int = 10, retry_delay: int = 5) -> bool:
        """
        ElasticSearch 연결 (재시도 로직 포함)
        
        Args:
            max_retries: 최대 재시도 횟수
            retry_delay: 재시도 간격 (초)
            
        Returns:
            연결 성공 여부
        """
        import time
        
        # 클라이언트 설정
        es_config = {
            "hosts": [f"{self.config.scheme}://{self.config.host}:{self.config.port}"],
            "verify_certs": False,
            "ssl_show_warn": False
        }
        
        # 인증 정보가 있는 경우 추가
        if self.config.username and self.config.password:
            es_config["basic_auth"] = (self.config.username, self.config.password)
        
        self.client = Elasticsearch(**es_config)
        
        for attempt in range(max_retries):
            try:
                # 클러스터 정보로 연결 확인 (ping 대신)
                info = self.client.info()
                logger.info(f"ElasticSearch 연결 성공: {info['cluster_name']} (v{info['version']['number']})")
                return True
                
            except Exception as e:
                logger.warning(f"ElasticSearch 연결 시도 {attempt + 1}/{max_retries} 실패: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        logger.error(f"ElasticSearch 연결 실패: {max_retries}회 시도 후 포기")
        return False
    
    def create_index(self, delete_existing: bool = False) -> bool:
        """
        인덱스 생성
        
        Args:
            delete_existing: 기존 인덱스 삭제 여부
            
        Returns:
            생성 성공 여부
        """
        if not self.client:
            logger.error("ElasticSearch 연결이 필요합니다.")
            return False
        
        try:
            index_name = self.config.index_name
            
            # 기존 인덱스 삭제
            if delete_existing and self.client.indices.exists(index=index_name):
                self.client.indices.delete(index=index_name)
                logger.info(f"기존 인덱스 삭제: {index_name}")
            
            # 인덱스 생성
            if not self.client.indices.exists(index=index_name):
                self.client.indices.create(index=index_name, body=self.INDEX_MAPPING)
                logger.info(f"인덱스 생성 완료: {index_name}")
            else:
                logger.info(f"인덱스 이미 존재: {index_name}")
            
            return True
            
        except RequestError as e:
            logger.error(f"인덱스 생성 실패: {e}")
            return False
    
    def index_document(self, document: ApartmentSchema) -> bool:
        """
        단일 문서 인덱싱
        
        Args:
            document: 인덱싱할 아파트 문서
            
        Returns:
            인덱싱 성공 여부
        """
        if not self.client:
            logger.error("ElasticSearch 연결이 필요합니다.")
            return False
        
        try:
            # 리뷰 텍스트 결합
            combined_review = ""
            if document.pros:
                combined_review += f"장점: {document.pros} "
            if document.cons:
                combined_review += f"단점: {document.cons}"
            
            # 임베딩 생성 (리뷰 텍스트 기반)
            embedding = None
            if combined_review.strip():
                embedding = self.embedding_model.encode_single(combined_review)
            
            # 문서 데이터 준비
            doc_dict = document.model_dump()
            doc_dict['combined_review'] = combined_review
            doc_dict['embedding'] = embedding
            
            # 인덱싱
            self.client.index(
                index=self.config.index_name,
                id=document.kapt_code,
                document=doc_dict
            )
            
            return True
            
        except Exception as e:
            logger.error(f"문서 인덱싱 실패: {e}")
            return False
    
    def bulk_index_documents(
        self,
        documents: List[Dict[str, Any]],
        generate_embeddings: bool = True,
        batch_size: int = 100
    ) -> Tuple[int, int]:
        """
        대량 문서 인덱싱
        
        Args:
            documents: 인덱싱할 문서 리스트
            generate_embeddings: 임베딩 생성 여부
            batch_size: 배치 크기
            
        Returns:
            (성공 수, 실패 수) 튜플
        """
        if not self.client:
            logger.error("ElasticSearch 연결이 필요합니다.")
            return 0, len(documents)
        
        success_count = 0
        fail_count = 0
        
        try:
            # 임베딩 생성 (배치 처리)
            if generate_embeddings:
                logger.info("임베딩 생성 중...")
                texts = []
                for doc in documents:
                    combined = ""
                    if doc.get('pros'):
                        combined += f"장점: {doc['pros']} "
                    if doc.get('cons'):
                        combined += f"단점: {doc['cons']}"
                    texts.append(combined if combined.strip() else "정보 없음")
                
                embeddings = self.embedding_model.encode(texts, batch_size=batch_size)
                
                for i, doc in enumerate(documents):
                    doc['embedding'] = embeddings[i].tolist()
                    doc['combined_review'] = texts[i]
            
            # 벌크 인덱싱 액션 생성
            actions = []
            for doc in documents:
                action = {
                    "_index": self.config.index_name,
                    "_id": doc.get('kapt_code', doc.get('id')),
                    "_source": doc
                }
                actions.append(action)
            
            # 벌크 인덱싱 실행
            logger.info(f"벌크 인덱싱 시작: {len(actions)} 문서")
            success, failed = helpers.bulk(
                self.client,
                actions,
                raise_on_error=False,
                raise_on_exception=False
            )
            
            success_count = success
            fail_count = len(failed) if isinstance(failed, list) else 0
            
            # 실패 원인 로깅
            if failed:
                for item in failed[:5]:  # 처음 5개만 출력
                    logger.error(f"인덱싱 실패: {item}")
            
            logger.info(f"벌크 인덱싱 완료: 성공 {success_count}, 실패 {fail_count}")
            
        except Exception as e:
            logger.error(f"벌크 인덱싱 오류: {e}")
            fail_count = len(documents)
        
        return success_count, fail_count
    
    def hybrid_search(
        self,
        query: SearchQuery,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        BM25 + Dense Vector 하이브리드 검색
        
        Args:
            query: 검색 쿼리
            bm25_weight: BM25 점수 가중치
            vector_weight: 벡터 검색 점수 가중치
            
        Returns:
            검색 결과 리스트
        """
        if not self.client:
            logger.error("ElasticSearch 연결이 필요합니다.")
            return []
        
        try:
            # 1. BM25 쿼리 구성
            bm25_query = self._build_bm25_query(query)
            
            # 2. 필터 조건 구성
            filter_conditions = self._build_filter_conditions(query)
            
            # 3. 벡터 검색 쿼리 텍스트 생성
            search_text = self._build_search_text(query)
            query_embedding = self.embedding_model.encode_single(search_text)
            
            # 4. 하이브리드 검색 쿼리 구성
            search_body = {
                "size": query.top_k,
                "query": {
                    "bool": {
                        "should": [
                            # BM25 검색
                            {
                                "bool": {
                                    "should": bm25_query,
                                    "boost": bm25_weight
                                }
                            }
                        ],
                        "filter": filter_conditions
                    }
                },
                # kNN 벡터 검색
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": query.top_k,
                    "num_candidates": query.top_k * 2,
                    "boost": vector_weight
                }
            }
            
            # 검색 실행
            response = self.client.search(
                index=self.config.index_name,
                body=search_body
            )
            
            # 결과 파싱
            results = []
            for hit in response['hits']['hits']:
                result = hit['_source']
                result['_score'] = hit['_score']
                result['_id'] = hit['_id']
                results.append(result)
            
            logger.info(f"하이브리드 검색 완료: {len(results)} 결과")
            return results
            
        except Exception as e:
            logger.error(f"하이브리드 검색 오류: {e}")
            return []
    
    def bm25_search(self, query: SearchQuery) -> List[Dict[str, Any]]:
        """
        BM25 키워드 검색만 수행
        
        Args:
            query: 검색 쿼리
            
        Returns:
            검색 결과 리스트
        """
        if not self.client:
            logger.error("ElasticSearch 연결이 필요합니다.")
            return []
        
        try:
            bm25_query = self._build_bm25_query(query)
            filter_conditions = self._build_filter_conditions(query)
            
            search_body = {
                "size": query.top_k,
                "query": {
                    "bool": {
                        "should": bm25_query,
                        "filter": filter_conditions,
                        "minimum_should_match": 1
                    }
                }
            }
            
            response = self.client.search(
                index=self.config.index_name,
                body=search_body
            )
            
            results = []
            for hit in response['hits']['hits']:
                result = hit['_source']
                result['_score'] = hit['_score']
                result['_id'] = hit['_id']
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"BM25 검색 오류: {e}")
            return []
    
    def vector_search(
        self,
        text: str,
        top_k: int = 10,
        filter_conditions: Optional[List[Dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        Dense Vector 검색만 수행
        
        Args:
            text: 검색 텍스트
            top_k: 반환할 결과 수
            filter_conditions: 필터 조건
            
        Returns:
            검색 결과 리스트
        """
        if not self.client:
            logger.error("ElasticSearch 연결이 필요합니다.")
            return []
        
        try:
            query_embedding = self.embedding_model.encode_single(text)
            
            search_body = {
                "size": top_k,
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": top_k,
                    "num_candidates": top_k * 2
                }
            }
            
            if filter_conditions:
                search_body["knn"]["filter"] = {"bool": {"filter": filter_conditions}}
            
            response = self.client.search(
                index=self.config.index_name,
                body=search_body
            )
            
            results = []
            for hit in response['hits']['hits']:
                result = hit['_source']
                result['_score'] = hit['_score']
                result['_id'] = hit['_id']
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"벡터 검색 오류: {e}")
            return []
    
    def _build_bm25_query(self, query: SearchQuery) -> List[Dict]:
        """BM25 쿼리 조건 생성"""
        should_queries = []
        
        # 자연어 쿼리 검색 (리뷰, 주소 등)
        if query.natural_query:
            should_queries.extend([
                {"match": {"combined_review": {"query": query.natural_query, "boost": 2.0}}},
                {"match": {"pros": {"query": query.natural_query, "boost": 1.5}}},
                {"match": {"cons": {"query": query.natural_query, "boost": 1.0}}},
                {"match": {"doro_juso": {"query": query.natural_query, "boost": 1.0}}},
                {"match": {"kapt_name": {"query": query.natural_query, "boost": 1.5}}}
            ])
        
        # 라이프스타일 키워드 검색
        if query.lifestyle_keywords:
            keyword_query = " ".join(query.lifestyle_keywords)
            should_queries.extend([
                {"match": {"combined_review": {"query": keyword_query, "boost": 2.5}}},
                {"match": {"pros": {"query": keyword_query, "boost": 2.0}}}
            ])
        
        # 동 검색
        if query.dong:
            should_queries.append(
                {"match": {"dong": {"query": query.dong, "boost": 1.5}}}
            )
        
        # 기본 쿼리가 없는 경우 match_all
        if not should_queries:
            should_queries.append({"match_all": {}})
        
        return should_queries
    
    def _build_filter_conditions(self, query: SearchQuery) -> List[Dict]:
        """필터 조건 생성"""
        filters = []
        
        # 구역 필터
        if query.districts:
            filters.append({"terms": {"gu": query.districts}})
        
        # 가격 범위 필터
        price_range = {}
        if query.min_price is not None:
            price_range["gte"] = query.min_price
        if query.max_price is not None:
            price_range["lte"] = query.max_price
        if price_range:
            filters.append({"range": {"price_manwon": price_range}})
        
        # 면적 범위 필터
        area_range = {}
        if query.min_area is not None:
            area_range["gte"] = query.min_area
        if query.max_area is not None:
            area_range["lte"] = query.max_area
        if area_range:
            filters.append({"range": {"area_m2": area_range}})
        
        # 층수 필터
        floor_range = {}
        if query.min_floor is not None:
            floor_range["gte"] = query.min_floor
        if query.max_floor is not None:
            floor_range["lte"] = query.max_floor
        if floor_range:
            filters.append({"range": {"floor": floor_range}})
        
        # 준공연도 필터
        if query.min_year_built is not None:
            filters.append({"range": {"year_built": {"gte": query.min_year_built}}})
        
        return filters
    
    def _build_search_text(self, query: SearchQuery) -> str:
        """벡터 검색용 텍스트 생성"""
        texts = []
        
        if query.natural_query:
            texts.append(query.natural_query)
        
        if query.lifestyle_keywords:
            texts.append(" ".join(query.lifestyle_keywords))
        
        if query.dong:
            texts.append(f"{query.dong} 아파트")
        
        if query.districts:
            texts.append(" ".join(query.districts))
        
        return " ".join(texts) if texts else "서울 아파트 추천"
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """문서 ID로 조회"""
        if not self.client:
            return None
        
        try:
            response = self.client.get(index=self.config.index_name, id=doc_id)
            return response['_source']
        except NotFoundError:
            return None
        except Exception as e:
            logger.error(f"문서 조회 오류: {e}")
            return None
    
    def delete_index(self) -> bool:
        """인덱스 삭제"""
        if not self.client:
            return False
        
        try:
            if self.client.indices.exists(index=self.config.index_name):
                self.client.indices.delete(index=self.config.index_name)
                logger.info(f"인덱스 삭제 완료: {self.config.index_name}")
            return True
        except Exception as e:
            logger.error(f"인덱스 삭제 오류: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """인덱스 통계 조회"""
        if not self.client:
            return {}
        
        try:
            stats = self.client.indices.stats(index=self.config.index_name)
            return {
                "doc_count": stats['indices'][self.config.index_name]['primaries']['docs']['count'],
                "size_bytes": stats['indices'][self.config.index_name]['primaries']['store']['size_in_bytes']
            }
        except Exception as e:
            logger.error(f"통계 조회 오류: {e}")
            return {}


# 데이터 로드 유틸리티 함수
def load_and_merge_data(
    apartments_csv: str,
    reviews_csv: str,
    deals_csv: str
) -> List[Dict[str, Any]]:
    """
    CSV 파일들을 로드하여 병합
    
    Args:
        apartments_csv: 아파트 기본 정보 CSV 경로
        reviews_csv: 리뷰 데이터 CSV 경로
        deals_csv: 실거래가 CSV 경로
        
    Returns:
        병합된 문서 리스트
    """
    import pandas as pd
    
    try:
        # 데이터 로드
        apartments_df = pd.read_csv(apartments_csv)
        reviews_df = pd.read_csv(reviews_csv)
        deals_df = pd.read_csv(deals_csv)
        
        # 리뷰 데이터 집계 (아파트별 평균 점수, 리뷰 통합)
        reviews_agg = reviews_df.groupby('kaptName').agg({
            'Score': 'mean',
            'Pros': lambda x: ' | '.join(x.dropna().astype(str)[:5]),  # 상위 5개 리뷰
            'Cons': lambda x: ' | '.join(x.dropna().astype(str)[:5])
        }).reset_index()
        reviews_agg.columns = ['kaptName', 'review_score', 'pros', 'cons']
        
        # 실거래가 최신 데이터 (아파트별 최근 거래)
        deals_df['deal_date'] = pd.to_datetime(deals_df['deal_date'])
        latest_deals = deals_df.sort_values('deal_date', ascending=False).drop_duplicates('apt_name')
        latest_deals = latest_deals[['apt_name', 'gu', 'dong', 'price_manwon', 'area_m2', 'floor', 'year_built']]
        latest_deals.columns = ['kaptName', 'gu', 'dong', 'price_manwon', 'area_m2', 'floor', 'year_built']
        
        # 데이터 병합
        merged = apartments_df.merge(reviews_agg, on='kaptName', how='left')
        merged = merged.merge(latest_deals, on='kaptName', how='left')
        
        # 필드명 정리
        merged = merged.rename(columns={
            'kaptCode': 'kapt_code',
            'kaptName': 'kapt_name',
            'doroJuso': 'doro_juso',
            '수집지역': 'gu'
        })
        
        # gu 필드 처리 (우선순위: 수집지역 > 실거래가 데이터)
        if 'gu_x' in merged.columns and 'gu_y' in merged.columns:
            merged['gu'] = merged['gu_x'].fillna(merged['gu_y'])
            merged = merged.drop(columns=['gu_x', 'gu_y'])
        
        # 문서 리스트로 변환
        documents = merged.to_dict('records')
        
        logger.info(f"데이터 병합 완료: {len(documents)} 문서")
        return documents
        
    except Exception as e:
        logger.error(f"데이터 로드 오류: {e}")
        return []


if __name__ == "__main__":
    """모듈 테스트"""
    
    # 설정
    config = ESConfig(
        host="localhost",
        port=9200,
        index_name="realhome_apartments_test"
    )
    
    # 검색 엔진 초기화
    engine = SearchEngine(config)
    
    # 연결 테스트
    if engine.connect():
        print("✅ ElasticSearch 연결 성공")
        
        # 인덱스 생성 테스트
        if engine.create_index(delete_existing=True):
            print("✅ 인덱스 생성 성공")
        
        # 샘플 검색 쿼리
        query = SearchQuery(
            districts=["송파구"],
            max_price=70000,
            lifestyle_keywords=["육아", "교통"],
            natural_query="아이 키우기 좋은 조용한 동네",
            top_k=5
        )
        
        print(f"\n검색 쿼리: {query.model_dump()}")
        
        # 통계 출력
        stats = engine.get_index_stats()
        print(f"인덱스 통계: {stats}")
    else:
        print("❌ ElasticSearch 연결 실패")
