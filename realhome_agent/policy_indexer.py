"""
정책 문서 PDF OCR 및 인덱싱 스크립트
====================================
PDF 파일에서 텍스트 추출 후 ElasticSearch 정책 인덱스에 저장

Author: RealHome Agent Team
Version: 1.0.0
"""

import os
import sys
import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import PyPDF2
import pdfplumber
from PIL import Image
from elasticsearch import Elasticsearch, helpers

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PolicyPDFExtractor:
    """PDF 정책 문서에서 텍스트 추출"""
    
    def __init__(self, use_ocr: bool = False):
        """
        Args:
            use_ocr: OCR 사용 여부 (이미지 기반 PDF용)
        """
        self.use_ocr = use_ocr
        
        if use_ocr:
            try:
                import pytesseract
                from pdf2image import convert_from_path
                self.pytesseract = pytesseract
                self.convert_from_path = convert_from_path
                logger.info("OCR 모드 활성화")
            except ImportError:
                logger.warning("OCR 라이브러리 없음. pytesseract, pdf2image 설치 필요")
                self.use_ocr = False
    
    def extract_text_pypdf(self, pdf_path: str) -> str:
        """PyPDF2로 텍스트 추출"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            return text.strip()
        except Exception as e:
            logger.error(f"PyPDF2 추출 실패: {e}")
            return ""
    
    def extract_text_pdfplumber(self, pdf_path: str) -> str:
        """pdfplumber로 텍스트 추출 (더 정확)"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            return text.strip()
        except Exception as e:
            logger.error(f"pdfplumber 추출 실패: {e}")
            return ""
    
    def extract_text_ocr(self, pdf_path: str) -> str:
        """OCR로 텍스트 추출 (이미지 기반 PDF)"""
        if not self.use_ocr:
            return ""
        
        try:
            logger.info(f"OCR 처리 중: {pdf_path}")
            images = self.convert_from_path(pdf_path)
            text = ""
            
            for i, image in enumerate(images):
                logger.info(f"페이지 {i+1}/{len(images)} OCR 처리 중...")
                page_text = self.pytesseract.image_to_string(image, lang='kor+eng')
                text += page_text + "\n\n"
            
            return text.strip()
        except Exception as e:
            logger.error(f"OCR 추출 실패: {e}")
            return ""
    
    def extract_text(self, pdf_path: str) -> str:
        """PDF에서 텍스트 추출 (자동 선택)"""
        logger.info(f"PDF 처리: {pdf_path}")
        
        # 1. pdfplumber 시도 (가장 정확)
        text = self.extract_text_pdfplumber(pdf_path)
        
        # 2. 텍스트가 부족하면 PyPDF2 시도
        if not text or len(text) < 100:
            logger.warning("pdfplumber 결과 부족, PyPDF2 시도")
            text = self.extract_text_pypdf(pdf_path)
        
        # 3. 여전히 부족하면 OCR 시도
        if not text or len(text) < 100:
            if self.use_ocr:
                logger.warning("텍스트 추출 실패, OCR 시도")
                text = self.extract_text_ocr(pdf_path)
            else:
                logger.error("OCR 비활성화 - 이미지 기반 PDF는 처리 불가")
        
        logger.info(f"추출된 텍스트 길이: {len(text)} 문자")
        return text


class PolicyDocumentParser:
    """정책 문서를 구조화된 데이터로 파싱"""
    
    @staticmethod
    def parse_filename(filename: str) -> Dict[str, str]:
        """파일명에서 정보 추출 (예: R25_1015.pdf -> 2025년 10월 15일)"""
        match = re.match(r'R(\d{2})_(\d{2})(\d{2})', filename)
        if match:
            year = f"20{match.group(1)}"
            month = match.group(2)
            day = match.group(3)
            return {
                "year": year,
                "month": month,
                "day": day,
                "date": f"{year}-{month}-{day}"
            }
        return {}
    
    @staticmethod
    def extract_sections(text: str) -> List[Dict[str, str]]:
        """텍스트를 섹션별로 분리"""
        sections = []
        
        # 제목 패턴 (번호. 제목, 1. 제목, 가. 제목 등)
        patterns = [
            r'(\d+\.\s+[^\n]+)',  # 1. 제목
            r'([가-힣]\.\s+[^\n]+)',  # 가. 제목
            r'(\d+-\d+\.\s+[^\n]+)',  # 1-1. 제목
        ]
        
        lines = text.split('\n')
        current_section = {"title": "서론", "content": ""}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 제목 패턴 매칭
            is_title = False
            for pattern in patterns:
                if re.match(pattern, line):
                    # 이전 섹션 저장
                    if current_section["content"]:
                        sections.append(current_section)
                    # 새 섹션 시작
                    current_section = {"title": line, "content": ""}
                    is_title = True
                    break
            
            if not is_title:
                current_section["content"] += line + " "
        
        # 마지막 섹션 저장
        if current_section["content"]:
            sections.append(current_section)
        
        return sections
    
    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        """정책 관련 키워드 추출"""
        keywords = []
        
        # 부동산 정책 키워드
        policy_keywords = [
            "LTV", "DSR", "대출", "규제", "완화",
            "생애최초", "신혼부부", "청약", "분양",
            "재개발", "재건축", "리모델링",
            "종합부동산세", "취득세", "양도세",
            "조정대상지역", "투기과열지구", "규제지역",
            "공시가격", "실거래가", "분양가상한제"
        ]
        
        for keyword in policy_keywords:
            if keyword in text:
                keywords.append(keyword)
        
        return list(set(keywords))
    
    def parse_document(self, pdf_path: str, text: str) -> Dict[str, Any]:
        """PDF 문서를 구조화된 데이터로 변환"""
        filename = Path(pdf_path).name
        date_info = self.parse_filename(filename)
        sections = self.extract_sections(text)
        keywords = self.extract_keywords(text)
        
        return {
            "document_id": filename.replace('.pdf', ''),
            "filename": filename,
            "title": f"부동산 정책 문서 ({date_info.get('date', 'Unknown')})",
            "date": date_info.get('date', datetime.now().strftime('%Y-%m-%d')),
            "year": date_info.get('year', ''),
            "month": date_info.get('month', ''),
            "full_text": text,
            "sections": sections,
            "keywords": keywords,
            "indexed_at": datetime.now().isoformat(),
            "document_type": "policy",
            "source": "PDF"
        }


class PolicyIndexer:
    """ElasticSearch 정책 인덱스 관리"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        index_name: str = "realhome_policies"
    ):
        self.host = host
        self.port = port
        self.index_name = index_name
        self.es = None
    
    def connect(self) -> bool:
        """ElasticSearch 연결"""
        try:
            self.es = Elasticsearch([f"http://{self.host}:{self.port}"])
            if self.es.ping():
                logger.info(f"ElasticSearch 연결 성공: {self.host}:{self.port}")
                return True
            else:
                logger.error("ElasticSearch ping 실패")
                return False
        except Exception as e:
            logger.error(f"ElasticSearch 연결 실패: {e}")
            return False
    
    def create_index(self, delete_existing: bool = False) -> bool:
        """정책 인덱스 생성"""
        if delete_existing and self.es.indices.exists(index=self.index_name):
            logger.warning(f"기존 인덱스 삭제: {self.index_name}")
            self.es.indices.delete(index=self.index_name)
        
        if self.es.indices.exists(index=self.index_name):
            logger.info(f"인덱스 이미 존재: {self.index_name}")
            return True
        
        # 인덱스 매핑 정의
        mapping = {
            "mappings": {
                "properties": {
                    "document_id": {"type": "keyword"},
                    "filename": {"type": "keyword"},
                    "title": {"type": "text", "analyzer": "nori"},
                    "date": {"type": "date"},
                    "year": {"type": "keyword"},
                    "month": {"type": "keyword"},
                    "full_text": {
                        "type": "text",
                        "analyzer": "nori",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    },
                    "sections": {
                        "type": "nested",
                        "properties": {
                            "title": {"type": "text", "analyzer": "nori"},
                            "content": {"type": "text", "analyzer": "nori"}
                        }
                    },
                    "keywords": {"type": "keyword"},
                    "indexed_at": {"type": "date"},
                    "document_type": {"type": "keyword"},
                    "source": {"type": "keyword"}
                }
            }
        }
        
        try:
            self.es.indices.create(index=self.index_name, body=mapping)
            logger.info(f"인덱스 생성 완료: {self.index_name}")
            return True
        except Exception as e:
            logger.error(f"인덱스 생성 실패: {e}")
            return False
    
    def index_document(self, document: Dict[str, Any]) -> bool:
        """단일 문서 인덱싱"""
        try:
            response = self.es.index(
                index=self.index_name,
                id=document['document_id'],
                document=document
            )
            logger.info(f"문서 인덱싱 완료: {document['document_id']}")
            return True
        except Exception as e:
            logger.error(f"문서 인덱싱 실패: {e}")
            return False
    
    def bulk_index_documents(self, documents: List[Dict[str, Any]]) -> tuple:
        """벌크 문서 인덱싱"""
        actions = []
        for doc in documents:
            actions.append({
                "_index": self.index_name,
                "_id": doc['document_id'],
                "_source": doc
            })
        
        try:
            success, failed = helpers.bulk(self.es, actions, raise_on_error=False)
            logger.info(f"벌크 인덱싱 완료: 성공 {success}, 실패 {len(failed)}")
            return success, len(failed)
        except Exception as e:
            logger.error(f"벌크 인덱싱 실패: {e}")
            return 0, len(documents)
    
    def search(self, query: str, size: int = 10) -> List[Dict]:
        """정책 검색"""
        try:
            response = self.es.search(
                index=self.index_name,
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["title^3", "full_text", "keywords^2", "sections.content"],
                            "type": "best_fields"
                        }
                    },
                    "size": size,
                    "highlight": {
                        "fields": {
                            "full_text": {},
                            "sections.content": {}
                        }
                    }
                }
            )
            
            results = []
            for hit in response['hits']['hits']:
                results.append({
                    "score": hit['_score'],
                    "document": hit['_source'],
                    "highlights": hit.get('highlight', {})
                })
            
            return results
        except Exception as e:
            logger.error(f"검색 실패: {e}")
            return []


def main():
    """메인 실행 함수"""
    
    # 환경변수 설정 (Docker 컨테이너에서는 /app/policy_docs 사용)
    DATA_DIR = os.getenv("DATA_DIR", "/app/policy_docs")
    ES_HOST = os.getenv("ES_HOST", "localhost")
    ES_PORT = int(os.getenv("ES_PORT", "9200"))
    USE_OCR = os.getenv("USE_OCR", "false").lower() == "true"
    
    # PDF 파일 찾기
    pdf_files = list(Path(DATA_DIR).glob("*.pdf"))
    
    # 현재 디렉토리에서도 검색 (로컬 실행 시)
    if not pdf_files:
        pdf_files = list(Path(".").glob("*.pdf"))
    if not pdf_files:
        pdf_files = list(Path("..").glob("*.pdf"))
    
    if not pdf_files:
        logger.error(f"PDF 파일 없음. 검색 경로: {DATA_DIR}, ., ..")
        logger.info("PDF 파일을 다음 위치에 배치하세요:")
        logger.info("  - Docker: /app/policy_docs/*.pdf (상위 디렉토리)")
        logger.info("  - 로컬: 프로젝트 루트 디렉토리")
        return
    
    logger.info(f"발견된 PDF 파일: {len(pdf_files)}개")
    for pdf in pdf_files:
        logger.info(f"  - {pdf.name}")
    
    # PDF 텍스트 추출
    extractor = PolicyPDFExtractor(use_ocr=USE_OCR)
    parser = PolicyDocumentParser()
    
    documents = []
    for pdf_path in pdf_files:
        logger.info(f"\n처리 중: {pdf_path.name}")
        
        # 텍스트 추출
        text = extractor.extract_text(str(pdf_path))
        
        if not text:
            logger.error(f"텍스트 추출 실패: {pdf_path.name}")
            continue
        
        # 문서 파싱
        document = parser.parse_document(str(pdf_path), text)
        documents.append(document)
        
        logger.info(f"  - 섹션 수: {len(document['sections'])}")
        logger.info(f"  - 키워드: {', '.join(document['keywords'][:5])}")
    
    if not documents:
        logger.error("처리된 문서 없음")
        return
    
    # ElasticSearch 인덱싱
    indexer = PolicyIndexer(host=ES_HOST, port=ES_PORT)
    
    if not indexer.connect():
        logger.error("ElasticSearch 연결 실패")
        return
    
    # 인덱스 생성
    if not indexer.create_index(delete_existing=True):
        logger.error("인덱스 생성 실패")
        return
    
    # 문서 인덱싱
    success, failed = indexer.bulk_index_documents(documents)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"정책 문서 인덱싱 완료!")
    logger.info(f"  - 총 문서: {len(documents)}개")
    logger.info(f"  - 성공: {success}개")
    logger.info(f"  - 실패: {failed}개")
    logger.info(f"{'='*60}")
    
    # 테스트 검색
    logger.info("\n테스트 검색: 'LTV 규제'")
    results = indexer.search("LTV 규제", size=3)
    for i, result in enumerate(results, 1):
        logger.info(f"\n[{i}] {result['document']['title']}")
        logger.info(f"    점수: {result['score']:.2f}")
        logger.info(f"    키워드: {', '.join(result['document']['keywords'][:5])}")


if __name__ == "__main__":
    main()
