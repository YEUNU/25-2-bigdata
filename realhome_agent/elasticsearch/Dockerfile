# =============================================================================
# ElasticSearch with Nori (한국어 형태소 분석기) Plugin
# =============================================================================
FROM docker.elastic.co/elasticsearch/elasticsearch:8.11.0

# Nori 플러그인 설치 (한국어 형태소 분석)
RUN elasticsearch-plugin install analysis-nori

# 기본 설정
ENV discovery.type=single-node
ENV xpack.security.enabled=false
