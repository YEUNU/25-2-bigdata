#!/usr/bin/env bash

# Run these steps from project root
# 1) Install deps
# python -m venv .venv && source .venv/bin/activate
# pip install -r rag_example/requirements.txt

# 2) Build index
python rag_example/build_index.py --csv 리뷰_구조화_결과.csv --out rag_example/vectorstore

# 3) Query (local HF LLM)
export LOCAL_LLM_MODEL=google/flan-t5-small
python rag_example/query_rag.py --index rag_example/vectorstore --k 4 --query "역세권이 좋은 집을 추천해줘"

# 4) Another example query (local LLM)
python rag_example/query_rag.py --index rag_example/vectorstore --k 3 --query "장점이 많은 아파트를 알려줘"
