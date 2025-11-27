# LangChain RAG example (FAISS + Sentence-Transformers)

This example shows how to build a simple Retrieval-Augmented Generation (RAG) pipeline using LangChain:

- Build embeddings with `google/embeddinggemma-300m` (local HF embedding model)
- Index with FAISS
- Retrieve top-k documents from the vector DB (no LLM required)

Quick start
-----------

1. Install dependencies (recommended in a virtual env)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Build the FAISS index from the CSV

```bash
python build_index.py --csv ../리뷰_구조화_결과.csv --out ./vectorstore
```

This will create `./vectorstore` folder with saved index and metadata.

3. Retrieve nearest docs (no LLM)

If you only want the vector DB and to inspect the top-k nearest chunks, use `retrieve.py`:

```bash
python retrieve.py --index ./vectorstore --k 4 --query "역세권이 좋은 집을 추천해줘"
```

(`query_rag.py` still exists for those who want to experiment with local LLMs and RAG, but this example focuses on building the vector DB.)

Notes
-----

- The default embedding model is `google/embeddinggemma-300m` (suggested local embedding model). You can override the local model by setting the `LOCAL_EMBEDDING_MODEL` env var:

```bash
export LOCAL_EMBEDDING_MODEL=google/embeddinggemma-300m
```
- This demo only uses local LLMs — set `LOCAL_LLM_MODEL` to the desired HF model name (default: `google/flan-t5-small`).
- The `query_rag.py` shows sources and a final generated answer.

Troubleshooting
---------------

- FAISS build errors often happen if vector dimensions mismatch — re-run `build_index.py` to rebuild.
- If the local HF model is slow or OOM, prefer OpenAI or a smaller model (flan-t5-small), or run on GPU.

If you want an alternative to FAISS, consider `Chroma` or `Weaviate`.
