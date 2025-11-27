"""
Query an existing FAISS vectorstore with LangChain RetrievalQA.

Only local HuggingFace LLM is supported; OpenAI is not used.

Usage:
    pip install -r requirements.txt
    export LOCAL_LLM_MODEL=google/flan-t5-small
    python query_rag.py --index ./vectorstore --k 4 --query "역세권이 좋은 곳이 어디야?"
"""
import argparse
import os
from pathlib import Path

# [MODIFIED] 최신 LangChain 구조에 맞게 import 경로 수정
# - Embeddings와 LLM 래퍼는 langchain_huggingface로 이동
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
# - VectorStore는 langchain_community로 이동
from langchain_community.vectorstores import FAISS
# - Chain은 유지 (RetrievalQA는 여전히 유효하지만, 추후 create_retrieval_chain 사용 권장)
from langchain.chains import RetrievalQA


def load_vectorstore(path: str):
    emb_model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "google/embeddinggemma-300m")
    print(f"Loading Embeddings: {emb_model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=emb_model_name)
    
    # [CRITICAL UPDATE] 최신 FAISS는 로컬 파일 로드 시 보안을 위해 allow_dangerous_deserialization=True가 필요함
    # 신뢰할 수 있는(직접 만든) 인덱스이므로 True로 설정합니다.
    import pandas as pd
    return FAISS.load_local(
        path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )


def build_llm():
    # Use local Hugging Face text-to-text model only
    print("Using local Hugging Face LLM (LOCAL_LLM_MODEL or default google/flan-t5-small)")
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        model_name = os.getenv("LOCAL_LLM_MODEL", "google/flan-t5-small")
        
        # GPU 가용성 확인 로직 개선
        if torch.cuda.is_available():
            device = 0 
        elif torch.backends.mps.is_available(): # Mac M1/M2/M3 지원
            device = "mps"
        else:
            device = -1
            
        print(f"Loading local model: {model_name} on device {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        gen_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            min_length=10,
            max_length=256,
            do_sample=False,
        )
        
        # [MODIFIED] langchain_huggingface의 래퍼 클래스 사용
        llm = HuggingFacePipeline(pipeline=gen_pipeline)
        return llm
    except Exception as e:
        raise RuntimeError("Failed to create a local HF LLM. Please install required HF packages") from e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, required=True, help="Path to saved FAISS index folder")
    parser.add_argument("--k", type=int, default=4, help="Top-K docs to retrieve for RAG")
    parser.add_argument("--query", type=str, required=True, help="Query text")
    args = parser.parse_args()

    index_path = Path(args.index)
    if not index_path.exists():
        raise FileNotFoundError(f"Index folder not found: {index_path}")

    # 1. VectorStore 로드
    vectorstore = load_vectorstore(str(index_path))
    retriever = vectorstore.as_retriever(search_kwargs={"k": args.k})

    # 2. LLM 빌드
    llm = build_llm()

    # 3. QA Chain 생성
    chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True
    )

    print("Running retrieval and generation...")
    # [MODIFIED] chain() 직접 호출은 deprecated 되었습니다. .invoke()를 사용해야 합니다.
    result = chain.invoke({"query": args.query})

    # 결과 파싱 로직 유지
    if isinstance(result, dict):
        answer = result.get("result") or result.get("answer")
        docs = result.get("source_documents") or result.get("source_documents", [])
    else:
        answer = result
        docs = []

    print("\n--- Answer ---\n")
    print(answer)
    print("\n--- Sources ---\n")
    for i, d in enumerate(docs[:5]):
        text_snippet = d.page_content[:200].replace('\n', ' ')
        print(f"[{i}] metadata: {d.metadata} | text: {text_snippet}...\n")


if __name__ == "__main__":
    main()