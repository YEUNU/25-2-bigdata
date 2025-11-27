"""
Build an embeddings index from CSV and save as FAISS vectorstore.

Usage:
    pip install -r requirements.txt
    export OPENAI_API_KEY=...
    python build_index.py --csv ../리뷰_구조화_결과.csv --out ./vectorstore

This script uses sentence-transformers all-MiniLM-L6-v2 for embeddings (local) and FAISS for vector store.
"""
import argparse
import os
from pathlib import Path
import pandas as pd

# [MODIFIED] 최신 LangChain 구조에 맞게 import 경로 수정
# langchain-text-splitters는 v0.1부터 분리된 패키지 사용 (Good)
from langchain_text_splitters import CharacterTextSplitter
# [FIXED] langchain.embeddings는 deprecated 되었습니다. langchain_huggingface를 사용해야 합니다.
from langchain_huggingface import HuggingFaceEmbeddings
# langchain_community는 v0.1부터 커뮤니티 통합을 담당 (Good)
from langchain_community.vectorstores import FAISS
# langchain_core는 LangChain의 핵심 추상화 계층 (Good)
from langchain_core.documents import Document


def load_csv_as_documents(csv_path: str, text_columns: list, meta_columns: list | None = None):
    # on_bad_lines='skip': 파싱 오류가 있는 행은 건너뜀
    df = pd.read_csv(csv_path, dtype=str, on_bad_lines='skip').fillna("")
    print(f"  -> Loaded {len(df)} rows (bad lines skipped)")

    docs: list[Document] = []
    for idx, row in df.iterrows():
        # Join the chosen text columns with safe string conversion
        parts = []
        for col in text_columns:
            if col in row.index:
                val = row.get(col, "")
                s = str(val).strip()
                if s:
                    parts.append(s)
        content = "\n\n".join(parts)
        if not content.strip():
            continue
        metadata = {c: row[c] for c in meta_columns if c in row.index} if meta_columns else {}
        metadata["row_idx"] = int(idx)
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Input CSV path")
    parser.add_argument("--out", type=str, default="./vectorstore", help="Output folder to save vector store")
    args = parser.parse_args()

    csv_path = args.csv
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose columns to join as text
    text_columns = ["Pros", "Cons"]
    meta_columns = ["kaptName", "doroJuso", "Score"]

    print(f"Loading CSV: {csv_path}")
    docs = load_csv_as_documents(csv_path, text_columns, meta_columns)
    print(f"Loaded {len(docs)} documents from CSV")

    # Optional: split long documents into smaller chunks for better retrieval
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splitted_docs = []
    for d in docs:
        chunks = text_splitter.split_text(d.page_content)
        for i, chunk in enumerate(chunks):
            md = dict(d.metadata)
            md["chunk_id"] = i
            splitted_docs.append(Document(page_content=chunk, metadata=md))

    print(f"Created {len(splitted_docs)} chunks after splitting")

    # Create embeddings using Hugging Face embeddings model
    # Use the model requested by the user (LOCAL_EMBEDDING_MODEL env var override)
    emb_model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "google/embeddinggemma-300m")
    print(f"Creating embeddings: {emb_model_name}")
    
    # langchain_huggingface의 HuggingFaceEmbeddings를 사용합니다.
    embeddings = HuggingFaceEmbeddings(model_name=emb_model_name)

    # Build FAISS vector store
    print("Building FAISS index (this may take a while)")
    vectorstore = FAISS.from_documents(splitted_docs, embeddings)

    # Save vector store locally
    print(f"Saving vectorstore to: {out_dir}")
    vectorstore.save_local(str(out_dir))
    print("Done")


if __name__ == "__main__":
    main()