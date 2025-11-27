"""
Retrieve nearest documents from FAISS vectorstore without using an LLM.

Usage:
    python retrieve.py --index ./vectorstore --k 4 --query "역세권이 좋은 곳이 어디야?"

This script loads the FAISS index created by `build_index.py`, runs a similarity search and prints the results
(with metadata and similarity scores).
"""
import argparse
import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_vectorstore(path: str):
    emb_model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "google/embeddinggemma-300m")
    embeddings = HuggingFaceEmbeddings(model_name=emb_model_name)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, required=True, help="Path to saved FAISS index folder")
    parser.add_argument("--k", type=int, default=4, help="Top-K docs to retrieve")
    parser.add_argument("--query", type=str, required=True, help="Query text")
    args = parser.parse_args()

    index_path = Path(args.index)
    if not index_path.exists():
        raise FileNotFoundError(f"Index folder not found: {index_path}")

    vectorstore = load_vectorstore(str(index_path))

    # Use similarity_search_with_score to return scores and docs without LLM
    docs_and_scores = vectorstore.similarity_search_with_score(args.query, k=args.k)

    print("Nearest documents (without LLM):\n")
    for i, (doc, score) in enumerate(docs_and_scores):
        text_snippet = doc.page_content[:250].replace("\n", " ")
        print(f"[{i}] score: {score:.5f} | metadata: {doc.metadata}\n{text_snippet}...\n\n")


if __name__ == "__main__":
    main()
