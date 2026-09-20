"""
rag_pipeline.py — the retrieval and generation core.

Exposes:
  Embedder      : text -> vectors (sentence-transformers, TF-IDF fallback)
  RAGPipeline   : question -> (grounded answer, sources)
"""

import json
import os
import pickle

import faiss
import numpy as np
import requests

VECTORSTORE_DIR = "vectorstore"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5                    # how many chunks to feed the LLM
MIN_SCORE = 0.18             # below this, we treat retrieval as "nothing relevant found"

SYSTEM_PROMPT = """You are the IEEE RAS AI Assistant, answering questions about the \
IEEE Robotics and Automation Society.

Rules you must follow:
1. Answer ONLY using the CONTEXT provided below. It comes from official IEEE RAS sources.
2. If the context does not contain the answer, say clearly: "I could not find that in my \
IEEE RAS knowledge base." Then suggest where the user might look on ieee-ras.org. Never \
guess, and never fill gaps from your own general knowledge.
3. Never invent statistics, dates, names, prices or URLs.
4. If a question is vague, answer the most likely interpretation and say which one you chose.
5. Use earlier conversation turns to resolve follow-up questions like "what about students?"
6. Explain technical terms in plain language.
7. Be concise: 2-5 short paragraphs or a short bullet list."""


class Embedder:
    def __init__(self, backend=None):
        self.backend = backend
        self.model = None

        if self.backend in (None, "sentence-transformers"):
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(EMBED_MODEL)
                self.backend = "sentence-transformers"
                return
            except Exception as exc:
                if backend == "sentence-transformers":
                    raise
                print(f"[Embedder] sentence-transformers unavailable ({exc}); using TF-IDF.")

        from sklearn.feature_extraction.text import TfidfVectorizer
        self.backend = "tfidf"
        self.model = TfidfVectorizer(max_features=4096, stop_words="english",
                                     ngram_range=(1, 2))

    def fit_embed(self, texts):
        if self.backend == "sentence-transformers":
            vecs = self.model.encode(texts, show_progress_bar=True,
                                     convert_to_numpy=True, batch_size=32)
        else:
            vecs = self.model.fit_transform(texts).toarray().astype("float32")
        return _normalize(vecs.astype("float32"))

    def embed(self, texts):
        if self.backend == "sentence-transformers":
            vecs = self.model.encode(texts, convert_to_numpy=True)
        else:
            vecs = self.model.transform(texts).toarray().astype("float32")
        return _normalize(vecs.astype("float32"))

    def save(self, directory):
        if self.backend == "tfidf":
            with open(os.path.join(directory, "tfidf.pkl"), "wb") as f:
                pickle.dump(self.model, f)

    @classmethod
    def load(cls, directory, backend):
        obj = cls(backend=backend)
        if backend == "tfidf":
            with open(os.path.join(directory, "tfidf.pkl"), "rb") as f:
                obj.model = pickle.load(f)
        return obj


def _normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return vecs / norms


def build_faiss_index(vectors, path):
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, path)
    return index


class RAGPipeline:
    def __init__(self, api_key, model=None, vectorstore_dir=VECTORSTORE_DIR):
        self.api_key = api_key
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        meta_path = os.path.join(vectorstore_dir, "chunks.json")
        index_path = os.path.join(vectorstore_dir, "index.faiss")
        if not (os.path.exists(meta_path) and os.path.exists(index_path)):
            raise FileNotFoundError(
                "Vector store not found. Run `python ingest.py` and commit the "
                "vectorstore/ folder."
            )

        meta = json.load(open(meta_path, encoding="utf-8"))
        self.chunks = meta["chunks"]
        self.index = faiss.read_index(index_path)
        self.embedder = Embedder.load(vectorstore_dir, meta["backend"])

    def retrieve(self, query, top_k=TOP_K):
        qvec = self.embedder.embed([query])
        scores, idxs = self.index.search(qvec, top_k)

        hits = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            hits.append(chunk)
        return hits

    def answer(self, question, history=None):
        hits = self.retrieve(question)
        strong = [h for h in hits if h["score"] >= MIN_SCORE]

        if not strong:
            return (
                "I could not find anything relevant to that in my IEEE RAS knowledge "
                "base, so I won't guess. Try rephrasing, or check the official site at "
                "https://www.ieee-ras.org/.",
                [],
            )

        context = "\n\n".join(
            f"[{i+1}] Source: {h['title']} ({h['url']})\n{h['text']}"
            for i, h in enumerate(strong)
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in (history or [])[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({
            "role": "user",
            "content": f"CONTEXT FROM IEEE RAS SOURCES:\n{context}\n\nQUESTION: {question}",
        })

        answer = self._call_llm(messages)

        sources, seen = [], set()
        for h in strong:
            if h["url"] not in seen:
                seen.add(h["url"])
                sources.append({"title": h["title"], "url": h["url"],
                                "score": round(h["score"], 3)})
        return answer, sources

    def _call_llm(self, messages):
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages,
                  "temperature": 0.2, "max_tokens": 900},
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"LLM API error {resp.status_code}: {resp.text[:300]}")
        return resp.json()["choices"][0]["message"]["content"].strip()