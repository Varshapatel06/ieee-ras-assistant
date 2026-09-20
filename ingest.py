"""
ingest.py — builds the IEEE RAS knowledge base.

Pipeline:
  1. Crawl public IEEE RAS pages (polite, depth-limited, same-domain only)
  2. Strip HTML down to clean readable text
  3. Load any local .txt documents in data/documents/
  4. Split everything into overlapping chunks
  5. Embed each chunk with sentence-transformers
  6. Store vectors in a FAISS index + chunk metadata in JSON

Run this ONCE on your own machine, then commit vectorstore/ to GitHub.
    python ingest.py
"""

import json
import os
import re
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from rag_pipeline import Embedder, build_faiss_index, VECTORSTORE_DIR

# ---------------------------------------------------------------- config
START_URLS = [
    "https://www.ieee-ras.org/",
    "https://www.ieee-ras.org/about-ras",
    "https://www.ieee-ras.org/membership",
    "https://www.ieee-ras.org/publications",
    "https://www.ieee-ras.org/conferences-workshops",
    "https://www.ieee-ras.org/technical-committees",
    "https://www.ieee-ras.org/educational-resources",
    "https://www.ieee-ras.org/awards-recognition",
    "https://www.ieee-ras.org/chapters",
    "https://www.ieee-ras.org/industry",
]
ALLOWED_DOMAIN = "ieee-ras.org"
MAX_PAGES = 45          # keep the crawl small so ingestion finishes in ~2 minutes
CRAWL_DEPTH = 2
REQUEST_DELAY = 0.7     # be polite to the server
TIMEOUT = 20
HEADERS = {"User-Agent": "IEEE-RAS-Student-RAG-Project/1.0 (educational use)"}

CHUNK_SIZE = 900        # characters — big enough for a full idea, small enough to stay precise
CHUNK_OVERLAP = 150     # overlap stops a fact being cut in half at a boundary

DOCS_DIR = os.path.join("data", "documents")


# ------------------------------------------------------------ 2. cleaning
def html_to_text(html: str):
    """Return (title, clean_text) from raw HTML."""
    soup = BeautifulSoup(html, "lxml")

    # Remove everything that is navigation or noise, not content.
    for tag in soup(["script", "style", "nav", "footer", "header", "form",
                     "noscript", "iframe", "svg"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""
    text = soup.get_text(separator="\n")
    return title, clean_text(text)


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    # Drop menu fragments and stray single words
    lines = [ln for ln in lines if len(ln) > 2]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def links_from(html: str, base_url: str):
    soup = BeautifulSoup(html, "lxml")
    found = []
    for a in soup.find_all("a", href=True):
        url = urljoin(base_url, a["href"]).split("#")[0].rstrip("/")
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            continue
        if ALLOWED_DOMAIN not in parsed.netloc:
            continue
        if re.search(r"\.(pdf|jpg|jpeg|png|gif|zip|docx?|pptx?)$", url, re.I):
            continue
        found.append(url)
    return found


# ------------------------------------------------------------- 1. crawling
def crawl():
    """Breadth-first crawl of the public site. Failures are skipped, never fatal."""
    seen, queue, pages = set(), [(u, 0) for u in START_URLS], []

    while queue and len(pages) < MAX_PAGES:
        url, depth = queue.pop(0)
        key = url.rstrip("/")
        if key in seen:
            continue
        seen.add(key)

        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
                print(f"  skip  [{resp.status_code}] {url}")
                continue
        except Exception as exc:
            print(f"  fail  {url} -> {exc}")
            continue

        title, text = html_to_text(resp.text)
        if len(text) > 400:                       # ignore near-empty pages
            pages.append({"url": url, "title": title or url, "text": text})
            print(f"  ok    ({len(text):>6} chars) {url}")

        if depth < CRAWL_DEPTH:
            for link in links_from(resp.text, url):
                if link.rstrip("/") not in seen:
                    queue.append((link, depth + 1))

        time.sleep(REQUEST_DELAY)

    return pages


# --------------------------------------------------- 3. local seed documents
def load_local_docs():
    """Load .txt files from data/documents/. 'SOURCE:' / 'TITLE:' headers set citations."""
    docs = []
    if not os.path.isdir(DOCS_DIR):
        return docs

    for name in sorted(os.listdir(DOCS_DIR)):
        if not name.lower().endswith(".txt"):
            continue
        path = os.path.join(DOCS_DIR, name)
        raw = open(path, encoding="utf-8").read()

        # Split the file on SOURCE: markers so each section keeps its own URL
        blocks = re.split(r"\nSOURCE:\s*", "\n" + raw)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            lines = block.split("\n")
            url = lines[0].strip() if lines[0].startswith("http") else name
            body = "\n".join(lines[1:])
            title_match = re.search(r"TITLE:\s*(.+)", body)
            title = title_match.group(1).strip() if title_match else name
            body = re.sub(r"TITLE:\s*.+", "", body)
            body = clean_text(body)
            if len(body) > 100:
                docs.append({"url": url, "title": title, "text": body})
                print(f"  local ({len(body):>6} chars) {title}")
    return docs


# ------------------------------------------------------------- 4. chunking
def chunk_document(doc):
    """Split on paragraph boundaries, packing into ~CHUNK_SIZE windows with overlap."""
    text = doc["text"]
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks, buffer = [], ""
    for para in paragraphs:
        if len(buffer) + len(para) + 1 <= CHUNK_SIZE:
            buffer += para + "\n"
        else:
            if buffer.strip():
                chunks.append(buffer.strip())
            buffer = (buffer[-CHUNK_OVERLAP:] if len(buffer) > CHUNK_OVERLAP else "") + para + "\n"
    if buffer.strip():
        chunks.append(buffer.strip())

    return [
        {"text": c, "url": doc["url"], "title": doc["title"]}
        for c in chunks
        if len(c) > 120                           # discard scraps
    ]


# ------------------------------------------------------------------- main
def main():
    print("\n[1/5] Crawling public IEEE RAS pages ...")
    pages = crawl()
    print(f"      -> {len(pages)} pages retrieved")

    print("\n[2/5] Loading local verified documents ...")
    pages += load_local_docs()

    if not pages:
        raise SystemExit("No documents collected. Check your internet connection.")

    print("\n[3/5] Chunking ...")
    chunks = []
    for doc in pages:
        chunks.extend(chunk_document(doc))

    # Drop exact duplicates (site templates repeat text across pages)
    unique, seen_text = [], set()
    for c in chunks:
        sig = c["text"][:200]
        if sig not in seen_text:
            seen_text.add(sig)
            unique.append(c)
    chunks = unique
    print(f"      -> {len(chunks)} unique chunks")

    print("\n[4/5] Generating embeddings ...")
    embedder = Embedder()
    vectors = embedder.fit_embed([c["text"] for c in chunks])
    print(f"      -> vectors of shape {vectors.shape} (backend: {embedder.backend})")

    print("\n[5/5] Building FAISS index ...")
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    build_faiss_index(vectors, os.path.join(VECTORSTORE_DIR, "index.faiss"))

    with open(os.path.join(VECTORSTORE_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump({"backend": embedder.backend, "chunks": chunks}, f, ensure_ascii=False)
    embedder.save(VECTORSTORE_DIR)

    print(f"\nDone. Knowledge base written to {VECTORSTORE_DIR}/")
    print("Commit that folder to GitHub so the deployed app can load it.\n")


if __name__ == "__main__":
    main()