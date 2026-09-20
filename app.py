"""app.py — IEEE RAS AI Assistant (Streamlit frontend)."""

import os

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import RAGPipeline

load_dotenv()

st.set_page_config(page_title="IEEE RAS AI Assistant", page_icon="🤖",
                   layout="centered", initial_sidebar_state="expanded")

# ------------------------------------------------------------------- styling
st.markdown("""
<style>
  .block-container { padding-top: 2.2rem; max-width: 880px; }
  .ras-header {
      background: linear-gradient(120deg, #00629B 0%, #003B5C 100%);
      padding: 1.6rem 1.8rem; border-radius: 14px; color: #fff;
      margin-bottom: 1.2rem;
  }
  .ras-header h1 { margin: 0; font-size: 1.85rem; font-weight: 700; color: #fff; }
  .ras-header p  { margin: .45rem 0 0; opacity: .9; font-size: .95rem; }
  .ras-badge {
      display:inline-block; background: rgba(255,255,255,.16); color:#fff;
      padding:.18rem .6rem; border-radius:999px; font-size:.72rem;
      margin-right:.4rem; margin-top:.7rem;
  }
  .src-item { font-size:.85rem; padding:.3rem 0; border-bottom:1px solid #eee; }
  .stChatMessage { border-radius: 12px; }
  footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ras-header">
  <h1>🤖 IEEE RAS AI Assistant</h1>
  <p>A retrieval-augmented chatbot answering questions about the IEEE Robotics and
     Automation Society, grounded in text retrieved from official public IEEE RAS
     sources — with citations on every answer.</p>
  <span class="ras-badge">FAISS vector search</span>
  <span class="ras-badge">MiniLM embeddings</span>
  <span class="ras-badge">Groq LLM</span>
</div>
""", unsafe_allow_html=True)

EXAMPLE_QUESTIONS = [
    "What is the IEEE Robotics and Automation Society?",
    "When was IEEE RAS founded and how did it start?",
    "What is the field of interest of IEEE RAS?",
    "Which journals does IEEE RAS publish?",
    "What are ICRA and IROS?",
    "What are IEEE RAS Technical Committees?",
    "How do I become a member of IEEE RAS?",
    "What benefits do student members get?",
    "What educational resources does IEEE RAS offer?",
    "What awards does IEEE RAS give out?",
    "What is an IEEE RAS chapter?",
    "How does IEEE RAS support industry engagement?",
]


# --------------------------------------------------------------- resources
def get_api_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY")


@st.cache_resource(show_spinner="Loading the IEEE RAS knowledge base…")
def load_pipeline(api_key):
    return RAGPipeline(api_key=api_key)


api_key = get_api_key()
if not api_key:
    st.error("**No API key found.** Add `GROQ_API_KEY` to your `.env` file locally, "
             "or to *Settings → Secrets* on Streamlit Cloud. Free keys: "
             "https://console.groq.com/keys")
    st.stop()

try:
    rag = load_pipeline(api_key)
except FileNotFoundError as exc:
    st.error(f"**Knowledge base missing.** {exc}")
    st.stop()
except Exception as exc:
    st.error(f"**Could not start the assistant:** {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("About")
    st.caption(
        "Every answer is generated from text chunks retrieved out of a FAISS vector "
        "index built from public IEEE RAS pages. If the index has no relevant text, "
        "the assistant says so instead of guessing."
    )
    st.metric("Indexed chunks", len(rag.chunks))
    st.caption(f"Embeddings: `{rag.embedder.backend}` · LLM: `{rag.model}`")

    st.divider()
    st.subheader("Example questions")
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if st.button(q, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = q
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear chat", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()

    st.caption("Unofficial student project. Not affiliated with IEEE. "
               "Always verify on [ieee-ras.org](https://www.ieee-ras.org/).")


# ------------------------------------------------------------ chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "🧑"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(
                        f"<div class='src-item'>🔗 <a href='{s['url']}' target='_blank'>"
                        f"{s['title']}</a> &nbsp;<code>similarity {s['score']}</code></div>",
                        unsafe_allow_html=True)

if not st.session_state.messages:
    st.info("👋 Ask me anything about IEEE RAS — or pick an example question "
            "from the sidebar to get started.")


# -------------------------------------------------------------- user input
typed = st.chat_input("Ask about IEEE RAS…")
question = typed or st.session_state.pending
st.session_state.pending = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        try:
            with st.spinner("Searching IEEE RAS sources…"):
                history = [{"role": m["role"], "content": m["content"]}
                           for m in st.session_state.messages[:-1]]
                answer, sources = rag.answer(question, history)

            st.markdown(answer)
            if sources:
                with st.expander(f"📚 Sources ({len(sources)})", expanded=True):
                    for s in sources:
                        st.markdown(
                            f"<div class='src-item'>🔗 <a href='{s['url']}' target='_blank'>"
                            f"{s['title']}</a> &nbsp;<code>similarity {s['score']}</code></div>",
                            unsafe_allow_html=True)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources})

        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
            st.caption("Check your API key and internet connection, then try again.")