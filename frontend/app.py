import streamlit as st
import requests
import os
import uuid

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Autonomous Energy Researcher Agent",
    layout="centered",
    page_icon="⚡"
)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e0f7fa, #fce4ec);
    min-height: 100vh;
}
.block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
    max-width: 740px;
}
h1 {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 700;
    color: #1a237e;
    margin-bottom: 0.3rem;
}
.tagline {
    text-align: center;
    font-size: 1.05rem;
    color: #37474f;
    margin-bottom: 2rem;
}
.card {
    background: white;
    padding: 28px;
    border-radius: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
}
.section-label {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #78909c;
    margin-bottom: 0.5rem;
}
.stTextArea textarea {
    border-radius: 14px;
    border: 2px solid #b2dfdb;
    padding: 14px;
    font-size: 16px;
}
.stButton>button {
    background: linear-gradient(90deg, #26c6da, #ec407a);
    color: white;
    font-weight: 600;
    border-radius: 14px;
    height: 3em;
    border: none;
    width: 100%;
    transition: 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.03);
    opacity: 0.9;
}
.result-box {
    margin-top: 20px;
    padding: 20px;
    background-color: #ffffff;
    border-radius: 15px;
    border: 1px solid #eeeeee;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
}
.upload-success {
    background: #e8f5e9;
    border: 1px solid #a5d6a7;
    border-radius: 12px;
    padding: 12px 16px;
    color: #2e7d32;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>⚡ Autonomous Energy Researcher Agent</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='tagline'>Save Energy. Think Smart. Build Sustainable Futures.</div>",
    unsafe_allow_html=True
)

# ── Document Upload ──────────────────────────────────────────────────────────
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-label'>📄 Upload Reference Documents (RAG)</div>", unsafe_allow_html=True)
st.caption("Upload energy sector PDFs to enrich research with your own document context.")

uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"], key="doc_upload")

if uploaded_file and st.button("Index Document", key="btn_upload"):
    with st.spinner("Indexing document into vector store..."):
        resp = requests.post(
            f"{API_URL}/upload",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
            data={"project": "energy-intelligence"},
        )
        if resp.status_code == 200:
            data = resp.json()
            st.markdown(
                f"<div class='upload-success'>✅ <strong>{uploaded_file.name}</strong> indexed — "
                f"{data['chunk_count']} chunks stored.</div>",
                unsafe_allow_html=True
            )
        else:
            st.error(f"Upload failed: {resp.text}")

st.markdown("</div>", unsafe_allow_html=True)

# ── PDF Q&A ──────────────────────────────────────────────────────────────────
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-label'>💬 Ask Your PDF</div>", unsafe_allow_html=True)
st.caption("Ask questions and get answers directly from your uploaded document.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + ", ".join(
                f"{s['source']} chunk {s['chunk_index']}"
                for s in msg["sources"] if s.get("source")
            ))

question = st.chat_input("Ask a question about your uploaded PDF...")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching document..."):
            try:
                resp = requests.post(
                    f"{API_URL}/rag/query",
                    data={
                        "question": question,
                        "project": "energy-intelligence",
                        "top_k": 5,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "No answer found.")
                    sources = data.get("sources", [])
                    st.write(answer)
                    if sources:
                        st.caption("Sources: " + ", ".join(
                            f"{s['source']} chunk {s['chunk_index']}"
                            for s in sources if s.get("source")
                        ))
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                else:
                    st.error(f"Error: {resp.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

st.markdown("</div>", unsafe_allow_html=True)


# ── Research Query ───────────────────────────────────────────────────────────
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-label'>🔬 Research Query</div>", unsafe_allow_html=True)

query = st.text_area("Enter your energy research topic", height=130, key="query_input")

if st.button("Generate Insight", key="btn_research"):
    if not query.strip():
        st.warning("Please enter a topic first.")
    else:
        with st.spinner("Analyzing global energy signals..."):
            try:
                response = requests.post(
                    f"{API_URL}/research",
                    json={
                        "query": query,
                        "thread_id": st.session_state.thread_id,
                    },
                )
                if response.status_code != 200:
                    st.error(f"Backend Error {response.status_code}")
                    st.write(response.text)
                else:
                    data = response.json()
                    st.markdown(
                        f"""
                        <div class="result-box">
                        <h3>📄 Energy Intelligence Report</h3>
                        <p><strong>Topic:</strong> {data.get("query", query)}</p>
                        <hr>
                        <p>{data.get("result", "No result returned.")}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    if data.get("suggestions"):
                        st.markdown("### 💡 Explore Further")
                        for s in data["suggestions"]:
                            st.markdown(f"- {s}")
            except Exception as e:
                st.error(f"Connection error: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# ── Recent History ───────────────────────────────────────────────────────────
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-label'>🕘 Recent History</div>", unsafe_allow_html=True)

if st.button("Load History", key="btn_history"):
    resp = requests.get(f"{API_URL}/history")
    if resp.status_code == 200:
        entries = resp.json()
        if not entries:
            st.info("No history yet.")
        for entry in entries:
            with st.expander(f"🔍 {entry['query'][:60]}"):
                text = entry["result"]
                st.write(text[:500] + "..." if len(text) > 500 else text)
    else:
        st.error("Could not load history.")

st.markdown("</div>", unsafe_allow_html=True)