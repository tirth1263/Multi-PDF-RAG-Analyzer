"""
Chat with Multiple PDFs — a finance-aware RAG analyst.

Upload annual reports / financial statements, build a FAISS vector index from
them with Google's embedding models, and interrogate them with Gemini.
"""

import html
import os
from datetime import datetime

import markdown as md
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

FAISS_INDEX_DIR = "faiss_index"

# gemini-1.5-flash is kept first for parity with the original project, but newer
# API keys often no longer have access to the 1.5 family — hence the picker.
CHAT_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
]

EMBEDDING_MODELS = [
    "models/text-embedding-004",
    "models/embedding-001",
]

PROMPT_TEMPLATE = """
You are a meticulous financial analyst reviewing corporate disclosures such as
annual reports, quarterly results and financial statements of listed companies
(with an emphasis on the Indian stock market).

Answer the question as completely as possible using ONLY the context provided
below. The context is extracted from PDF documents the user uploaded.

Ground rules:
1. Never invent a number. Every figure you quote must appear in the context.
2. If the answer is not in the context, say exactly: "The answer is not available
   in the provided documents." Do not guess and do not fill gaps from memory.
3. Quote figures with their units and reporting period (e.g. "Rs. 1,240 crore,
   FY2023") and name the company whenever more than one is in scope.
4. Where the question is analytical, show the arithmetic you used so the reader
   can verify it.

When the question touches any of the areas below, be especially rigorous:
- Financial health: revenue, margins, debt-to-equity, interest coverage,
  working capital, and the trend across the periods available.
- Red flags: cash flow from operations diverging from reported net profit,
  ballooning receivables or inventory, frequent auditor or CFO changes,
  qualified audit opinions, contingent liabilities, and pledged promoter shares.
- Related party transactions: identify the parties, the amounts, the nature of
  the dealing, and flag anything that looks circular, unusually large, or poorly
  explained.
- Managerial remuneration: track Key Managerial Personnel (KMP) pay, its growth
  rate, and how it compares with profit growth and median employee pay.

Present the answer in clean markdown. Use a table when comparing periods or
companies, and finish with a short "Key takeaways" list when the question is
analytical.

Context:
{context}

Question:
{question}

Answer:
"""

CHAT_CSS = """
<style>
.chat-row {
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    padding: 1rem 1.1rem;
    border-radius: 0.85rem;
    margin-bottom: 0.85rem;
    border: 1px solid rgba(128, 128, 128, 0.18);
}
.chat-row.user   { background: rgba(99, 102, 241, 0.10); }
.chat-row.bot    { background: rgba(16, 185, 129, 0.10); }
.chat-avatar {
    flex: 0 0 42px;
    width: 42px;
    height: 42px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.35rem;
    background: rgba(128, 128, 128, 0.16);
}
.chat-body { flex: 1 1 auto; min-width: 0; }
.chat-name {
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.65;
    margin-bottom: 0.3rem;
}
.chat-text { line-height: 1.6; word-wrap: break-word; }
.chat-text table { width: 100%; border-collapse: collapse; margin: 0.6rem 0; }
.chat-text th, .chat-text td {
    border: 1px solid rgba(128, 128, 128, 0.3);
    padding: 0.4rem 0.6rem;
    text-align: left;
}
</style>
"""


def get_pdf_text(pdf_docs):
    """Concatenate the text of every page of every uploaded PDF.

    Each page is prefixed with its source so the model can cite the document it
    came from — essential when several annual reports are loaded at once.
    """
    text = ""
    stats = []
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        pages_with_text = 0
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages_with_text += 1
                text += f"\n\n[Source: {pdf.name} | Page {page_number}]\n{page_text}"
        stats.append(
            {
                "File": pdf.name,
                "Pages": len(reader.pages),
                "Pages with text": pages_with_text,
            }
        )
    return text, stats


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)


def get_vector_store(chunks, api_key, embedding_model):
    embeddings = GoogleGenerativeAIEmbeddings(
        model=embedding_model, google_api_key=api_key
    )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(FAISS_INDEX_DIR)
    return vector_store


def load_vector_store(api_key, embedding_model):
    """Return the in-session index, falling back to one persisted on disk."""
    if st.session_state.get("vector_store") is not None:
        return st.session_state.vector_store
    if os.path.isdir(FAISS_INDEX_DIR):
        embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model, google_api_key=api_key
        )
        return FAISS.load_local(
            FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
    return None


def get_conversational_chain(api_key, chat_model, temperature):
    """Build the LCEL chain: prompt -> Gemini -> plain string.

    Composed from langchain_core primitives only, so it does not depend on the
    legacy chain helpers that the LangChain 1.x rewrite removed.
    """
    model = ChatGoogleGenerativeAI(
        model=chat_model, temperature=temperature, google_api_key=api_key
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt | model | StrOutputParser()


def format_docs(docs):
    """Flatten retrieved chunks into the prompt's context block.

    Each chunk already carries its own [Source: file | Page N] header from
    extraction, so the separator just keeps them visually distinct.
    """
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def answer_question(question, api_key, chat_model, embedding_model, temperature, k):
    vector_store = load_vector_store(api_key, embedding_model)
    if vector_store is None:
        return None, "No documents indexed yet. Upload PDFs and click **Process Documents** first."

    docs = vector_store.similarity_search(question, k=k)
    chain = get_conversational_chain(api_key, chat_model, temperature)
    answer = chain.invoke({"context": format_docs(docs), "question": question})
    # langchain-core 1.x returns a str subclass here; normalise it so the CSV
    # export and markdown renderer always receive a plain string.
    return str(answer), None


def render_message(role, content):
    """Draw one chat bubble.

    The model replies in markdown, but markdown nested inside a raw HTML block
    is not parsed by Streamlit — so it is converted to HTML here, which also
    lets the bubble's CSS style the generated tables.
    """
    avatar = "🧑‍💼" if role == "user" else "🤖"
    name = "You" if role == "user" else "Gemini Analyst"
    if role == "user":
        body = f"<p>{html.escape(content)}</p>"
    else:
        body = md.markdown(content, extensions=["tables", "fenced_code", "sane_lists"])

    st.markdown(
        f"""
        <div class="chat-row {role}">
            <div class="chat-avatar">{avatar}</div>
            <div class="chat-body">
                <div class="chat-name">{name}</div>
                <div class="chat-text">{body}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Chat with Multiple PDFs",
        page_icon="📄",
        layout="wide",
    )
    st.markdown(CHAT_CSS, unsafe_allow_html=True)

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("vector_store", None)
    st.session_state.setdefault("doc_stats", [])

    st.title("📄 Chat with Multiple PDFs")
    st.caption(
        "Upload annual reports and financial statements, then ask Gemini to "
        "analyse them — ratios, red flags, related party transactions and KMP pay."
    )

    with st.sidebar:
        st.header("⚙️ Configuration")

        api_key = st.text_input(
            "Google AI API Key",
            type="password",
            help="Create one for free at https://ai.google.dev/",
            value=os.getenv("GOOGLE_API_KEY", ""),
        )
        st.markdown(
            "[Get an API key →](https://aistudio.google.com/app/apikey)",
            unsafe_allow_html=True,
        )

        chat_model = st.selectbox("Chat model", CHAT_MODELS, index=0)
        embedding_model = st.selectbox("Embedding model", EMBEDDING_MODELS, index=0)

        with st.expander("Advanced"):
            temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
            k = st.slider("Chunks retrieved per question", 2, 12, 5)

        st.divider()
        st.header("📄 Your documents")
        pdf_docs = st.file_uploader(
            "Upload PDFs", type="pdf", accept_multiple_files=True
        )

        if st.button("Process Documents", type="primary", use_container_width=True):
            if not api_key:
                st.error("Enter your Google AI API key first.")
            elif not pdf_docs:
                st.error("Upload at least one PDF.")
            else:
                try:
                    with st.spinner("Extracting text…"):
                        raw_text, stats = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error(
                            "No selectable text found. These PDFs are likely "
                            "scanned images and would need OCR first."
                        )
                    else:
                        with st.spinner("Chunking and embedding…"):
                            chunks = get_text_chunks(raw_text)
                            st.session_state.vector_store = get_vector_store(
                                chunks, api_key, embedding_model
                            )
                        st.session_state.doc_stats = stats
                        st.success(
                            f"Indexed {len(chunks)} chunks from {len(pdf_docs)} document(s)."
                        )
                except Exception as exc:  # surface API/parse errors in the UI
                    st.error(f"Processing failed: {exc}")

        if st.session_state.doc_stats:
            st.dataframe(
                pd.DataFrame(st.session_state.doc_stats),
                hide_index=True,
                use_container_width=True,
            )

        st.divider()
        if st.session_state.messages:
            history = pd.DataFrame(st.session_state.messages)
            st.download_button(
                "📥 Export chat as CSV",
                data=history.to_csv(index=False).encode("utf-8"),
                file_name=f"pdf_chat_{datetime.now():%Y%m%d_%H%M%S}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if not st.session_state.messages:
        st.info(
            "**Try asking:**\n"
            "- Compare the debt-to-equity ratio across all uploaded reports.\n"
            "- List every related party transaction and flag anything unusual.\n"
            "- How well did cash flow from operations convert from net profit?\n"
            "- How much did Key Managerial Personnel pay rise year on year?"
        )

    for message in st.session_state.messages:
        render_message(message["role"], message["content"])

    question = st.chat_input("Ask a question about your documents…")
    if question:
        if not api_key:
            st.error("Enter your Google AI API key in the sidebar.")
            return

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )
        render_message("user", question)

        with st.spinner("Analysing…"):
            try:
                answer, warning = answer_question(
                    question, api_key, chat_model, embedding_model, temperature, k
                )
                reply = warning or answer
            except Exception as exc:
                reply = f"⚠️ Request failed: {exc}"

        st.session_state.messages.append(
            {
                "role": "bot",
                "content": reply,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )
        render_message("bot", reply)


if __name__ == "__main__":
    main()
