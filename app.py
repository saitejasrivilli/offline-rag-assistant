import streamlit as st
from rag import RAGSystem

st.set_page_config(page_title="Offline RAG", layout="wide")

st.title("📄 Offline RAG System")
st.write("Ask questions about your local documents. Runs fully offline.")

# Initialize RAG only once
if "rag" not in st.session_state:
    with st.spinner("Loading RAG system..."):
        st.session_state.rag = RAGSystem()
        st.session_state.rag.ingest_documents(force_rebuild=False)

question = st.text_input("Enter your question:")

if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents..."):
            result = st.session_state.rag.query(question)

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources")
        for src in result["sources"]:
            st.write(
                f"- **{src['source']}** (Page {src['page']}) — Distance: {src['distance']:.4f}"
            )

        st.subheader("Confidence")
        st.write(result["confidence"])
