import streamlit as st
import os
import tempfile
import asyncio
from typing import List

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration & Styling ---
st.set_page_config(page_title="ScholarAI Agent", page_icon="📖", layout="wide")


def local_css():
    st.markdown("""
        <style>
        .stChatMessage { border-radius: 15px; padding: 10px; margin: 5px 0; }
        .st-emotion-cache-1c7n2ka { max-width: 95%; }
        </style>
    """, unsafe_allow_html=True)


# --- Logic Classes ---

class PaperAgent:
    def __init__(self, api_key: str, model_name: str):
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=0.7,
            streaming=True
        )

        # 2. 替换为 HuggingFace 的免费本地 Embedding 模型
        # 首次运行时会自动下载大约 90MB 的模型权重到本地
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

    def ingest_pdf(self, uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            path = tmp.name

        try:
            loader = PDFPlumberLoader(path)
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(data)

            vector_db = FAISS.from_documents(chunks, self.embeddings)
            return vector_db
        finally:
            os.remove(path)

    def get_qa_chain(self, vector_db):
        template = """You are a world-class AI Research Assistant. Analyze the provided paper sections to answer questions with high academic precision.

        Guidelines:
        - If the answer isn't in the context, say "I cannot find this specific detail in the paper."
        - Use LaTeX for mathematical notations.
        - Structure long answers with bullet points.

        Context: {context}
        Chat History: {chat_history}
        Question: {question}
        Helpful Academic Answer:"""

        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_db.as_retriever(search_kwargs={"k": 4}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": ChatPromptTemplate.from_template(template)},
            return_source_documents=True
        )


# --- Streamlit UI Main ---

async def main():
    local_css()
    st.title("⚡ ScholarAI: Powered by Groq & Llama 3")
    st.subheader("Ultra-fast Retrieval-Augmented Generation")

    # Sidebar for Setup
    with st.sidebar:
        st.header("🔑 Authentication")
        api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API Key")

        # 更新下拉菜单为 Groq 支持的模型
        model_choice = st.selectbox(
            "Model",
            [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it"
            ]
        )

        st.divider()
        uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type="pdf")

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Session State Init
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None

    # Processing Logic
    if uploaded_file and api_key:
        if st.session_state.vector_db is None:
            with st.status("Analyzing Paper & Generating Local Embeddings...") as status:

                agent = PaperAgent(api_key, model_choice)
                st.session_state.vector_db = agent.ingest_pdf(uploaded_file)
                st.session_state.agent_instance = agent
                status.update(label="Analysis Complete!", state="complete")

    # Chat Interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about methodology, results, or conclusions..."):
        if not api_key:
            st.warning("Please provide a Groq API Key in the sidebar.")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.vector_db:
            with st.chat_message("assistant"):
                agent = st.session_state.agent_instance
                chain = agent.get_qa_chain(st.session_state.vector_db)

                # Execution
                response = await chain.ainvoke({"question": prompt})
                full_response = response["answer"]

                st.markdown(full_response)

                with st.expander("View Evidence (Sources)"):
                    for doc in response["source_documents"]:
                        st.caption(f"Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:300]}...")

                st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.info("Please upload a PDF to start the conversation.")


if __name__ == "__main__":
    asyncio.run(main())