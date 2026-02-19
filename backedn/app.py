import streamlit as st
import os
import tempfile
import asyncio
from typing import List

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0,
            streaming=True
        )
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            openai_api_base=base_url
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
    st.title("📖 ScholarAI: Advanced Paper Agent")
    st.subheader("Deep Insight through Retrieval-Augmented Generation")

    # Sidebar for Setup
    with st.sidebar:
        st.header("🔑 Authentication")
        api_key = st.text_input("API Key", type="password", help="Enter your OpenAI or DeepSeek Key")
        base_url = st.text_input("Base URL", value="https://api.openai.com/v1")
        model_choice = st.selectbox("Model", ["gpt-4o", "deepseek-chat", "gpt-3.5-turbo"])

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
            with st.status("Analyzing Paper Geometry...") as status:
                agent = PaperAgent(api_key, base_url, model_choice)
                st.session_state.vector_db = agent.ingest_pdf(uploaded_file)
                st.session_state.agent_instance = agent
                status.update(label="Analysis Complete!", state="complete")

    # Chat Interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about methodology, results, or conclusions..."):
        if not api_key:
            st.warning("Please provide an API Key in the sidebar.")
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