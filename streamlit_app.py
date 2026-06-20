import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI PDF RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #334155
    );
}

h1, h2, h3 {
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

# -----------------------------
# SESSION STATE
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# HEADER
# -----------------------------
st.title("🤖 AI PDF RAG Assistant")
st.write("Upload a PDF and ask questions about it.")

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

# -----------------------------
# PDF PROCESSING
# -----------------------------
db = None

if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    st.info(f"Pages Loaded: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        chunks,
        embedding,
        persist_directory="./chroma_db"
    )

# -----------------------------
# GEMINI
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# -----------------------------
# SHOW CHAT HISTORY
# -----------------------------
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------
# CHAT INPUT
# -----------------------------
query = None

if db is not None:
    query = st.chat_input(
        "Ask a question from the uploaded PDF..."
    )

# -----------------------------
# ANSWER QUESTIONS
# -----------------------------
if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.write(query)

    docs = db.similarity_search(
        query,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not present in the context,
say:

'I couldn't find this information in the uploaded PDF.'

Context:
{context}

Question:
{query}
"""

    with st.spinner("🤖 Thinking..."):

        response = llm.invoke(prompt)
        answer = response.content

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):
        st.write(answer)

    with st.expander("📚 Sources"):

        for i, doc in enumerate(docs):

            page_no = doc.metadata.get(
                "page",
                "Unknown"
            )

            st.write(
                f"Source {i+1} | Page {page_no}"
            )

            st.write(
                doc.page_content[:500]
            )

            st.markdown("---")
