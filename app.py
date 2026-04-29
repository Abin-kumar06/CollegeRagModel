import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage
from ingestion import load_uploaded_files

st.set_page_config(page_title="College Student Knowledge Assistant", page_icon="🎓", layout="wide")

# Custom CSS for Premium UI
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Main Background & Text */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(20, 20, 24) 0%, rgb(10, 10, 14) 90%);
        color: #e0e0e0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(25, 25, 30, 0.6);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Chat Messages Glassmorphism */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Specific to user and assistant to differentiate slightly */
    [data-testid="chatAvatarIcon-user"] {
        background-color: #4f46e5;
    }
    
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #10b981;
    }

    /* Inputs and Buttons */
    .stTextInput>div>div>input, .stChatInputContainer>div {
        border-radius: 8px;
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
    }
    
    .stButton>button {
        border-radius: 8px;
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
        color: white;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎓 College Knowledge Assistant")
st.caption("A premium RAG-powered assistant for college-related queries.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your college assistant. Ask me anything based on the uploaded documents."}
    ]

# Sidebar for documents
with st.sidebar:
    st.header("📚 Knowledge Base")
    st.write("Upload course policies, schedules, etc.")
    
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if "vectorstore" not in st.session_state or st.button("Process Documents"):
            with st.spinner("Processing & Indexing..."):
                docs = load_uploaded_files(uploaded_files)
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                chunks = splitter.split_documents(docs)
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
                st.session_state.vectorstore = vectorstore
                st.success(f"Successfully indexed {len(chunks)} chunks!")
                
    if "vectorstore" in st.session_state:
        st.info("✅ Documents are loaded and ready for queries.")
        if st.button("Clear Memory & DB"):
            del st.session_state.vectorstore
            st.session_state.messages = [{"role": "assistant", "content": "Memory cleared. Please upload new documents."}]
            st.rerun()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.markdown(source)

# Accept user input
if prompt := st.chat_input("What is the minimum attendance required?"):
    if "vectorstore" not in st.session_state:
        st.warning("Please upload and process documents in the sidebar first.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            vectorstore = st.session_state.vectorstore
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            with st.spinner("Thinking..."):
                retrieved_docs = retriever.invoke(prompt)
                
            if not retrieved_docs:
                response_text = "I could not find the answer in the provided documents."
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            else:
                context = "\n\n".join(doc.page_content for doc in retrieved_docs)
                
                # Format previous messages for context (last 3 pairs)
                chat_history = ""
                for msg in st.session_state.messages[-7:-1]: # exclude the current prompt and the assistant's initial
                    role = "User" if msg["role"] == "user" else "Assistant"
                    chat_history += f"{role}: {msg['content']}\n"
                
                llm = ChatOllama(model="phi3", temperature=0.1)
                
                full_prompt = f"""You are a helpful college assistant. Answer the user's question based ONLY on the provided Context.
If the answer is not in the context, say "I could not find the answer in the provided documents."

Chat History:
{chat_history}

Context:
{context}

Question:
{prompt}
"""
                
                # Stream the response
                response_placeholder = st.empty()
                full_response = ""
                
                for chunk in llm.stream(full_prompt):
                    full_response += chunk.content
                    response_placeholder.markdown(full_response + "▌")
                    
                response_placeholder.markdown(full_response)
                
                # Show sources
                source_strings = []
                with st.expander("View Sources"):
                    for i, doc in enumerate(retrieved_docs, start=1):
                        source_text = f"**Source {i}: {doc.metadata.get('source','Unknown')}**\n\n{doc.page_content}\n\n---"
                        st.markdown(source_text)
                        source_strings.append(source_text)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": source_strings
                })