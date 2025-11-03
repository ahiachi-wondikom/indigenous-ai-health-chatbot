"""
Indigenous Medical Chatbot - Streamlit UI
English Only Version (for deployment testing)
"""

import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load environment variables
load_dotenv()

def get_secret(key):
    """Get secret from Streamlit Cloud or local .env"""
    try:
        return st.secrets[key]
    except:
        return os.getenv(key)

st.set_page_config(
    page_title="Indigenous Medical Chatbot",
    page_icon="🏥",
    layout="wide"
)

# Simple CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #212121;
    }
    .user-message {
        background-color: #E8F4F8;
        border-left: 4px solid #0288D1;
    }
    .bot-message {
        background-color: #E8F5E9;
        border-left: 4px solid #43A047;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    """Load only essential models"""
    with st.spinner("🔄 Loading AI models..."):
        # Get API keys
        pinecone_key = get_secret("PINECONE_API_KEY")
        groq_key = get_secret("GROQ_API_KEY")
        
        # Pinecone
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index("medicalbot")
        
        # Embedding model (small, fast)
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Groq
        groq_client = Groq(api_key=groq_key)
    
    return index, embedding_model, groq_client

index, embedding_model, groq_client = load_models()

def generate_answer(query, context):
    """Generate answer using Groq"""
    system_prompt = f"""You are a medical assistant. Use this context to answer:

{context}

Be clear, simple, and include a medical disclaimer."""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content

def medical_chatbot(query):
    """Process query"""
    # Search Pinecone
    query_embedding = embedding_model.encode(query).tolist()
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    
    # Get context
    texts = [match['metadata']['text'] for match in results['matches']]
    context = "\n\n".join(texts)
    
    # Generate answer
    answer = generate_answer(query, context)
    
    return {
        "answer": answer,
        "sources": [
            {"text": match['metadata']['text'][:200] + "...", "score": match['score']}
            for match in results['matches']
        ]
    }

# Session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("🏥 Medical Chatbot")
    st.caption("English Only (Translation coming soon)")
    st.markdown("---")
    
    st.subheader("💡 Try These:")
    samples = [
        "What causes malaria?",
        "How to treat diabetes?",
        "Symptoms of high blood pressure?",
        "How to prevent flu?"
    ]
    
    for q in samples:
        if st.button(q, key=q):
            st.session_state.current_question = q
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.caption("📚 2,735 medical articles")
    st.warning("⚠️ Educational purposes only. Consult healthcare professionals.")

# Main
st.title("🏥 Medical Chatbot")
st.caption("Ask medical questions in English")

# Display messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-message user-message"><strong>You:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message bot-message"><strong>🤖 Assistant:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander("📚 View Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source {i}** ({src['score']:.1%})\n\n{src['text']}\n\n---")

# Input
user_input = st.chat_input("Ask your medical question...")

if 'current_question' in st.session_state:
    user_input = st.session_state.current_question
    del st.session_state.current_question

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.spinner("🔍 Searching..."):
        result = medical_chatbot(user_input)
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
    
    st.rerun()

st.caption("Built with Pinecone, Groq, and Sentence Transformers")