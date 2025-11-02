import streamlit as st
import os

st.title("🔧 Debug Mode")

# Test 1: Secrets
st.header("1. Testing Secrets")
try:
    pinecone_key = st.secrets["PINECONE_API_KEY"]
    groq_key = st.secrets["GROQ_API_KEY"]
    st.success(f"✅ Pinecone key found: {pinecone_key[:10]}...")
    st.success(f"✅ Groq key found: {groq_key[:10]}...")
except Exception as e:
    st.error(f"❌ Secrets error: {e}")

# Test 2: Pinecone
st.header("2. Testing Pinecone")
try:
    from pinecone import Pinecone
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    st.success("✅ Pinecone connected!")
    
    index = pc.Index("medicalbot")
    st.success("✅ Index connected!")
    
    stats = index.describe_index_stats()
    st.success(f"✅ Index has {stats['total_vector_count']} vectors!")
except Exception as e:
    st.error(f"❌ Pinecone error: {e}")

# Test 3: Groq
st.header("3. Testing Groq")
try:
    from groq import Groq
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    st.success("✅ Groq connected!")
except Exception as e:
    st.error(f"❌ Groq error: {e}")

# Test 4: Models
st.header("4. Testing Models")
try:
    from sentence_transformers import SentenceTransformer
    st.write("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    st.success("✅ Embedding model loaded!")
except Exception as e:
    st.error(f"❌ Model error: {e}")

st.header("✅ All Tests Complete!")
st.balloons()