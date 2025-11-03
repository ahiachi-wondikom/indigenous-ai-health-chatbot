"""
Indigenous Medical Chatbot - Streamlit UI
Supports: English, Igbo, Hausa, Yoruba
"""

import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from groq import Groq

# Load environment variables
load_dotenv()

# ========== HELPER FUNCTION FOR SECRETS ==========
def get_secret(key):
    """Get secret from Streamlit Cloud or local .env"""
    try:
        # Try Streamlit Cloud secrets first
        return st.secrets[key]
    except:
        # Fall back to environment variable (local)
        return os.getenv(key)

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Indigenous Medical Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #00695C;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle */
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #424242;
        margin-bottom: 2rem;
    }
    
    /* Chat message container */
    .chat-message {
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        font-size: 1rem;
        line-height: 1.6;
        color: #212121;
    }
    
    /* User message */
    .user-message {
        background-color: #E8F4F8;
        border-left: 4px solid #0288D1;
    }
    
    .user-message strong {
        color: #01579B;
    }
    
    /* Bot message */
    .bot-message {
        background-color: #E8F5E9;
        border-left: 4px solid #43A047;
    }
    
    .bot-message strong {
        color: #1B5E20;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        height: 2.8rem;
        font-weight: 600;
    }
    
    /* Disclaimer */
    .disclaimer {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #FFA000;
        margin-top: 1rem;
        color: #F57F17;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ========== INITIALIZE MODELS ==========
@st.cache_resource
def load_models():
    """Load all models once and cache them"""
    
    with st.spinner("🔄 Loading AI models... (First time takes 2-3 minutes)"):
        # Get API keys
        pinecone_key = get_secret("PINECONE_API_KEY")
        groq_key = get_secret("GROQ_API_KEY")
        
        # Pinecone
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index("medicalbot")
        
        # Embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Translation model
        translator = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-yo")  # Just Yoruba for testing
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-yo")
        
        # Groq
        groq_client = Groq(api_key=groq_key)
    
    return index, embedding_model, translator, tokenizer, groq_client

# Load models
index, embedding_model, translator, tokenizer, groq_client = load_models()

# Language codes
LANGUAGES = {
    "english": "eng_Latn",
    "igbo": "ibo_Latn",
    "hausa": "hau_Latn",
    "yoruba": "yor_Latn"
}

# ========== TRANSLATION FUNCTION ==========
def translate_text(text, source_lang, target_lang):
    """Translate text between languages"""
    if source_lang == target_lang:
        return text
    
    src_code = LANGUAGES[source_lang]
    tgt_code = LANGUAGES[target_lang]
    
    tokenizer.src_lang = src_code
    
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=512
    )
    
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)
    
    translated = translator.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=512
    )
    
    result = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    return result

# ========== SMART ANSWER GENERATION ==========
def generate_smart_answer(query_english, context_english):
    """Generate smart answer using Groq"""
    
    system_prompt = f"""You are a helpful medical assistant. Use the medical information below to answer the user's question clearly and accurately.

Medical Context:
{context_english}

Instructions:
- Answer in 2-3 clear sentences
- Use simple language
- If the context doesn't fully answer the question, say so
- Add a disclaimer that this is educational info and they should see a doctor"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_english}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content

# ========== MAIN CHATBOT FUNCTION ==========
def medical_chatbot(query, user_language="english"):
    """Process medical query and return answer"""
    
    # Translate to English if needed
    if user_language != "english":
        query_english = translate_text(query, user_language, "english")
    else:
        query_english = query
    
    # Search Pinecone
    query_embedding = embedding_model.encode(query_english).tolist()
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    
    # Get context
    texts = [match['metadata']['text'] for match in results['matches']]
    context_english = "\n\n".join(texts)
    
    # Generate answer
    answer_english = generate_smart_answer(query_english, context_english)
    
    # Translate back if needed
    if user_language != "english":
        answer_translated = translate_text(answer_english, "english", user_language)
    else:
        answer_translated = answer_english
    
    return {
        "original_query": query,
        "english_query": query_english if user_language != "english" else None,
        "answer": answer_translated,
        "answer_english": answer_english if user_language != "english" else None,
        "language": user_language,
        "num_sources": len(texts),
        "sources": [
            {
                "text": match['metadata']['text'][:300] + "...",
                "score": match['score']
            }
            for match in results['matches']
        ]
    }

# ========== INITIALIZE SESSION STATE ==========
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'language' not in st.session_state:
    st.session_state.language = 'english'

# ========== SIDEBAR ==========
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2785/2785482.png", width=120)
    st.title("🏥 Medical Chatbot")
    st.markdown("---")
    
    # Language Selection
    st.subheader("🌍 Select Language")
    language = st.selectbox(
        "Choose your language:",
        options=["english", "igbo", "hausa", "yoruba"],
        format_func=lambda x: {
            "english": "🇬🇧 English",
            "igbo": "🇳🇬 Igbo (Asụsụ Igbo)",
            "hausa": "🇳🇬 Hausa (Harshen Hausa)",
            "yoruba": "🇳🇬 Yoruba (Èdè Yorùbá)"
        }[x],
        key="language_selector"
    )
    st.session_state.language = language
    
    st.markdown("---")
    
    # Sample Questions
    st.subheader("💡 Sample Questions")
    
    sample_questions = {
        "english": [
            "What causes malaria?",
            "How to treat diabetes?",
            "What are symptoms of high blood pressure?",
            "How to prevent flu?"
        ],
        "igbo": [
            "Gịnị na-akpata ịba?",
            "Kedu ka esi agwọ ọrịa shuga?",
            "Kedu ihe mgbaàmà nke ọbara mgbali elu?",
            "Kedu ka m ga-esi gbochie flu?"
        ],
        "hausa": [
            "Menene ke haifar da zazzabin cizon sauro?",
            "Ta yaya ake maganin ciwon sikari?",
            "Menene alamun hawan jini?",
            "Ta yaya zan iya rigakafin mura?"
        ],
        "yoruba": [
            "Kini o nfa iba?",
            "Bawo ni a ṣe le ṣe itọju aisan suga?",
            "Kini awọn ami ẹjẹ giga?",
            "Bawo ni MO ṣe le yago fun aisan atari?"
        ]
    }
    
    for question in sample_questions.get(language, sample_questions["english"]):
        if st.button(question, key=f"sample_{question}"):
            st.session_state.current_question = question
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    show_english = st.checkbox("Show English translation", value=True)
    show_sources = st.checkbox("Show medical sources", value=True)
    
    st.markdown("---")
    
    # Clear Chat
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # Info
    st.caption("📚 **Knowledge Base:**")
    st.caption("Gale Encyclopedia of Medicine")
    st.caption("2,735 medical articles")
    
    st.markdown("---")
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Medical Disclaimer</strong><br>
        This chatbot provides educational information only. 
        Always consult healthcare professionals for medical advice.
    </div>
    """, unsafe_allow_html=True)

# ========== MAIN CONTENT ==========
st.markdown('<div class="main-header">🏥 Indigenous Medical Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-header">Ask medical questions in English, Igbo, Hausa, or Yoruba | '
    f'Currently using: <strong>{language.title()}</strong></div>',
    unsafe_allow_html=True
)

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You ({message['language'].title()}):</strong><br>
            {message['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 Medical Assistant:</strong><br>
            {message['content']}
        </div>
        """, unsafe_allow_html=True)
        
        # Show English translation if enabled
        if show_english and message.get('english_answer') and message['language'] != "english":
            with st.expander("📖 View English Translation"):
                st.write(message['english_answer'])
        
        # Show sources if enabled
        if show_sources and message.get('sources'):
            with st.expander(f"📚 View {len(message['sources'])} Medical Sources"):
                for i, source in enumerate(message['sources'], 1):
                    st.markdown(f"""
                    **Source {i}** (Relevance: {source['score']:.1%})
                    
                    {source['text']}
                    
                    ---
                    """)
 
# Chat input
user_input = st.chat_input(
    f"Type your medical question in {language.title()}..." 
)

# Handle sample question clicks
if 'current_question' in st.session_state:
    user_input = st.session_state.current_question
    del st.session_state.current_question

# Process input
if user_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "language": language
    })
    
    # Get bot response
    with st.spinner(f"🔍 Searching medical knowledge and processing in {language.title()}..."):
        result = medical_chatbot(user_input, language)
    
    # Add bot message
    st.session_state.messages.append({
        "role": "assistant",
        "content": result['answer'],
        "english_answer": result.get('answer_english'),
        "sources": result.get('sources'),
        "language": language
    })
    
    # Rerun to update display
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>About This Chatbot</strong></p>
    <p>AI-powered medical information in Nigerian languages</p>
    <p>🔬 Powered by: Pinecone • NLLB Translation • Groq • Sentence Transformers</p>
    <p style='color: #d32f2f; margin-top: 1rem;'>
        ⚠️ <strong>Always consult qualified healthcare providers for medical decisions</strong>
    </p>
</div>
""", unsafe_allow_html=True)