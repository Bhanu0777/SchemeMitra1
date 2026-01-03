"""
🏛️ SchemeMitra - AI Government Scheme Finder
A modern web application to help Indians discover government schemes they are eligible for.
Built with Streamlit, Azure OpenAI, and Azure Text Analytics.
"""

import streamlit as st
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import requests
from typing import List, Dict, Tuple

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="SchemeMitra - AI Government Scheme Finder",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'bookmarked_schemes' not in st.session_state:
    st.session_state.bookmarked_schemes = []

if 'language' not in st.session_state:
    st.session_state.language = 'en'

if 'accessibility_mode' not in st.session_state:
    st.session_state.accessibility_mode = False

if 'search_history' not in st.session_state:
    st.session_state.search_history = []

if 'expanded_schemes' not in st.session_state:
    st.session_state.expanded_schemes = []

# ============================================================================
# AZURE AI SERVICES CONFIGURATION
# ============================================================================

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo")
AZURE_TEXTANALYTICS_KEY = os.getenv("AZURE_TEXTANALYTICS_KEY")
AZURE_TEXTANALYTICS_ENDPOINT = os.getenv("AZURE_TEXTANALYTICS_ENDPOINT")

# API version for Azure OpenAI
AZURE_OPENAI_API_VERSION = "2023-05-15"

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_schemes() -> List[Dict]:
    """Load schemes from JSON file."""
    try:
        with open('schemes.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('schemes', [])
    except FileNotFoundError:
        st.error("❌ schemes.json not found. Please ensure the file exists in the project directory.")
        return []

# Load schemes
SCHEMES = load_schemes()

# Categories mapping
CATEGORIES = {
    'Farmers': '🌾',
    'Women': '👩‍💼',
    'Youth': '👨‍🎓',
    'MSME': '🏭',
    'Education': '📚',
    'Senior Citizens': '👴'
}

CATEGORY_NAMES = list(CATEGORIES.keys())

# ============================================================================
# AZURE AI FUNCTIONS
# ============================================================================

def call_azure_openai(prompt: str, max_tokens: int = 200) -> str:
    """
    Call Azure OpenAI API to generate responses.
    Used for eligibility explanations and scheme matching.
    """
    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT]):
        return "⚠️ Azure OpenAI not configured. Please set your API credentials in .env file."
    
    try:
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_API_KEY
        }
        
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that explains Indian government schemes in simple, non-legal language. Be concise and clear."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "top_p": 0.95
        }
        
        url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
        
        response = requests.post(url, json=data, headers=headers, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    
    except requests.exceptions.RequestException as e:
        return f"⚠️ Error calling Azure OpenAI: {str(e)}"
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"

def analyze_text_azure(text: str) -> Dict:
    """
    Analyze user input using Azure Text Analytics.
    Extracts key entities and sentiment.
    """
    if not all([AZURE_TEXTANALYTICS_KEY, AZURE_TEXTANALYTICS_ENDPOINT]):
        return {"error": "Azure Text Analytics not configured"}
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": AZURE_TEXTANALYTICS_KEY
        }
        
        data = {
            "documents": [
                {
                    "id": "1",
                    "language": "en",
                    "text": text
                }
            ]
        }
        
        url = f"{AZURE_TEXTANALYTICS_ENDPOINT}/text/analytics/v3.1/entities/recognition/general"
        
        response = requests.post(url, json=data, headers=headers, timeout=10)
        response.raise_for_status()
        
        return response.json()
    
    except requests.exceptions.RequestException as e:
        return {"error": f"Error calling Azure Text Analytics: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def generate_eligibility_explanation(scheme: Dict, user_profile: str) -> Tuple[str, int]:
    """
    Generate AI-powered eligibility explanation and match score.
    Returns: (explanation_text, match_score_percentage)
    """
    prompt = f"""
    Scheme Name: {scheme['name']}
    Ministry: {scheme['ministry']}
    Beneficiary Type: {scheme['beneficiary']}
    Benefit: {scheme['benefit']}
    
    User Profile: {user_profile}
    
    Based on the scheme details and user profile:
    1. Briefly explain (2-3 sentences) why this user MIGHT be eligible
    2. Mention any potential eligibility gaps
    3. Suggest next steps
    
    Keep language simple and non-legal.
    """
    
    explanation = call_azure_openai(prompt, max_tokens=150)
    
    # Calculate match score (in production, this would be more sophisticated)
    # For now, based on keyword matching
    match_score = calculate_match_score(scheme, user_profile)
    
    return explanation, match_score

def calculate_match_score(scheme: Dict, user_profile: str) -> int:
    """Calculate match percentage based on keyword matching."""
    scheme_text = f"{scheme['name']} {scheme['beneficiary']} {scheme['category']}".lower()
    profile_text = user_profile.lower()
    
    keywords = [
        'farmer', 'women', 'youth', 'student', 'senior', 'elder', 'msme', 'business',
        'entrepreneur', 'girl', 'female', 'young', 'old', 'small', 'enterprise'
    ]
    
    matches = sum(1 for keyword in keywords if keyword in scheme_text and keyword in profile_text)
    
    # Base score + keyword matches
    base_score = 50
    additional_score = matches * 5
    
    return min(95, base_score + additional_score)  # Cap at 95%

# ============================================================================
# UI STYLING & EMBEDDED CSS
# ============================================================================

def inject_css():
    """Inject custom CSS with India-inspired color palette and dark theme."""
    css = """
    <style>
    /* Color Palette - India Inspired with Dark Theme */
    :root {
        --primary-blue: #0B5ED7;
        --secondary-saffron: #FF9933;
        --success-green: #138808;
        --warning-navy: #003366;
        --background: #0F172A;
        --card-bg: #1F2937;
        --text-primary: #F3F4F6;
        --text-secondary: #D1D5DB;
        --border-light: #374151;
    }
    
    /* Global styles */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: var(--background);
        color: var(--text-primary);
    }
    
    /* Smooth transitions and animations */
    a {
        transition: all 0.3s ease;
        color: var(--primary-blue);
    }
    
    a:hover {
        color: var(--secondary-saffron);
    }
    
    button {
        transition: all 0.3s ease !important;
    }
    
    button:hover {
        transform: translateY(-2px) !important;
    }
    
    /* Input focus glow - Saffron accent */
    input:focus {
        box-shadow: 0 0 0 3px rgba(255, 153, 51, 0.15) !important;
        border-color: var(--secondary-saffron) !important;
    }
    
    input {
        border-color: var(--border-light) !important;
        background-color: var(--card-bg) !important;
        color: var(--text-primary) !important;
    }
    
    /* Selection color */
    [data-testid="stSelectbox"] {
        transition: all 0.3s ease;
    }
    
    /* Smooth divider */
    hr {
        margin: 2rem 0;
        border-color: var(--border-light);
    }
    
    /* Container padding */
    [data-testid="stContainer"] {
        padding: 1rem 0;
    }
    
    /* Streamlit button styling */
    [data-testid="stButton"] button {
        background: linear-gradient(135deg, var(--primary-blue), var(--secondary-saffron)) !important;
        color: white !important;
        border: none !important;
    }
    
    [data-testid="stButton"] button:hover {
        background: linear-gradient(135deg, var(--secondary-saffron), var(--primary-blue)) !important;
    }
    
    /* Link button styling */
    [data-testid="stLinkButton"] button {
        background: var(--primary-blue) !important;
        color: white !important;
    }
    
    [data-testid="stLinkButton"] button:hover {
        background: var(--secondary-saffron) !important;
    }
    
    /* Success styling */
    .stSuccess {
        border-left: 4px solid var(--success-green) !important;
        background-color: rgba(19, 136, 8, 0.1) !important;
    }
    
    /* Info styling */
    .stInfo {
        border-left: 4px solid var(--primary-blue) !important;
        background-color: rgba(11, 94, 215, 0.1) !important;
    }
    
    /* Warning styling */
    .stWarning {
        border-left: 4px solid var(--secondary-saffron) !important;
        background-color: rgba(255, 153, 51, 0.1) !important;
    }
    
    /* Error styling */
    .stError {
        border-left: 4px solid var(--warning-navy) !important;
        background-color: rgba(0, 51, 102, 0.1) !important;
    }
    
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_navbar():
    """Render the top navigation bar."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0B5ED7 0%, #003366 100%); padding: 1.5rem 2rem; border-bottom: 3px solid #FF9933; margin: -1rem -1rem 2rem -1rem;">
        <div style="font-size: 2rem; font-weight: 800; color: #ffffff; text-align: center; letter-spacing: 0.5px; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">🏛️ SchemeMitra</div>
        <div style="font-size: 0.9rem; color: #D1D5DB; text-align: center; margin-top: 0.5rem; font-weight: 500;">AI-Powered Government Scheme Finder</div>
    </div>
    """, unsafe_allow_html=True)

def render_disclaimer():
    """Render the important disclaimer."""
    st.markdown("""
    <div style="background: #FFF3E0; border-left: 4px solid #FF9933; padding: 1.2rem; border-radius: 6px; margin-bottom: 2rem; font-size: 0.9rem; color: #5F3300;">
        <div style="font-weight: 700; margin-bottom: 0.5rem;">⚠️ Important Disclaimer</div>
        <div>
            SchemeMitra is an independent application and is <strong>NOT an official government portal</strong>.
            This platform provides guidance only. Always verify information on official government portals.
            We are not responsible for inaccuracies. Consult official government offices for clarification.
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_category_selector():
    """Render category selector with circular icons."""
    col1, col2, col3 = st.columns([1, 1, 1])
    
    category_cols = {
        'Farmers': col1,
        'Women': col2,
        'Youth': col3,
        'MSME': col1,
        'Education': col2,
        'Senior Citizens': col3
    }
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <p style="font-weight: 700; color: #FF9933; margin-bottom: 1rem;">Browse by Category</p>
    </div>
    """, unsafe_allow_html=True)
    
    for idx, category in enumerate(CATEGORY_NAMES):
        if idx % 3 == 0:
            col1, col2, col3 = st.columns([1, 1, 1])
        
        cols = [col1, col2, col3]
        with cols[idx % 3]:
            if st.button(f"{CATEGORIES[category]}\n{category}", key=f"cat_{category}", use_container_width=True):
                st.session_state.selected_category = category

def render_search_section():
    """Render the search section with input and filters."""
    st.markdown("""
    <div style="background: #1F2937; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); margin-bottom: 2rem; border: 1px solid #374151;">
        <p style="font-size: 1.1rem; font-weight: 700; color: #FF9933; margin-bottom: 1rem;">
            🔍 Find Your Perfect Scheme
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        search_query = st.text_input(
            "Search by scheme name, ministry, or keyword...",
            key="search_input",
            placeholder="e.g., Farmer support, Women entrepreneur, Student scholarship..."
        )
    
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    return search_query, search_button

def render_filters():
    """Render filter panel."""
    st.markdown("""
    <div style="margin-bottom: 2rem; font-weight: 700; color: #FF9933; font-size: 1.1rem;">
        ⚙️ Refine Your Search
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_ministry = st.selectbox(
            "Ministry / Department",
            options=["All Ministries"] + sorted(set(s['ministry'] for s in SCHEMES)),
            key="filter_ministry"
        )
    
    with col2:
        selected_beneficiary = st.selectbox(
            "Beneficiary Type",
            options=["All Types"] + sorted(set(s['beneficiary'] for s in SCHEMES)),
            key="filter_beneficiary"
        )
    
    with col3:
        selected_category = st.selectbox(
            "Category",
            options=["All Categories"] + CATEGORY_NAMES,
            key="filter_category"
        )
    
    return selected_ministry, selected_beneficiary, selected_category

def render_scheme_card(scheme: Dict, idx: int):
    """Render a single scheme card with all features."""
    is_bookmarked = scheme['id'] in st.session_state.bookmarked_schemes
    is_expanded = scheme['id'] in st.session_state.expanded_schemes
    
    # Determine match score
    user_profile = st.session_state.get('last_user_profile', 'General user')
    _, match_score = generate_eligibility_explanation(scheme, user_profile)
    
    # Create card container with proper styling
    with st.container():
        # Header with title and status
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"### {scheme['name']}")
        with col2:
            st.markdown(f'<div style="background-color: #138808; color: white; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; text-align: center; white-space: nowrap;">✓ Active</div>', unsafe_allow_html=True)
        
        # Ministry and Beneficiary info
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**📍 Ministry:**  \n{scheme['ministry']}")
        with col2:
            st.markdown(f"**👥 Beneficiary:**  \n{scheme['beneficiary']}")
        
        # Benefit section
        st.markdown(f"""
        <div style="background: rgba(255, 153, 51, 0.15); padding: 1rem; border-left: 3px solid #FF9933; border-radius: 6px; margin: 1rem 0; color: #F3F4F6;">
            <strong>💰 Benefit:</strong> {scheme['benefit']}
        </div>
        """, unsafe_allow_html=True)
        
        # Match score display
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin: 1rem 0; padding: 0.8rem; background: rgba(11, 94, 215, 0.15); border-radius: 6px; border: 1px solid #374151;">
            <span style="font-size: 0.9rem; color: #D1D5DB;">Eligibility Match:</span>
            <div style="flex: 1; height: 8px; background: #374151; border-radius: 4px; overflow: hidden;">
                <div style="height: 100%; width: {match_score}%; background: linear-gradient(90deg, #138808, #FF9933);"></div>
            </div>
            <span style="font-weight: 700; color: #FF9933; min-width: 45px; text-align: right;">{match_score}%</span>
        </div>
        """, unsafe_allow_html=True)
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💡 Why I'm Eligible", key=f"expand_{scheme['id']}", use_container_width=True):
                if scheme['id'] in st.session_state.expanded_schemes:
                    st.session_state.expanded_schemes.remove(scheme['id'])
                else:
                    st.session_state.expanded_schemes.append(scheme['id'])
                st.rerun()
        
        with col2:
            if st.button(f"{'⭐ Bookmarked' if is_bookmarked else '☆ Bookmark'}", 
                         key=f"bookmark_{scheme['id']}", use_container_width=True):
                if is_bookmarked:
                    st.session_state.bookmarked_schemes.remove(scheme['id'])
                else:
                    st.session_state.bookmarked_schemes.append(scheme['id'])
                st.rerun()
        
        with col3:
            st.link_button("🔗 Official Source", scheme['source_url'], use_container_width=True)
        
        # Show eligibility explanation if expanded
        if scheme['id'] in st.session_state.expanded_schemes:
            with st.spinner("✨ Generating AI explanation..."):
                explanation, _ = generate_eligibility_explanation(scheme, user_profile)
                st.info(f"**Why you might be eligible:**\n\n{explanation}")
        
        st.divider()

def render_bookmarked_schemes():
    """Render bookmarked schemes section."""
    if st.session_state.bookmarked_schemes:
        st.markdown("""
        <div style="margin-bottom: 2rem; margin-top: 3rem;">
            <h2 style="color: #FF9933; border-bottom: 3px solid #0B5ED7; padding-bottom: 0.5rem; font-size: 1.8rem;">
                ⭐ My Bookmarked Schemes ({count})
            </h2>
        </div>
        """.format(count=len(st.session_state.bookmarked_schemes)), unsafe_allow_html=True)
        
        st.info(f"📚 You have {len(st.session_state.bookmarked_schemes)} scheme(s) saved. Review them below:")
        
        bookmarked = [s for s in SCHEMES if s['id'] in st.session_state.bookmarked_schemes]
        
        if bookmarked:
            for idx, scheme in enumerate(bookmarked, 1):
                render_scheme_card(scheme, idx)
        else:
            st.warning("No bookmarked schemes found. Bookmark schemes from the Finder tab!")

def render_feedback_section():
    """Render feedback section."""
    st.markdown("""
    <div style="background: rgba(11, 94, 215, 0.15); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #0B5ED7; margin-top: 2rem; border: 1px solid #374151;">
        <p style="font-weight: 700; color: #FF9933; margin-bottom: 0.5rem;">📝 Was this helpful?</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("👍 Yes, very helpful!", key="feedback_yes", use_container_width=True):
            st.success("Thank you for your feedback! It helps us improve SchemeMitra.")
    
    with col2:
        if st.button("😐 Somewhat helpful", key="feedback_neutral", use_container_width=True):
            st.info("Thank you! We'll work on improving the experience.")
    
    with col3:
        if st.button("👎 Not helpful", key="feedback_no", use_container_width=True):
            st.warning("We'd love to hear your suggestions. Please reach out!")

def render_footer():
    """Render footer."""
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #D1D5DB; font-size: 0.9rem; border-top: 1px solid #374151; margin-top: 3rem;">
        <p><strong>🏛️ SchemeMitra</strong> © 2026 - AI Government Scheme Finder</p>
        <p>Built with ❤️ for the Imagine Cup | Data from official government portals</p>
        <p style="font-size: 0.8rem; margin-top: 1rem; color: #9CA3AF;">
            This is an educational MVP and not officially affiliated with the Government of India.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FILTERING & SEARCH LOGIC
# ============================================================================

def filter_schemes(schemes: List[Dict], 
                  search_query: str = "",
                  ministry_filter: str = "All Ministries",
                  beneficiary_filter: str = "All Types",
                  category_filter: str = "All Categories") -> List[Dict]:
    """
    Filter schemes based on search query and filters.
    """
    filtered = schemes.copy()
    
    # Search filter
    if search_query:
        search_lower = search_query.lower()
        filtered = [
            s for s in filtered
            if (search_lower in s['name'].lower() or
                search_lower in s['description'].lower() or
                search_lower in s['ministry'].lower() or
                search_lower in s['beneficiary'].lower())
        ]
    
    # Ministry filter
    if ministry_filter != "All Ministries":
        filtered = [s for s in filtered if s['ministry'] == ministry_filter]
    
    # Beneficiary filter
    if beneficiary_filter != "All Types":
        filtered = [s for s in filtered if s['beneficiary'] == beneficiary_filter]
    
    # Category filter
    if category_filter != "All Categories":
        filtered = [s for s in filtered if s['category'] == category_filter]
    
    return filtered

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def render_landing_page():
    """Render the SchemeMitra landing/homepage with professional design."""
    
    # Hero Section using Streamlit
    st.markdown("""
    <style>
        .hero-section {
            background: linear-gradient(135deg, #0B5ED7 0%, #003366 100%);
            padding: 5rem 3rem;
            border-radius: 16px;
            color: white;
            text-align: center;
            margin-bottom: 4rem;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
        }
        .hero-title {
            font-size: 3.5rem;
            font-weight: 900;
            margin: 0;
            letter-spacing: -1px;
        }
        .hero-subtitle {
            font-size: 1.6rem;
            color: #FFD700;
            margin: 1.5rem 0 0.5rem;
            font-weight: 700;
        }
        .hero-description {
            font-size: 1.1rem;
            color: #D1D5DB;
            margin: 0.5rem 0 0;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.6;
        }
    </style>
    
    <div class="hero-section">
        <div class="hero-title">🏛️ SchemeMitra</div>
        <div class="hero-subtitle">Your AI Assistant for Government Schemes</div>
        <p class="hero-description">
            Discover and apply for government schemes you're eligible for using advanced AI-powered analysis. 
            No login required. Completely free. Your privacy is our priority.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features Section
    st.markdown("""
    <div style="margin-bottom: 4rem;">
        <h2 style="color: #FF9933; font-size: 2.2rem; text-align: center; margin-bottom: 3rem; font-weight: 800;">✨ Key Features</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown("""
        <div style="background: #1F2937; padding: 2.5rem; border-radius: 12px; border: 2px solid #374151; height: 100%; transition: all 0.3s ease;">
            <div style="font-size: 3rem; margin-bottom: 1rem; text-align: center;">🤖</div>
            <h3 style="color: #FF9933; margin: 0 0 0.75rem 0; text-align: center; font-size: 1.2rem;">AI-Powered Matching</h3>
            <p style="color: #D1D5DB; margin: 0; text-align: center; line-height: 1.6;">Azure OpenAI analyzes your profile and matches you with relevant schemes instantly.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #1F2937; padding: 2.5rem; border-radius: 12px; border: 2px solid #374151; height: 100%; transition: all 0.3s ease;">
            <div style="font-size: 3rem; margin-bottom: 1rem; text-align: center;">📊</div>
            <h3 style="color: #FF9933; margin: 0 0 0.75rem 0; text-align: center; font-size: 1.2rem;">Real-Time Analysis</h3>
            <p style="color: #D1D5DB; margin: 0; text-align: center; line-height: 1.6;">Advanced Text Analytics extract your key information to determine eligibility accurately.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: #1F2937; padding: 2.5rem; border-radius: 12px; border: 2px solid #374151; height: 100%; transition: all 0.3s ease;">
            <div style="font-size: 3rem; margin-bottom: 1rem; text-align: center;">📚</div>
            <h3 style="color: #FF9933; margin: 0 0 0.75rem 0; text-align: center; font-size: 1.2rem;">12+ Schemes</h3>
            <p style="color: #D1D5DB; margin: 0; text-align: center; line-height: 1.6;">Verified schemes from government portals covering all major beneficiary categories.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")  # Spacer
    
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown("""
        <div style="background: #1F2937; padding: 2.5rem; border-radius: 12px; border: 2px solid #374151; height: 100%; transition: all 0.3s ease;">
            <div style="font-size: 3rem; margin-bottom: 1rem; text-align: center;">⭐</div>
            <h3 style="color: #FF9933; margin: 0 0 0.75rem 0; text-align: center; font-size: 1.2rem;">Bookmarking</h3>
            <p style="color: #D1D5DB; margin: 0; text-align: center; line-height: 1.6;">Save your favorite schemes for later and build your personalized opportunities list.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #1F2937; padding: 2.5rem; border-radius: 12px; border: 2px solid #374151; height: 100%; transition: all 0.3s ease;">
            <div style="font-size: 3rem; margin-bottom: 1rem; text-align: center;">🔐</div>
            <h3 style="color: #FF9933; margin: 0 0 0.75rem 0; text-align: center; font-size: 1.2rem;">Privacy First</h3>
            <p style="color: #D1D5DB; margin: 0; text-align: center; line-height: 1.6;">No login required. No personal data stored. No external tracking. Ever.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: #1F2937; padding: 2.5rem; border-radius: 12px; border: 2px solid #374151; height: 100%; transition: all 0.3s ease;">
            <div style="font-size: 3rem; margin-bottom: 1rem; text-align: center;">🌐</div>
            <h3 style="color: #FF9933; margin: 0 0 0.75rem 0; text-align: center; font-size: 1.2rem;">Official Links</h3>
            <p style="color: #D1D5DB; margin: 0; text-align: center; line-height: 1.6;">Every scheme links directly to official government portals for verification.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # How It Works Section
    st.markdown("""
    <div style="margin-bottom: 4rem;">
        <h2 style="color: #FF9933; font-size: 2.2rem; text-align: center; margin-bottom: 3rem; font-weight: 800;">🚀 How It Works</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    
    with col1:
        st.markdown("""
        <div style="text-align: center;">
            <div style="width: 80px; height: 80px; background: linear-gradient(135deg, #FF9933, #FFB366); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-size: 2rem; font-weight: 900; color: #0F172A;">1</div>
            <h4 style="color: #F3F4F6; margin: 0 0 0.5rem 0; font-size: 1rem;">Tell Us About Yourself</h4>
            <p style="color: #D1D5DB; font-size: 0.85rem; margin: 0; line-height: 1.5;">Share basic information like age, category, and skills.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <div style="width: 80px; height: 80px; background: linear-gradient(135deg, #0B5ED7, #003366); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-size: 2rem; font-weight: 900; color: #fff;">2</div>
            <h4 style="color: #F3F4F6; margin: 0 0 0.5rem 0; font-size: 1rem;">AI Analyzes</h4>
            <p style="color: #D1D5DB; font-size: 0.85rem; margin: 0; line-height: 1.5;">Our AI engine matches you with relevant schemes.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center;">
            <div style="width: 80px; height: 80px; background: linear-gradient(135deg, #138808, #1FBF1F); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-size: 2rem; font-weight: 900; color: #fff;">3</div>
            <h4 style="color: #F3F4F6; margin: 0 0 0.5rem 0; font-size: 1rem;">Review Results</h4>
            <p style="color: #D1D5DB; font-size: 0.85rem; margin: 0; line-height: 1.5;">Check eligibility and learn why you match.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center;">
            <div style="width: 80px; height: 80px; background: linear-gradient(135deg, #FF9933, #FFB366); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-size: 2rem; font-weight: 900; color: #0F172A;">4</div>
            <h4 style="color: #F3F4F6; margin: 0 0 0.5rem 0; font-size: 1rem;">Apply Now</h4>
            <p style="color: #D1D5DB; font-size: 0.85rem; margin: 0; line-height: 1.5;">Bookmark and apply via official portals.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Coverage Section
    st.markdown("""
    <div style="margin-bottom: 4rem;">
        <h2 style="color: #FF9933; font-size: 2.2rem; text-align: center; margin-bottom: 3rem; font-weight: 800;">📋 We Cover All Categories</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6, gap="small")
    
    categories = [
        ("🌾", "Farmers"),
        ("👩‍💼", "Women"),
        ("👨‍🎓", "Youth"),
        ("🏭", "MSME"),
        ("📚", "Education"),
        ("👴", "Seniors"),
    ]
    
    for (emoji, name), col in zip(categories, [col1, col2, col3, col4, col5, col6]):
        with col:
            st.markdown(f"""
            <div style="text-align: center; padding: 1.5rem; background: #1F2937; border-radius: 10px; border: 2px solid #374151;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{emoji}</div>
                <p style="color: #F3F4F6; font-weight: 600; margin: 0; font-size: 0.9rem;">{name}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Tech Stack Section
    st.markdown("""
    <div style="margin-bottom: 4rem;">
        <h2 style="color: #FF9933; font-size: 2.2rem; text-align: center; margin-bottom: 3rem; font-weight: 800;">⚙️ Built With</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6, gap="small")
    
    tech_stack = [
        ("Streamlit", "Frontend"),
        ("Python 3.9+", "Language"),
        ("Azure OpenAI", "AI"),
        ("Text Analytics", "NLP"),
        ("Gov Data", "Source"),
        ("Cloud", "Deploy"),
    ]
    
    for (tech, category), col in zip(tech_stack, [col1, col2, col3, col4, col5, col6]):
        with col:
            st.markdown(f"""
            <div style="text-align: center; padding: 1.5rem; background: #1F2937; border-radius: 10px; border: 2px solid #374151;">
                <p style="color: #FF9933; font-weight: 700; margin: 0; font-size: 0.85rem;">{tech}</p>
                <p style="color: #D1D5DB; font-size: 0.75rem; margin: 0.5rem 0 0 0;">{category}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # CTA Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0B5ED7 0%, #FF9933 100%); padding: 3.5rem 2.5rem; border-radius: 12px; text-align: center; margin-bottom: 2rem; box-shadow: 0 15px 40px rgba(11, 94, 215, 0.25);">
        <h2 style="color: white; font-size: 2.2rem; margin: 0 0 1rem 0; font-weight: 900;">🎯 Ready to Find Your Schemes?</h2>
        <p style="color: #F3F4F6; font-size: 1.1rem; margin: 0; line-height: 1.6;">Click the <strong>Finder</strong> tab above to discover government schemes tailored to your profile in seconds.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div style="background: rgba(255, 153, 51, 0.1); border: 2px solid #FF9933; padding: 2rem; border-radius: 10px;">
        <p style="color: #FF9933; font-weight: 800; margin: 0 0 0.75rem 0; font-size: 1.1rem;">⚠️ Important Disclaimer</p>
        <p style="color: #D1D5DB; margin: 0; line-height: 1.8;">
            <strong>SchemeMitra</strong> is an independent educational application and is <strong>NOT an official government portal</strong>. 
            This platform provides AI-powered guidance based on your input. 
            <strong>Always verify all information on official government websites</strong> before applying for any scheme. 
            We are not responsible for inaccuracies or outdated information. 
            For official details, eligibility criteria, and applications, please visit government portals directly.
        </p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application entry point."""
    
    # Inject custom CSS FIRST
    inject_css()
    
    # Add custom styling
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #0F172A;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Render navbar
    render_navbar()
    
    # Create tabs
    tab1, tab2 = st.tabs(["🏠 Home", "🔍 Finder"])
    
    # ====== HOME TAB ======
    with tab1:
        render_landing_page()
    
    # ====== FINDER TAB ======
    with tab2:
        # Render disclaimer
        render_disclaimer()
        
        # Sidebar for settings
        with st.sidebar:
            st.title("⚙️ Settings")
            
            # Language selector
            language = st.radio(
                "Language",
                options=["🇬🇧 English", "🇮🇳 Hindi"],
                key="language_select"
            )
            st.session_state.language = "en" if "English" in language else "hi"
            
            # Accessibility mode
            accessibility = st.checkbox(
                "♿ Accessibility Mode (Large Text + High Contrast)",
                value=st.session_state.accessibility_mode
            )
            st.session_state.accessibility_mode = accessibility
            
            st.divider()
            
            st.markdown("""
            ### 📱 About SchemeMitra
            
            AI-powered scheme discovery using:
            - **Azure OpenAI** for intelligent matching
            - **Text Analytics** for input analysis
            - **Real government data** from official portals
            
            ### 🔐 Privacy
            - No login required
            - No personal data stored
            - Session-based bookmarks only
            - No external tracking
            
            ### 📚 Need Help?
            - Check official scheme sources
            - Contact government offices directly
            - Review application disclaimers
            """)
        
        # Main content
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <p style="font-size: 1.1rem; color: #D1D5DB; font-weight: 500;">
                    🎯 Discover government schemes tailored to your profile
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Search section
        search_query, search_button = render_search_section()
        
        # User profile input for AI analysis
        with st.expander("📋 Tell us about yourself (Optional - for better matching)", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=0, max_value=100, value=30)
            
            with col2:
                category = st.selectbox("Select your category", CATEGORY_NAMES + ["Other"])
            
            skills = st.text_input("Your skills/profession (optional)")
            
            # Create user profile for AI
            user_profile = f"{age} years old, {category} category"
            if skills:
                user_profile += f", skills: {skills}"
            
            st.session_state.last_user_profile = user_profile
        
        st.divider()
        
        # Category selector
        render_category_selector()
        
        # Filters
        selected_ministry, selected_beneficiary, selected_category = render_filters()
        
        st.divider()
        
        # Get selected category from button clicks
        selected_category = st.session_state.get('selected_category', selected_category)
        
        # Filter schemes
        filtered_schemes = filter_schemes(
            SCHEMES,
            search_query=search_query if search_button or search_query else "",
            ministry_filter=selected_ministry,
            beneficiary_filter=selected_beneficiary,
            category_filter=selected_category
        )
        
        # Display results
        st.markdown("""
        <div style="margin-bottom: 2rem; margin-top: 2rem;">
            <h2 style="color: #FF9933; border-bottom: 3px solid #0B5ED7; padding-bottom: 0.5rem;">
                🎯 Available Schemes
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        if filtered_schemes:
            st.markdown(f"""
            <div style="padding: 0.8rem; background: rgba(11, 94, 215, 0.15); border-radius: 6px; margin-bottom: 1.5rem; text-align: center; font-weight: 600; color: #0B5ED7; border: 1px solid #374151;">
                Found {len(filtered_schemes)} scheme(s) matching your criteria
            </div>
            """, unsafe_allow_html=True)
            
            for idx, scheme in enumerate(filtered_schemes, 1):
                render_scheme_card(scheme, idx)
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem 2rem; background: #1F2937; border-radius: 10px; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.3); border: 1px solid #374151;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #FF9933; margin-bottom: 0.5rem;">No Schemes Found</div>
                <div style="color: #D1D5DB; font-size: 1rem; margin-bottom: 1.5rem;">
                    Try adjusting your filters or search terms. You can also:<br>
                    • Remove filters to see all schemes<br>
                    • Try different keywords<br>
                    • Browse by category above
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Bookmarked schemes section
        if st.session_state.bookmarked_schemes:
            render_bookmarked_schemes()
        
        st.divider()
        
        # Feedback section
        render_feedback_section()
        
        st.divider()
        
        st.divider()
        
        # Footer
        render_footer()

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
