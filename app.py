import os
import json
import re
import pandas as pd
import logging
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from uuid import uuid4
from io import BytesIO
from PyPDF2 import PdfReader
from paddleocr import PaddleOCR
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# --------------- CONFIGURATION ---------------
USE_GEMINI = True  # Toggle manually in code
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
DB_FILE = "C:/Users/archi/Desktop/hackday/outputsNew/all_results_new.json"
GEMINI_API_KEY = "AIzaSyAAPOvYkR2jVEj6s7cYZlCbPhzD7tdTZrk"

# --------------- LOGGER SETUP ---------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("App")

# --------------- PAGE CONFIG ---------------
st.set_page_config(
    page_title="MedLab Analytics Pro",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for theme
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# --------------- THEME SYSTEM ---------------
def get_theme_colors():
    if st.session_state.dark_mode:
        return {
            'primary_bg': '#0f1419',
            'secondary_bg': '#1a1f2e',
            'card_bg': '#242b3d',
            'text_primary': "#A489C6",  # Orange text
            'text_secondary': "#92a1be",
            'accent': "#5576aa",
            'success': '#50c878',
            'warning': '#ffa500',
            'error': '#ff6b6b',
            'border': '#3a4553'
        }
    else:
        return {
            'primary_bg': '#ffffff',
            'secondary_bg': '#f8fafc',
            'card_bg': '#ffffff',
            'text_primary': '#ff7f50',  # Orange text
            'text_secondary': '#64748b',
            'accent': '#667eea',
            'success': '#10b981',
            'warning': '#f59e0b',
            'error': '#ef4444',
            'border': '#e2e8f0'
        }

# --------------- ENHANCED CSS ---------------
def get_dynamic_css():
    colors = get_theme_colors()
    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* CSS Variables for Theme */
    :root {{
        --primary-bg: {colors['primary_bg']};
        --secondary-bg: {colors['secondary_bg']};
        --card-bg: {colors['card_bg']};
        --text-primary: {colors['text_primary']};
        --text-secondary: {colors['text_secondary']};
        --accent: {colors['accent']};
        --success: {colors['success']};
        --warning: {colors['warning']};
        --error: {colors['error']};
        --border: {colors['border']};
    }}
    
    /* Override Streamlit text color */
    .st-emotion-cache-13k62yr {{
        color: var(--text-primary) !important;
    }}
    
    /* Global Styles */
    .main .block-container {{
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
        transition: all 0.3s ease;
    }}
    
    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', sans-serif;
        background: var(--primary-bg) !important;
        color: var(--text-primary) !important;
        transition: all 0.3s ease;
    }}
    
    /* Loading Animation */
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}
    
    @keyframes slideIn {{
        from {{ transform: translateY(20px); opacity: 0; }}
        to {{ transform: translateY(0); opacity: 1; }}
    }}
    
    .loading-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }}
    
    .loading-spinner {{
        width: 40px;
        height: 40px;
        border: 3px solid var(--border);
        border-top: 3px solid var(--accent);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }}
    
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    
    /* Enhanced Header */
    .main-header {{
        # background: linear-gradient(135deg, var(--accent) 0%, #764ba2 100%);
background: linear-gradient(135deg, #3a3d5a, #6b5876);


        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
        animation: slideIn 0.6s ease-out;
    }}
    
    .main-header::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shimmer 3s infinite;
    }}
    
    @keyframes shimmer {{
        0% {{ left: -100%; }}
        100% {{ left: 100%; }}
    }}
    
    .main-title {{
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-align: center;
        position: relative;
        z-index: 1;
    }}
    
    .main-subtitle {{
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }}
    
    /* Enhanced Cards */
    .custom-card {{
        background: var(--card-bg);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid var(--border);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        animation: slideIn 0.6s ease-out;
    }}
    
    .custom-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }}
    
    /* Interactive Upload Zone */
    .upload-zone {{
        background: linear-gradient(135deg, var(--secondary-bg) 0%, var(--card-bg) 100%);
        padding: 3rem;
        border-radius: 20px;
        border: 3px dashed var(--accent);
        text-align: center;
        margin-bottom: 2rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .upload-zone:hover {{
        border-color: var(--text-primary);
        transform: scale(1.02);
    }}
    
    /* Enhanced Metrics */
    .metric-card {{
        background: linear-gradient(135deg, var(--card-bg) 0%, var(--secondary-bg) 100%);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        border: 1px solid var(--border);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, var(--accent), var(--text-primary));
    }}
    
    .metric-card:hover {{
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 12px 32px rgba(0,0,0,0.12);
    }}
    
    .metric-title {{
        font-size: 0.9rem;
        color: var(--text-secondary);
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .metric-value {{
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
    }}
    
    /* Status Pills */
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    .status-normal {{
        background: linear-gradient(135deg, var(--success), #059669);
        color: white;
    }}
    
    .status-abnormal {{
        background: linear-gradient(135deg, var(--error), #dc2626);
        color: white;
    }}
    
    .status-pill:hover {{
        transform: scale(1.05);
    }}
    
    /* Enhanced Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, var(--accent), var(--text-primary));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
    }}
    
    .stButton > button:active {{
        transform: translateY(0);
    }}
    
    /* Section Headers with Icons */
    .section-header {{
        display: flex;
        align-items: center;
        margin-bottom: 1.5rem;
        padding: 1rem 0;
        border-bottom: 2px solid var(--border);
        position: relative;
    }}
    
    .section-icon {{
        font-size: 1.8rem;
        margin-right: 1rem;
        background: linear-gradient(135deg, var(--accent), var(--text-primary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    .section-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }}
    
    /* Alert System */
    .custom-alert {{
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid;
        display: flex;
        align-items: center;
        gap: 1rem;
        animation: slideIn 0.5s ease-out;
    }}
    
    .alert-success {{
        background: rgba(16, 185, 129, 0.1);
        border-color: var(--success);
        color: var(--success);
    }}
    
    .alert-warning {{
        background: rgba(245, 158, 11, 0.1);
        border-color: var(--warning);
        color: var(--warning);
    }}
    
    .alert-error {{
        background: rgba(239, 68, 68, 0.1);
        border-color: var(--error);
        color: var(--error);
    }}
    
    /* Hide Streamlit Elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .stDeployButton {{display: none;}}
    
    /* Responsive Design */
    @media (max-width: 768px) {{
        .main-title {{
            font-size: 2rem;
        }}
        
        .custom-card {{
            padding: 1rem;
        }}
    }}
</style>
"""

# --------------- MODEL CALL FUNCTIONS ---------------
def call_gemini(prompt):
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model.generate_content(prompt).text.strip()
    except Exception as e:
        logger.error(f"Gemini call failed: {e}")
        return "Error: Unable to process with Gemini API"

def call_ollama(prompt):
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        return response.json().get("response", "")
    except Exception as e:
        logger.error("Ollama call failed: %s", e)
        return "Error: Unable to connect to Ollama"

# --------------- OCR FUNCTIONS ---------------
def extract_text_from_image(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert("RGB")
        temp_path = f"temp_{uuid4()}.png"
        image.save(temp_path)
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        result = ocr.ocr(temp_path, cls=True)
        if result and result[0]:
            text = "\n".join([line[1][0] for line in result[0]])
        else:
            text = "No text found in image"
        os.remove(temp_path) if os.path.exists(temp_path) else None
        return text
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return "Error: Could not extract text from image"

def extract_text_from_pdf(uploaded_pdf):
    try:
        reader = PdfReader(uploaded_pdf)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text if text.strip() else "No text found in PDF"
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return "Error: Could not extract text from PDF"

# --------------- DATA UTILS ---------------
def load_db():
    try:
        if os.path.exists(DB_FILE):
            with open(DB_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Database load failed: {e}")
    return []

def save_to_db(entry):
    try:
        db = load_db()
        entry["id"] = str(uuid4())
        entry["timestamp"] = datetime.now().isoformat()
        db.append(entry)
        os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
        with open(DB_FILE, "w", encoding='utf-8') as f:
            json.dump(db, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Database save failed: {e}")
        return False

# --------------- ENHANCED LOADING SYSTEM ---------------
def show_loading(message="Processing...", duration=2):
    """Enhanced loading animation"""
    placeholder = st.empty()
    with placeholder.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="loading-container">
                <div class="loading-spinner"></div>
                <div style="margin-left: 1rem; color: var(--text-primary); font-weight: 600;">
                    {message}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    progress_bar = st.progress(0)
    for i in range(duration * 10):
        progress_bar.progress((i + 1) / (duration * 10))
        time.sleep(0.1)
    
    placeholder.empty()

# --------------- NLP QUERY RUNNER ---------------
def run_nlp_query_simple(natural_query):
    db = load_db()
    if not db:
        st.warning("📊 No data available in database. Please upload some medical reports first.")
        return

    # Summarize database to keep prompt short but useful
    summarized_data = [
        {
            "Patient Name": p.get("Patient Name", ""),
            "Age": p.get("Age"),
            "Gender": p.get("Gender"),
            "Tests": [
                {
                    "Test Name": t.get("Test Name"),
                    "Result": t.get("Result"),
                    "Reference Range": t.get("Reference Range")
                } for t in p.get("Tests", [])[:3]  # Limit tests per patient
            ]
        } for p in db[:10]  # Limit number of patients in prompt
    ]

    query_prompt = f"""
You are a medical data assistant. Analyze the following summarized patient data and answer the question.

PATIENT DATA SAMPLE:
{json.dumps(summarized_data, indent=2)}

QUESTION: "{natural_query}"

Respond with a clear, human-friendly explanation. Avoid code or JSON formatting.
"""

    try:
        show_loading("🔍 Analyzing your query...", 2)
        response = call_gemini(query_prompt) if USE_GEMINI else call_ollama(query_prompt)
        
        if not response or response.strip() == "":
            st.error("⚠️ The AI model did not return a response. Try rephrasing your query.")
            return

        st.markdown('<div class="section-header"><span class="section-icon">💡</span><h3 class="section-title">Query Results</h3></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="custom-card">{response.strip()}</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="custom-alert alert-success">
            <span>✅</span>
            <span>Query completed successfully!</span>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.markdown(f"""
        <div class="custom-alert alert-error">
            <span>❌</span>
            <span>Query failed: {str(e)}</span>
        </div>
        """, unsafe_allow_html=True)


# --------------- ANALYTICS FUNCTIONS ---------------
def create_test_results_chart(df_data):
    if df_data.empty:
        return None
    
    colors = get_theme_colors()
    numeric_results = pd.to_numeric(df_data['Result'], errors='coerce')
    valid_data = df_data[~numeric_results.isna()].copy()
    valid_data['Numeric_Result'] = numeric_results[~numeric_results.isna()]
    
    if valid_data.empty:
        return None
    
    fig = px.bar(
        valid_data, 
        x='Test Name', 
        y='Numeric_Result',
        color='In Range',
        color_discrete_map={True: colors['success'], False: colors['error']},
        title="📊 Test Results Analysis",
        labels={'Numeric_Result': 'Result Value', 'In Range': 'Status'}
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family="Inter",
        title_font_size=18,
        title_x=0.5,
        font_color=colors['text_primary']
    )
    
    return fig

def get_database_stats():
    db = load_db()
    if not db:
        return {"total_patients": 0, "total_tests": 0, "avg_age": 0, "gender_dist": {}}
    
    total_patients = len(db)
    total_tests = sum(len(patient.get("Tests", [])) for patient in db)
    ages = [patient.get("Age") for patient in db if patient.get("Age") and isinstance(patient.get("Age"), (int, float))]
    avg_age = sum(ages) / len(ages) if ages else 0
    
    genders = [patient.get("Gender") for patient in db if patient.get("Gender")]
    gender_dist = {gender: genders.count(gender) for gender in set(genders)}
    
    return {
        "total_patients": total_patients,
        "total_tests": total_tests,
        "avg_age": round(avg_age, 1),
        "gender_dist": gender_dist
    }

def analyze_test_trends():
    """Analyze trends in test results"""
    db = load_db()
    if not db:
        return None, None
    
    all_tests = []
    for patient in db:
        patient_tests = patient.get("Tests", [])
        for test in patient_tests:
            test_data = {
                'Patient': patient.get('Patient Name', 'Unknown'),
                'Age': patient.get('Age'),
                'Gender': patient.get('Gender'),
                'Test Name': test.get('Test Name'),
                'Result': test.get('Result'),
                'Unit': test.get('Unit'),
                'Reference Range': test.get('Reference Range')
            }
            all_tests.append(test_data)
    
    df = pd.DataFrame(all_tests)
    if df.empty:
        return None, None
    
    # Count test frequency
    test_counts = df['Test Name'].value_counts().head(10)
    
    # Analyze abnormal results
    abnormal_patterns = df.groupby('Test Name').apply(
        lambda x: len([r for r in x['Result'] if 'abnormal' in str(r).lower() or 'high' in str(r).lower() or 'low' in str(r).lower()])
    ).sort_values(ascending=False).head(5)
    
    return test_counts, abnormal_patterns

# --------------- MAIN UI ---------------

# Apply dynamic CSS
st.markdown(get_dynamic_css(), unsafe_allow_html=True)

# Theme Toggle in Header
col_theme, col_title = st.columns([1, 8])
with col_theme:
    theme_icon = "🌙" if not st.session_state.dark_mode else "☀️"
    if st.button(f"{theme_icon}", key="theme_toggle", help="Toggle Dark/Light Mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# Enhanced Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🏥 MediSaarthi</h1>
    <p class="main-subtitle">A Smart AI Assistant for Medical Report Structuring & AI Analytics Platform</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with Enhanced Statistics
with st.sidebar:
    st.markdown('<div class="section-header"><span class="section-icon">📊</span><h3 class="section-title">Dashboard</h3></div>', unsafe_allow_html=True)
    
    stats = get_database_stats()
    
    # Enhanced Metrics with animations
    metrics_data = [
        ("👥 Total Patients", stats['total_patients']),
        ("🧪 Total Tests", stats['total_tests']),
        ("📅 Avg Age", f"{stats['avg_age']} yrs" if stats['avg_age'] > 0 else "N/A")
    ]
    
    for title, value in metrics_data:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Gender Distribution with enhanced chart
    if stats['gender_dist']:
        st.markdown('<div class="section-header"><span class="section-icon">👥</span><h4 class="section-title">Demographics</h4></div>', unsafe_allow_html=True)
        gender_df = pd.DataFrame(list(stats['gender_dist'].items()), columns=['Gender', 'Count'])
        colors = get_theme_colors()
        fig_gender = px.pie(
            gender_df, 
            values='Count', 
            names='Gender',
            color_discrete_sequence=[colors['accent'], colors['text_primary'], '#f093fb'],
            title="Gender Distribution"
        )
        fig_gender.update_layout(
            height=280, 
            showlegend=True,
            font_color=colors['text_primary'],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_gender, use_container_width=True)
    
    # Enhanced Sample Queries
    st.markdown('<div class="section-header"><span class="section-icon">💬</span><h4 class="section-title">Quick Queries</h4></div>', unsafe_allow_html=True)
    
    sample_queries = [
        ("📈 Hemoglobin Analysis", "Show average Hemoglobin levels for female patients"),
        ("⚠️ Abnormal Results", "List all tests where the result is out of range"),
        ("🔬 RBC Analysis", "How many patients have RBC count > 4.5?"),
        ("🏥 TSH Results", "Show abnormal TSH results"),
        ("👫 Gender Comparison", "Compare test results by gender")
    ]
    
    for icon_title, query in sample_queries:
        if st.button(icon_title, key=f"sample_{hash(query)}", use_container_width=True):
            st.session_state.sample_query = query

# Main Content with Enhanced Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "🔍 Query Database", "📈 Analytics Dashboard", "⚙️ Settings"])
def check_test_range(result, ref_range):
    """Check if test result is within reference range"""
    try:
        if not result or not ref_range or ref_range == "N/A":
            return True
        
        # Extract numeric value from result
        result_match = re.search(r'(\d+\.?\d*)', str(result))
        if not result_match:
            return True
        
        result_val = float(result_match.group(1))
        
        # Parse reference range
        range_match = re.search(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', str(ref_range))
        if range_match:
            min_val, max_val = float(range_match.group(1)), float(range_match.group(2))
            return min_val <= result_val <= max_val
        
        return True
    except:
        return True

with tab1:
    st.markdown('<div class="section-header"><span class="section-icon">📤</span><h2 class="section-title">Document Upload & AI Processing</h2></div>', unsafe_allow_html=True)
    
    # Enhanced Upload Zone
    st.markdown("""
    <div class="upload-zone">
        <h3>🎯 Drop Your Medical Reports Here</h3>
        <p>Supports: PDF, PNG, JPG, JPEG | Max size: 200MB</p>
        <p>✨ Powered by Advanced OCR & AI Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded = st.file_uploader("Choose file", type=["pdf", "png", "jpg", "jpeg"], label_visibility="collapsed")
    
    if uploaded:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="section-header"><span class="section-icon">📄</span><h3 class="section-title">Document Preview</h3></div>', unsafe_allow_html=True)
            
            # Processing with enhanced loading
            show_loading("🔍 Extracting text from document...", 3)
            
            if uploaded.type == "application/pdf":
                st.markdown("""
                <div class="custom-alert alert-success">
                    <span>📄</span>
                    <span>PDF document processed successfully</span>
                </div>
                """, unsafe_allow_html=True)
                text = extract_text_from_pdf(uploaded)
            else:
                st.image(uploaded, caption="📸 Uploaded Medical Report", use_container_width=True)
                text = extract_text_from_image(uploaded)
            
            with st.expander("👁️ View Extracted Text", expanded=False):
                st.text_area("Raw OCR Output", text, height=200, disabled=True)
        
        with col2:
            st.markdown('<div class="section-header"><span class="section-icon">🤖</span><h3 class="section-title">AI Analysis Results</h3></div>', unsafe_allow_html=True)
            
            show_loading("🤖 AI is analyzing the medical report...", 4)
            
            prompt = f"""
You are a medical report extractor. From the following report text, extract structured JSON like:
{{
  "Patient Name": "",
  "Age": null,
  "Gender": "",
  "Tests": [
    {{
      "Test Name": "",
      "Result": "",
      "Unit": "",
      "Reference Range": ""
    }}
  ]
}}

Extract all test results found. If values are missing, use null for numbers and empty string for text.
TEXT: <<< {text} >>>

Return only valid JSON, no additional text.
"""
            
            try:
                response = call_gemini(prompt) if USE_GEMINI else call_ollama(prompt)
                match = re.search(r'\{[\s\S]*\}', response)
                
                if match:
                    data = json.loads(match.group())
                    
                    # Enhanced Patient Info Display
                    st.markdown("#### 👤 Patient Information")
                    info_cols = st.columns(3)
                    with info_cols[0]:
                        st.metric("Name", data.get("Patient Name", "Unknown"))
                    with info_cols[1]:
                        st.metric("Age", f"{data.get('Age', 'N/A')} years" if data.get('Age') else "N/A")
                    with info_cols[2]:
                        st.metric("Gender", data.get("Gender", "Unknown"))
                    
                    # Enhanced Test Results Display
                    if data.get("Tests"):
                        st.markdown("#### 🧪 Test Results")
                        
                        # Create DataFrame for better display
                        tests_df = pd.DataFrame(data["Tests"])
                        
                        # Add status column
                        # tests_df['In Range
                        tests_df['In Range'] = tests_df.apply(
                            lambda row: check_test_range(row['Result'], row['Reference Range']), 
                            axis=1
                        )
                        
                        # Display tests in enhanced format
                        for idx, test in tests_df.iterrows():
                            status_class = "status-normal" if test.get('In Range', False) else "status-abnormal"
                            status_text = "✅ Normal" if test.get('In Range', False) else "⚠️ Abnormal"
                            
                            st.markdown(f"""
                            <div class="custom-card">
                                <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 0.5rem;">
                                    <h4 style="margin: 0; color: var(--text-primary);">{test['Test Name']}</h4>
                                    <span class="status-pill {status_class}">{status_text}</span>
                                </div>
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                                    <div><strong>Result:</strong> {test['Result']} {test.get('Unit', '')}</div>
                                    <div><strong>Reference:</strong> {test.get('Reference Range', 'N/A')}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Save to database
                        if st.button("💾 Save Analysis to Database", type="primary", use_container_width=True):
                            if save_to_db(data):
                                st.success("✅ Report saved successfully!")
                                st.balloons()
                            else:
                                st.error("❌ Failed to save report")
                    
                    else:
                        st.markdown("""
                        <div class="custom-alert alert-warning">
                            <span>⚠️</span>
                            <span>No test results found in the document</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                else:
                    st.markdown("""
                    <div class="custom-alert alert-error">
                        <span>❌</span>
                        <span>Failed to parse medical report. Please try again with a clearer image.</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.markdown(f"""
                <div class="custom-alert alert-error">
                    <span>❌</span>
                    <span>Analysis failed: {str(e)}</span>
                </div>
                """, unsafe_allow_html=True)


with tab2:
    st.markdown('<div class="section-header"><span class="section-icon">🔍</span><h2 class="section-title">Natural Language Database Query</h2></div>', unsafe_allow_html=True)
    
    # Enhanced Query Interface
    st.markdown("""
    <div class="custom-card">
        <h3>🗣️ Ask Questions About Your Medical Data</h3>
        <p>Use natural language to query your medical database. Ask about specific patients, test results, trends, or comparisons.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Query input with enhanced styling
    query_input = st.text_input(
        "💬 Enter your question:",
        placeholder="e.g., Show me all patients with high cholesterol levels",
        key="main_query"
    )
    
    # Handle sample query from sidebar
    if 'sample_query' in st.session_state:
        query_input = st.session_state.sample_query
        del st.session_state.sample_query
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔍 Execute Query", type="primary", use_container_width=True):
            if query_input.strip():
                run_nlp_query_simple(query_input)
            else:
                st.warning("⚠️ Please enter a query first")
    
    with col2:
        if st.button("🧹 Clear Results", use_container_width=True):
            st.rerun()

with tab3:
    st.markdown('<div class="section-header"><span class="section-icon">📈</span><h2 class="section-title">Advanced Analytics Dashboard</h2></div>', unsafe_allow_html=True)
    
    db = load_db()
    if not db:
        st.markdown("""
        <div class="custom-alert alert-warning">
            <span>📊</span>
            <span>No data available for analytics. Please upload some medical reports first.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Create comprehensive analytics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header"><span class="section-icon">🧪</span><h3 class="section-title">Test Frequency Analysis</h3></div>', unsafe_allow_html=True)
            
            test_counts, abnormal_patterns = analyze_test_trends()
            if test_counts is not None:
                colors = get_theme_colors()
                fig_tests = px.bar(
                    x=test_counts.values,
                    y=test_counts.index,
                    orientation='h',
                    title="Most Common Tests",
                    color=test_counts.values,
                    color_continuous_scale=['#667eea', '#764ba2']
                )
                fig_tests.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color=colors['text_primary']
                )
                st.plotly_chart(fig_tests, use_container_width=True)
        
        with col2:
            st.markdown('<div class="section-header"><span class="section-icon">⚠️</span><h3 class="section-title">Abnormal Results Pattern</h3></div>', unsafe_allow_html=True)
            
            if abnormal_patterns is not None and not abnormal_patterns.empty:
                


                if abnormal_patterns is not None and not abnormal_patterns.empty:
                    abnormal_df = pd.DataFrame({
                        "Test Name": abnormal_patterns.index,
                        "Abnormal Count": abnormal_patterns.values
                    })

                    if len(abnormal_df) > 1:
                        fig_abnormal = px.pie(
            abnormal_df,
            names="Test Name",
            values="Abnormal Count",
            title="Tests with Abnormal Results",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
                fig_abnormal.update_layout(
            height=400,
            font_color=colors['text_primary'],
            paper_bgcolor='rgba(0,0,0,0)'
        )
                st.plotly_chart(fig_abnormal, use_container_width=True)
            else:
                st.info("✅ Only one test shows abnormal results — no pie chart needed.")





                fig_abnormal.update_layout(
                    height=400,
                    font_color=colors['text_primary'],
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_abnormal, use_container_width=True)
        
        # Age Distribution Analysis
        st.markdown('<div class="section-header"><span class="section-icon">📊</span><h3 class="section-title">Patient Demographics</h3></div>', unsafe_allow_html=True)
        
        ages = [p.get('Age') for p in db if p.get('Age') and isinstance(p.get('Age'), (int, float))]
        if ages:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("👥 Total Patients", len(db))
            with col2:
                st.metric("📅 Average Age", f"{sum(ages)/len(ages):.1f} years")
            with col3:
                st.metric("📈 Age Range", f"{min(ages)}-{max(ages)} years")
            
            # Age distribution histogram
            fig_age = px.histogram(
                x=ages,
                nbins=10,
                title="Age Distribution of Patients",
                color_discrete_sequence=[colors['accent']]
            )
            fig_age.update_layout(
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=colors['text_primary']
            )
            st.plotly_chart(fig_age, use_container_width=True)
        
        # Test Results Summary Table
        # st.markdown('<div class="section-header"><span class="section-icon">📋</span><h3 class="section-title">Recent Reports Summary</h3></div>', unsafe_allow_html=True)
        
        # Create summary table

        # Full Patient Data Table in Analytics Dashboard
st.markdown('<div class="section-header"><span class="section-icon">📋</span><h3 class="section-title">All Patient Records</h3></div>', unsafe_allow_html=True)

full_data = []
for patient in db:
    for test in patient.get('Tests', []):
        full_data.append({
            'Patient Name': patient.get('Patient Name', 'Unknown'),
            'Age': patient.get('Age', 'N/A'),
            'Gender': patient.get('Gender', 'Unknown'),
            'Test Name': test.get('Test Name', 'N/A'),
            'Result': test.get('Result', 'N/A'),
            'Unit': test.get('Unit', ''),
            'Reference Range': test.get('Reference Range', ''),
            'Timestamp': patient.get('timestamp', 'N/A')[:10] if patient.get('timestamp') else 'N/A'
        })

if full_data:
    df_full = pd.DataFrame(full_data)
    st.dataframe(df_full, use_container_width=True)
else:
    st.info("📂 No patient data available.")

        

with tab4:
    st.markdown('<div class="section-header"><span class="section-icon">⚙️</span><h2 class="section-title">Application Settings</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header"><span class="section-icon">🤖</span><h3 class="section-title">AI Model Configuration</h3></div>', unsafe_allow_html=True)
        
        current_model = "Gemini 1.5 Flash" if USE_GEMINI else "Ollama Llama3"
        st.info(f"🔧 Current AI Model: **{current_model}**")
        
        st.markdown("""
        <div class="custom-card">
            <h4>Model Information</h4>
            <ul>
                <li><strong>Gemini:</strong> Google's advanced AI model for medical text analysis</li>
                <li><strong>Ollama:</strong> Local LLM for privacy-focused processing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-header"><span class="section-icon">🗃️</span><h3 class="section-title">Database Management</h3></div>', unsafe_allow_html=True)
        
        db_stats = get_database_stats()
        st.metric("📁 Database Size", f"{db_stats['total_patients']} patients")
        st.metric("💾 Storage Location", DB_FILE.split('/')[-1])
        
        if st.button("🗑️ Clear Database", type="secondary", use_container_width=True):
            if st.button("⚠️ Confirm Delete All Data", use_container_width=True):
                try:
                    if os.path.exists(DB_FILE):
                        os.remove(DB_FILE)
                    st.success("✅ Database cleared successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to clear database: {e}")
    
    # Export Options
    st.markdown('<div class="section-header"><span class="section-icon">📤</span><h3 class="section-title">Data Export</h3></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export as CSV", use_container_width=True):
            db = load_db()
            if db:
                # Flatten data for CSV export
                export_data = []
                for patient in db:
                    for test in patient.get('Tests', []):
                        export_data.append({
                            'Patient_Name': patient.get('Patient Name'),
                            'Age': patient.get('Age'),
                            'Gender': patient.get('Gender'),
                            'Test_Name': test.get('Test Name'),
                            'Result': test.get('Result'),
                            'Unit': test.get('Unit'),
                            'Reference_Range': test.get('Reference Range'),
                            'Timestamp': patient.get('timestamp')
                        })
                
                df_export = pd.DataFrame(export_data)
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="💾 Download CSV",
                    data=csv,
                    file_name=f"medlab_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with col2:
        if st.button("📋 Export as JSON", use_container_width=True):
            db = load_db()
            if db:
                json_str = json.dumps(db, indent=2, ensure_ascii=False)
                st.download_button(
                    label="💾 Download JSON",
                    data=json_str,
                    file_name=f"medlab_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    with col3:
        if st.button("📈 Generate Report", use_container_width=True):
            db = load_db()
            if db:
                report = generate_medical_report(db)
                st.download_button(
                    label="💾 Download Report",
                    data=report,
                    file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

def generate_medical_report(db_data):
    """Generate a comprehensive medical report"""
    report = f"""
MEDLAB ANALYTICS PRO - COMPREHENSIVE MEDICAL REPORT
Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
================================================

SUMMARY STATISTICS:
- Total Patients: {len(db_data)}
- Total Tests Processed: {sum(len(p.get('Tests', [])) for p in db_data)}
- Report Generation Date: {datetime.now().strftime('%Y-%m-%d')}

PATIENT DEMOGRAPHICS:
"""
    
    ages = [p.get('Age') for p in db_data if p.get('Age')]
    if ages:
        report += f"- Average Age: {sum(ages)/len(ages):.1f} years\n"
        report += f"- Age Range: {min(ages)} - {max(ages)} years\n"
    
    genders = [p.get('Gender') for p in db_data if p.get('Gender')]
    if genders:
        gender_counts = {g: genders.count(g) for g in set(genders)}
        for gender, count in gender_counts.items():
            report += f"- {gender}: {count} patients\n"
    
    report += "\nDETAILED PATIENT RECORDS:\n" + "="*50 + "\n"
    
    for i, patient in enumerate(db_data, 1):
        report += f"\nPATIENT #{i}:\n"
        report += f"Name: {patient.get('Patient Name', 'Unknown')}\n"
        report += f"Age: {patient.get('Age', 'N/A')}\n"
        report += f"Gender: {patient.get('Gender', 'Unknown')}\n"
        report += f"Tests Performed: {len(patient.get('Tests', []))}\n"
        
        if patient.get('Tests'):
            report += "\nTest Results:\n"
            for test in patient.get('Tests', []):
                status = "Normal" if check_test_range(test.get('Result'), test.get('Reference Range')) else "Abnormal"
                report += f"  • {test.get('Test Name', 'Unknown')}: {test.get('Result', 'N/A')} {test.get('Unit', '')} [{status}]\n"
                report += f"    Reference: {test.get('Reference Range', 'N/A')}\n"
        
        report += "-" * 40 + "\n"
    
    report += f"\nReport generated by MedLab Analytics Pro v2.0\n"
    report += f"For technical support, contact: support@medlabanalytics.com\n"
    
    return report

# Footer
st.markdown("""
<div style="margin-top: 3rem; padding: 2rem; text-align: center; border-top: 1px solid var(--border);">
    <p style="color: var(--text-secondary); margin: 0;">
        🏥 <strong>MedLab Analytics Pro</strong> v2.0 | 
        Powered by Advanced AI & OCR Technology | 
        <span style="color: var(--accent);">Next-Generation Medical Analysis</span>
    </p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh for real-time updates
if st.button("🔄 Refresh Dashboard", key="refresh_main"):
    st.rerun()
