import os
import json
import re
import pandas as pd
import easyocr
import logging
import streamlit as st
import matplotlib.pyplot as plt
import requests
from uuid import uuid4
from io import BytesIO
from PyPDF2 import PdfReader
from paddleocr import PaddleOCR
from PIL import Image

# --------------- CONFIGURATION ---------------
USE_GEMINI = True  # Toggle manually in code
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
DB_FILE = "C:/Users/archi/Desktop/hackday/outputsNew/all_results_new.json"
GEMINI_API_KEY = "AIzaSyAAPOvYkR2jVEj6s7cYZlCbPhzD7tdTZrk"

# --------------- LOGGER SETUP ---------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("App")

# --------------- MODEL CALL FUNCTIONS ---------------
def call_gemini(prompt):
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    return model.generate_content(prompt).text.strip()

def call_ollama(prompt):
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        return response.json().get("response", "")
    except Exception as e:
        logger.error("Ollama call failed: %s", e)
        return ""

# --------------- OCR FUNCTIONS ---------------
def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image.save("temp.png")
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    result = ocr.ocr("temp.png", cls=True)
    text = "\n".join([line[1][0] for line in result[0]])
    return text

def extract_text_from_pdf(uploaded_pdf):
    reader = PdfReader(uploaded_pdf)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

# --------------- DATA UTILS ---------------
def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE) as f:
            return json.load(f)
    return []

def save_to_db(entry):
    db = load_db()
    entry["id"] = str(uuid4())
    db.append(entry)
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=2)

# --------------- NLP QUERY RUNNER ---------------
def run_nlp_query_simple(natural_query):
    db = load_db()
    query_prompt = f"""

{json.dumps(db, indent=2)}

Answer this user question by analyzing the data:
"{natural_query}"

Return only the final answer as a sentence or table (no JSON).
"""
    try:
        response = call_gemini(query_prompt) if USE_GEMINI else call_ollama(query_prompt)
        st.subheader("📥 Answer to your query")
        st.markdown(response.strip())
    except Exception as e:
        st.error(f"❌ Failed: {e}")

# --------------- STREAMLIT UI ---------------
st.set_page_config(page_title="Lab Report Extractor", layout="wide")
st.title("🧬 Lab Report Extractor")

uploaded = st.file_uploader("📤 Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])
if uploaded:
    if uploaded.type == "application/pdf":
        text = extract_text_from_pdf(uploaded)
    else:
        text = extract_text_from_image(uploaded)

    st.subheader("📄 Extracted Text")

    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
You are a medical report extractor. From the following report text, extract structured JSON like:
{{
  "Patient Name": "",
  "Age": ,
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
If values are missing, use null. TEXT: <<< {text} >>>
"""
    try:
        response = model.generate_content(prompt).text
        match = re.search(r'\{[\s\S]+\}', response)
        data = json.loads(match.group()) if match else {}
        st.subheader("🧬 Extracted Structured Data (JSON)")
        st.json(data)

        st.subheader("📊 Extracted Data Table")
        df_data = pd.DataFrame(data.get("Tests", []))
        st.dataframe(df_data)

        def is_within_range(row):
            ref = row['Reference Range']
            try:
                result = float(row['Result'])
                match = re.findall(r"([\d.]+)[\s-]+([\d.]+)", ref)
                if match:
                    low, high = map(float, match[0])
                    return low <= result <= high
            except:
                pass
            return False

        df_data['In Range'] = df_data.apply(is_within_range, axis=1)
        bar_colors = df_data['In Range'].map({True: 'green', False: 'red'})

        fig, ax = plt.subplots()
        ax.bar(df_data['Test Name'], pd.to_numeric(df_data['Result'], errors='coerce'), color=bar_colors)
        ax.set_ylabel("Result")
        ax.set_title("Test Results vs Reference Range")
        ax.set_xticklabels(df_data['Test Name'], rotation=45, ha='right')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Failed to extract structured data: {e}")

st.subheader("📊 Natural Language Query on Lab Data")
query = st.text_input("Ask a question (e.g. Show average Hemoglobin levels for female patients)")
if query:
    run_nlp_query_simple(query)

# --------------- SAMPLE QUESTIONS ---------------
with st.sidebar.expander("💬 Sample Questions", expanded=True):
    st.markdown("""
- **Show average Hemoglobin levels for female patients**
- **List all tests where the result is out of the reference range**
- **How many patients have an RBC count greater than 4.5?**
- **Show all test results for patient named Niketa**
- **List all abnormal TSH results**
""")

# --------------- DATABASE VIEWER ---------------
with st.expander("📂 Show Raw Patient Database", expanded=False):
    try:
        db_raw = load_db()
        for entry in db_raw:
            with st.container(border=True):
                st.markdown(f"### 🧾 {entry.get('Patient Name', 'N/A')}")
                col1, col2 = st.columns(2)
                col1.markdown(f"**Age:** {entry.get('Age', 'N/A')}")
                col2.markdown(f"**Gender:** {entry.get('Gender', 'N/A')}")
                tests = entry.get("Tests", [])
                if tests:
                    df_tests = pd.DataFrame(tests)
                    st.dataframe(df_tests, use_container_width=True, height=min(35 * len(tests), 400))
                else:
                    st.info("No tests found for this patient.")
    except Exception as e:
        st.warning("Could not load database.")
        st.error(e)

