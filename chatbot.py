import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader

GOOGLE_API_KEY = "AIzaSyAldrlmJDsWzdwKoHUwFel_T2KlMMh49uM"
genai.configure(api_key=GOOGLE_API_KEY)

pdf_file_path = r"D:\Projects\Brain_Tumour\Brain_tumor_Final\brain_tumor_pdf.pdf"  

def page_setup():
    st.markdown("""
        <style>
            .sidebar .sidebar-content {
                background-color: #f0f2f6;
            }
            .stButton > button {
                background-color: #007bff;
                color: white;
                border-radius: 4px;
                padding: 10px 20px;
            }
            .stTextInput input {
                border-radius: 4px;
                padding: 10px;
            }
        </style>
        """, unsafe_allow_html=True)

    st.header("🧠 Brain Tumor ChatBot")
    st.markdown("Interact with the chatbot to get insights.")

def get_llminfo():
    st.sidebar.header("Options")
    model = st.sidebar.radio(
        "Choose LLM:",
        ("gemini-1.5-flash", "gemini-1.5-pro"),
        help="Select a model you want to use."
    )
    temperature = st.sidebar.slider(
        "Temperature:",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.25,
        help="Lower temperatures are good for less open-ended responses."
    )
    top_p = st.sidebar.slider(
        "Top P:",
        min_value=0.0,
        max_value=1.0,
        value=0.94,
        step=0.01,
        help="Used for nucleus sampling."
    )
    max_tokens = st.sidebar.slider(
        "Maximum Tokens:",
        min_value=100,
        max_value=5000,
        value=2000,
        step=100,
        help="Number of response tokens."
    )
    return model, temperature, top_p, max_tokens


def app():
    page_setup()
    model, temperature, top_p, max_tokens = get_llminfo()

    # Extract text from PDF
    text = ""
    pdf_reader = PdfReader(pdf_file_path)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_tokens,
        "response_mime_type": "text/plain",
    }
    model_instance = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
    )

    question = st.text_input("Enter your question:")
    if st.button("Submit"):
        response = model_instance.generate_content([question, "response the answer similar to the text given:", text])
        st.write(response.text)
