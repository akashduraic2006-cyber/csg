import streamlit as st
import fitz  # PyMuPDF for PDF reading
import re
import pyttsx3
import speech_recognition as sr

# MOCK IBM WatsonX API
class MockModel:
    def __init__(self, *args, **kwargs):
        pass
    def generate(self, prompt=None, **kwargs):
        return {'results': [{'generated_text': f"MOCK RESPONSE: {prompt[:50]}..."}]}

# Use mock model for development
Model = MockModel
Credentials = object


# ================= CONFIG =================
# IBM Watsonx credentials (replace with yours)
WATSONX_API_KEY = "your_api_key"
PROJECT_ID = "your_project_id"
MODEL_ID = "mistralai/mixtral-8x3b-instruct"



# Initialize mock model
model = Model(
    model_id=MODEL_ID,
    params={"max_new_tokens": 512, "temperature": 0.5},
    credentials=None,
    project_id=PROJECT_ID
)

# ================= HELPERS =================
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500):
    """Split text into smaller chunks for efficient LLM queries."""
    sentences = re.split(r'(?<=[.?!]) +', text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) <= chunk_size:
            current += " " + sentence
        else:
            chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())
    return chunks

def ask_watsonx(question, context=""):
    """Send question + context to IBM Watsonx LLM."""
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer in simple terms:"
    response = model.generate(prompt=prompt)
    return response['results'][0]['generated_text']

def generate_quiz(content, q_type="mcq"):
    """Generate quizzes/flashcards/MCQs from content."""
    prompt = f"Create 5 {q_type.upper()} from the following study material:\n{content}"
    response = model.generate(prompt=prompt)
    return response['results'][0]['generated_text']

def speech_to_text():
    """Record voice and convert to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak your question...")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
    try:
        return recognizer.recognize_google(audio)
    except:
        return "Sorry, I couldn't understand."

def text_to_speech(text):
    """Read out text answer."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# ================= WEB APP =================
st.set_page_config(page_title="CSG - AI Study Guide", layout="wide")

st.title("📘 CSG - Cognitive Study Guide")
st.caption("AI Powered PDF Q&A System (IBM Watsonx Mixtral 8x3B)")

pdf_file = st.file_uploader("📂 Upload a PDF file", type=["pdf"])

if pdf_file:
    raw_text = extract_text_from_pdf(pdf_file)
    chunks = chunk_text(raw_text)

    st.success("✅ PDF processed successfully!")

    # ----------- Q&A -----------
    st.header("❓ Ask Questions from PDF")
    col1, col2 = st.columns([2, 1])

    with col1:
        user_q = st.text_input("Enter your question here:")
    with col2:
        if st.button("🎙️ Ask by Voice"):
            user_q = speech_to_text()
            st.write(f"Your question: {user_q}")

    if user_q:
        context = " ".join(chunks[:5])  # basic context
        answer = ask_watsonx(user_q, context)
        st.subheader("🧾 Answer")
        st.write(answer)

        if st.button("🔊 Read Answer Aloud"):
            text_to_speech(answer)

    # ----------- Study Aids -----------
    st.header("📑 Auto-Generate Study Aids")

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("📘 Generate MCQs"):
            st.write(generate_quiz(raw_text[:2000], "mcq"))

    with c2:
        if st.button("📝 Generate Flashcards"):
            st.write(generate_quiz(raw_text[:2000], "flashcards"))

    with c3:
        if st.button("🎯 Generate Quizzes"):
            st.write(generate_quiz(raw_text[:2000], "quiz"))