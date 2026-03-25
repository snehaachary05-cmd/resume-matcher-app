import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import string

# Setup
st.set_page_config(page_title="Resume Matcher", page_icon="📄")
nltk.download('stopwords', quiet=True)

# -------------------------------
# Function to extract text from PDF
# -------------------------------
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# -------------------------------
# Text preprocessing
# -------------------------------
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

# -------------------------------
# UI
# -------------------------------
st.title("📄 Resume Matcher")
st.write("Upload a Job Description and multiple resumes to compare.")

job_description = st.file_uploader("Upload Job Description (TXT or PDF)", type=["txt", "pdf"])
resumes = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

# -------------------------------
# Processing
# -------------------------------
if job_description and resumes:

    # Handle JD (PDF or TXT)
    if job_description.type == "application/pdf":
        jd_text = extract_text(job_description)
    else:
        jd_text = job_description.read().decode("utf-8")

    jd_text = preprocess(jd_text)

    resume_texts = []
    resume_names = []

    for resume in resumes:
        text = extract_text(resume)
        text = preprocess(text)
        resume_texts.append(text)
        resume_names.append(resume.name)

    # Combine JD + resumes
    documents = [jd_text] + resume_texts

    # TF-IDF
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)

    jd_vector = vectors[0]

    # Similarity scores
    scores = []
    for i in range(1, vectors.shape[0]):
        score = cosine_similarity(jd_vector, vectors[i])[0][0]
        scores.append(score)

    # Sort results
    results = list(zip(resume_names, scores))
    results.sort(key=lambda x: x[1], reverse=True)

    # -------------------------------
    # Display Results
    # -------------------------------
    st.subheader("📊 Resume Ranking")

    for name, score in results:
        percentage = round(score * 100, 2)
        st.write(f"**{name}** — {percentage}% match")
        st.progress(int(percentage))
