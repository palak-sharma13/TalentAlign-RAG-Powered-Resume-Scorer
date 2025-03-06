import os
import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
import pandas as pd
import torch
import spacy
from sentence_transformers import SentenceTransformer, util

# Load SBERT Model for resume-job description similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load NLP Model (NER) for keyword extraction
nlp = spacy.load("en_core_web_sm")

# Define file paths for resume and job description CSV files
resume_file = "fixed_resume.csv"
job_desc_file = "fixed_jd.csv"

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Function to extract keywords using Named Entity Recognition (NER)
def extract_keywords(text):
    doc = nlp(text)
    keywords = [ent.text.lower() for ent in doc.ents if ent.label_ in ["PRODUCT", "SKILL", "WORK_OF_ART", "EVENT"]]
    return list(set(keywords))

# Load existing Resume & Job Description data (if files exist)
resume_data, job_data = None, None

if os.path.exists(resume_file):
    resume_data = pd.read_csv(resume_file)
    resume_data.columns = resume_data.columns.str.strip().str.lower()

if os.path.exists(job_desc_file):
    job_data = pd.read_csv(job_desc_file)
    job_data.columns = job_data.columns.str.strip().str.lower()

# Validate required columns in CSV files
required_resume_columns = {"resume id", "name", "skills"}
required_jd_columns = {"job role", "keywords"}

if job_data is not None and not required_jd_columns.issubset(set(job_data.columns)):
    st.error("Error: 'fixed_jd.csv' must contain 'Job Role' and 'Keywords' columns.")
    job_data = None

if resume_data is not None and not required_resume_columns.issubset(set(resume_data.columns)):
    st.error("Error: 'fixed_resume.csv' must contain 'Resume ID', 'Name', and 'Skills' columns.")
    resume_data = None

# Streamlit UI
st.title("TalentAlign Resume Ranking System")

# Sidebar Menu
menu = ["Home", "Upload Job Description", "Upload Resumes", "Rank Resumes"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("Welcome to TalentAlign Resume Scorer")
    st.write("Upload job descriptions and resumes, and get ranked matches.")

elif choice == "Upload Job Description":
    st.subheader("Upload Job Description (JD)")
    job_role = st.text_input("Enter Job Role")
    job_keywords = st.text_area("Enter Keywords (comma-separated)")

    if st.button("Save Job Description"):
        if job_role and job_keywords:
            new_jd = pd.DataFrame({"Job Role": [job_role], "Keywords": [job_keywords]})

            if os.path.exists(job_desc_file):
                old_jd = pd.read_csv(job_desc_file)
                new_jd = pd.concat([old_jd, new_jd], ignore_index=True)

            new_jd.to_csv(job_desc_file, index=False)
            st.success("Job Description saved successfully!")
        else:
            st.warning("Please enter both Job Role and Keywords.")

elif choice == "Upload Resumes":
    st.subheader("Upload Resumes (PDF)")
    uploaded_files = st.file_uploader("Upload multiple resumes", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        resume_texts = []
        resume_names = []

        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            if text:
                resume_texts.append(text)
                resume_names.append(file.name)

                # Display extracted text
                st.subheader(f"Extracted Text from {file.name}:")
                st.text_area("Resume Content", text, height=150)

        if resume_texts:
            st.success(f"{len(resume_texts)} resumes uploaded successfully!")

elif choice == "Rank Resumes":
    st.subheader("Resume Ranking System")

    if job_data is not None and resume_data is not None:
        uploaded_files = st.file_uploader("Upload resumes for ranking", type=["pdf"], accept_multiple_files=True)

        if uploaded_files:
            new_resume_texts = []
            new_resume_names = []

            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                if text:
                    new_resume_texts.append(text)
                    new_resume_names.append(file.name)

            if new_resume_texts:
                # Extract keywords for new resumes
                new_resume_keywords = [extract_keywords(text) for text in new_resume_texts]

                # Extract keywords for existing resumes
                existing_resume_keywords = resume_data["skills"].astype(str).apply(extract_keywords)

                # Convert uploaded resumes & existing resumes to embeddings
                new_resume_embeddings = model.encode(
                    [" ".join(keywords) for keywords in new_resume_keywords], convert_to_tensor=True
                )
                existing_resume_embeddings = model.encode(
                    [" ".join(keywords) for keywords in existing_resume_keywords], convert_to_tensor=True
                )

                # Extract job description embeddings
                jd_embeddings = model.encode(
                    job_data["keywords"].astype(str).tolist(), convert_to_tensor=True
                )

                # Compute similarity between uploaded resumes & job descriptions
                similarity_with_jobs = util.cos_sim(new_resume_embeddings, jd_embeddings)

                # Compute similarity between uploaded resumes & existing resumes
                similarity_with_existing = util.cos_sim(new_resume_embeddings, existing_resume_embeddings)

                # Get best job match
                top_job_matches = similarity_with_jobs.argmax(dim=1).tolist()
                top_job_scores = similarity_with_jobs.max(dim=1).values.cpu().numpy().tolist()

                # Get best existing resume match
                top_existing_matches = similarity_with_existing.argmax(dim=1).tolist()
                top_existing_scores = similarity_with_existing.max(dim=1).values.cpu().numpy().tolist()

                # Create DataFrame for ranking results
                results_df = pd.DataFrame({
                    "Uploaded Resume": new_resume_names,
                    "Best Matching Job Role": [job_data.iloc[i]["job role"] for i in top_job_matches],
                    "Job Match Score": [f"{round(score * 100, 2)}%" for score in top_job_scores],
                    "Most Similar Existing Resume": [resume_data.iloc[i]["name"] for i in top_existing_matches],
                    "Resume Similarity Score": [f"{round(score * 100, 2)}%" for score in top_existing_scores]
                })

                # Display ranked results
                st.write("### Ranked Resumes Compared to Existing Resumes")
                st.dataframe(results_df)

                # Download results
                st.download_button(
                    label="Download Ranked Resumes",
                    data=results_df.to_csv(index=False),
                    file_name="Ranked_Resumes.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No valid resume data found. Please upload resumes.")
    else:
        st.warning("Job descriptions and existing resumes are required. Please upload them first.")

# Footer
st.sidebar.text("TalentAlign © 2025")
