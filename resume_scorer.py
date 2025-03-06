import os
import streamlit as st
import pandas as pd
import torch
import spacy
from sentence_transformers import SentenceTransformer, util

# Load SBERT model for similarity checking
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load NLP model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# File paths
resume_file = "STRUCTURED_DATA.csv"
job_desc_file = "JD.csv"

# Check if files exist before proceeding
if not os.path.exists(resume_file):
    print(f"Error: File '{resume_file}' not found. Please check the filename and location.")
    exit()

if not os.path.exists(job_desc_file):
    print(f"Error: File '{job_desc_file}' not found. Please check the filename and location.")
    exit()

# Load CSV files
resume_data = pd.read_csv(resume_file)
job_data = pd.read_csv(job_desc_file)

# Standardize column names (lowercase, remove spaces)
resume_data.columns = resume_data.columns.str.strip().str.lower()
job_data.columns = job_data.columns.str.strip().str.lower()

# Identify necessary columns
skills_column = "skills" if "skills" in resume_data.columns else resume_data.columns[-1]
keyword_column = "keywords" if "keywords" in job_data.columns else job_data.columns[3]
resume_id_column = "resume id" if "resume id" in resume_data.columns else resume_data.columns[0]

# Fill missing values
resume_data[skills_column] = resume_data[skills_column].fillna("").astype(str)
job_data[keyword_column] = job_data[keyword_column].fillna("").astype(str)

# Function to extract keywords using NER
def extract_keywords(text):
    doc = nlp(text)
    keywords = [ent.text.lower() for ent in doc.ents if ent.label_ in ["PRODUCT", "SKILL", "WORK_OF_ART", "EVENT"]]
    return list(set(keywords))

# Extract keywords from job descriptions and resumes
job_data["extracted keywords"] = job_data[keyword_column].astype(str).apply(extract_keywords)
resume_data["extracted keywords"] = resume_data[skills_column].astype(str).apply(extract_keywords)

# Convert skills and extracted keywords into SBERT embeddings
resume_embeddings = model.encode(
    resume_data["skills"] + " " + resume_data["extracted keywords"].apply(lambda x: " ".join(x)), 
    convert_to_tensor=True
)
jd_embeddings = model.encode(
    job_data["keywords"] + " " + job_data["extracted keywords"].apply(lambda x: " ".join(x)), 
    convert_to_tensor=True
)

# Compute cosine similarity
similarity_scores = util.cos_sim(resume_embeddings, jd_embeddings)

# Convert similarity matrix to DataFrame
similarity_df = pd.DataFrame(
    similarity_scores.cpu().numpy(),
    index=resume_data[resume_id_column],
    columns=job_data["job role"]
)

# Find best matches
top_matches = similarity_df.idxmax(axis=1)  # Best job match
top_scores = similarity_df.max(axis=1)  # Highest similarity score

# Ensure scores are numeric
top_scores = pd.to_numeric(top_scores, errors="coerce")

# Ensure row count consistency
min_length = min(len(resume_data), len(top_matches), len(top_scores))

# Create results DataFrame
results_df = pd.DataFrame({
    "Resume ID": resume_data[resume_id_column][:min_length],
    "Candidate Name": resume_data["name"][:min_length],
    "Best Matching Job Role": top_matches.values[:min_length],
    "Match Score": (top_scores.values[:min_length] * 100).round(2).astype(str) + "%"
})

# Save results to a CSV file
output_file = "Resume_Job_Matching.csv"
results_df.to_csv(output_file, index=False)

print(f"\n✅ Resume Matching Completed! Results saved to '{output_file}'.")
