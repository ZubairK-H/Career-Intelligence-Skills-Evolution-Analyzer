# Imports
import ast
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit App Config
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Career Intelligence & Skills Evolution Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths and Constants
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Skill Dictionaries and Role Maps
CANONICAL_SKILLS = [
    "Python", "SQL", "Excel", "Power BI", "Tableau", "Machine Learning", "Deep Learning",
    "NLP", "AWS", "Azure", "GCP", "Git", "Statistics", "Communication", "Leadership",
    "Prompt Engineering", "Data Analysis", "Data Visualization", "Pandas", "NumPy",
    "Scikit-Learn", "TensorFlow", "PyTorch", "ETL", "Data Cleaning", "Reporting",
    "Dashboarding", "Business Analysis", "Problem Solving", "Critical Thinking"
]

SKILL_ALIAS = {
    "python": "Python",
    "sql": "SQL",
    "mysql": "SQL",
    "postgresql": "SQL",
    "sqlite": "SQL",
    "excel": "Excel",
    "microsoft excel": "Excel",
    "ms excel": "Excel",
    "powerbi": "Power BI",
    "power bi": "Power BI",
    "tableau": "Tableau",
    "machine learning": "Machine Learning",
    "ml": "Machine Learning",
    "deep learning": "Deep Learning",
    "nlp": "NLP",
    "natural language processing": "NLP",
    "aws": "AWS",
    "amazon web services": "AWS",
    "azure": "Azure",
    "gcp": "GCP",
    "google cloud": "GCP",
    "git": "Git",
    "statistics": "Statistics",
    "statistical analysis": "Statistics",
    "communication": "Communication",
    "leadership": "Leadership",
    "prompt engineering": "Prompt Engineering",
    "data analysis": "Data Analysis",
    "data analyst": "Data Analysis",
    "data visualization": "Data Visualization",
    "visualization": "Data Visualization",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "scikit learn": "Scikit-Learn",
    "scikit-learn": "Scikit-Learn",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "etl": "ETL",
    "data cleaning": "Data Cleaning",
    "reporting": "Reporting",
    "dashboard": "Dashboarding",
    "dashboarding": "Dashboarding",
    "business analysis": "Business Analysis",
    "problem solving": "Problem Solving",
    "critical thinking": "Critical Thinking",
}

ROLE_SKILL_MAP = {
    "data analyst": ["Excel", "SQL", "Power BI", "Statistics", "Data Analysis", "Communication"],
    "business analyst": ["Excel", "SQL", "Power BI", "Business Analysis", "Communication", "Problem Solving"],
    "data scientist": ["Python", "SQL", "Statistics", "Machine Learning", "Pandas", "Scikit-Learn"],
    "ml engineer": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "AWS"],
    "bi analyst": ["Excel", "SQL", "Power BI", "Tableau", "Reporting", "Data Visualization"],
    "data engineer": ["Python", "SQL", "ETL", "AWS", "Azure", "Git"],
}

LEARNING_PATH_MAP = {
    "Excel": "Master formulas, pivot tables, lookup functions, and dashboard-ready spreadsheet design.",
    "SQL": "Practice SELECT, JOIN, GROUP BY, CTEs, window functions, and real business query problems.",
    "Python": "Learn Python syntax, pandas, NumPy, file handling, and project-based data cleaning workflows.",
    "Power BI": "Build interactive dashboards, DAX measures, data modeling, and drill-down reports in Power BI.",
    "Tableau": "Create story dashboards, filters, calculations, and KPI visuals in Tableau.",
    "Statistics": "Study descriptive statistics, probability, distributions, hypothesis testing, and regression basics.",
    "Machine Learning": "Learn supervised learning, model evaluation, feature engineering, and deployment basics.",
    "Deep Learning": "Study neural networks, backpropagation, CNNs, sequence models, and practical use cases.",
    "NLP": "Cover text preprocessing, TF-IDF, embeddings, classification, and information extraction.",
    "AWS": "Learn cloud basics, S3, EC2, IAM, and model or data pipeline deployment.",
    "Azure": "Study Azure storage, compute, ML services, and dashboard integration options.",
    "GCP": "Learn BigQuery, storage, notebooks, and cloud ML workflows.",
    "Git": "Practice version control, branching, pull requests, and collaborative project workflows.",
    "Communication": "Improve presentation, reporting, stakeholder communication, and business storytelling.",
    "Leadership": "Develop ownership, prioritization, teamwork, and project coordination skills.",
    "Prompt Engineering": "Learn LLM prompting, evaluation, prompt patterns, and AI workflow integration.",
    "Data Analysis": "Practice exploratory analysis, KPI framing, business questions, and insight generation.",
    "Data Visualization": "Build clean charts, choose correct plots, and improve dashboard readability.",
    "Pandas": "Work on filtering, grouping, joining, feature creation, and efficient tabular analysis.",
    "NumPy": "Learn arrays, vectorized operations, linear algebra basics, and data transformations.",
    "Scikit-Learn": "Practice preprocessing, pipelines, cross-validation, and classification/regression models.",
    "TensorFlow": "Build deep learning models and understand training loops and evaluation.",
    "PyTorch": "Implement tensors, datasets, training pipelines, and neural network experimentation.",
    "ETL": "Learn extraction, transformation, loading, scheduling, and quality checks for pipelines.",
    "Data Cleaning": "Practice null handling, duplicates, outliers, text cleaning, and schema standardization.",
    "Reporting": "Create concise reporting templates, business summaries, and KPI-based analysis.",
    "Dashboarding": "Design interactive dashboard layouts, filters, drill-throughs, and user-focused insights.",
    "Business Analysis": "Learn requirement gathering, process mapping, and KPI translation into analytics tasks.",
    "Problem Solving": "Build structured thinking through case studies and business scenario analysis.",
    "Critical Thinking": "Practice evaluating assumptions, comparing alternatives, and defending analytical decisions.",
}

RISK_SCORE_MAP = {"low_risk": 0.2, "medium_risk": 0.5, "high_risk": 0.8}
RISK_LABEL_PRETTY = {"low_risk": "Low", "medium_risk": "Medium", "high_risk": "High"}

# File Reading Helper Functions
def safe_read_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None

# Parquet Reading Helper Function
def safe_read_parquet(path: Path):
    if path.exists():
        return pd.read_parquet(path)
    return None

# Pickle Loading Helper Function
def safe_load_pickle(path: Path):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

# Safe List Conversion Function
def safe_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return [i.strip() for i in x.split(",") if i.strip()]
    return []

# Skill Token Cleaning Function
def clean_skill_token(token):
    token = str(token).strip().lower().replace('"', "").replace("'", "")
    token = token.replace("[", "").replace("]", "")
    token = " ".join(token.split())
    if token in {"", "nan", "none", "null", "[]"}:
        return None
    if len(token) == 1 and token not in {"r", "c"}:
        return None
    return SKILL_ALIAS.get(token, token.title())

# Skill Normalization Function
def normalize_skills(skill_value):
    raw = safe_list(skill_value)
    cleaned = []
    for item in raw:
        val = clean_skill_token(item)
        if val:
            cleaned.append(val)
    return sorted(list(dict.fromkeys(cleaned)))

# Flexible Column Finder Function
def extract_skills_from_text(text: str):
    text = str(text).lower()
    found = []
    for raw, canon in SKILL_ALIAS.items():
        if raw in text:
            found.append(canon)
    return sorted(list(dict.fromkeys(found)))

# Flexible Column Finder Function
def find_first_column(df, candidates, default=None):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return default
    

# Jobs Schema Standardization Function
def ensure_jobs_schema(df: pd.DataFrame):
    df = df.copy()

    rename_map = {}
    mapping = {
        "job_title": ["job_title", "title", "role", "job profession", "profession"],
        "location": ["location", "region", "city", "country"],
        "industry": ["industry", "category", "domain", "sector"],
        "job_description": ["job_description", "description", "job_desc", "text"],
        "experience_years": ["experience_years", "experience", "years_experience", "years experience"],
        "salary_usd": ["salary_usd", "salary", "salary_in_usd", "monthly_salary", "annual_salary"],
        "date_posted": ["date_posted", "posted_date", "date", "posting_date"],
        "skills": ["skills", "keywords", "skill_keywords"],
        "automation_risk_label": ["automation_risk_label", "predicted_risk", "risk_label", "automation_risk"],
    }

    for standard, candidates in mapping.items():
        col = find_first_column(df, candidates)
        if col and col != standard:
            rename_map[col] = standard
    df = df.rename(columns=rename_map)

    for col, default in {
        "job_title": "Unknown",
        "location": "Unknown",
        "industry": "Unknown",
        "job_description": "",
        "experience_years": np.nan,
        "salary_usd": np.nan,
        "date_posted": pd.NaT,
        "skills": [[] for _ in range(len(df))],
        "automation_risk_label": "medium_risk",
    }.items():
        if col not in df.columns:
            df[col] = default

    df["job_title"] = df["job_title"].fillna("Unknown").astype(str).str.strip()
    df["location"] = df["location"].fillna("Unknown").astype(str).str.strip()
    df["industry"] = df["industry"].fillna("Unknown").astype(str).str.strip()
    df["job_description"] = df["job_description"].fillna("").astype(str)
    df["experience_years"] = pd.to_numeric(df["experience_years"], errors="coerce")
    df["salary_usd"] = pd.to_numeric(df["salary_usd"], errors="coerce")
    df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce")
    df["skills"] = df["skills"].apply(normalize_skills)
    df["automation_risk_label"] = df["automation_risk_label"].fillna("medium_risk").astype(str).str.lower()

    if "job_id" not in df.columns:
        df["job_id"] = [f"job_{i}" for i in range(len(df))]

    desc_skill_fill = df["skills"].apply(len).eq(0)
    df.loc[desc_skill_fill, "skills"] = df.loc[desc_skill_fill, "job_description"].apply(extract_skills_from_text)
    
def ensure_jobs_schema(df: pd.DataFrame):
    df = df.copy()

    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[\/\-\.\(\)]", "_", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )

    rename_map = {}
    mapping = {
        "job_title": [
            "job_title", "title", "role", "job_role", "job_profession",
            "profession", "designation", "current_role", "job"
        ],
        "location": [
            "location", "job_location", "region", "city", "country", "place"
        ],
        "industry": [
            "industry", "category", "domain", "sector", "company_sector", "field"
        ],
        "job_description": [
            "job_description", "description", "job_desc", "text",
            "summary", "details", "job_description_clean"
        ],
        "experience_years": [
            "experience_years", "experience", "years_experience",
            "years_experience_required", "years_experience_needed",
            "required_experience", "min_experience"
        ],
        "salary_usd": [
            "salary_usd", "salary", "salary_in_usd", "monthly_salary",
            "annual_salary", "avg_salary", "median_salary_usd"
        ],
        "date_posted": [
            "date_posted", "posted_date", "date", "posting_date",
            "job_posted_date", "posting_month"
        ],
        "skills": [
            "skills", "keywords", "skill_keywords", "required_skills",
            "skill_set", "skillset"
        ],
        "automation_risk_label": [
            "automation_risk_label", "predicted_risk", "risk_label",
            "automation_risk", "risk"
        ],
    }

    for standard, candidates in mapping.items():
        col = find_first_column(df, candidates)
        if col and col != standard:
            rename_map[col] = standard

    df = df.rename(columns=rename_map)

    defaults = {
        "job_title": "Unknown",
        "location": "Unknown",
        "industry": "Unknown",
        "job_description": "",
        "experience_years": np.nan,
        "salary_usd": np.nan,
        "date_posted": pd.NaT,
        "skills": [[] for _ in range(len(df))],
        "automation_risk_label": "medium_risk",
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    df["job_title"] = df["job_title"].fillna("Unknown").astype(str).str.strip()
    df["location"] = df["location"].fillna("Unknown").astype(str).str.strip()
    df["industry"] = df["industry"].fillna("Unknown").astype(str).str.strip()
    df["job_description"] = df["job_description"].fillna("").astype(str).str.strip()

    df["experience_years"] = pd.to_numeric(df["experience_years"], errors="coerce")
    df["salary_usd"] = pd.to_numeric(df["salary_usd"], errors="coerce")
    df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce")

    df["skills"] = df["skills"].apply(normalize_skills)
    empty_skill_mask = df["skills"].apply(len).eq(0)
    df.loc[empty_skill_mask, "skills"] = df.loc[empty_skill_mask, "job_description"].apply(extract_skills_from_text)

    df["automation_risk_label"] = (
        df["automation_risk_label"]
        .fillna("medium_risk")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "low": "low_risk",
            "medium": "medium_risk",
            "high": "high_risk"
        })
    )

    df["job_title"] = df["job_title"].replace({"": "Unknown", "nan": "Unknown", "none": "Unknown"})
    df["location"] = df["location"].replace({"": "Unknown", "nan": "Unknown", "none": "Unknown"})
    df["industry"] = df["industry"].replace({"": "Unknown", "nan": "Unknown", "none": "Unknown"})

    if "job_id" not in df.columns:
        df["job_id"] = [f"job_{i}" for i in range(len(df))]

    df = df.drop_duplicates(
        subset=["job_title", "location", "industry", "job_description"],
        keep="first"
    ).reset_index(drop=True)

    return df

# Students Schema Standardization Function
def ensure_students_schema(df: pd.DataFrame):
    df = df.copy()
    rename_map = {}
    mapping = {
        "student_id": ["student_id", "student", "user_id", "sr.no.", "sr no", "id"],
        "course": ["course", "current_role"],
        "target_role": ["target_role", "job profession", "preferred_role", "profession", "current_role"],
        "country": ["country", "location"],
        "years_experience": ["years_experience", "experience_years", "years experience"],
        "skills": ["skills", "current_skills"],
        "linguistic": ["linguistic"],
        "logical": ["logical - mathematical", "logical", "logical_mathematical"],
        "interpersonal": ["interpersonal"],
        "intrapersonal": ["intrapersonal"],
        "spatial": ["spatial-visualization", "spatial"],
    }

    for standard, candidates in mapping.items():
        col = find_first_column(df, candidates)
        if col and col != standard:
            rename_map[col] = standard
    df = df.rename(columns=rename_map)

    if "student_id" not in df.columns:
        df["student_id"] = [f"student_{i}" for i in range(len(df))]
    else:
        df["student_id"] = df["student_id"].astype(str).str.strip()
        df["student_id"] = [f"student_{i}" if x == "" else x for i, x in enumerate(df["student_id"])]

    if "target_role" not in df.columns:
        df["target_role"] = "Data Analyst"
    if "course" not in df.columns:
        df["course"] = "General"
    if "years_experience" not in df.columns:
        df["years_experience"] = 0
    if "country" not in df.columns:
        df["country"] = "Unknown"
    if "skills" not in df.columns:
        df["skills"] = [[] for _ in range(len(df))]

    df["target_role"] = df["target_role"].fillna("Data Analyst").astype(str).str.strip()
    df["course"] = df["course"].fillna("General").astype(str).str.strip()
    df["country"] = df["country"].fillna("Unknown").astype(str).str.strip()
    df["years_experience"] = pd.to_numeric(df["years_experience"], errors="coerce").fillna(0)
    df["skills"] = df["skills"].apply(normalize_skills)

    return df

# Demo Jobs Data Builder Function
def build_demo_jobs():
    return pd.DataFrame([
        {
            "job_id": "job_1",
            "job_title": "Data Analyst",
            "location": "Bengaluru, India",
            "industry": "Analytics",
            "job_description": "Analyze data, build dashboards in Power BI, work with SQL and Excel, communicate insights.",
            "experience_years": 2,
            "salary_usd": 14000,
            "date_posted": "2026-01-15",
            "skills": ["SQL", "Excel", "Power BI", "Data Analysis", "Communication"],
            "automation_risk_label": "medium_risk",
        },
        {
            "job_id": "job_2",
            "job_title": "Business Analyst",
            "location": "Hyderabad, India",
            "industry": "Consulting",
            "job_description": "Gather requirements, build reports, use Excel and SQL, support stakeholders with dashboards.",
            "experience_years": 1,
            "salary_usd": 12000,
            "date_posted": "2026-02-10",
            "skills": ["Excel", "SQL", "Reporting", "Business Analysis", "Communication"],
            "automation_risk_label": "medium_risk",
        },
        {
            "job_id": "job_3",
            "job_title": "Data Scientist",
            "location": "Pune, India",
            "industry": "AI",
            "job_description": "Build ML models, Python pipelines, feature engineering, statistics, NLP experimentation.",
            "experience_years": 3,
            "salary_usd": 22000,
            "date_posted": "2026-03-08",
            "skills": ["Python", "Statistics", "Machine Learning", "NLP", "Pandas"],
            "automation_risk_label": "low_risk",
        },
        {
            "job_id": "job_4",
            "job_title": "BI Analyst",
            "location": "Chennai, India",
            "industry": "Retail",
            "job_description": "Develop Tableau and Power BI dashboards, reporting automation, SQL extraction, KPI analysis.",
            "experience_years": 2,
            "salary_usd": 16000,
            "date_posted": "2026-04-12",
            "skills": ["SQL", "Power BI", "Tableau", "Reporting", "Data Visualization"],
            "automation_risk_label": "medium_risk",
        },
        {
            "job_id": "job_5",
            "job_title": "ML Engineer",
            "location": "Remote",
            "industry": "Technology",
            "job_description": "Deploy deep learning models, build APIs, Python, AWS, Git, model monitoring.",
            "experience_years": 4,
            "salary_usd": 28000,
            "date_posted": "2026-04-20",
            "skills": ["Python", "Deep Learning", "AWS", "Git", "Machine Learning"],
            "automation_risk_label": "low_risk",
        },
        {
            "job_id": "job_6",
            "job_title": "Reporting Executive",
            "location": "Mumbai, India",
            "industry": "Operations",
            "job_description": "Daily MIS reporting, Excel sheets, formatting, manual updates, and repetitive status reporting.",
            "experience_years": 1,
            "salary_usd": 9000,
            "date_posted": "2026-01-22",
            "skills": ["Excel", "Reporting", "Communication"],
            "automation_risk_label": "high_risk",
        },
                {
            "job_id": "job_7",
            "job_title": "Junior Data Engineer",
            "location": "Bengaluru, India",
            "industry": "Technology",
            "job_description": "Build ETL pipelines, work with SQL databases, Python scripts, and version control for analytics systems.",
            "experience_years": 2,
            "salary_usd": 18000,
            "date_posted": "2026-02-02",
            "skills": ["Python", "SQL", "ETL", "Git", "Data Engineering"],
            "automation_risk_label": "low_risk",
        },
        {
            "job_id": "job_8",
            "job_title": "Product Analyst",
            "location": "Gurugram, India",
            "industry": "Product",
            "job_description": "Analyze product usage data, create dashboards, define KPIs, and communicate business insights.",
            "experience_years": 2,
            "salary_usd": 17000,
            "date_posted": "2026-02-18",
            "skills": ["SQL", "Excel", "Power BI", "Communication", "Data Analysis"],
            "automation_risk_label": "medium_risk",
        },
        {
            "job_id": "job_9",
            "job_title": "AI Engineer",
            "location": "Remote",
            "industry": "AI",
            "job_description": "Develop AI applications using Python, machine learning, prompt engineering, and cloud deployment.",
            "experience_years": 3,
            "salary_usd": 26000,
            "date_posted": "2026-03-12",
            "skills": ["Python", "Machine Learning", "AWS", "Prompt Engineering", "Git"],
            "automation_risk_label": "low_risk",
        },
        {
            "job_id": "job_10",
            "job_title": "SQL Analyst",
            "location": "Noida, India",
            "industry": "Finance",
            "job_description": "Write SQL queries, create MIS reports, clean records, and support reporting teams.",
            "experience_years": 1,
            "salary_usd": 11000,
            "date_posted": "2026-01-28",
            "skills": ["SQL", "Excel", "Reporting", "Data Cleaning"],
            "automation_risk_label": "medium_risk",
        },
        {
            "job_id": "job_11",
            "job_title": "Marketing Data Analyst",
            "location": "Mumbai, India",
            "industry": "Marketing",
            "job_description": "Analyze campaign data, measure ROI, create reports, and build dashboards for stakeholders.",
            "experience_years": 2,
            "salary_usd": 15000,
            "date_posted": "2026-03-01",
            "skills": ["Excel", "SQL", "Power BI", "Communication", "Data Analysis"],
            "automation_risk_label": "medium_risk",
        },
        {
            "job_id": "job_12",
            "job_title": "Data Visualization Specialist",
            "location": "Pune, India",
            "industry": "Analytics",
            "job_description": "Create interactive dashboards and executive reports using Tableau, Power BI, and storytelling techniques.",
            "experience_years": 2,
            "salary_usd": 16500,
            "date_posted": "2026-03-14",
            "skills": ["Tableau", "Power BI", "Data Visualization", "Communication", "Reporting"],
            "automation_risk_label": "medium_risk",
        },
        {
            "job_id": "job_13",
            "job_title": "NLP Engineer",
            "location": "Remote",
            "industry": "AI",
            "job_description": "Build NLP pipelines, text classification systems, and transformer-based experimentation in Python.",
            "experience_years": 3,
            "salary_usd": 24000,
            "date_posted": "2026-04-02",
            "skills": ["Python", "NLP", "Machine Learning", "Deep Learning", "Git"],
            "automation_risk_label": "low_risk",
        },
        {
            "job_id": "job_14",
            "job_title": "Operations Analyst",
            "location": "Chennai, India",
            "industry": "Operations",
            "job_description": "Track KPIs, maintain reports, improve process visibility, and coordinate with business teams.",
            "experience_years": 1,
            "salary_usd": 10500,
            "date_posted": "2026-02-25",
            "skills": ["Excel", "Reporting", "Communication", "Problem Solving"],
            "automation_risk_label": "high_risk",
        },
        {
            "job_id": "job_15",
            "job_title": "Cloud Data Engineer",
            "location": "Hyderabad, India",
            "industry": "Cloud",
            "job_description": "Manage cloud data pipelines, storage systems, ETL jobs, and deployment workflows on AWS.",
            "experience_years": 3,
            "salary_usd": 25000,
            "date_posted": "2026-04-08",
            "skills": ["Python", "SQL", "AWS", "ETL", "Git"],
            "automation_risk_label": "low_risk",
        },
        {
            "job_id": "job_16",
            "job_title": "Research Analyst",
            "location": "Delhi, India",
            "industry": "Research",
            "job_description": "Collect data, perform descriptive analysis, summarize findings, and prepare stakeholder reports.",
            "experience_years": 1,
            "salary_usd": 11500,
            "date_posted": "2026-01-30",
            "skills": ["Excel", "Statistics", "Communication", "Reporting"],
            "automation_risk_label": "medium_risk",
        },
        {
            "job_id": "job_17",
            "job_title": "Junior ML Analyst",
            "location": "Bengaluru, India",
            "industry": "AI",
            "job_description": "Support ML model building, data preprocessing, feature creation, and experiment tracking.",
            "experience_years": 2,
            "salary_usd": 20000,
            "date_posted": "2026-03-20",
            "skills": ["Python", "Machine Learning", "Statistics", "Git"],
            "automation_risk_label": "low_risk",
        },
       
    ])


def build_demo_students():
    return pd.DataFrame([
        {"student_id": "student_0",  "course": "BSc Data Science",     "target_role": "Data Analyst",      "country": "India", "years_experience": 0, "skills": ["Excel", "Communication"]},
        {"student_id": "student_1",  "course": "BTech CSE",            "target_role": "Data Scientist",    "country": "India", "years_experience": 1, "skills": ["Python", "Statistics"]},
        {"student_id": "student_2",  "course": "BBA Analytics",        "target_role": "Business Analyst",  "country": "India", "years_experience": 0, "skills": ["Excel", "Communication", "Problem Solving"]},
        {"student_id": "student_3",  "course": "BTech AI",             "target_role": "ML Engineer",       "country": "India", "years_experience": 1, "skills": ["Python", "Machine Learning", "Git"]},
        {"student_id": "student_4",  "course": "BCom",                 "target_role": "Data Analyst",      "country": "India", "years_experience": 0, "skills": ["Excel", "SQL"]},
        {"student_id": "student_5",  "course": "BSc Statistics",       "target_role": "Data Scientist",    "country": "India", "years_experience": 0, "skills": ["Python", "Statistics", "SQL"]},
        {"student_id": "student_6",  "course": "BCA",                  "target_role": "BI Analyst",        "country": "India", "years_experience": 1, "skills": ["Power BI", "Excel", "Communication"]},
        {"student_id": "student_7",  "course": "BTech IT",             "target_role": "Data Engineer",     "country": "India", "years_experience": 1, "skills": ["Python", "SQL", "Git"]},
        {"student_id": "student_8",  "course": "MBA",                  "target_role": "Business Analyst",  "country": "India", "years_experience": 0, "skills": ["Communication", "Leadership", "Excel"]},
        {"student_id": "student_9",  "course": "BSc Computer Science", "target_role": "ML Engineer",       "country": "India", "years_experience": 1, "skills": ["Python", "Machine Learning", "Statistics"]},
        {"student_id": "student_10", "course": "BSc Mathematics",      "target_role": "Data Scientist",    "country": "India", "years_experience": 0, "skills": ["Python", "Statistics", "Problem Solving"]},
        {"student_id": "student_11", "course": "BTech ECE",            "target_role": "Data Analyst",      "country": "India", "years_experience": 0, "skills": ["Excel", "SQL", "Communication"]},
        {"student_id": "student_12", "course": "BCA",                  "target_role": "BI Analyst",        "country": "India", "years_experience": 1, "skills": ["Power BI", "SQL", "Excel"]},
        {"student_id": "student_13", "course": "BTech CSE",            "target_role": "Data Engineer",     "country": "India", "years_experience": 1, "skills": ["Python", "SQL", "Docker"]},
        {"student_id": "student_14", "course": "BSc AI",               "target_role": "ML Engineer",       "country": "India", "years_experience": 1, "skills": ["Python", "Machine Learning", "Git", "Statistics"]},
        {"student_id": "student_15", "course": "BBA",                  "target_role": "Business Analyst",  "country": "India", "years_experience": 0, "skills": ["Excel", "Communication", "Leadership"]},
        {"student_id": "student_16", "course": "BSc Data Science",     "target_role": "Data Analyst",      "country": "India", "years_experience": 0, "skills": ["Excel", "Python", "SQL"]},
        {"student_id": "student_17", "course": "BTech AI",             "target_role": "AI Engineer",       "country": "India", "years_experience": 1, "skills": ["Python", "Machine Learning", "AWS"]},
        {"student_id": "student_18", "course": "MBA Analytics",        "target_role": "Business Analyst",  "country": "India", "years_experience": 1, "skills": ["Excel", "Power BI", "Communication", "Problem Solving"]},
        {"student_id": "student_19", "course": "BSc Statistics",       "target_role": "Data Scientist",    "country": "India", "years_experience": 0, "skills": ["Python", "Statistics", "Machine Learning"]}
    ])

# Main Data Loading Function
@st.cache_data(show_spinner=False)
def load_all_data():
    jobs = build_demo_jobs()
    students = None

    student_paths = [
        DATA_DIR / "clean_students.parquet",
        DATA_DIR / "clean_students.csv",
        DATA_DIR / "student_data.xlsx",
        DATA_DIR / "raw" / "student_data.xlsx",
    ]

    for p in student_paths:
        if p.exists():
            if p.suffix == ".parquet":
                students = pd.read_parquet(p)
            elif p.suffix == ".csv":
                students = pd.read_csv(p)
            elif p.suffix in [".xlsx", ".xls"]:
                students = pd.read_excel(p)
            if students is not None:
                break

    if students is None:
        students = build_demo_students()

    jobs = ensure_jobs_schema(jobs)
    students = ensure_students_schema(students)

    st.write("Jobs loaded:", len(jobs))
    st.write("Students loaded:", len(students))

    return jobs, students

# Model Artifact Loading Function
@st.cache_resource(show_spinner=False)
def load_artifacts():
    salary_model = safe_load_pickle(MODEL_DIR / "best_salary_model.pkl")
    risk_model = safe_load_pickle(MODEL_DIR / "best_risk_model.pkl")
    tfidf = safe_load_pickle(MODEL_DIR / "tfidf_vectorizer.pkl")
    mlb = safe_load_pickle(MODEL_DIR / "skill_binarizer.pkl")
    return salary_model, risk_model, tfidf, mlb

# Currency Formatting Function
def format_money(x):
    if pd.isna(x):
        return "N/A"
    return f"{float(x):,.0f}"

# Role Inference From Text Function
def role_from_text(text: str):
    text = str(text).lower()
    for role in ROLE_SKILL_MAP.keys():
        if role in text:
            return role
    if "analyst" in text:
        return "data analyst"
    if "scientist" in text:
        return "data scientist"
    if "engineer" in text:
        return "ml engineer"
    return "data analyst"

# Student Skill Inference Function
def infer_student_skills(student_row):
    direct_skills = normalize_skills(student_row.get("skills", []))
    role = role_from_text(student_row.get("target_role", ""))
    role_skills = ROLE_SKILL_MAP.get(role, [])
    merged = sorted(list(dict.fromkeys(direct_skills + role_skills[:3])))
    return merged

# Job Text Builder Function
def build_job_text(row):
    return " ".join([
        str(row.get("job_title", "")),
        str(row.get("industry", "")),
        str(row.get("location", "")),
        str(row.get("job_description", "")),
        " ".join(normalize_skills(row.get("skills", []))),
    ]).strip()

# Student Text Builder Function
def build_student_text(student_row):
    return " ".join([
        str(student_row.get("target_role", "")),
        str(student_row.get("course", "")),
        str(student_row.get("country", "")),
        " ".join(infer_student_skills(student_row)),
    ]).strip()

# Skill Similarity Calculation Function
def simple_similarity(student_skills, job_skills):
    s = set(student_skills)
    j = set(job_skills)
    if not s and not j:
        return 0.0
    return len(s & j) / max(len(j), 1)

# Salary Prediction Function
def predict_salary(description, experience, found_skills, salary_model=None, tfidf=None, mlb=None, jobs_df=None):
    if salary_model is not None and tfidf is not None and mlb is not None:
        try:
            desc_vec = tfidf.transform([description])
            skill_vec = mlb.transform([found_skills])
            exp_df = pd.DataFrame({"experience_years": [experience]})
            desc_df = pd.DataFrame(desc_vec.toarray())
            skill_df = pd.DataFrame(skill_vec, columns=mlb.classes_)
            X = pd.concat([exp_df.reset_index(drop=True), desc_df.reset_index(drop=True), skill_df.reset_index(drop=True)], axis=1)
            pred = float(salary_model.predict(X)[0])
            return max(pred, 0)
        except Exception:
            pass

    base = 8000 + (float(experience) * 2200)
    premium_map = {
        "Python": 2500, "SQL": 1800, "Excel": 900, "Power BI": 1600, "Tableau": 1600,
        "Machine Learning": 3500, "Deep Learning": 4500, "NLP": 3000, "AWS": 2800,
        "Azure": 2200, "GCP": 2200, "Git": 800, "Statistics": 1800, "Prompt Engineering": 2200,
    }
    premium = sum(premium_map.get(skill, 500) for skill in found_skills[:8])
    if jobs_df is not None and "salary_usd" in jobs_df.columns and jobs_df["salary_usd"].notna().any():
        market_anchor = float(jobs_df["salary_usd"].median())
        pred = 0.55 * base + 0.45 * max(market_anchor, 0) + premium
    else:
        pred = base + premium
    return max(pred, 0)

# Automation Risk Prediction Function
def predict_risk(description, found_skills, risk_model=None, tfidf=None, mlb=None):
    if risk_model is not None and tfidf is not None and mlb is not None:
        try:
            desc_vec = tfidf.transform([description])
            skill_vec = mlb.transform([found_skills])
            exp_df = pd.DataFrame({"experience_years": [0]})
            desc_df = pd.DataFrame(desc_vec.toarray())
            skill_df = pd.DataFrame(skill_vec, columns=mlb.classes_)
            X = pd.concat([exp_df.reset_index(drop=True), desc_df.reset_index(drop=True), skill_df.reset_index(drop=True)], axis=1)
            risk_pred = str(risk_model.predict(X)[0]).lower()
            risk_pred = risk_pred if risk_pred in RISK_SCORE_MAP else "medium_risk"
            return risk_pred
        except Exception:
            pass

    text = str(description).lower()
    repetitive_terms = ["manual", "repetitive", "data entry", "formatting", "copy paste", "status reporting", "mis reporting"]
    advanced_terms = ["machine learning", "model", "strategy", "stakeholder", "forecasting", "nlp", "deployment", "architecture"]
    repetitive_score = sum(term in text for term in repetitive_terms)
    advanced_score = sum(term in text for term in advanced_terms)
    skill_score = sum(skill in found_skills for skill in ["Python", "Machine Learning", "NLP", "AWS", "Statistics", "Prompt Engineering"])

    score = repetitive_score - advanced_score - skill_score * 0.4
    if score >= 2:
        return "high_risk"
    if score <= -1:
        return "low_risk"
    return "medium_risk"

# Required Skills Extraction For Role
def get_required_skills_for_role(target_role, jobs_df):
    role = str(target_role).lower().strip()
    role_jobs = jobs_df[jobs_df["job_title"].str.lower().str.contains(role, na=False)].copy()

    if not role_jobs.empty:
        exploded = role_jobs["skills"].explode().dropna()
        if not exploded.empty:
            return exploded.value_counts().head(12).index.tolist()

    inferred_role = role_from_text(role)
    return ROLE_SKILL_MAP.get(inferred_role, ["Excel", "SQL", "Communication"])

# Student Gap Analysis Function
def get_gap_analysis(student_row, target_role, jobs_df):
    current_skills = infer_student_skills(student_row)
    required_skills = get_required_skills_for_role(target_role, jobs_df)
    missing_skills = [s for s in required_skills if s not in current_skills]
    readiness = round(100 * len(set(current_skills) & set(required_skills)) / max(len(required_skills), 1), 1)
    learning_path = [LEARNING_PATH_MAP.get(skill, f"Build a practical mini-project in {skill}.") for skill in missing_skills]
    return current_skills, required_skills, missing_skills, readiness, learning_path

# Personalized Job Recommendation Function
def recommend_jobs_for_student(student_row, jobs_df, top_n=5, region_filter="All", industry_filter="All"):
    temp = jobs_df.copy()
    student_skills = infer_student_skills(student_row)
    target_role = str(student_row.get("target_role", "")).lower()
    student_text = build_student_text(student_row)

    temp["required_skills"] = temp["skills"].apply(normalize_skills)
    temp["skill_similarity"] = temp["required_skills"].apply(lambda x: simple_similarity(student_skills, x))
    temp["missing_skills"] = temp["required_skills"].apply(lambda x: sorted(list(set(x) - set(student_skills))))
    temp["skill_overlap"] = temp["required_skills"].apply(lambda x: len(set(x) & set(student_skills)))
    temp["role_bonus"] = temp["job_title"].str.lower().apply(lambda x: 1.0 if target_role in x else 0.4 if any(word in x for word in target_role.split()) else 0.0)
    temp["salary_score"] = temp["salary_usd"].fillna(temp["salary_usd"].median() if temp["salary_usd"].notna().any() else 0)
    salary_max = temp["salary_score"].max() if temp["salary_score"].max() > 0 else 1
    temp["salary_score"] = temp["salary_score"] / salary_max
    temp["risk_score"] = temp["automation_risk_label"].map({"low_risk": 1.0, "medium_risk": 0.55, "high_risk": 0.15}).fillna(0.5)

    if region_filter != "All":
        temp["region_match"] = (temp["location"] == region_filter).astype(float)
    else:
        student_country = str(student_row.get("country", "")).lower()
        temp["region_match"] = temp["location"].str.lower().apply(lambda x: 0.8 if student_country and student_country in x else 0.4)

    if industry_filter != "All":
        temp["industry_match"] = (temp["industry"] == industry_filter).astype(float)
    else:
        pref_role = role_from_text(student_row.get("target_role", ""))
        temp["industry_match"] = temp["industry"].str.lower().apply(lambda x: 0.8 if pref_role.split()[0] in x else 0.4)

    years_exp = float(student_row.get("years_experience", 0))
    temp["experience_fit"] = temp["experience_years"].fillna(years_exp).apply(lambda x: 1 / (1 + abs(float(x) - years_exp)))

    temp["match_score"] = (
        0.35 * temp["skill_similarity"] +
        0.20 * temp["role_bonus"] +
        0.15 * temp["salary_score"] +
        0.10 * temp["risk_score"] +
        0.10 * temp["region_match"] +
        0.10 * temp["experience_fit"]
    )

    temp["predicted_salary_usd"] = temp["salary_usd"].fillna(temp["salary_usd"].median() if temp["salary_usd"].notna().any() else 0)
    temp["predicted_automation_risk"] = temp["automation_risk_label"].fillna("medium_risk")

    cols = [
        "job_title", "location", "industry", "match_score", "predicted_salary_usd",
        "predicted_automation_risk", "required_skills", "missing_skills"
    ]
    return temp.sort_values(["match_score", "predicted_salary_usd"], ascending=[False, False])[cols].head(top_n).reset_index(drop=True)

# Streamlit Metric Card Function
def metric_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)

# Overview Dashboard Renderer
def render_overview(filtered_jobs, students):
    st.subheader("Overview")
    c1, c2 = st.columns([1.6, 1])

    with c1:
        skills_series = filtered_jobs["skills"].explode().dropna()
        if not skills_series.empty:
            skill_counts = skills_series.value_counts().head(10).reset_index()
            skill_counts.columns = ["skill", "count"]
            fig = px.bar(skill_counts.sort_values("count"), x="count", y="skill", orientation="h", title="Top skills in filtered jobs")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No skill data available for current filters.")

    with c2:
        salary_data = filtered_jobs["salary_usd"].dropna()
        if not salary_data.empty:
            bins = pd.cut(salary_data, bins=4, labels=["Low", "Medium", "High", "Very High"])
            salary_dist = bins.value_counts().reset_index()
            salary_dist.columns = ["tier", "count"]
            fig = px.pie(salary_dist, names="tier", values="count", title="Salary distribution")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No salary data available for filtered jobs.")

    sample_cols = ["job_title", "location", "industry", "salary_usd", "automation_risk_label"]
    sample = filtered_jobs[sample_cols].copy().head(10)
    sample["salary_usd"] = sample["salary_usd"].apply(format_money)
    sample["automation_risk_label"] = sample["automation_risk_label"].map(RISK_LABEL_PRETTY).fillna("Medium")
    st.write("**Representative job records**")
    st.dataframe(sample, use_container_width=True)

# Skills Heatmap Renderer
def render_skills_heatmap(filtered_jobs):
    st.subheader("Skills Heatmap")
    exploded = filtered_jobs[["industry", "location", "skills"]].explode("skills").dropna(subset=["skills"])
    if exploded.empty:
        st.warning("No cleaned skill data available for the selected filters.")
        return

    top_skills = exploded["skills"].value_counts().head(15).index.tolist()
    pivot = exploded[exploded["skills"].isin(top_skills)].pivot_table(index="skills", columns="industry", values="location", aggfunc="count", fill_value=0)

    fig = px.imshow(
        pivot,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Top skill demand by industry"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.write("**Top cleaned skills**")
    top_table = exploded["skills"].value_counts().head(20).reset_index()
    top_table.columns = ["skill", "count"]
    st.dataframe(top_table, use_container_width=True)

# Gap Analysis Page Renderer
def render_gap_analysis(filtered_jobs, students):
    st.subheader("Gap Analysis Tool")
    student_ids = students["student_id"].tolist()
    selected_student_id = st.selectbox("Select student", student_ids)
    student_row = students[students["student_id"] == selected_student_id].iloc[0]

    role_options = sorted(filtered_jobs["job_title"].dropna().unique().tolist())
    default_role = str(student_row.get("target_role", role_options[0] if role_options else "Data Analyst"))
    default_idx = role_options.index(default_role) if default_role in role_options else 0
    selected_role = st.selectbox("Choose a dream job from dataset", role_options, index=default_idx if role_options else 0)

    current_skills, required_skills, missing_skills, readiness, learning_path = get_gap_analysis(student_row, selected_role, filtered_jobs)

    st.text_input("Target role", value=selected_role, disabled=True)
    st.text_area("Current skills", value=", ".join(current_skills), height=90, disabled=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Current skills**")
        st.write(current_skills if current_skills else ["No current skills found"])
    with col2:
        st.write("**Missing skills**")
        st.write(missing_skills if missing_skills else ["No major gaps"])

    st.write("**Required skills**")
    st.write(required_skills if required_skills else ["No role skills found"])

    st.write("**Readiness score**")
    st.progress(int(min(readiness, 100)))
    st.caption(f"Readiness score: {readiness}%")

    st.write("**Recommended learning path**")
    if learning_path:
        for i, step in enumerate(learning_path, start=1):
            st.write(f"{i}. {step}")
    else:
        st.success("This student already matches most critical skills for the selected role.")

    sim_jobs = recommend_jobs_for_student(student_row, filtered_jobs, top_n=5)
    sim_view = sim_jobs[["job_title", "location", "industry", "match_score"]].copy()
    sim_view["match_score"] = sim_view["match_score"].round(4)
    st.write("**Similar job matches**")
    st.dataframe(sim_view, use_container_width=True)

# Future Trends Page Renderer
def render_future_trends(filtered_jobs):
    st.subheader("Future Trends")
    dated = filtered_jobs.copy()
    dated = dated[dated["date_posted"].notna()].copy()

    if dated.empty:
        st.info("No valid date column found. Showing skill evolution proxy instead.")
        exploded = filtered_jobs["skills"].explode().dropna()
        if exploded.empty:
            st.warning("No skill data available.")
            return
        proxy = exploded.value_counts().head(12).reset_index()
        proxy.columns = ["skill", "count"]
        fig = px.line(proxy, x="skill", y="count", markers=True, title="Proxy skill trend view")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        return

    dated["month"] = dated["date_posted"].dt.to_period("M").astype(str)
    exploded = dated[["month", "skills"]].explode("skills").dropna()
    if exploded.empty:
        st.warning("No skill data available after filtering.")
        return

    top_skills = exploded["skills"].value_counts().head(8).index.tolist()
    trend = exploded[exploded["skills"].isin(top_skills)].groupby(["month", "skills"]).size().reset_index(name="count")
    fig = px.line(trend, x="month", y="count", color="skills", markers=True, title="Skill demand over time")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# Risk And Salary Prediction Renderer
def render_risk_salary(filtered_jobs, salary_model, risk_model, tfidf, mlb):
    st.subheader("Automation Risk & Salary Prediction")
    job_text = st.text_area(
        "Paste a job description",
        "Responsible for data cleaning, reporting, dashboard creation, SQL querying, Excel automation, and stakeholder communication.",
        height=150,
    )
    experience = st.slider("Experience years", 0, 15, 2)

    found_skills = extract_skills_from_text(job_text)
    c1, c2 = st.columns(2)

    with c1:
        if st.button("Predict automation risk", use_container_width=True):
            risk_label = predict_risk(job_text, found_skills, risk_model=risk_model, tfidf=tfidf, mlb=mlb)
            risk_score = int(RISK_SCORE_MAP.get(risk_label, 0.5) * 100)
            if risk_label == "low_risk":
                st.success(f"Predicted automation risk: {RISK_LABEL_PRETTY[risk_label]} ({risk_score}/100)")
            elif risk_label == "medium_risk":
                st.warning(f"Predicted automation risk: {RISK_LABEL_PRETTY[risk_label]} ({risk_score}/100)")
            else:
                st.error(f"Predicted automation risk: {RISK_LABEL_PRETTY[risk_label]} ({risk_score}/100)")
            st.write("Detected skills:", found_skills if found_skills else ["No mapped skills detected"])

    with c2:
        if st.button("Predict salary", use_container_width=True):
            salary_pred = predict_salary(job_text, experience, found_skills, salary_model=salary_model, tfidf=tfidf, mlb=mlb, jobs_df=filtered_jobs)
            st.success(f"Predicted salary: {format_money(salary_pred)}")
            st.write("Detected skills:", found_skills if found_skills else ["No mapped skills detected"])

    salary_chart_df = filtered_jobs[filtered_jobs["salary_usd"].notna()].copy()
    if not salary_chart_df.empty:
        fig = px.histogram(salary_chart_df, x="salary_usd", nbins=20, title="Salary distribution across filtered jobs")
        fig.update_layout(height=420, xaxis_title="Salary (USD)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

# Job Recommender Page Renderer
def render_job_recommender(filtered_jobs, students):
    st.subheader("Personalized Job Recommender")
    student_ids = students["student_id"].tolist()
    selected_student_id = st.selectbox("Choose a student profile", student_ids, key="job_recommender_student")
    n_recs = st.slider("Number of recommendations", 3, 15, 5)

    student_row = students[students["student_id"] == selected_student_id].iloc[0]
    result = recommend_jobs_for_student(student_row, filtered_jobs, top_n=n_recs)

    display_df = result.copy()
    display_df["match_score"] = display_df["match_score"].round(4)
    display_df["predicted_salary_usd"] = display_df["predicted_salary_usd"].apply(format_money)
    display_df["predicted_automation_risk"] = display_df["predicted_automation_risk"].map(RISK_LABEL_PRETTY).fillna("Medium")

    st.dataframe(
        display_df[["job_title", "location", "industry", "match_score", "predicted_salary_usd", "predicted_automation_risk"]],
        use_container_width=True,
    )

    fig = px.bar(
        display_df.sort_values("match_score"),
        x="match_score",
        y="job_title",
        orientation="h",
        title="Top job matches",
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show required and missing skills for recommendations"):
        detail_df = result[["job_title", "required_skills", "missing_skills"]].copy()
        detail_df["required_skills"] = detail_df["required_skills"].apply(lambda x: ", ".join(x))
        detail_df["missing_skills"] = detail_df["missing_skills"].apply(lambda x: ", ".join(x) if x else "None")
        st.dataframe(detail_df, use_container_width=True)

# Main Application Runner
def main():
    jobs, students = load_all_data()
    salary_model, risk_model, tfidf, mlb = load_artifacts()

    st.title("Career Intelligence & Skills Evolution Analyzer")
    st.caption("Advanced workforce analytics for skills demand, automation risk, salary forecasting, and personalized career guidance.")

    with st.sidebar:
        st.header("Navigate")
        page = st.radio(
            "Go to",
            ["Overview", "Skills Heatmap", "Gap Analysis", "Future Trends", "Risk & Salary", "Job Recommender"],
        )

        regions = ["All"] + sorted(jobs["location"].dropna().astype(str).unique().tolist())
        industries = ["All"] + sorted(jobs["industry"].dropna().astype(str).unique().tolist())

        region_filter = st.selectbox("Filter by region", regions)
        industry_filter = st.selectbox("Filter by industry", industries)

    filtered_jobs = jobs.copy()
    if region_filter != "All":
        filtered_jobs = filtered_jobs[filtered_jobs["location"] == region_filter]
    if industry_filter != "All":
        filtered_jobs = filtered_jobs[filtered_jobs["industry"] == industry_filter]

    if filtered_jobs.empty:
        st.warning("No records found for current filters. Reset filters to view dashboard content.")
        return

    k1, k2, k3, k4 = st.columns(4)
    #metric_card("Jobs", f"{len(filtered_jobs):,}")
    with k2:
        metric_card("Students", f"{len(students):,}")
    with k3:
        total_skills = int(filtered_jobs["skills"].apply(len).sum())
        metric_card("Skills found", f"{total_skills:,}")
    with k4:
        median_salary = filtered_jobs["salary_usd"].median() if filtered_jobs["salary_usd"].notna().any() else np.nan
        metric_card("Median salary", format_money(median_salary))

    # Fix first metric alignment
    k1.empty()
    with k1:
        metric_card("Jobs", f"{len(filtered_jobs):,}")

    if page == "Overview":
        render_overview(filtered_jobs, students)
    elif page == "Skills Heatmap":
        render_skills_heatmap(filtered_jobs)
    elif page == "Gap Analysis":
        render_gap_analysis(filtered_jobs, students)
    elif page == "Future Trends":
        render_future_trends(filtered_jobs)
    elif page == "Risk & Salary":
        render_risk_salary(filtered_jobs, salary_model, risk_model, tfidf, mlb)
    elif page == "Job Recommender":
        render_job_recommender(filtered_jobs, students)

    st.markdown("---")
    st.caption("Built for career intelligence, skills analytics, automation-risk assessment, and personalized job-market forecasting.")


if __name__ == "__main__":
    main()