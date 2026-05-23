# Career Intelligence & Skills Evolution Analyzer

An end-to-end data science and AI-powered career guidance system that analyzes job market trends, extracts skill requirements, identifies skill gaps, and recommends suitable career paths for students and job seekers.

## Project Overview

The Career Intelligence & Skills Evolution Analyzer is designed to connect education with industry demand. It processes job market and student profile data, performs exploratory data analysis, applies NLP and machine learning techniques, and presents insights through an interactive Streamlit dashboard.

## Objectives

- Analyze job market data to identify trending skills and roles.
- Compare student profiles with industry requirements.
- Detect skill gaps between current and required competencies.
- Recommend suitable career roles based on profile and job similarity.
- Visualize insights using interactive charts and dashboards.

## Features

- Data cleaning and preprocessing.
- Exploratory data analysis.
- Skill extraction and text normalization.
- NLP-based job description analysis.
- Career recommendation engine.
- Salary and automation risk insights.
- Interactive Streamlit dashboard.
- Exportable cleaned datasets and model outputs.

## Project Structure

```text
career-intelligence-analyzer/
├── data/
│   ├── raw/
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_nlp_and_modeling.ipynb
│   ├── 03_career_recommendation_engine.ipynb
|   ├── models/
|   ├── output/
|   ├── recommendations_outputs
├── dashboard_app.py
├── requirements.txt
└── README.md
```

## Datasets Used

This project uses job market and student profile datasets, including Kaggle-based sources and project-specific cleaned data files.
job_market.csv
job_posts.csv 
student_data.xlsx 
tech_jobs_data.csv

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- NLP techniques
- Machine learning

## Installation

```bash
git clone https://github.com/your-username/career-intelligence-analyzer.git
cd career-intelligence-analyzer
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

If using notebooks, execute them in order:

1. `01_eda_and_preprocessing.ipynb`
2. `02_nlp_and_modeling.ipynb`
3. `03_career_recommendation_engine.ipynb`

## Results

The project provides:
- Career recommendations.
- Skill gap analysis.
- Job market trend insights.
- Visual analytics for students and recruiters.
- Model-based predictions and dashboard reporting.


## Screenshots

Add screenshots of:
- Home dashboard.
  <img width="1895" height="978" alt="Screenshot 2026-05-22 233652" src="https://github.com/user-attachments/assets/c799fdf1-448a-44ca-84a5-953b4c187abf" />

- Skill gap analysis page.
  <img width="1874" height="973" alt="Screenshot 2026-05-22 233759" src="https://github.com/user-attachments/assets/75e4403b-2e73-481d-b3d8-b82d9b2046c8" />

- Recommendation results.
  <img width="1873" height="953" alt="Screenshot 2026-05-23 000311" src="https://github.com/user-attachments/assets/f36802d3-fe18-416a-9c15-f306e501fa7d" />

- Salary or automation risk charts.
  <img width="1878" height="951" alt="Screenshot 2026-05-23 000223" src="https://github.com/user-attachments/assets/1d9fd386-75e5-44ab-9e00-51d5ffe15ecc" />



## Future Scope

- Live job scraping and real-time updates.
- Resume parsing and matching.
- Advanced recommendation models.
- Multi-country job market analysis.
- Deployment on cloud platforms.

## License

This project is created for academic and internship purposes.

## Author

Zubair KH  
Data Analyst Intern
