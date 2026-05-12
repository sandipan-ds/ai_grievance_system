## 1. AI-Powered Civic Grievance Classification System

An NLP-based multi-output complaint routing system that automatically predicts:
- responsible civic department
- complaint severity

Built using FastAPI, Streamlit, DistilBERT, LinearSVC, Supabase, and Hugging Face Spaces.

## 2. Live Demo

- Web App: https://huggingface.co/spaces/sandipanarnab/grievance_iq
- GitHub: https://github.com/sandipan-ds/ai_grievance_system

## 3. Application Preview

The application provides an interactive civic grievance dashboard for automated complaint analysis and prioritization.

### Main Features

- Complaint department prediction
- Complaint severity classification
- Interactive analytics dashboard
- Real-time prediction using trained NLP models
- Complaint distribution visualization
- Severity-wise complaint monitoring
- Department-level complaint insights

### Dashboard Overview

The Streamlit dashboard enables users to:

- Submit civic complaints in natural language
- Automatically classify complaints into departments
- Predict severity levels:
  - Critical
  - High
  - Medium
  - Low
- Visualize complaint trends using interactive charts
- Monitor complaint statistics across departments

### Screenshots

#### Home Dashboard
![Dashboard Screenshot](dashboard_screenshots\complaint_distribution.png)

#### Complaint Prediction Interface
![Prediction Screenshot](dashboard_screenshots\complaint_register.png)

#### Severity Analytics
![Analytics Screenshot](dashboard_screenshots\complaint_distribution.png)

#### FastAPI Swagger Documentation
![API Screenshot](dashboard_screenshots\fast_api_response.png)

## 4. Problem Statement

Large civic grievance systems receive thousands of complaints daily.
Manual routing and prioritization are slow and inconsistent.

This project automates:
- complaint department classification
- severity prioritization
- complaint analytics

## 5. Features

- Multi-output NLP classification
- Department prediction
- Severity prediction
- DistilBERT severity classifier
- LinearSVC department classifier
- FastAPI backend
- Streamlit dashboard
- Supabase PostgreSQL integration
- Hugging Face deployment
- Cross-validation pipeline
- Data augmentation pipeline

## 6. Tech Stack

### 6.1 Machine Learning
- DistilBERT
- Scikit-learn
- PyTorch
- TF-IDF
- NLTK

### 6.2 Backend
- FastAPI
- Pydantic

### 6.3 Frontend
- Streamlit

### 6.4 Database
- Supabase PostgreSQL

### 6.5 Deployment
- Hugging Face Spaces
- Docker

### 6.6 Experiment Tracking
- DVC

## 7. System Workflow

Complaint Text
      ↓
Text Preprocessing
      ↓
Department Classifier (LinearSVC)
      ↓
Severity Classifier (DistilBERT)
      ↓
FastAPI Backend
      ↓
Streamlit Dashboard

## 8. Dataset

The project uses civic grievance complaint data collected and consolidated into Supabase PostgreSQL.

Dataset includes:
- complaint descriptions
- departments
- categories
- severity labels
- timestamps

## 9. Model Performance

### 9.1 Department Classification

| Model | Accuracy | Macro F1 |
|---|---|---|
| LinearSVC | 90.41% | 0.774 |
| Logistic Regression | 89.12% | 0.762 |

### 9.2 Severity Classification

| Model | Accuracy | Macro F1 |
|---|---|---|
| DistilBERT | 79% | 0.73 |
| Logistic Regression | 75% | 0.68 |

## 10. Deployment

The application is containerized using Docker and deployed on Hugging Face Spaces.

### Deployment Stack
- Docker
- Hugging Face Spaces
- FastAPI
- Streamlit

### Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit.py

