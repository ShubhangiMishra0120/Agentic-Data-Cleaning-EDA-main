# 🤖 Agentic Data Cleaning & EDA Pipeline

## 📖 Project Overview

This project is a comprehensive **Data Cleaning & Analysis Pipeline** that integrates **Agentic AI** with traditional data processing techniques. It is designed to ingest messy data from multiple sources (CSV, Database, API), clean it using a hybrid approach, and provide automated **Exploratory Data Analysis (EDA)**.

The system utilizes **FastAPI** for backend processing, **Streamlit** for an interactive frontend, and **Google Gemini (via LangChain)** to provide intelligent data imputation and narrative insights.

## 🚀 Features

### 1. Hybrid Data Cleaning
**Rule-Based Cleaning (Fast)**
* Automatically fills missing values (Mean/Median/Mode).
* Removes exact duplicates.
* Standardizes data types and formats.

**AI-Powered Cleaning (Smart)**
* Uses **Google Gemini** to handle semantic errors (e.g., spelling mistakes, inconsistent naming).
* Context-aware imputation for complex missing data.
* Batch processing handles large datasets efficiently.

### 2. Automated Exploratory Data Analysis (EDA)
* **Key Metrics:** Instant calculation of total rows, columns, and missing value counts.
* **Visualizations:**
    * *Correlation Heatmaps:* Identify relationships between numeric variables.
    * *Distribution Plots:* Analyze the spread of data columns.
* **AI Insights:** Generates a narrative report identifying trends, anomalies, and actionable conclusions.

### 3. Multi-Source Ingestion
* **CSV/Excel Upload:** Drag-and-drop interface for local files.
* **Database Query:** Connects to PostgreSQL to fetch and clean SQL data.
* **API Integration:** Fetches JSON data from external APIs.
* **Before vs. After Comparison:** Displays raw data alongside the AI-cleaned version for immediate verification.

## 🏗️ System Architecture: The Agents

The project employs a multi-agent architecture to balance speed and intelligence.


### 🔹 Rule-Based Agent
* **Technical Class:** `DataCleaning`
* **Role:** The Fast Pre-processor
* **Tasks:** Handles deterministic tasks like duplicate removal, type casting, and mathematical imputation instantly.

### 🔹 AI Cleaning Agent
* **Technical Class:** `AIAgent` (process_data)
* **Role:** The Contextual Cleaner
* **Tasks:** Uses Gemini LLM to fix spelling errors, standardize inconsistent formats (e.g., "60k" → 60000), and impute values based on context.

### 🔹 AI Data Analyst
* **Technical Class:** `AIAgent` (analyze_data)
* **Role:** The Insight Generator
* **Tasks:** Analyzes the statistical summary of the clean data to find anomalies, trends, and generate a text-based report.

## 🛠️ Tech Stack

### Backend
* **FastAPI:** High-performance asynchronous API framework.
* **SQLAlchemy:** Database ORM for PostgreSQL connection.
* **Pydantic:** Data validation and settings management.

### Frontend
* **Streamlit:** Interactive web interface for data ingestion and dashboarding.
* **Plotly:** Engine for dynamic and interactive charts.

### AI & Data
* **LangChain / LangGraph:** Orchestration framework for AI workflows.
* **Google Gemini:** Large Language Model (LLM) for reasoning and text generation.
* **Pandas:** Core library for data manipulation.