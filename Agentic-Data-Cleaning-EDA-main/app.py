import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import json

# If you import backend logic directly into Streamlit:
from scripts.ai_agent import AIAgent 
# FastAPI Backend URL
FASTAPI_URL = "http://127.0.0.1:8000"

# Streamlit UI Configuration
st.set_page_config(page_title="AI-Powered Data Cleaning & EDA", layout="wide")

# Create Tabs
cleaning_tab, eda_tab = st.tabs(["Data Cleaning", "EDA Dashboard"])

# Data Cleaning Tab
with cleaning_tab:
    st.sidebar.header("📊 Data Source Selection")
    data_source = st.sidebar.radio(
        "Select Data Source:",
        ["CSV/Excel", "Database Query", "API Data"],
        index=0
    )

    st.markdown("""
    # ✏️ **AI-Powered Data Cleaning**
    *Clean your data effortlessly using AI-powered processing!*
    """)

    # ✅ Handling CSV/Excel Upload
    if data_source == "CSV/Excel":
        st.subheader("📂 Upload File for Cleaning")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

        if uploaded_file is not None:
            file_extension = uploaded_file.name.split(".")[-1]
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write("### 🔍 Raw Data Preview:")
            st.dataframe(df)

            if st.button("🚀 Clean Data"):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                print(f"[FRONTEND] Sending file '{uploaded_file.name}' to backend...")
                try:
                    response = requests.post(f"{FASTAPI_URL}/clean-data", files=files)
                    print(f"[FRONTEND] STATUS: {response.status_code}")
                    if response.status_code != 200:
                        print(f"[FRONTEND] ERROR RESPONSE: {response.text}")
                    
                    if response.status_code == 200:
                        try:
                            cleaned_data_raw = response.json()["cleaned_data"]
                            if isinstance(cleaned_data_raw, str):
                                cleaned_data = pd.DataFrame(json.loads(cleaned_data_raw))
                            else:
                                cleaned_data = pd.DataFrame(cleaned_data_raw)

                            st.session_state.cleaned_data = cleaned_data  # Store cleaned data in session state
                            st.session_state.current_analysis_df = cleaned_data  # Update shared state

                            st.subheader("✅ Cleaned Data:")
                            st.dataframe(cleaned_data)
                        except Exception as e:
                            print(f"[FRONTEND] Error converting response to DataFrame: {e}")
                            st.error(f"❌ Error converting response to DataFrame: {e}")
                    else:
                        st.error(f"❌ Failed to clean data. Status: {response.status_code}")
                except Exception as e:
                    print(f"[FRONTEND] NETWORK ERROR: {str(e)}")
                    st.error(f"❌ Could not connect to backend: {str(e)}")

    # ✅ Handling Database Query
    elif data_source == "Database Query":
        st.subheader("🔍 Enter Database Query")
        db_url = st.text_input("Database Connection URL:", "postgresql://user:password@localhost:5432/db")
        query = st.text_area("Enter SQL Query:", "SELECT * FROM my_table;")

        if st.button("📥 Fetch & Clean Data"):
            response = requests.post(f"{FASTAPI_URL}/clean-db", json={"db_url": db_url, "query": query})

            if response.status_code == 200:
                try:
                    # Parse the response
                    raw_data = pd.DataFrame(response.json()["raw_data"])
                    cleaned_data = pd.DataFrame(response.json()["cleaned_data"])

                    # Update shared state
                    st.session_state.cleaned_data = cleaned_data
                    st.session_state.current_analysis_df = cleaned_data

                    # Display Before vs After Comparison
                    st.subheader("🔄 Before vs After Cleaning")

                    st.write("### Raw Data (Before)")
                    st.dataframe(raw_data)

                    st.write("### AI Cleaned Data (After)")
                    st.dataframe(cleaned_data)

                except Exception as e:
                    st.error(f"❌ Error processing response: {e}")
            else:
                st.error("❌ Failed to fetch/clean data from database.")

    # ✅ Handling API Data
    elif data_source == "API Data":
        st.subheader("🌐 Fetch Data from API")
        api_url = st.text_input("Enter API Endpoint:", "https://jsonplaceholder.typicode.com/posts")

        if st.button("📥 Fetch & Clean Data"):
            response = requests.post(f"{FASTAPI_URL}/clean-api", json={"api_url": api_url})

            if response.status_code == 200:
                try:
                    # Parse the response
                    raw_data = pd.DataFrame(response.json()["raw_data"])
                    cleaned_data = pd.DataFrame(response.json()["cleaned_data"])

                    # Update shared state
                    st.session_state.cleaned_data = cleaned_data
                    st.session_state.current_analysis_df = cleaned_data

                    # Display Before vs After Comparison
                    st.subheader("### Raw Data (Before)")
                    st.dataframe(raw_data)

                    st.markdown("---")  # Visual separator

                    st.subheader("### AI Cleaned Data (After)")
                    st.dataframe(cleaned_data)

                except Exception as e:
                    st.error(f"❌ Error processing response: {e}")
            else:
                st.error("❌ Failed to fetch/clean data from API.")

# EDA Dashboard Tab
with eda_tab:
    st.markdown("""
    # 📊 **EDA Dashboard**
    *Explore your data with visualizations and insights!*
    """)

    if "current_analysis_df" in st.session_state:
        current_analysis_df = st.session_state.current_analysis_df

        # Key Metrics
        st.subheader("🔑 Key Metrics")
        st.write(f"**Total Rows:** {current_analysis_df.shape[0]}")
        st.write(f"**Total Columns:** {current_analysis_df.shape[1]}")
        st.write(f"**Missing Values:** {current_analysis_df.isnull().sum().sum()}")

        # Correlation Heatmap
        st.subheader("📈 Correlation Heatmap")
        numeric_cols = current_analysis_df.select_dtypes(include=['float64', 'int64']).columns
        if not numeric_cols.empty:
            corr = current_analysis_df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
            st.plotly_chart(fig)
        else:
            st.write("No numeric columns available for correlation heatmap.")

        # Distribution Plot
        st.subheader("📊 Distribution Plot")
        column = st.selectbox("Select a column to visualize:", current_analysis_df.columns)
        if column:
            fig = px.histogram(current_analysis_df, x=column, title=f"Distribution of {column}")
            st.plotly_chart(fig)

        # AI-Powered Insights
        st.subheader("🤖 AI-Powered Insights")
        agent = AIAgent()  # Initialize the AI Agent
        if st.button("🤖 Generate AI Insights"):
            with st.spinner("Analyzing data patterns..."):
                insights = agent.analyze_data(current_analysis_df)
                st.success("### AI Insights")
                st.markdown(insights)
    else:
        st.warning("⚠️ No data available for analysis. Please clean or fetch data first.")

# Footer
st.markdown("""
---
🚀 *Built with **Streamlit + FastAPI + AI** for automated data cleaning and EDA* 🔥
""")
