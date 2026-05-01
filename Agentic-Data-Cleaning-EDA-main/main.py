import sys
import os
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

# ✅ Clean Imports
from scripts.data_ingestions import DataIngestion
from scripts.data_cleaning import DataCleaning
from scripts.ai_agent import AIAgent

# ✅ Database Configuration - All values from .env file
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# Validate all required database credentials are present
if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
    raise ValueError("❌ Missing database credentials. Please set DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, and DB_NAME in your .env file.")

# Construct the connection string
DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ✅ Initialize Components
try:
    ingestion = DataIngestion(DB_URL)
    cleaner = DataCleaning()
    ai_agent = AIAgent()
    print("✅ All components initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing components: {e}")
    sys.exit(1)

### === 1️⃣ Load and Clean CSV Data === ###
# Ensure sample_data.csv exists in your 'data/' folder before running
df_csv = ingestion.load_csv("sample_data.csv")
if df_csv is not None:
    print("\n ♦ Cleaning CSV Data...")
    df_csv = cleaner.clean_data(df_csv)
    # Pass to AI Agent
    print("🤖 AI Agent is processing CSV data...")
    df_csv = ai_agent.process_data(df_csv)
    print("\n✅ AI-Cleaned CSV Data:\n", df_csv)

### === 2️⃣ Load and Clean Excel Data === ###
# Only runs if you have created sample_data.xlsx
df_excel = ingestion.load_excel("sample_data.xlsx")
if df_excel is not None:
    print("\n ♦ Cleaning Excel Data...")
    df_excel = cleaner.clean_data(df_excel)
    df_excel = ai_agent.process_data(df_excel)
    print("\n✅ AI-Cleaned Excel Data:\n", df_excel)
else:
    print("\n⚠️ Skipping Excel test (sample_data.xlsx not found).")

### === 3️⃣ Load and Clean Database Data === ###
# We fetch from 'my_table' which we know exists in demoDb
df_db = ingestion.load_from_database("SELECT * FROM my_table")
if df_db is not None:
    print("\n ♦ Cleaning Database Data...")
    df_db = cleaner.clean_data(df_db)
    df_db = ai_agent.process_data(df_db)
    print("\n✅ AI-Cleaned Database Data:\n", df_db)

### === 4️⃣ Fetch and Clean API Data === ###
# ✅ Fetch API Data
API_URL = " "
df_api = ingestion.fetch_from_api(API_URL)

if df_api is not None:
    print("\n ♦ Cleaning API Data...")

    # ✅ Keep only first 10 rows to save money/tokens on OpenAI
    df_api = df_api.head(10)

    # ✅ Reduce long text fields before sending to OpenAI
    if "body" in df_api.columns:
        df_api["body"] = df_api["body"].apply(lambda x: x[:100] + "..." if isinstance(x, str) else x)

    df_api = cleaner.clean_data(df_api)
    df_api = ai_agent.process_data(df_api)

    print("\n✅ AI-Cleaned API Data:\n", df_api)
