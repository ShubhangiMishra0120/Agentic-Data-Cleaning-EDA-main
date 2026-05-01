import pandas as pd
import io
import aiohttp
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv

from sqlalchemy import create_engine

# Load environment variables
load_dotenv()

# ✅ Clean Imports (No sys.path hacks needed if file is in root)
from scripts.ai_agent import AIAgent
from scripts.data_cleaning import DataCleaning

app = FastAPI()

# Initialize AI agent and rule-based data cleaner
ai_agent = AIAgent()
cleaner = DataCleaning()

# ----------------------- CSV / Excel Cleaning Endpoint -----------------------------

@app.post("/clean-data")

async def clean_data(file: UploadFile = File(...)):
    """Receives file from UI, cleans it using rule-based & AI methods, and returns cleaned JSON."""
    print("STEP 1: Request received")
    try:
        contents = await file.read()
        print("STEP 2: File read")
        file_extension = file.filename.split(".")[-1]

        # Load file into Pandas DataFrame
        if file_extension == "csv":
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        elif file_extension == "xlsx":
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or Excel.")

        print("STEP 3: Data loaded")
        df_cleaned = cleaner.clean_data(df)

        print("STEP 4: Calling AI agent")
        try:
            df_ai_cleaned = ai_agent.process_data(df_cleaned)
            print("STEP 5: AI agent finished")
        except Exception as e:
            print("AGENT ERROR:", str(e))
            return {"error": str(e)}

        # Ensure AI output is converted back to a DataFrame
        # Note: This assumes the AI returns a valid CSV string.
        if isinstance(df_ai_cleaned, str):
            from io import StringIO
            # Wrap in try-except in case AI returns conversational text
            try:
                df_ai_cleaned = pd.read_csv(StringIO(df_ai_cleaned))
            except Exception:
                # Fallback: if AI output isn't perfect CSV, return the raw text for debugging
                return {"cleaned_data": [], "raw_ai_response": df_ai_cleaned}

        print(f"🚀 [BACKEND] Successfully returning cleaned data. Output format: {type(df_ai_cleaned)}")
        return {"cleaned_data": df_ai_cleaned.fillna("").to_dict(orient="records")}

    except Exception as e:
        print(f"❌ [BACKEND ERROR] Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# ----------------------- Database Query Cleaning Endpoint -----------------------------

class DBQuery(BaseModel):
    db_url: str
    query: str

@app.post("/clean-db")
async def clean_db(query: DBQuery):
    """Fetches data from a database, cleans it using AI, and returns raw and cleaned JSON."""
    try:
        engine = create_engine(query.db_url)
        df = pd.read_sql(query.query, engine)

        # Step 1: Rule-Based Cleaning
        df_cleaned = cleaner.clean_data(df)

        # Step 2: AI-Powered Cleaning
        df_ai_cleaned = ai_agent.process_data(df_cleaned)

        # Convert AI cleaned data to DataFrame
        if isinstance(df_ai_cleaned, str):
            from io import StringIO
            df_ai_cleaned = pd.read_csv(StringIO(df_ai_cleaned))

        return {
            "raw_data": df.fillna("").to_dict(orient="records"),
            "cleaned_data": df_ai_cleaned.fillna("").to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data from database: {str(e)}")

# ----------------------- API Data Cleaning Endpoint -----------------------------

class APIRequest(BaseModel):
    api_url: str

@app.post("/clean-api")
async def clean_api(api_request: APIRequest):
    """Fetches data from an API, cleans it using AI, and returns comparison JSON."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(api_request.api_url) as response:
                if response.status != 200:
                    raise HTTPException(status_code=400, detail="Failed to fetch data from API.")

                data = await response.json()
                
                # 1. Handle Nested JSON (e.g. {'products': [...]})
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, list):
                            data = value
                            break
                
                df = pd.DataFrame(data)

                # 2. FIX: Convert Lists/Dicts to Strings to avoid "Unhashable type" error
                for col in df.columns:
                    if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                        df[col] = df[col].astype(str)

        # Step 1: Rule-Based Cleaning
        df_cleaned = cleaner.clean_data(df)

        # Step 2: AI-Powered Cleaning
        df_ai_cleaned = ai_agent.process_data(df_cleaned)

        # Convert AI cleaned data back to DataFrame if it's a string
        if isinstance(df_ai_cleaned, str):
            from io import StringIO
            try:
                df_ai_cleaned = pd.read_csv(StringIO(df_ai_cleaned))
            except Exception:
                # Fallback if AI returns text instead of CSV
                df_ai_cleaned = pd.DataFrame([{"error": "AI response was not valid CSV", "raw": df_ai_cleaned}])

        # Return BOTH raw and cleaned data
        return {
            "raw_data": df.fillna("").to_dict(orient="records"),
            "cleaned_data": df_ai_cleaned.fillna("").to_dict(orient="records")
        }

    except Exception as e:
        print(f"❌ API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing API data: {str(e)}")
# ----------------------- Run Server -----------------------------

if __name__ == "__main__":
    import uvicorn
    # Host on 127.0.0.1 (Localhost)
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)