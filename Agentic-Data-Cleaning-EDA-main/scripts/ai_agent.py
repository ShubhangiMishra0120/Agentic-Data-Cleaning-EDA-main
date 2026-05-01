import os
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY", "")

# Add a debug print/log to verify key is present in environment
print("API KEY LOADED:", bool(groq_api_key))

# Clean the key (removes any accidental trailing spaces or quotes from .env)
groq_api_key = groq_api_key.strip().strip('"').strip("'")

# Define AI Model (Groq)
llm = None
if groq_api_key:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        api_key=groq_api_key,
        temperature=0
    )
else:
    print("⚠️ WARNING: GROQ_API_KEY not found. AI features will not work.")

class CleaningState(BaseModel):
    input_text: str
    structured_response: str = ""

class AIAgent:
    def __init__(self):
        self.graph = self.create_graph()

    def create_graph(self):
        graph = StateGraph(CleaningState)

        def agent_logic(state: CleaningState) -> CleaningState:
            try:
                if llm is None:
                    return CleaningState(
                        input_text=state.input_text,
                        structured_response="Error: GROQ_API_KEY not configured."
                    )
                
                # DEBUG PRINT: Check if Agent is receiving input
                print(f"\n🤖 Agent Input (Preview): {state.input_text[:50]}...")
                
                response_msg = llm.invoke(state.input_text)
                
                # --- FIX START: Handle List vs String content ---
                content = response_msg.content
                
                # If content is a list (e.g. [{'type': 'text', ...}]), extract the text
                if isinstance(content, list):
                    # Extract 'text' field if it exists, otherwise convert to string
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict):
                            text_parts.append(block.get("text", ""))
                        else:
                            text_parts.append(str(block))
                    content = "".join(text_parts)
                
                # Ensure it is definitively a string
                content = str(content)
                # --- FIX END ---

                # DEBUG PRINT: Check what Groq actually replied
                print(f"✅ Agent Output: {content[:100]}...")

                return CleaningState(
                    input_text=state.input_text, 
                    structured_response=content
                )
            except Exception as e:
                print(f"❌ Error in Agent: {str(e)}")
                return CleaningState(
                    input_text=state.input_text,
                    structured_response=f"Error: {str(e)}"
                )

        graph.add_node("cleaning_agent", agent_logic)
        graph.add_edge("cleaning_agent", END)
        graph.set_entry_point("cleaning_agent")
        return graph.compile()

    def process_data(self, df, batch_size=100):
        # DEBUG PRINT: Check if DataFrame is valid
        print(f"\n📊 Processing Data... Rows: {len(df)}")
        if len(df) == 0:
            return "Error: DataFrame is empty."

        cleaned_responses = []

        for i in range(0, len(df), batch_size):
            df_batch = df.iloc[i:i + batch_size]
            
            prompt = f"""
            You are an AI Data Cleaning Agent. 
            Input Data (CSV format):
            {df_batch.to_string()}

            Task:
            1. Identify missing values and impute them (mean/median/mode).
            2. Fix inconsistent formatting.
            3. Remove duplicates.

            Output:
            Return ONLY the cleaned dataset in CSV format. 
            NO explanations. NO markdown code blocks (like ```csv).
            """
            
            print(f"🤖 [AI AGENT] Starting AI processing for batch {i//batch_size + 1}...")
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    state = CleaningState(input_text=prompt, structured_response="")
                    response = self.graph.invoke(state)
                    
                    # Extract content properly if response is a dict or state object
                    if isinstance(response, dict):
                        response = CleaningState(**response)

                    # Check for rate limit error in response
                    if "RESOURCE_EXHAUSTED" in response.structured_response or "429" in response.structured_response:
                        if attempt < max_retries - 1:
                            print(f"⚠️ Rate limit hit. Retrying in 10 seconds... (Attempt {attempt+1}/{max_retries})")
                            import time
                            time.sleep(10)
                            continue
                        else:
                            raise Exception(response.structured_response)

                    cleaned_responses.append(response.structured_response)
                    print(f"✅ [AI AGENT] Finished AI processing for batch {i//batch_size + 1}.")
                    break # Success, exit retry loop
                except Exception as e:
                    print(f"❌ [AI AGENT ERROR]: {str(e)}")
                    cleaned_responses.append(f"Error: {str(e)}")
                    break
            
            # Anti-Rate-Limiting (Free tier has 15-20 RPM limit)
            import time
            time.sleep(4)

        return "\n".join(cleaned_responses)
    def analyze_data(self, df):
        """
        Analyze the given DataFrame and return AI-generated insights.
        """
        # 1. Build context from the DataFrame
        column_info = df.dtypes.to_string()
        basic_stats = df.describe().to_string()
        first_rows = df.head().to_string()

        # 2. Construct the AI prompt
        prompt = f"""
        You are an AI Data Scientist. Analyze the following dataset summary and provide insights:

        Column Information:
        {column_info}

        Basic Statistics:
        {basic_stats}

        First 5 Rows of Data:
        {first_rows}

        Task:
        1. Identify 3 key trends in the data.
        2. Highlight any potential anomalies.
        3. Provide a brief conclusion about the dataset.
        
        Output:
        Return a clean Markdown formatted response.
        """

        try:
            if llm is None:
                return "❌ Error: GROQ_API_KEY not configured. AI Analysis is unavailable."
                
            # 3. Pass the prompt to the Groq model
            response_msg = llm.invoke(prompt)
            
            # --- CRITICAL FIX: Handle List vs String content ---
            content = response_msg.content
            
            # If content is a list (e.g. [{'type': 'text', ...}]), extract the text
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        text_parts.append(block.get("text", ""))
                    else:
                        text_parts.append(str(block))
                content = "".join(text_parts)
            
            # Ensure it is definitively a string
            return str(content)
            # ---------------------------------------------------

        except Exception as e:
            return f"❌ Error in AI Analysis: {str(e)}"