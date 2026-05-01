import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONNECTION DETAILS ---
DB_HOST = "your_database_host"      
DB_PORT = "your_database_port"     
DB_NAME = "your_database_name" 
DB_USER = "your_database_user"    
DB_PASSWORD = "your_database_password"

try:
    print(f"🔌 Attempting to connect to '{DB_NAME}' as '{DB_USER}'...")
    
    # Connect to PostgreSQL database
    connection = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    
    cursor = connection.cursor()
    print("✅ PostgreSQL Connection Successful!")

    # Execute a test query to list all tables in the 'public' schema
    print(f"\n📊 Tables in '{DB_NAME}':")
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
    tables = cursor.fetchall() #to fetch all table names
    
    if tables:
        for table in tables:
            print(f"   - {table[0]}")  # Should print 'my_table'
    else:
        print("   (No tables found)")

    # Close connection
    cursor.close()
    connection.close() # Closing the connection
    print("\n🔒 Connection Closed.")

except Exception as e:
    print(f"\n❌ Error connecting to PostgreSQL: {e}")
    print("💡 Tip: Check if your PostgreSQL server is running with: 'brew services list'")
