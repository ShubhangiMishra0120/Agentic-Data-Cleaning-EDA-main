import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
from dotenv import load_dotenv

# Load your password from .env
load_dotenv()

# Configuration
DB_USER = "your_database_username"
DB_PASSWORD = "your_database_password"  
DB_HOST = "your_database_host"
NEW_DB_NAME = "your_database_name"

def create_database():
    """Creates the demoDb database if it doesn't exist."""
    try:
        # Connect to the default 'postgres' database to create a new one
        con = psycopg2.connect(
            dbname="postgres", 
            user=DB_USER, 
            password=DB_PASSWORD, 
            host=DB_HOST
        )
        con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = con.cursor()

        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{NEW_DB_NAME}'")
        if cursor.fetchone():
            print(f"⚠️  Database '{NEW_DB_NAME}' already exists.")
        else:
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(NEW_DB_NAME)))
            print(f"✅ Database '{NEW_DB_NAME}' created successfully.")
            
        cursor.close()
        con.close()
    except Exception as e:
        print(f"❌ Error creating database: {e}")

def create_table_and_data():
    """Connects to demoDb, creates my_table, and inserts data."""
    try:
        con = psycopg2.connect(
            dbname=NEW_DB_NAME, 
            user=DB_USER, 
            password=DB_PASSWORD, 
            host=DB_HOST
        )
        cursor = con.cursor()

        # 1. Create the Table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS my_table (
            id SERIAL PRIMARY KEY,
            name VARCHAR(50),
            age INTEGER,
            city VARCHAR(50),
            salary DOUBLE PRECISION
        );
        """
        cursor.execute(create_table_query)
        print("✅ Table 'my_table' created.")

        # 2. Insert Data (The Clean Data)
        data = [
            (1, 'Alice', 25, 'New York', 50000),
            (2, 'Bob', 30, 'Los Angeles', 60000),
            (3, 'Charlie', 28, 'Chicago', 70000),
            (4, 'David', 35, 'Houston', 80000),
            (5, 'Eve', 40, 'San Francisco', 90000)
        ]
        
        insert_query = """
        INSERT INTO my_table (id, name, age, city, salary) 
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING;
        """
        
        cursor.executemany(insert_query, data)
        con.commit()
        print(f"✅ {len(data)} rows inserted into 'my_table'.")

        cursor.close()
        con.close()

    except Exception as e:
        print(f"❌ Error managing table/data: {e}")

if __name__ == "__main__":
    create_database()
    create_table_and_data()
