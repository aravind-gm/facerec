from supabase import create_client, Client
from dotenv import load_dotenv
import os
import logging

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Connected to Supabase")

    test_data = {
        "name": "Test User",
        "employee_id": "TEST1234",
        "department": "Test Dept",
        "position": "Tester",
        "face_embedding": "test_embedding",  # Replace with a real embedding if testing that column
        "active": True
    }

    response = supabase.table("people").insert(test_data).execute()

    if response.error:
        logger.error(f"Supabase error: {response.error}")
    else:
        logger.info(f"Test data inserted successfully: {response.data}")

except Exception as e:
    logger.error(f"Error: {e}")