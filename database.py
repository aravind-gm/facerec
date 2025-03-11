import os
from supabase import create_client
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get Supabase credentials
SUPABASE_URL = "https://iohaewsejnmakbmejqzt.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlvaGFld3Nlam5tYWtibWVqcXp0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDEwMjMyNTUsImV4cCI6MjA1NjU5OTI1NX0.uZXDOjx6Y1Iei9g6RA-dvWVpAsjvz3qvdVMRHGy6H0M"

try:
    # Initialize Supabase client
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Successfully connected to Supabase")
except Exception as e:
    logger.error(f"Error connecting to Supabase: {str(e)}")
    raise