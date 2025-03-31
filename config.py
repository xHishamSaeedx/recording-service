from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
JAEGER_HOST = os.getenv("JAEGER_HOST", "localhost")  # Default to localhost if not set
JAEGER_PORT = int(os.getenv("JAEGER_PORT", "6831"))  # Default to 6831 if not set
SERVICE_ACCOUNT_EMAIL = os.getenv('SERVICE_ACCOUNT_EMAIL')
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE")
SENTENCE_TRANSFORMERS_HOME = os.getenv("SENTENCE_TRANSFORMERS_HOME")
HF_HOME = os.getenv("HF_HOME")
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")