from supabase import create_client, Client

url: str = "https://grtbcjgjalbtrejqmzvo.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdydGJjamdqYWxidHJlanFtenZvIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczMDk1NTM2NywiZXhwIjoyMDQ2NTMxMzY3fQ.Bx8RUsFO2cYlKvhE42OnzwU4gNWmG2LPkKGKHqBuXZY"
supabase: Client = create_client(url, key) 