import os
from urllib.parse import quote


def postgres_url_from_env(default_host='four-brain-postgres', default_port=5433):
    # If SUPABASE_URL is present, prefer it (local Supabase strategy)
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_pw = os.getenv('SUPABASE_DB_PASSWORD') or os.getenv('POSTGRES_PASSWORD')
    db = os.getenv('POSTGRES_DB', 'ai_system')
    user = os.getenv('POSTGRES_USER', 'postgres')
    pw = quote((supabase_pw or 'ai_secure_2024'), safe='')
    if supabase_url:
        return f"postgresql://{user}:{pw}@{supabase_url}:{os.getenv('SUPABASE_DB_PORT','5432')}/{db}"
    # Fallback to compose Postgres on remapped host port 5433
    return f"postgresql://{user}:{pw}@{os.getenv('POSTGRES_HOST', default_host)}:{os.getenv('POSTGRES_PORT', default_port)}/{db}"

