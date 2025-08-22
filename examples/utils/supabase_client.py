from supabase import create_client

class SupabaseHelper:
    def __init__(self, url: str, key: str):
        self.client = create_client(url, key)

    def match_documents(self, embedding, match_count=10, similarity_threshold=0.3):
        return self.client.rpc("match_documents", {
            "query_embedding": embedding,
            "match_count": match_count,
            "similarity_threshold": similarity_threshold,
        }).execute()

