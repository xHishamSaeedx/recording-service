from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from supabase_client import supabase
import asyncio

async def find_similar_feedbacks(embeddings: list[float], workspace_id: str, threshold: float = 0.55):
    """
    Find similar feedbacks using cosine similarity
    Returns feedbacks with similarity score above threshold and the most similar feedback_insight_id
    """
    try:
        response = await asyncio.to_thread(
            lambda: supabase.table('feedback_insights')
                    .select('feedback_insights_id, feedback, embeddings')
                    .eq('workspace_id', workspace_id)
                    .execute()
        )
        
        if not response.data:
            return [], None

        query_embedding = np.array(embeddings).reshape(1, -1)
        
        similar_feedbacks = []
        for row in response.data:
            if row.get('embeddings'):
                stored_embedding = np.array(row['embeddings']).reshape(1, -1)
                similarity = cosine_similarity(query_embedding, stored_embedding)[0][0]
                
                if similarity > threshold:
                    similar_feedbacks.append({
                        'feedback_insights_id': row['feedback_insights_id'],
                        'feedback': row['feedback'],
                        'similarity_score': float(similarity)
                    })
        
        similar_feedbacks.sort(key=lambda x: x['similarity_score'], reverse=True)
        most_similar_id = similar_feedbacks[0]['feedback_insights_id'] if similar_feedbacks else None
        return similar_feedbacks, most_similar_id

    except Exception as e:
        print(f"Error finding similar feedbacks: {str(e)}")
        return [], None