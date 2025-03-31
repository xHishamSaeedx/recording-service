from sentence_transformers import SentenceTransformer
import numpy as np
import os
TRANSFORMERS_CACHE="./cache/transformers"
SENTENCE_TRANSFORMERS_HOME="./cache/sentence_transformers"
HF_HOME="./cache/huggingface"
MODEL_NAME="all-MiniLM-L6-v2"

# Create cache directories if they don't exist
cache_dirs = [
    TRANSFORMERS_CACHE,
    SENTENCE_TRANSFORMERS_HOME,
    HF_HOME
]

for cache_dir in cache_dirs:
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

# Initialize the model with cache directory
model = SentenceTransformer(
    MODEL_NAME,
    cache_folder=SENTENCE_TRANSFORMERS_HOME
)

async def get_embeddings(text: str) -> list[float]:
    """
    Generate embeddings using sentence-transformers.
    Returns a list of floats representing the text embedding.
    """
    try:
        # Generate embeddings
        embedding = model.encode(text, convert_to_numpy=True)
        # Convert to list and return
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        # Return a zero vector of the same dimension as a fallback
        return [0.0] * 384  # all-MiniLM-L6-v2 produces 384-dimensional embeddings