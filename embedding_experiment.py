import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

## Top K retrieval 
K = 3

def cosine_similarity(a, b):
    ## magnitude of A and B
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")

    if mag_a == 0 or mag_b == 0:
        raise ValueError("Cannot calculate cosine similarity with a zero vector")
    
    return (sum(a * b) /
            np.dot(mag_a, mag_b) )

sentences = [
    "ATP synthase produces ATP using a proton gradient.",
    "Cats are obligate carnivores.",
    "The electron transport chain creates a proton gradient.",
    "Bacteria reproduce through binary fission."
]

question = "How do cells use hydrogen ions to produce ATP"

embeddings = model.encode(sentences)

q_embeddings = model.encode(question)

similarities = []

## Get the similarities to sentence from the question
for emb in embeddings:
    similarities.append(cosine_similarity(q_embeddings, emb))
    
## sort index from most to least similar
ranked_indices = np.argsort(similarities)[::-1]

for i in ranked_indices:
    print(str(i) + ": " + sentences[i])