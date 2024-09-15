import random 
from numpy import dot
from numpy.linalg import norm

"""
Average pairwise cosine similarity between activation sentences.
"""
def get_semantic_similarity(sentences, model):
  embeddings = model.encode(sentences)
  avg_cos_sim = 0.0
  total_count = 0
  # Brute force N^2 average cosine similarity between sentence embedding pairs
  for i in range(len(embeddings)):
    for j in range(i, len(embeddings)):
      avg_cos_sim += dot(embeddings[i], embeddings[j])/(norm(embeddings[i])*norm(embeddings[j]))
      total_count += 1
  return avg_cos_sim/total_count

"""
Average pairwise cosine similarity between activation sentences with shuffled words.
Process repeated for multiple iterations to get a more robust estimate.
"""
def get_naive_similarity(sentences, model, iterations=10, seed=8888):
  random.seed(seed)
  avg_naive_sim = 0.0
  for _ in range(iterations):
    shuffled_sentences = []
    for sentence in sentences:
      tokens = sentence.split()
      random.shuffle(tokens)
      shuffled_sentences.append(" ".join(tokens))
    avg_naive_sim += get_semantic_similarity(shuffled_sentences, model)
  return avg_naive_sim/iterations

"""
Conceptual level of a feature: ratio of semantic to naive similarity between activation sentences.
"""
def get_conceptual_level(sentences, model, iterations=10, seed=8888):
    semantic_sim = get_semantic_similarity(sentences, model)
    naive_sim = get_naive_similarity(sentences, model, iterations, seed)
    return (semantic_sim/naive_sim) * semantic_sim, semantic_sim, naive_sim