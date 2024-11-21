import random 
from numpy import dot
from numpy.linalg import norm
from activations import get_activations, get_activations_word_by_word, get_activations_by_token, load_gemma2b_model
import numpy as np

"""
# Brute force N^2 average cosine similarity between sentence embedding pairs
"""
def mean_cos_sim(embeddings):
  avg_cos_sim = 0.0
  total_count = 0
  for i in range(len(embeddings)):
    for j in range(i, len(embeddings)):
      avg_cos_sim += dot(embeddings[i], embeddings[j])/(norm(embeddings[i])*norm(embeddings[j]))
      total_count += 1
  return avg_cos_sim/total_count

"""
Average pairwise cosine similarity between activation sentences.
"""
def get_semantic_similarity(sentences, model):
  embeddings = model.encode(sentences)
  return mean_cos_sim(embeddings)
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
Util
"""
def filter_empty_weights(sentences, weights):
  new_sentences  = []
  new_weights  = []
  for sentence, weight in zip(sentences, weights):
    if sum(weight) > 0:
      new_sentences.append(sentence)
      new_weights.append(weight)
  return new_sentences, new_weights

# Adjust the weights array
def adjust_weights(w, target_length):
    if len(w) > target_length:
        # Trim the weights
        return w[:target_length]
    elif len(w) < target_length:
        # Zero-pad the weights
        return np.pad(w, (0, target_length - len(w)))
    return w

"""

"""
def get_activation_semantic_similarity(model, tokenizer, sentences, weights):
  sentences, weights = filter_empty_weights(sentences, weights)
  activations = [get_activations(s, model, tokenizer).cpu() for s in sentences]
  embeddings = []
  for a, w, s in zip(activations, weights, sentences):
    a = np.array(a)
    w = np.array(w)
    # print(f"activations: {a.shape}")
    # print(f"weights: {w.shape}")
    # print(w.sum())
    # print(s)
    w = adjust_weights(w, a.shape[1])
    # Compute weighted average of output activations
    embeddings.append(np.squeeze(np.average(a, axis=1, weights=w)))
  return mean_cos_sim(embeddings)

"""

"""
def get_activation_naive_similarity_by_token(model, tokenizer, sentences, weights):
  sentences, weights = filter_empty_weights(sentences, weights)
  activations = [get_activations_by_token(s, w, model, tokenizer).cpu() for s, w in zip(sentences, weights)]
  embeddings = []
  for a, w in zip(activations, weights):
    w = [weight for weight in w if weight > 0.0]
    a = np.array(a)
    w = np.array(w)
    w = adjust_weights(w, a.shape[0])
    # Compute weighted average of output activations
    embeddings.append(np.squeeze(np.average(a, axis=0, weights=w)))
  return mean_cos_sim(embeddings)

"""

"""
def get_activation_naive_similarity(model, tokenizer, sentences, weights):
  word_activations = [get_activations_word_by_word(s, model, tokenizer).cpu() for s in sentences]
  embeddings = []
  for sentence, weight, a in zip(sentences, weights, word_activations):
    stripped_sentence = " ".join(sentence.split())
    # sentence_tokens = tokenizer(sentence, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze(0).tolist()
    sentence_tokens = tokenizer.tokenize(sentence, add_special_tokens=False)

    sentence_tokens = " ".join([token.strip() for token in sentence_tokens])

    # New weights per word
    word_weights = [0]
    # Index in stipped_sentences, which is seperated by word
    word_index = 0
    # Index in sentence_tokens, which is seperated by token
    token_index = 0
    # Index in weight, the current weight to add.
    weight_index = 0
    # Count num tokens we've seen at this word to take average
    token_count = 0
    while word_index < len(stripped_sentence):
      while stripped_sentence[word_index] == " " and word_index < len(stripped_sentence): 
        if token_count:
          word_weights[-1] /= token_count
          token_count = 0
        word_index += 1
        word_weights.append(0)
      if word_index == len(stripped_sentence): break

      while sentence_tokens[token_index] == " " and token_index < len(sentence_tokens):
        weight_index += 1
        token_index += 1
        token_count = 0

      word_weights[-1] += weight[weight_index]

      word_index += 1
      token_index += 1
      token_count += 1

    a = np.array(a)
    w = np.array(word_weights)
    # Compute weighted average of output activations
    embeddings.append(np.squeeze(np.average(a, axis=1, weights=w)))
  return mean_cos_sim(embeddings)

"""
Conceptual level of a feature: ratio of semantic to naive similarity between activation sentences.
"""
def get_conceptual_level(sentences, model, iterations=10, seed=8888):
    semantic_sim = get_semantic_similarity(sentences, model)
    naive_sim = get_naive_similarity(sentences, model, iterations, seed)
    return (semantic_sim/naive_sim) * semantic_sim, semantic_sim, naive_sim

if __name__ == "__main__":
  tokenizer, model = load_gemma2b_model()
  sentences = ["Hello friend.", "Kill me."]
  print(tokenizer.tokenize(sentences[1]))
  weights = [[1, 1, 0], [0, 0, 1]]

  print(get_activation_naive_similarity_by_token(model, tokenizer, sentences, weights))