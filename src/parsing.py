import http.client
import json
from langdetect import detect, detect_langs

"""
Parse activation sentences/tokens from feature's json data.
"""
def get_activation_sentences(json_data, window, num_samples):
  # germanic_languages = {'en', 'de', 'nl', 'sv', 'da', 'no', 'is', 'af'}
  activation_sentences = []
  activation_token_counts = {}
  if "activations" not in json_data: return activation_sentences, activation_token_counts
  for sentence_data in json_data["activations"]:
    if "tokens" not in sentence_data or "maxValueTokenIndex" not in sentence_data: continue
    tokens = sentence_data["tokens"]
    maxValueIndex = sentence_data["maxValueTokenIndex"]
    activating_token = tokens[maxValueIndex]
    tokens = tokens[max(0, maxValueIndex - window):min(len(tokens), maxValueIndex+window)]
    sentence = "".join(tokens).replace('▁', ' ')
    try:
      # if detect(sentence) in germanic_languages:
      activation_token_counts[activating_token] = activation_token_counts.get(activating_token, 0) + 1
      if len(activation_sentences) == 0 or activation_sentences[-1] != sentence:
        activation_sentences.append(sentence)
    except:
      continue
    if len(activation_sentences) == num_samples: break
  return activation_sentences, activation_token_counts

"""
Parse activation sentences/values from feature's json data
"""
def get_activation_sentences_with_weights(json_data, num_samples=1000):
  sentences = []
  weights = []
  if "activations" not in json_data: return sentences, weights
  for sentence_data in json_data["activations"]:
    if "tokens" not in sentence_data or  "values" not in sentence_data: continue
    sentences.append("".join(sentence_data["tokens"]).replace('▁', ' '))
    weights.append(sentence_data["values"])
    if len(sentences) == num_samples: break
  return sentences, weights


"""
Get feature data using Neuronpedia API.
See https://www.neuronpedia.org/api-doc#tag/lookup/GET/api/feature/{modelId}/{layer}/{index}
"""
def get_feature(model, layer, index, api_key):
  print(model, layer, index)
  conn = http.client.HTTPSConnection("www.neuronpedia.org")
  headers = {"X-Api-Key": api_key}
  conn.request("GET", f"/api/feature/{model}/{layer}/{index}", headers=headers)
  res = conn.getresponse()
  data = res.read()
  # data = data.decode("utf-8")
  return json.loads(data)

"""
Get feature from a Neuronpedia link.
"""
def get_feature_from_link(link, api_key):
  link_parts = link.split("/")
  return get_feature(link_parts[-3], link_parts[-2], link_parts[-1], api_key)

"""
Read features from local file.
"""
def get_features_from_disk(filename):
  with open(filename, 'r') as f:
    features = json.load(f)
    for f in features: yield f

"""
Create a Neuronpedia link from model/ayer/index.
"""
def create_link(model, layer, index):
  return f"https://www.neuronpedia.org/{model}/{layer}/{index}"

def json_to_link(feature):
  return create_link(feature["modelId"], feature["layer"], feature["index"])