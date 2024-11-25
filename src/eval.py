from metrics import get_activation_naive_similarity_by_token, get_activation_semantic_similarity
from activations import load_gemma2b_model
from parsing import get_features_from_disk, json_to_link, get_activation_sentences_with_weights
import tqdm
import json
import os
import sys
sys.path.append("..")

def evaluate_metric(directory, model, tokenizer):
    files = os.listdir(directory)
    final_data = []
    for i in tqdm.tqdm(range(len(files))):
        file = files[i]
        if not file.endswith(".json"): continue
        for feature in get_features_from_disk(os.path.join(directory, file)):
            # print feature info
            print(f'{feature["modelId"]} {feature["layer"]} {feature["index"]}')
            sentences, weights = get_activation_sentences_with_weights(feature, 10)
            semantic_sim = get_activation_semantic_similarity(model, tokenizer, sentences, weights)
            naive_sim = get_activation_naive_similarity_by_token(model, tokenizer, sentences, weights)
            final_data.append({
                "modelId": feature["modelId"],
                "layer": feature["layer"],
                "index": feature["index"],
                "link": json_to_link(feature),
                "semantic_similarity": semantic_sim,
                "naive_similarity": naive_sim,
                "conceptual_level": semantic_sim/naive_sim,
                "num_sentences": len(sentences),
            })

    with open(os.path.join(directory, "conceptual_level_by_activations.json"), "w") as outfile:
        json.dump(final_data, outfile)

    final_data.sort(key=lambda x: x["conceptual_level"], reverse=True)
    print("Top 20 conceptual features:")
    for data in final_data[:20]:
        print(f"{data['link']}: {data['conceptual_level']}")

    print("\nBottom 20 conceptual features:")
    for data in final_data[-20:]:
        print(f"{data['link']}: {data['conceptual_level']}")

    return final_data

if __name__ == "__main__":
    tokenizer, model = load_gemma2b_model(device="cuda")
    directory = "data/gemma-2-2b-gemmascope-res-16k/20-gemmascope-res-16k"
    evaluate_metric(directory, model, tokenizer)
            