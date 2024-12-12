import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os

def load_gemma2b_model(device="cuda"):
    # Load the tokenizer and the model for Gemma2 2B
    model_name = "google/gemma-2-2b"  # You need to replace with the correct path or model identifier
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ['HF_TOKEN'])
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, token=os.environ['HF_TOKEN'])
    model = model.to(device)
    return tokenizer, model

def get_activations(input_text, model, tokenizer, device="cuda"):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors='pt', add_special_tokens=False).to(device)
    
    # Run the model and get the output
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the final hidden states (activations)
    hidden_states = outputs.hidden_states  # A tuple of hidden states from all layers
    final_layer_activations = hidden_states[-1]  # Get the last layer's activations
    return final_layer_activations

"""

"""
def get_activations_by_token(input_text, weights, model, tokenizer, device="cuda"):
    # Tokenize the input text
    tokens = tokenizer(input_text, return_tensors='pt', add_special_tokens=False).to(device)
    input_ids = tokens['input_ids'].squeeze(0)
    activations_list = []
    # Iterate through each token and get the activations if weight > 0
    for token_id, weight in zip(input_ids, weights):
        if weight > 0.0:
            token_tensor = token_id.unsqueeze(0).unsqueeze(0)  # Shape to match batch and sequence dimensions

            # Move token to device
            token_tensor = token_tensor.to(device)

            # Run the model on the single token
            with torch.no_grad():
                outputs = model(input_ids=token_tensor)

            # Extract the final hidden states (activations)
            hidden_states = outputs.hidden_states  # A tuple of hidden states from all layers
            final_layer_activations = hidden_states[-1].squeeze(0)  # Get the last layer's activations
            activations_list.append(final_layer_activations.cpu())

    # Concatenate activations along the sequence dimension
    if activations_list:
        activations = torch.cat(activations_list, dim=0)
    else:
        activations = torch.empty(0)  # Return an empty tensor if no activations were collected

    return activations
        
"""

"""
def get_activations_word_by_word(input_text, weights, model, tokenizer):
    # remove tokens with zero weight
    tokens = tokenizer.tokenize(input_text, add_special_tokens=False)
    input_text = ""
    for token, weight in zip(tokens, weights):
        if weight>0:
            input_text += token
    
    input_text.replace("▁", " ")

    # Split the input text into words
    words = input_text.split()
    activations_list = []
    
    # Iterate through each word and get the activations
    for word in words:
        inputs = tokenizer(word, return_tensors='pt').to("cuda")
        
        # Run the model and get the output
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract the final hidden states (activations)
        hidden_states = outputs.hidden_states  # A tuple of hidden states from all layers
        final_layer_activations = hidden_states[-1].mean(dim=1, keepdim=True)  # Get the last layer's activations
        activations_list.append(final_layer_activations.cpu())
    
    # Concatenate activations along the sequence dimension
    activations = torch.cat(activations_list, dim=1)
    return activations


if __name__ == "__main__":
    tokenizer, model = load_gemma2b_model()
    input_text = "This is an example input text."
    weights = [1] * len(tokenizer.tokenize(input_text, add_special_tokens=False))
    weights[2] = 0
    print(weights)
    token_activations = get_activations_by_token(input_text, weights, model, tokenizer)
    
    # Print the shape of the activations
    print("Shape of activations:", token_activations.shape) 

    # # Get activations for each word in the input text
    # word_activations = get_activations_word_by_word(input_text, model, tokenizer)
    
    # # Print the shape of the activations
    # print("Shape of activations:", word_activations.shape)  # Should be [batch_size, total_sequence_length, hidden_size]
    
    # If you want to manipulate or access specific parts of the activations, you can continue from here.
