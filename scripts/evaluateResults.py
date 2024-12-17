import os
import random
import torch
import pickle
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    RobertaTokenizer,
    RobertaModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import re

out_dir = "/n/home03/ahmadazim/WORKING/synthData/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open(f"{out_dir}baseline/holdout_results_per_iteration.pkl", "rb") as f:
    baseline_res_iter = pickle.load(f)

with open(f"{out_dir}perplexitySampling_inverse/holdout_results_per_iteration.pkl", "rb") as f:
    highperpSample_res_iter = pickle.load(f)

with open(f"{out_dir}referenceSampling/holdout_results_per_iteration.pkl", "rb") as f:
    refSample_res_iter = pickle.load(f)

with open(f"{out_dir}perplexitySampling/holdout_results_per_iteration.pkl", "rb") as f:
    lowperpSample_res_iter = pickle.load(f)

def process_results(results, method_name):
    rows = []
    for iteration, prompts in results.items():
        for prompt_idx, (prompt, candidates) in enumerate(prompts):
            for candidate_idx, candidate in enumerate(candidates):
                row = {
                    "iteration": iteration,
                    "prompt_idx": prompt_idx,
                    "candidate_idx": candidate_idx,
                    "prompt": prompt,
                    "candidate": candidate,
                    "method": method_name
                }
                rows.append(row)
    return pd.DataFrame(rows)

baseline_df = process_results(baseline_res_iter, "baseline")
highperpSample_df = process_results(highperpSample_res_iter, "highPerplexitySampling")
refSample_df = process_results(refSample_res_iter, "referenceSampling")
lowperpSample_df = process_results(lowperpSample_res_iter, "lowPerplexitySampling")

final_df = pd.concat([baseline_df, highperpSample_df, refSample_df, lowperpSample_df], ignore_index=True)

model_name = "facebook/opt-2.7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

# def calculate_log_probability(prompt, completion, tokenizer, model):
#     """Calculate the log probability of a completion given a prompt."""
#     full_text = prompt + completion
#     tokens = tokenizer(full_text, return_tensors="pt")
#     input_ids = tokens.input_ids
#     attention_mask = tokens.attention_mask
#     prompt_length = len(tokenizer(prompt)["input_ids"])
#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         logits = outputs.logits
#     log_probs = torch.log_softmax(logits, dim=-1)
#     completion_ids = input_ids[0, prompt_length:]  # Get the IDs for completion
#     completion_log_probs = log_probs[0, torch.arange(prompt_length, input_ids.size(1) - 1), completion_ids[:-1]]
#     total_log_prob = completion_log_probs.sum().item()  # Sum log probabilities for the completion
#     return total_log_prob
def calculate_log_probability(prompt, completion, tokenizer, model, device):
    """Calculate the log probability of a completion given a prompt."""
    full_text = prompt + completion
    tokens = tokenizer(full_text, return_tensors="pt")
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    prompt_length = len(tokenizer(prompt)["input_ids"])
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)
    completion_ids = input_ids[0, prompt_length:]
    positions = torch.arange(prompt_length, input_ids.size(1) - 1, device=device)
    completion_ids_for_probs = completion_ids[:-1]
    completion_log_probs = log_probs[0, positions, completion_ids_for_probs]
    return completion_log_probs.sum().item()

log_probs = []
for _, row in tqdm(final_df.iterrows(), total=len(final_df)):
    prompt = row["prompt"]
    candidate = row["candidate"]
    log_prob = calculate_log_probability(prompt, candidate, tokenizer, model, device)
    log_probs.append(log_prob)

final_df["log_prob"] = log_probs
final_df.to_csv(f"{out_dir}holdout_results_with_log_probs.csv", index=False)



## Embeddings
# load roberta model 
print("Loading embedding model and tokenizer...")
embedding_model_name = "roberta-base"
embedding_tokenizer = RobertaTokenizer.from_pretrained(embedding_model_name)
embedding_model = RobertaModel.from_pretrained(embedding_model_name).to(device)
embedding_model.eval()  # Set embedding model to evaluation mode

import pandas as pd
from sklearn.decomposition import PCA
import umap.umap_ as umap

def get_embeddings(sentences, batch_size=32):
    """
    Compute normalized mean-pooled embeddings from RoBERTa using batching and GPU acceleration.
    Args:
        sentences (List[str]): List of sentences to embed.
        batch_size (int): Number of sentences to process in each batch.
    Returns:
        np.ndarray: Array of normalized embeddings.
    """
    embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Computing Embeddings"):
        batch_sentences = sentences[i:i + batch_size]
        encoded_input = embedding_tokenizer(
            batch_sentences,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            outputs = embedding_model(**encoded_input)
        last_hidden_states = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
        attention_mask = encoded_input['attention_mask']  # (batch_size, seq_length)
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask  # (batch_size, hidden_size)
        # Normalize embeddings
        mean_pooled = mean_pooled / mean_pooled.norm(p=2, dim=1, keepdim=True)
        embeddings.append(mean_pooled.cpu().numpy())
    embeddings_np = np.vstack(embeddings)
    return embeddings_np

def get_pca_umap_embeddings_iter(embeddings, name, embedding_model, embedding_tokenizer, device):
    # Extract synthetic data for the given iteration
    pca = PCA(n_components=2, random_state=42)
    pca_embeddings = pca.fit_transform(embeddings)
    reducer = umap.UMAP(n_neighbors=100, min_dist=0.2, random_state=42, n_components=2)
    umap_embeddings = reducer.fit_transform(embeddings)
    df = pd.DataFrame({
        'pca_1': pca_embeddings[:, 0],
        'pca_2': pca_embeddings[:, 1],
        'umap_1': umap_embeddings[:, 0],
        'umap_2': umap_embeddings[:, 1],
        'name': name
    })
    return df

def get_pca_umap_embeddings(embeddings, name):
    df = get_pca_umap_embeddings_iter(
        embeddings=embeddings,
        name=name,
        embedding_model=embedding_model,
        embedding_tokenizer=embedding_tokenizer,
        device=device
    )
    return df

reference_data = random.sample(train_data, int(len(train_data) * 0.15))
reference_embeddings = get_embeddings(reference_data)

def get_embeddings_dimRed(synthetic_data_per_iteration, name, reference_embeddings):
    synthetic_embeddings_per_iteration = {}
    for iteration in range(1, N_ITER + 1):
        print(f"Computing embeddings for synthetic data from Iteration {iteration}...")
        iteration_synthetic = synthetic_data_per_iteration.get(iteration, [])
        synthetic_embeddings = get_embeddings(iteration_synthetic)
        synthetic_embeddings_per_iteration[iteration] = synthetic_embeddings
    labels = []
    embeddings = []
    labels.extend(["Reference"] * len(reference_embeddings))
    embeddings.append(reference_embeddings)
    for iteration in range(1, N_ITER + 1):
        iter_label = f"Iteration {iteration}"
        iter_embeddings = synthetic_embeddings_per_iteration.get(iteration, [])
        if len(iter_embeddings) == 0:
            continue
        labels.extend([iter_label] * len(iter_embeddings))
        embeddings.append(synthetic_embeddings_per_iteration[iteration])
    all_embeddings = np.vstack(embeddings)
    dimRed_embed = get_pca_umap_embeddings(all_embeddings, name)
    # add labels to dimRed_embed
    dimRed_embed["label"] = labels
    return dimRed_embed

with open(f"{out_dir}referenceSampling/synthetic_data_per_iteration.pkl", "rb") as f:
    refSample_synthetic_data_per_iteration = pickle.load(f)

with open(f"{out_dir}baseline/synthetic_data_per_iteration.pkl", "rb") as f:
    baseline_synthetic_data_per_iteration = pickle.load(f)

with open(f"{out_dir}perplexitySampling_inverse/synthetic_data_per_iteration.pkl", "rb") as f:
    highperpSample_synthetic_data_per_iteration = pickle.load(f)

with open(f"{out_dir}perplexitySampling/synthetic_data_per_iteration.pkl", "rb") as f:
    lowperpSample_synthetic_data_per_iteration = pickle.load(f)

baseline_dimRed = get_embeddings_dimRed(baseline_synthetic_data_per_iteration, "baseline", reference_embeddings)
refSample_dimRed = get_embeddings_dimRed(refSample_synthetic_data_per_iteration, "referenceSampling", reference_embeddings)
highperpSample_dimRed = get_embeddings_dimRed(highperpSample_synthetic_data_per_iteration, "highPerplexitySampling", reference_embeddings)
lowperpSample_dimRed = get_embeddings_dimRed(lowperpSample_synthetic_data_per_iteration, "lowPerplexitySampling", reference_embeddings)

# change names
baseline_dimRed["name"] = "baseline"
refSample_dimRed["name"] = "referenceSampling"
highperpSample_dimRed["name"] = "highPerplexitySampling"
lowperpSample_dimRed["name"] = "lowPerplexitySampling"

dimRed_df = pd.concat([baseline_dimRed, refSample_dimRed, highperpSample_dimRed, lowperpSample_dimRed], ignore_index=True)
dimRed_df.to_csv(f"{out_dir}dimRed_embeddings.csv", index=False)