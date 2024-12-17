import os
import random
import torch
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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

out_dir = "/n/home03/ahmadazim/WORKING/synthData/referenceSampling/"

# -----------------------------
# Configuration and Setup
# -----------------------------
BLOCK_SIZE = 64
N_CAND = 5             # Number of candidates per prompt
N_ITER = 10            # Total number of recursive generations
SEED = 42
BATCH_SIZE = 4
LEARNING_RATE = 5e-5

# Set seeds for reproducibility
random.seed(SEED)
torch.manual_seed(SEED)

# Determine device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Model Loading
# -----------------------------
print("\nLoading language model and tokenizer...")
lm_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").to(device)

print("Loading embedding model and tokenizer...")
embedding_model_name = "roberta-base"
embedding_tokenizer = RobertaTokenizer.from_pretrained(embedding_model_name)
embedding_model = RobertaModel.from_pretrained(embedding_model_name).to(device)
embedding_model.eval()  # Set embedding model to evaluation mode

# -----------------------------
# Data Loading and Splitting
# -----------------------------
print("Loading dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
all_data = dataset['train']['text']

# quality filtering
def is_meaningful(line, 
                  min_tokens=5, 
                  min_alpha_ratio=0.3, 
                  min_alpha_word=3, 
                  markup_pattern=r"^=+\s?.*\s?=+$"):
    """
    Check if a line is meaningful according to several heuristics:
    Args:
        line (str): Input text line.
        min_tokens (int): Minimum number of tokens required.
        min_alpha_ratio (float): Minimum ratio of alphabetic chars to total chars required.
        min_alpha_word (int): Minimum length of at least one alphabetic word.
        markup_pattern (str): Regex pattern to identify lines that are just markup headers.
    Returns:
        bool: True if line passes all filters, False otherwise.
    """
    # Strip whitespace
    line = line.strip()
    if not line:
        return False
    tokens = lm_tokenizer.tokenize(line)
    if len(tokens) < min_tokens:
        return False
    alpha_chars = sum(c.isalpha() for c in line)
    total_chars = len(line)
    if total_chars == 0 or (alpha_chars / total_chars) < min_alpha_ratio:
        return False
    alpha_long_enough = any(len(re.findall(r"[A-Za-z]", t)) >= min_alpha_word for t in tokens)
    if not alpha_long_enough:
        return False
    if re.match(markup_pattern, line):
        return False
    return True

filtered_data = []
for line in all_data:
    if is_meaningful(line):
        filtered_data.append(line)

print(f"Original data size: {len(all_data)}")
print(f"Filtered data size: {len(filtered_data)}")

# some manually-picked good examples
holdout_data = [
    dataset['test']['text'][98], 
    dataset['test']['text'][11],
    dataset['test']['text'][12],
    dataset['test']['text'][17],
    dataset['test']['text'][101],
    dataset['test']['text'][211],
    dataset['test']['text'][311],
    dataset['test']['text'][411],
    dataset['test']['text'][49],
    dataset['test']['text'][67]
]

# Reserve 15% for reference distribution
ref_size = int(len(filtered_data) * 0.15)
reference_data = random.sample(filtered_data, ref_size)
remaining_data = list(set(filtered_data) - set(reference_data))

# Split the remaining 85% into train (80%) and validation (20%)
train_size = int(len(remaining_data) * 0.8)
train_data = remaining_data[:train_size]
val_data = remaining_data[train_size:]

print(f"Total (filtered) data points: {len(filtered_data)}")
print(f"Reference data points (15%): {len(reference_data)}")
print(f"Training data points (68%): {len(train_data)}")
print(f"Validation data points (17%): {len(val_data)}")


# -----------------------------
# Helper Functions
# -----------------------------

def create_sequences(data, block_size):
    """Split text into non-overlapping sequences of length block_size."""
    sequences = []
    for text in data:
        tokenized = lm_tokenizer(text, truncation=True, max_length=block_size, add_special_tokens=False)['input_ids']
        sequences.extend([tokenized[i:i + block_size] for i in range(0, len(tokenized), block_size)])
    return sequences

class LanguageModelingDataset(Dataset):
    """Custom dataset for language modeling."""
    def __init__(self, tokenized_sequences, block_size):
        self.examples = []
        for seq in tokenized_sequences:
            if len(seq) == block_size:
                self.examples.append({
                    "input_ids": torch.tensor(seq[:-1], dtype=torch.long),
                    "labels": torch.tensor(seq[1:], dtype=torch.long)
                })
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]

def fine_tune_model(model, tokenized_sequences, output_dir, block_size):
    """Fine-tune the model using Hugging Face Trainer."""
    train_dataset = LanguageModelingDataset(tokenized_sequences, block_size)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=lm_tokenizer,
        mlm=False,
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=100,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        learning_rate=LEARNING_RATE,
        fp16=True,  # Enable mixed precision if using GPU
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir)

def calculate_perplexity(model, sequences):
    """Calculate perplexity on a set of token sequences."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for seq in sequences:
            input_ids = torch.tensor([seq[:-1]]).to(model.device)
            labels = torch.tensor([seq[1:]]).to(model.device)
            outputs = model(input_ids, labels=labels)
            total_loss += outputs.loss.item()
    avg_loss = total_loss / len(sequences)
    return torch.exp(torch.tensor(avg_loss))

def get_embeddings(sentences, batch_size=16):
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
        del encoded_input
        torch.cuda.empty_cache()
        sum_embeddings = torch.sum(last_hidden_states * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask  # (batch_size, hidden_size)
        # Normalize embeddings
        mean_pooled = mean_pooled / mean_pooled.norm(p=2, dim=1, keepdim=True)
        embeddings.append(mean_pooled.cpu().numpy())
        del last_hidden_states, attention_mask, sum_embeddings, sum_mask, mean_pooled
        torch.cuda.empty_cache()  # Free GPU memory
    embeddings_np = np.vstack(embeddings)
    return embeddings_np

def compute_reference_centroid(ref_texts):
    """
    Compute a single centroid embedding by averaging all reference embeddings.
    Args:
        ref_texts (List[str]): List of reference texts.
    Returns:
        np.ndarray: Normalized centroid embedding.
    """
    ref_embeddings = get_embeddings(ref_texts)
    centroid = ref_embeddings.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    return centroid

def generate_fitb_candidates_batch(model, prompt_sequences, n_cand, batch_size_gen, max_new_tokens=64):
    """
    Generate N_CAND candidates for each prompt in batches.
    Args:
        model: Language model to generate text.
        prompt_sequences (List[str]): List of prompt strings.
        n_cand (int): Number of candidates per prompt.
        batch_size_gen (int): Batch size for generation.
        max_new_tokens (int): Maximum number of tokens to generate.
    Returns:
        List[List[str]]: Nested list containing candidates for each prompt.
    """
    all_candidates = []
    lm_tokenizer.padding_side = 'left'
    for i in tqdm(range(0, len(prompt_sequences), batch_size_gen), desc="Generating Candidates", unit="batch"):
        batch_prompts = prompt_sequences[i:i + batch_size_gen]
        encoded_prompts = lm_tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        with torch.no_grad():
            generated_outputs = model.generate(
                input_ids=encoded_prompts.input_ids,
                attention_mask=encoded_prompts.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                num_return_sequences=n_cand  # Generate n_cand per prompt
            )
        # Each prompt has n_cand generated sequences
        for j in range(len(batch_prompts)):
            candidates = []
            for k in range(n_cand):
                generated_sequence = generated_outputs[j * n_cand + k]
                # Decode only the newly generated tokens
                generated_text = lm_tokenizer.decode(
                    generated_sequence[len(encoded_prompts.input_ids[j]):],
                    skip_special_tokens=True
                )
                candidates.append(generated_text.strip())
            all_candidates.append(candidates)
    return all_candidates

# Evaluate holdout prompts after initial training
def evaluate_holdout_prompts(model, prompts, n_cand=3, max_new_tokens=64):
    model.eval()
    results = []
    batch_size_eval = 2  # Smaller batch size for evaluation
    for i in range(0, len(prompts), batch_size_eval):
        batch_prompts = prompts[i:i+batch_size_eval]
        encoded_inputs = lm_tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=encoded_inputs.input_ids,
                attention_mask=encoded_inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n_cand,
                do_sample=True,
                temperature=1.0
            )
        # Process generated outputs
        # outputs.shape: (batch_size*n_cand, seq_length_generated)
        # We can regroup them into sets of n_cand
        for j in range(len(batch_prompts)):
            prompt_candidates = []
            for k in range(n_cand):
                generated_seq = outputs[j*n_cand + k]
                generated_text = lm_tokenizer.decode(generated_seq[encoded_inputs.input_ids[j].shape[0]:], skip_special_tokens=True)
                prompt_candidates.append(generated_text)
            results.append((batch_prompts[j], prompt_candidates))
    return results

# -----------------------------
# Initial Fine-Tuning on Training Data
# -----------------------------
print("\nCreating training and validation sequences...")
train_sequences = create_sequences(train_data, BLOCK_SIZE)
val_sequences = create_sequences(val_data, BLOCK_SIZE)
holdout_sequences = create_sequences(holdout_data, BLOCK_SIZE)
holdout_prompts = [
    lm_tokenizer.decode(seq, skip_special_tokens=True) for seq in holdout_sequences
]

holdout_results_by_iter = {}
holdout_results_before = evaluate_holdout_prompts(base_model, holdout_prompts, n_cand=3, max_new_tokens=64)
holdout_results_by_iter[0] = holdout_results_before.copy()
print("Holdout evaluation before iterations:")
for prompt, candidates in holdout_results_before[:3]:  # just show a few
    print("Prompt:", prompt)
    for c_i, c in enumerate(candidates, 1):
        print(f"Candidate {c_i}: {c}")

print("Fine-tuning Model 0 on initial training data...")
fine_tune_model(base_model, train_sequences, output_dir=f"{out_dir}model_0", block_size=BLOCK_SIZE)
print("Model 0 fine-tuning complete.")

# -----------------------------
# Compute Reference Centroid Embedding
# -----------------------------
print("\nComputing reference centroid embedding...")
# To avoid memory issues, you might want to sample a subset of reference_data
# For demonstration, we'll use all reference_data
ref_centroid = compute_reference_centroid(reference_data)
print("Reference centroid computed.")

# -----------------------------
# Synthetic Data Generation Loop
# -----------------------------
synthetic_data = train_sequences.copy()  # Initialize synthetic data with training sequences
synthetic_data_per_iteration = {}         # To store synthetic data for each iteration
synthetic_data_per_iteration[0] = synthetic_data.copy()

synthetic_perplexities_by_block = {}  # Store synthetic perplexities for each generation

for iteration in range(1, N_ITER + 1):
    print(f"\n=== Iteration {iteration} ===")
    print("Generating Synthetic Data...")
    # Determine prompt_sequences based on iteration
    if iteration == 1:
        # synthetic_data is a list of token ID sequences
        prompt_sequences = [
            lm_tokenizer.decode(seq, skip_special_tokens=True) for seq in synthetic_data
        ]
    else:
        # synthetic_data is a list of strings
        prompt_sequences = synthetic_data
    # Generate N_CAND candidates for each prompt
    all_candidates = generate_fitb_candidates_batch(
        model=base_model,
        prompt_sequences=prompt_sequences,
        n_cand=N_CAND,
        batch_size_gen=32,
        max_new_tokens=BLOCK_SIZE
    )
    print("Evaluating Cosine Similarity and Sampling Candidates...")
    synthetic_data_iteration = []
    synthetic_block_perplexities = []  # Perplexities for synthetic blocks in this iteration
    for candidates in tqdm(all_candidates, desc="Processing Candidates", unit="prompt"):
        # Filter out candidates that are too short
        valid_candidates = [c for c in candidates if len(lm_tokenizer.encode(c, add_special_tokens=False)) > 1]
        if not valid_candidates:
            continue
        # Compute embeddings for candidates
        candidate_embeddings = get_embeddings(valid_candidates)
        # Compute cosine similarities to reference centroid
        similarities = candidate_embeddings @ ref_centroid  # Dot product since embeddings are normalized
        # Convert similarities to probabilities using softmax
        weights = torch.softmax(torch.tensor(similarities), dim=0).numpy()
        # Sample one candidate based on the computed weights
        chosen_index = random.choices(range(len(valid_candidates)), weights=weights, k=1)[0]
        selected_candidate = valid_candidates[chosen_index]
        synthetic_data_iteration.append(selected_candidate)
        # PERPLEXITY CALCULATION:
        tokenized_candidates = [
            lm_tokenizer.encode(c, add_special_tokens=False) for c in candidates
        ]
        valid_candidates = [tc for tc in tokenized_candidates if len(tc) > 1]
        if not valid_candidates:
            continue  # Skip if all candidates are invalid
        perplexities = [calculate_perplexity(base_model, [tc]) for tc in valid_candidates]
        synthetic_block_perplexities.append(perplexities)  # Save perplexity scores for this block
    synthetic_perplexities_by_block[iteration] = synthetic_block_perplexities  # Save all block perplexities for this generation
    synthetic_data = synthetic_data_iteration.copy()
    synthetic_data_per_iteration[iteration] = synthetic_data.copy()
    print(f"Generated {len(synthetic_data)} synthetic data points in Iteration {iteration}.")
    # Display a few examples
    print("Sample Synthetic Data:")
    for idx, example in enumerate(synthetic_data[:10], 1):
        print(f"Example {idx}: {example}")
    # Fine-tune the language model on the new synthetic data
    print(f"Fine-tuning Model {iteration} on synthetic data...")
    synthetic_tokenized = [
        lm_tokenizer.encode(seq, add_special_tokens=False) for seq in synthetic_data if len(lm_tokenizer.encode(seq, add_special_tokens=False)) > 1
    ]
    fine_tune_model(
        model=base_model,
        tokenized_sequences=synthetic_tokenized,
        output_dir=f"{out_dir}model_{iteration}",
        block_size=BLOCK_SIZE
    )
    # Evaluate holdout prompts
    holdout_results = evaluate_holdout_prompts(base_model, holdout_prompts, n_cand=3, max_new_tokens=64)
    holdout_results_by_iter[iteration] = holdout_results.copy()
    print(f"\nHoldout evaluation after iteration {iteration}:")
    for prompt, candidates in holdout_results[:3]:  # just show a few
        print("Prompt:", prompt)
        for c_i, c in enumerate(candidates, 1):
            print(f"Candidate {c_i}: {c}")
    print(f"Model {iteration} fine-tuning complete.")
    # Cleanup
    del all_candidates, synthetic_data_iteration, candidate_embeddings
    torch.cuda.empty_cache()

print("\nSynthetic Data Generation Pipeline Complete!")

# save important objects (synthetic data at each generation, reference data, etc.)
import pickle
with open(f"{out_dir}synthetic_data_per_iteration.pkl", "wb") as f:
    pickle.dump(synthetic_data_per_iteration, f)
with open(f"{out_dir}reference_data.pkl", "wb") as f:
    pickle.dump(reference_data, f)
with open(f"{out_dir}ref_centroid.pkl", "wb") as f:
    pickle.dump(ref_centroid, f)
with open(f"{out_dir}synthetic_perplexities_by_block.pkl", "wb") as f:
    pickle.dump(synthetic_perplexities_by_block, f)
with open(f"{out_dir}holdout_results_per_iteration.pkl", "wb") as f:
    pickle.dump(holdout_results_by_iter, f)

# -----------------------------
# UMAP Visualization
# -----------------------------
print("\nPreparing data for UMAP visualization...")

# Collect embeddings for reference data
print("Computing embeddings for reference data...")
reference_embeddings = get_embeddings(reference_data)

# Collect embeddings for synthetic data from each iteration
synthetic_embeddings_per_iteration = {}
for iteration in range(1, N_ITER + 1):
    print(f"Computing embeddings for synthetic data from Iteration {iteration}...")
    iteration_synthetic = synthetic_data_per_iteration.get(iteration, [])
    synthetic_embeddings = get_embeddings(iteration_synthetic)
    synthetic_embeddings_per_iteration[iteration] = synthetic_embeddings

# Prepare data for UMAP
print("Aggregating embeddings for UMAP...")
labels = []
embeddings = []

# Add reference data
labels.extend(["Reference"] * len(reference_embeddings))
embeddings.append(reference_embeddings)

# Add synthetic data from each iteration
for iteration in range(1, N_ITER + 1):
    iter_label = f"Iteration {iteration}"
    iter_embeddings = synthetic_embeddings_per_iteration.get(iteration, [])
    if len(iter_embeddings) == 0:
        continue
    labels.extend([iter_label] * len(iter_embeddings))
    embeddings.append(synthetic_embeddings_per_iteration[iteration])

# Concatenate all embeddings
all_embeddings = np.vstack(embeddings)

print("Running UMAP dimensionality reduction...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED)
umap_embeddings = reducer.fit_transform(all_embeddings)
print("Generating UMAP plot...")
plt.figure(figsize=(12, 8))
unique_labels = sorted(list(set(labels)))
palette = sns.color_palette("hsv", len(unique_labels))
label_color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

for label in unique_labels:
    idxs = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(
        umap_embeddings[idxs, 0],
        umap_embeddings[idxs, 1],
        c=[label_color_map[label]],
        label=label,
        s=10,
        alpha=0.6
    )

plt.title("UMAP Visualization of Reference and Synthetic Data Embeddings")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(markerscale=2, fontsize='small', loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{out_dir}umap_visualization.png", dpi=300)
plt.show()
print("UMAP visualization saved as 'umap_visualization.png'.")

## Trying PCA: 
print("Running PCA dimensionality reduction...")
pca = PCA(n_components=2, random_state=SEED)
pca_embeddings = pca.fit_transform(all_embeddings)

print("Generating PCA plot...")
plt.figure(figsize=(12, 8))

unique_labels = sorted(list(set(labels)))
palette = sns.color_palette("hsv", len(unique_labels))
label_color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

for label in unique_labels:
    idxs = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(
        pca_embeddings[idxs, 0],
        pca_embeddings[idxs, 1],
        c=[label_color_map[label]],
        label=label,
        s=10,
        alpha=0.6
    )

plt.title("PCA Visualization of Reference and Synthetic Data Embeddings")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.legend(markerscale=2, fontsize='small', loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{out_dir}pca_visualization.png", dpi=300)
plt.show()
print("PCA visualization saved as 'pca_visualization.png'.")

# -----------------------------
# Perplexity Visualization
# -----------------------------
import matplotlib
matplotlib.use("Agg")

plt.figure(figsize=(8, 4))
generations = list(synthetic_perplexities_by_block.keys())
for generation in generations:
    flattened_perplexities = [
        p.item() for block in synthetic_perplexities_by_block[generation] for p in block
    ]
    sns.kdeplot(
        flattened_perplexities,
        label=f"Generation {generation}",
        fill=True,
        alpha=0.3
    )

plt.title("Perplexity of Generated Data Points by Model Generations")
plt.xlabel("Perplexity of Generated Data Points")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.savefig(f"{out_dir}synthetic_perplexity.png")


