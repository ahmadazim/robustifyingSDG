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

out_dir = "/n/home03/ahmadazim/WORKING/synthData/baseline/"

BLOCK_SIZE = 64
N_CAND = 5   # Number of FITB candidates per block
N_ITER = 10  # Total number of recursive model generations
SEED = 42
BATCH_SIZE = 4
LEARNING_RATE = 5e-5

random.seed(SEED)
torch.manual_seed(SEED)

# Determine device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load roberta model 
print("Loading embedding model and tokenizer...")
embedding_model_name = "roberta-base"
embedding_tokenizer = RobertaTokenizer.from_pretrained(embedding_model_name)
embedding_model = RobertaModel.from_pretrained(embedding_model_name).to(device)
embedding_model.eval()  # Set embedding model to evaluation mode

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# Load Wikitext2 dataset
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
    tokens = tokenizer.tokenize(line)
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

train_size = int(len(filtered_data) * 0.8)
train_data = filtered_data[:train_size]
val_data = filtered_data[train_size:]

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

# # Take 10% of the data
# subset_size = int(len(all_data) * 1)
# subset_data = random.sample(all_data, subset_size)  # Randomly sample 10% of the data

# # Split the 10% subset into train (80%) and validation (20%)
# train_size = int(len(subset_data) * 0.8)
# train_data = subset_data[:train_size]
# val_data = subset_data[train_size:]


def create_sequences(data, block_size):
    """Split text into non-overlapping sequences of length block_size."""
    sequences = []
    for text in data:
        tokenized = tokenizer(text, truncation=True, max_length=block_size, add_special_tokens=False)['input_ids']
        sequences.extend([tokenized[i:i + block_size] for i in range(0, len(tokenized), block_size)])
    return sequences


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


def generate_fitb_candidates(model, prompt_seq, n_cand, max_new_tokens=BLOCK_SIZE):
    """Generate Fill-in-the-Blank (FITB) candidates for a given prompt."""
    model.eval()
    candidates = []
    prompt_input = torch.tensor([prompt_seq]).to(model.device)
    with torch.no_grad():
        for _ in range(n_cand):
            outputs = model.generate(
                prompt_input,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0
            )
            candidates.append(outputs[0].tolist())
    return candidates


def generate_fitb_candidates_batch(model, prompt_sequences, n_cand, batch_size, max_new_tokens=64):
    """
    Generate Fill-in-the-Blank (FITB) candidates in batches for a list of prompt sequences.
    """
    model.eval()
    all_candidates = []
    tokenizer.padding_side = 'left'
    # Process the prompts in batches
    for i in range(0, len(prompt_sequences), batch_size):
        batch_prompts = prompt_sequences[i:i + batch_size]
        prompt_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        batch_candidates = []
        with torch.no_grad():
            for _ in range(n_cand):
                outputs = model.generate(
                    input_ids=prompt_inputs.input_ids,
                    attention_mask=prompt_inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0
                )
                # Extract only the new tokens by slicing out the input length
                new_tokens_start = prompt_inputs.input_ids.shape[1]  # Length of original input
                decoded_outputs = [
                    tokenizer.decode(output[new_tokens_start:], skip_special_tokens=True)
                    for output in outputs
                ]
                batch_candidates.append(decoded_outputs)
        # Reshape the results: one list of candidates per prompt
        for j, candidates in enumerate(zip(*batch_candidates)):
            all_candidates.append(list(candidates))  # List of candidates for the j-th prompt
    return all_candidates


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

# def sample_synthetic_data(candidates, perplexity_scores):
#     """Sample synthetic data proportional to perplexity scores."""
#     probabilities = torch.softmax(torch.tensor([-p for p in perplexity_scores]), dim=0).numpy()
#     sampled_indices = random.choices(range(len(candidates)), weights=probabilities, k=1)
#     return candidates[sampled_indices[0]]
def sample_synthetic_data(candidates, perplexity_scores):
    """Sample synthetic data proportional to perplexity scores."""
    # Ensure all perplexity scores are finite
    valid_indices = [i for i, p in enumerate(perplexity_scores) if torch.isfinite(torch.tensor(p))]
    if not valid_indices:  # If no valid perplexities exist, return a random candidate
        return random.choice(candidates)
    valid_candidates = [candidates[i] for i in valid_indices]
    # valid_scores = [perplexity_scores[i] for i in valid_indices]
    # probabilities = torch.softmax(torch.tensor([p for p in valid_scores]), dim=0).numpy()
    # sampled_index = random.choices(range(len(valid_candidates)), weights=probabilities, k=1)[0]
    # return valid_candidates[sampled_index]
    probabilities = [1/len(valid_candidates)] * len(valid_candidates)
    sampled_index = random.choices(range(len(valid_candidates)), weights=probabilities, k=1)[0]
    return valid_candidates[sampled_index]


class LanguageModelingDataset(Dataset):
    """
    Custom dataset for language modeling.
    """
    def __init__(self, tokenized_sequences, block_size, tokenizer):
        self.examples = []
        self.block_size = block_size
        self.tokenizer = tokenizer
        # Prepare input_ids and labels from tokenized sequences
        for seq in tokenized_sequences:
            # Ensure sequences are exactly block_size length
            if len(seq) == block_size:
                self.examples.append({
                    "input_ids": torch.tensor(seq[:-1], dtype=torch.long),
                    "labels": torch.tensor(seq[1:], dtype=torch.long)
                })
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]


def fine_tune_model(model, tokenized_sequences, output_dir, tokenizer, block_size):
    """
    Fine-tune the model using the Hugging Face Trainer with a custom dataset.
    """
    # Create the dataset
    train_dataset = LanguageModelingDataset(tokenized_sequences, block_size, tokenizer)
    # Define DataCollator for dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're training a causal language model
    )
    # Define training arguments
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
    )
    # Create a Trainer and fine-tune the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir)

# Evaluate holdout prompts after initial training
def evaluate_holdout_prompts(model, prompts, n_cand=3, max_new_tokens=64):
    model.eval()
    results = []
    batch_size_eval = 2  # Smaller batch size for evaluation
    for i in range(0, len(prompts), batch_size_eval):
        batch_prompts = prompts[i:i+batch_size_eval]
        encoded_inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True).to(model.device)
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
                generated_text = tokenizer.decode(generated_seq[encoded_inputs.input_ids[j].shape[0]:], skip_special_tokens=True)
                prompt_candidates.append(generated_text)
            results.append((batch_prompts[j], prompt_candidates))
    return results

# Model 0 Training (on Wikitext2)
print("Training Model 0...")
train_sequences = create_sequences(train_data, BLOCK_SIZE)
val_sequences = create_sequences(val_data, BLOCK_SIZE)
holdout_sequences = create_sequences(holdout_data, BLOCK_SIZE)
holdout_prompts = [
    tokenizer.decode(seq, skip_special_tokens=True) for seq in holdout_sequences
]

holdout_results_by_iter = {}
holdout_results_before = evaluate_holdout_prompts(base_model, holdout_prompts, n_cand=3, max_new_tokens=64)
holdout_results_by_iter[0] = holdout_results_before.copy()
print("Holdout evaluation before iterations:")
for prompt, candidates in holdout_results_before[:3]:  # just show a few
    print("Prompt:", prompt)
    for c_i, c in enumerate(candidates, 1):
        print(f"Candidate {c_i}: {c}")


synthetic_perplexities_by_block = {}  # Store synthetic perplexities for each generation
real_perplexities_by_block = {}  # Store real perplexities for each generation

# Prepare dataset for Hugging Face Trainer
train_dataset = [{"input_ids": torch.tensor(seq[:-1]), "labels": torch.tensor(seq[1:])} for seq in train_sequences]
fine_tune_model(base_model, train_sequences, output_dir=f"{out_dir}model_0", tokenizer=tokenizer, block_size=BLOCK_SIZE)

# Recursive Generations
synthetic_data = train_sequences  # Start with training data as input
synthetic_data_per_iteration = {}
synthetic_data_per_iteration[0] = synthetic_data.copy()

for iteration in range(1, N_ITER + 1):
    print(f"\nIteration {iteration}: Generating Synthetic Data and Fine-Tuning Model {iteration}")
    # Step 1: Generate Synthetic Data with Parallel Batch Inference
    print("Generating Synthetic Data...")
    batch_size = 64  # Adjust based on available GPU/CPU memory
    synthetic_data_iteration = []
    # Prepare prompt sequences
    if iteration == 1:
        prompt_sequences = [tokenizer.decode(seq, skip_special_tokens=True) for seq in synthetic_data]
    elif iteration > 1:
        synthetic_data_tokenized = [
            tokenizer.encode(seq, add_special_tokens=False) for seq in synthetic_data
        ]
        prompt_sequences = [tokenizer.decode(seq, skip_special_tokens=True) for seq in synthetic_data_tokenized]
    # Generate FITB candidates in parallel with a progress bar
    all_candidates = []
    for i in tqdm(range(0, len(prompt_sequences), batch_size), desc="Generating Candidates", unit="batch"):
        batch_prompts = prompt_sequences[i:i + batch_size]
        batch_candidates = generate_fitb_candidates_batch(base_model, batch_prompts, N_CAND, batch_size)
        all_candidates.extend(batch_candidates)
    # Evaluate and sample candidates with a progress bar
    synthetic_block_perplexities = []  # Perplexities for synthetic blocks in this iteration
    for candidates in tqdm(all_candidates, desc="Sampling Candidates", unit="prompt"):
        tokenized_candidates = [
            tokenizer.encode(c, add_special_tokens=False) for c in candidates
        ]
        # Filter out candidates of length 1
        valid_candidates = [tc for tc in tokenized_candidates if len(tc) > 1]
        if not valid_candidates:
            continue  # Skip if all candidates are invalid
        # Calculate perplexity only for valid candidates
        perplexities = [calculate_perplexity(base_model, [tc]) for tc in valid_candidates]
        selected_candidate = sample_synthetic_data(candidates, perplexities)
        synthetic_data_iteration.append(selected_candidate)
        synthetic_block_perplexities.append(perplexities)  # Save perplexity scores for this block
    synthetic_perplexities_by_block[iteration] = synthetic_block_perplexities  # Save all block perplexities for this generation
    synthetic_data = synthetic_data_iteration
    synthetic_data_per_iteration[iteration] = synthetic_data.copy()
    print(f"Generated {len(synthetic_data)} synthetic data points.")
    print(f"A few examples:")
    for i in range(10):
        print(f"Example {i + 1}: {synthetic_data[i]}")
    # Step 2: Evaluate Real Data Perplexities
    print("Evaluating Perplexities on Real Data...")
    real_block_perplexities = []
    for seq in tqdm(val_sequences, desc="Evaluating Real Data Perplexities", unit="block"):
        tokenized_seq = tokenizer.encode(tokenizer.decode(seq, skip_special_tokens=True), add_special_tokens=False)
        if len(tokenized_seq) > 1:  # Skip sequences of length 1
            perplexity = calculate_perplexity(base_model, [tokenized_seq])  # Per-block perplexity
            real_block_perplexities.append(perplexity)
    real_perplexities_by_block[iteration] = real_block_perplexities  # Save real block perplexities for this generation
    # Step 3: Fine-tune the model on the aggregated synthetic data
    print("Fine-tuning the model...")
    synthetic_tokenized = [
        tokenizer.encode(seq, add_special_tokens=False) for seq in synthetic_data if len(seq) > 1
    ]
    synthetic_dataset = [{"input_ids": torch.tensor(seq[:-1]), "labels": torch.tensor(seq[1:])} for seq in synthetic_tokenized]
    fine_tune_model(base_model, synthetic_tokenized, output_dir=f"{out_dir}model_{iteration}", tokenizer=tokenizer, block_size=BLOCK_SIZE)
    # Step 4: Evaluate holdout prompts
    holdout_results = evaluate_holdout_prompts(base_model, holdout_prompts, n_cand=3, max_new_tokens=64)
    holdout_results_by_iter[iteration] = holdout_results.copy()
    print(f"\nHoldout evaluation after iteration {iteration}:")
    for prompt, candidates in holdout_results[:3]:  # just show a few
        print("Prompt:", prompt)
        for c_i, c in enumerate(candidates, 1):
            print(f"Candidate {c_i}: {c}")
    del all_candidates, synthetic_data_iteration, synthetic_block_perplexities, real_block_perplexities, synthetic_tokenized
    torch.cuda.empty_cache()

print("Pipeline complete!")

# save synthetic_perplexities_by_block
import pickle
with open(f"{out_dir}synthetic_perplexities_by_block.pkl", "wb") as f:
    pickle.dump(synthetic_perplexities_by_block, f)
with open(f"{out_dir}real_perplexities_by_block.pkl", "wb") as f:
    pickle.dump(real_perplexities_by_block, f)
with open(f"{out_dir}synthetic_data_per_iteration.pkl", "wb") as f:
    pickle.dump(synthetic_data_per_iteration, f)
with open(f"{out_dir}holdout_results_per_iteration.pkl", "wb") as f:
    pickle.dump(holdout_results_by_iter, f)

# # read 
# import pickle
# with open("synthetic_perplexities_by_block.pkl", "rb") as f:
#     synthetic_perplexities_by_block = pickle.load(f)

# with open("real_perplexities_by_block.pkl", "rb") as f:
#     real_perplexities_by_block = pickle.load(f)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Convert tensor values to floats and plot histograms of synthetic perplexity scores for each generation
import matplotlib
matplotlib.use("Agg")

plt.figure(figsize=(8, 4))
generations = list(synthetic_perplexities_by_block.keys())  # List of generations
for generation in generations:
    # Flatten the perplexity scores for this generation and convert tensors to floats
    flattened_perplexities = [
        p.item() for block in synthetic_perplexities_by_block[generation] for p in block
        # p.item() for p in real_perplexities_by_block[generation]
    ]
    # Plot histogram (density plot)
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


# Collect embeddings for reference data
# reference data = 15% of the training data
print("Computing embeddings for reference data...")
reference_data = random.sample(train_data, int(len(train_data) * 0.15))
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
