# Robustifying Synthetic Data Generation for LLMs
**Authors:** Ahmad Abdel-Azim, Evan Jiang, Kayla Huang, Erik Wang

*Final Project for CS 2281R*

Synthetic data generation is essential for scaling large language model training, as it reduces the need for expensive, human-labeled datasets. However, recursive training on synthetic data can lead to “model collapse,” characterized by diminished diversity and degraded output quality, particularly in the distributional tails. This paper investigates the collapse phenomenon and proposes a mitigation strategy that minimizes the data distribution shift. We evaluate three approaches: baseline single-candidate generation, perplexity-based sampling, and a novel reference-based sampling method that leverages embeddings to sample synthetic candidates most aligned with a reference distribution. Empirical results demonstrate that reference-based sampling preserves syntactic coherence and quality in synthetic data, helping mitigate collapse compared to the other methods. These findings provide new directions for high-quality synthetic data generation that minimizes model degradation using a statistical approach.
